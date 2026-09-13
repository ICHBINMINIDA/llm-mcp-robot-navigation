#!/usr/bin/env python3
"""Shared robot layer for architectures A, A1, A2, B and C.

This module is the SINGLE SOURCE OF TRUTH for everything that must be identical
across the five architectures, so that the only independent variable in the
experiment is the role given to the LLM:

* ROS 2 camera access, frame sequence numbers and frame-freshness rules
* the SAM perception pipeline (resize, JPEG quality, mask selection, geometry)
* the decision thresholds (left/right bands, camera stop condition)
* the kinematic constants (linear/angular speed)
* velocity execution: continuous motion, timed motion, and a motion watchdog
* structured JSONL metrics used for the evaluation chapter

Safety notes
------------
1. Every long running MCP tool must be declared ``async def`` and must offload
   the blocking work with :func:`run_blocking`.  A synchronous FastMCP tool is
   executed directly inside the server event loop, which blocks the whole
   session and makes ``stop_robot`` undeliverable while the robot is moving.
2. Continuous motion is protected by a dead-man watchdog: if no new motion
   command arrives within ``MAX_CONTINUOUS_MOTION_SEC`` the node stops the
   robot by itself, so a crashed client cannot leave the robot driving.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import anyio
import cv2
import numpy as np
import requests
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


# --------------------------------------------------------------------------- #
# configuration (identical defaults for every architecture)
# --------------------------------------------------------------------------- #
def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# public aliases, so the architecture files read their own extra parameters
# through the same helpers instead of re-implementing env parsing
env_float, env_int, env_bool = _f, _i, _b


# --- interfaces ------------------------------------------------------------ #
SAM_SERVICE_URL = os.getenv("SAM_SERVICE_URL", "http://localhost:15000/segment")
CAMERA_TOPIC = os.getenv("CAMERA_TOPIC", "camera/image")
CMD_VEL_TOPIC = os.getenv("CMD_VEL_TOPIC", "/cmd_vel")
OUTPUT_DIR = os.getenv("MCP_ROBOT_OUTPUT_DIR", "/tmp/mcp_robot")

# --- kinematics (shared by timed, continuous and skill-level motion) -------- #
LINEAR_SPEED = _f("ROBOT_LINEAR_SPEED", "0.10")
ANGULAR_SPEED = _f("ROBOT_ANGULAR_SPEED", "0.30")
MIN_FORWARD_M = _f("MIN_FORWARD_M", "0.02")
MAX_FORWARD_M = _f("MAX_FORWARD_M", "0.30")
MAX_BACKWARD_M = _f("MAX_BACKWARD_M", "0.15")
MIN_TURN_DEG = _f("MIN_TURN_DEG", "8")      # below the motor dead-band nothing moves
MAX_TURN_DEG = _f("MAX_TURN_DEG", "45")
MAX_MOTION_SEC = _f("MAX_MOTION_SEC", "6.0")
CMD_PUBLISH_HZ = _f("CMD_PUBLISH_HZ", "20")
CONTROL_HZ = _f("CONTROL_HZ", "30")
MAX_CONTINUOUS_MOTION_SEC = _f("MAX_CONTINUOUS_MOTION_SEC", "3.0")

# --- camera / perception ---------------------------------------------------- #
STALE_FRAME_SEC = _f("STALE_FRAME_SEC", "1.5")
FRESH_FRAME_TIMEOUT_SEC = _f("FRESH_FRAME_TIMEOUT_SEC", "0.8")
SETTLE_SEC = _f("SETTLE_SEC", "0.25")
SAM_TIMEOUT_SEC = _f("SAM_TIMEOUT_SEC", "60")
SAM_MAX_SIDE = _i("SAM_MAX_SIDE", "416")
SAM_JPEG_QUALITY = _i("SAM_JPEG_QUALITY", "80")
SAM_EVERY_SEC = _f("SAM_EVERY_SEC", "0.10")
MIRROR_IMAGE = _b("MIRROR_IMAGE", "false")
WRITE_DEBUG_IMAGE = _b("WRITE_DEBUG_IMAGE", "false")

# --- decision thresholds ---------------------------------------------------- #
LEFT_TH = _f("LEFT_TH", "0.40")
RIGHT_TH = _f("RIGHT_TH", "0.60")
STOP_AREA_RATIO = _f("STOP_AREA_RATIO", "0.05")
STOP_BBOX_H_RATIO = _f("STOP_BBOX_H_RATIO", "0.55")

# --- visualization (off while measuring, on for figures and demos) ---------- #
VISUALIZATION_ENABLED = _b("VISUALIZATION_ENABLED", "false")
VISUALIZATION_TOPIC = os.getenv("VISUALIZATION_TOPIC", "/robot_llm/visualization")
VISUALIZATION_HZ = _f("VISUALIZATION_HZ", "5")
VISUALIZATION_COMPRESSED = _b("VISUALIZATION_COMPRESSED", "true")
VISUALIZATION_JPEG_QUALITY = _i("VISUALIZATION_JPEG_QUALITY", "70")

# --- emergency stop flag (third, transport-independent stop layer) ---------- #
STOP_FLAG = os.getenv("ROBOT_STOP_FLAG", "/tmp/robot_llm_stop.flag")

# --- metrics ---------------------------------------------------------------- #
METRICS_PATH = os.getenv("METRICS_PATH", os.path.join(OUTPUT_DIR, "metrics.jsonl"))


def clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def stop_flag_set() -> bool:
    return os.path.exists(STOP_FLAG)


def set_stop_flag() -> None:
    """Raise the emergency flag (used by clients and by external scripts)."""
    try:
        os.makedirs(os.path.dirname(STOP_FLAG) or ".", exist_ok=True)
        with open(STOP_FLAG, "w", encoding="utf-8") as handle:
            handle.write("stop\n")
    except Exception:
        pass


def clear_stop_flag() -> None:
    try:
        os.remove(STOP_FLAG)
    except FileNotFoundError:
        pass
    except Exception:
        pass


async def run_blocking(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking function in a worker thread.

    Mandatory for every MCP tool that takes more than a few milliseconds,
    otherwise the FastMCP event loop is blocked and ``stop_robot`` cannot be
    served while the robot is moving.
    """
    return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class MetricsLogger:
    """Append-only JSONL trace, identical schema for all architectures."""

    def __init__(self, architecture: str, side: str, path: str = METRICS_PATH) -> None:
        self.architecture = architecture
        self.side = side
        self.path = path
        self.run_id = uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def new_run(self) -> str:
        self.run_id = uuid.uuid4().hex[:12]
        return self.run_id

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            "architecture": self.architecture,
            "side": self.side,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:      # metrics must never break the robot loop
            pass


# --------------------------------------------------------------------------- #
# ROS 2 node
# --------------------------------------------------------------------------- #
class RobotNode(Node):
    """Camera subscriber + velocity publisher with watchdog and stop event."""

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        # Visualization is published only when VISUALIZATION_ENABLED is true, so
        # that measurement runs are never charged for rendering or DDS traffic.
        self.viz_pub = None
        self.viz_lock = threading.Lock()
        self.last_viz_frame: Optional[np.ndarray] = None
        self._last_viz_time = 0.0
        if VISUALIZATION_ENABLED:
            if VISUALIZATION_COMPRESSED:
                self.viz_pub = self.create_publisher(
                    CompressedImage, VISUALIZATION_TOPIC + "/compressed", 5)
            else:
                self.viz_pub = self.create_publisher(Image, VISUALIZATION_TOPIC, 5)

        self.frame: Optional[np.ndarray] = None
        self.frame_time: Optional[float] = None
        self.frame_seq: int = 0
        self.frame_lock = threading.Lock()

        self.motion_lock = threading.Lock()
        self._linear = 0.0
        self._angular = 0.0
        self._motion = "stopped"
        self._deadline = 0.0
        self.motion_end_seq = 0

        self.stop_event = threading.Event()
        self._shutdown = threading.Event()

        self.create_subscription(Image, CAMERA_TOPIC, self._image_callback, 10)
        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()

        self.get_logger().info(f"camera={CAMERA_TOPIC} cmd_vel={CMD_VEL_TOPIC} "
                               f"lin={LINEAR_SPEED} ang={ANGULAR_SPEED}")

    # ------------------------------ camera --------------------------------- #
    def _image_callback(self, msg: Image) -> None:
        try:
            if msg.encoding not in {"bgr8", "rgb8"}:
                self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
                return
            expected = int(msg.height) * int(msg.width) * 3
            if len(msg.data) != expected:
                self.get_logger().error(
                    f"Invalid image buffer: got {len(msg.data)}, expected {expected}")
                return
            frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with self.frame_lock:
                self.frame = frame.copy()
                self.frame_time = time.time()
                self.frame_seq += 1
        except Exception as exc:
            self.get_logger().error(f"Failed to convert ROS Image: {exc}")

    def latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[float], int]:
        with self.frame_lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, self.frame_time, self.frame_seq

    def wait_for_new_frame(self, min_seq: int, timeout: float = FRESH_FRAME_TIMEOUT_SEC) -> bool:
        """Block until a frame newer than ``min_seq`` arrives."""
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            with self.frame_lock:
                if self.frame_seq > min_seq:
                    return True
            time.sleep(0.01)
        return False

    def wait_settled(self) -> bool:
        """Wait for the chassis to settle and for a post-motion camera frame.

        Used by the primitive architectures (A and A1) so that the LLM never
        reasons about a picture that was taken before the previous motion
        finished.  Without this the agent oscillates left/right.
        """
        with self.motion_lock:
            reference = self.motion_end_seq
        time.sleep(SETTLE_SEC)
        return self.wait_for_new_frame(reference)

    # ------------------------------ motion --------------------------------- #
    def _publish_once(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def _publish_loop(self) -> None:
        """Refresh the velocity setpoint at a fixed rate and run the watchdog.

        Zeros are published too: a single stop message can be lost, and most
        drivers keep the last command until a new one arrives.
        """
        period = 1.0 / max(CMD_PUBLISH_HZ, 1.0)
        while not self._shutdown.is_set():
            if stop_flag_set():
                # Third stop layer: works even if the MCP transport is busy or
                # the client process is gone.
                with self.motion_lock:
                    active = self._motion != "stopped"
                if active:
                    self.get_logger().warn("Emergency stop flag detected: stopping robot")
                    self.stop(reason="stop_flag")
                self._publish_once(0.0, 0.0)
                time.sleep(period)
                continue

            expired = False
            with self.motion_lock:
                linear, angular = self._linear, self._angular
                moving = self._motion != "stopped"
                if moving and time.time() > self._deadline:
                    expired = True
                    linear = angular = 0.0
            if expired:
                self.get_logger().warn("Continuous motion watchdog expired: stopping robot")
                self.stop(reason="watchdog")
            else:
                self._publish_once(linear, angular)
            time.sleep(period)

    def drive(self, linear: float, angular: float, motion: str = "servo") -> None:
        """Set the velocity setpoint and refresh the watchdog deadline."""
        with self.motion_lock:
            self._linear = float(linear)
            self._angular = float(angular)
            self._motion = "stopped" if (linear == 0.0 and angular == 0.0) else motion
            self._deadline = time.time() + MAX_CONTINUOUS_MOTION_SEC
        self._publish_once(linear, angular)

    def start_continuous(self, motion: str, linear: float, angular: float) -> Dict[str, Any]:
        self.clear_stop()
        self.drive(linear, angular, motion)
        return {
            "ok": True,
            "motion": motion,
            "mode": "continuous",
            "linear_x": round(float(linear), 4),
            "angular_z": round(float(angular), 4),
            "watchdog_sec": MAX_CONTINUOUS_MOTION_SEC,
        }

    def run_timed(self, linear: float, angular: float, duration: float, motion: str) -> Dict[str, Any]:
        """Blocking timed motion. Call it through :func:`run_blocking`."""
        duration = clamp(duration, 0.0, MAX_MOTION_SEC)
        self.clear_stop()
        start = time.monotonic()
        interrupted = False
        try:
            self.drive(linear, angular, motion)
            while time.monotonic() - start < duration:
                if self.stop_requested():
                    interrupted = True
                    break
                # keep the watchdog alive while the motion is legitimately running
                self.drive(linear, angular, motion)
                time.sleep(1.0 / max(CMD_PUBLISH_HZ, 1.0))
        finally:
            self.stop(reason="timed_motion_end")
        return {
            "mode": "timed",
            "elapsed_sec": round(time.monotonic() - start, 3),
            "interrupted": interrupted,
        }

    def stop(self, reason: str = "stop_robot") -> None:
        self.stop_event.set()
        with self.motion_lock:
            self._linear = 0.0
            self._angular = 0.0
            self._motion = "stopped"
            self._deadline = 0.0
        self._publish_once(0.0, 0.0)
        with self.frame_lock:
            seq = self.frame_seq
        with self.motion_lock:
            self.motion_end_seq = seq
        self.get_logger().debug(f"stop({reason})")

    def clear_stop(self) -> None:
        self.stop_event.clear()
        clear_stop_flag()

    def stop_requested(self) -> bool:
        """True if a stop was requested through EITHER layer.

        Layer 1: ``stop_robot`` sets ``stop_event`` (fast, in-process).
        Layer 2: a client raises the flag file (works even while the server is
        busy, and survives a broken stdio pipe).
        Layer 3: the motion watchdog in the publisher thread.
        """
        if self.stop_event.is_set():
            return True
        if stop_flag_set():
            self.stop(reason="stop_flag")
            return True
        return False

    def refresh_watchdog(self) -> Dict[str, Any]:
        """Extend the dead-man deadline of the motion that is already running.

        The cached-decision client uses this as a keep-alive: it proves that the
        client is still supervising the robot without changing the command.
        """
        with self.motion_lock:
            active = self._motion != "stopped"
            if active:
                self._deadline = time.time() + MAX_CONTINUOUS_MOTION_SEC
            return {"ok": True, "moving": active, "motion": self._motion,
                    "linear_x": round(self._linear, 4), "angular_z": round(self._angular, 4)}

    def is_moving(self) -> bool:
        with self.motion_lock:
            return self._motion != "stopped"

    # --------------------------- visualization ------------------------------ #
    def visualization_due(self) -> bool:
        """True when the next visualization frame may be published."""
        if self.viz_pub is None:
            return False
        return (time.monotonic() - self._last_viz_time) >= 1.0 / max(VISUALIZATION_HZ, 0.1)

    def publish_visualization(self, frame: Optional[np.ndarray], force: bool = False) -> bool:
        """Publish an annotated BGR frame, rate-limited to VISUALIZATION_HZ."""
        if self.viz_pub is None or frame is None or frame.ndim != 3:
            return False
        now = time.monotonic()
        if not force and (now - self._last_viz_time) < 1.0 / max(VISUALIZATION_HZ, 0.1):
            return False
        self._last_viz_time = now
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        with self.viz_lock:
            self.last_viz_frame = frame.copy()
        try:
            if VISUALIZATION_COMPRESSED:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), VISUALIZATION_JPEG_QUALITY])
                if not ok:
                    return False
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "robot_llm_camera"
                msg.format = "jpeg"
                msg.data = buf.tobytes()
            else:
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "robot_llm_camera"
                msg.height, msg.width = int(frame.shape[0]), int(frame.shape[1])
                msg.encoding = "bgr8"
                msg.is_bigendian = 0
                msg.step = int(frame.shape[1] * 3)
                msg.data = frame.tobytes()
            self.viz_pub.publish(msg)
            return True
        except Exception as exc:
            self.get_logger().debug(f"visualization publish failed: {exc}")
            return False

    def publish_banner(self, state: str, message: str = "") -> None:
        """Re-publish the last visualization with a new state banner."""
        if self.viz_pub is None:
            return
        with self.viz_lock:
            base = None if self.last_viz_frame is None else self.last_viz_frame.copy()
        if base is None:
            base, _, _ = self.latest_frame()
        if base is None:
            return
        draw_state_banner(base, state, message)
        self.publish_visualization(base, force=True)

    def shutdown(self) -> None:
        self.stop(reason="shutdown")
        self._shutdown.set()


def start_robot(node_name: str) -> RobotNode:
    """Initialise rclpy, build the node and spin it in a background thread."""
    rclpy.init()
    node = RobotNode(node_name)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    return node


def shutdown_robot(node: RobotNode) -> None:
    try:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# perception (identical for every architecture)
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    found: bool
    object: str
    direction: Optional[str] = None
    centroid_norm_x: Optional[float] = None
    area_ratio: Optional[float] = None
    bbox_h_ratio: Optional[float] = None
    bbox: Optional[Dict[str, int]] = None
    centroid: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    debug_image_path: Optional[str] = None
    frame_seq: Optional[int] = None
    frame_age_sec: Optional[float] = None
    sam_latency_sec: Optional[float] = None

    def compact(self) -> Dict[str, Any]:
        """Small payload handed to the LLM. Keys are identical everywhere."""
        return {
            "found": self.found,
            "object": self.object,
            "direction": self.direction,
            "centroid_norm_x": self.centroid_norm_x,
            "area_ratio": self.area_ratio,
            "bbox_h_ratio": self.bbox_h_ratio,
            "error": self.error,
        }

    def full(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}

    @property
    def close_enough(self) -> bool:
        return (float(self.area_ratio or 0.0) >= STOP_AREA_RATIO
                or float(self.bbox_h_ratio or 0.0) >= STOP_BBOX_H_RATIO)


def draw_state_banner(frame: np.ndarray, state: str, message: str = "") -> None:
    state = str(state).upper().strip() or "IDLE"
    height = 36 if not message else 62
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, height), (20, 20, 20), -1)
    cv2.putText(frame, f"STATE: {state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    if message:
        cv2.putText(frame, str(message)[:80], (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (220, 220, 220), 1)


def render_visualization(frame: np.ndarray, object_name: str, state: str,
                         detection: Optional["Detection"] = None, mask_data: Any = None,
                         scale: float = 1.0, message: str = "") -> np.ndarray:
    """Annotated view of what the robot currently perceives and does.

    Used for the figures and the demonstration video. It is never called while
    ``VISUALIZATION_ENABLED`` is false, so it cannot affect the measurements.
    """
    draw = frame.copy()
    if detection is not None and detection.found and detection.bbox and detection.centroid:
        if mask_data is not None:
            mask = np.asarray(mask_data, np.uint8)
            if scale != 1.0:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            selected = mask > 0
            if selected.shape == draw.shape[:2]:
                overlay = draw.copy()
                overlay[selected] = (0, 255, 0)
                draw = cv2.addWeighted(overlay, 0.42, draw, 0.58, 0)
        box, center = detection.bbox, detection.centroid
        cv2.rectangle(draw, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (255, 120, 0), 2)
        cv2.circle(draw, (center["x"], center["y"]), 6, (0, 0, 255), -1)
        cv2.putText(draw, f"{object_name} | {detection.direction or '-'}",
                    (box["x1"], max(82, box["y1"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
    elif object_name:
        cv2.putText(draw, f"Target: {object_name}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

    draw_state_banner(draw, state, message)

    if detection is not None:
        area = "-" if detection.area_ratio is None else f"{detection.area_ratio:.4f}"
        bbox_h = "-" if detection.bbox_h_ratio is None else f"{detection.bbox_h_ratio:.3f}"
        lines = [f"Target: {object_name}",
                 f"Direction: {detection.direction or 'not visible'}  "
                 f"Area: {area}  BBox-H: {bbox_h}"]
        base_y = draw.shape[0] - 54
        for index, text in enumerate(lines):
            cv2.putText(draw, text, (10, base_y + index * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    return draw


def _resize_for_sam(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    side = max(h, w)
    if SAM_MAX_SIDE <= 0 or side <= SAM_MAX_SIDE:
        return frame, 1.0
    scale = SAM_MAX_SIDE / float(side)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale


def _call_sam(frame: np.ndarray, object_name: str) -> List[Any]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), SAM_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode camera frame")
    response = requests.post(
        SAM_SERVICE_URL,
        files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        data={"object": object_name, "prompt": object_name},
        timeout=SAM_TIMEOUT_SEC,
    )
    response.raise_for_status()
    masks = response.json().get("masks", [])
    if isinstance(masks, dict) and "mask" in masks:
        masks = [masks["mask"]]
    return masks if isinstance(masks, list) else []


def _mask_geometry(mask_data: Any, width: int, height: int, scale: float) -> Optional[Dict[str, Any]]:
    mask = np.asarray(mask_data, dtype=np.uint8)
    if mask.ndim != 2:
        return None
    ys, xs = np.where(mask > 0)
    if not xs.size:
        return None
    inv = 1.0 / scale if scale > 0 else 1.0
    cx = int(clamp(round(float(xs.mean()) * inv), 0, width - 1))
    cy = int(clamp(round(float(ys.mean()) * inv), 0, height - 1))
    x1 = int(clamp(round(float(xs.min()) * inv), 0, width - 1))
    x2 = int(clamp(round(float(xs.max()) * inv), 0, width - 1))
    y1 = int(clamp(round(float(ys.min()) * inv), 0, height - 1))
    y2 = int(clamp(round(float(ys.max()) * inv), 0, height - 1))
    nx = (((width - 1) - cx) if MIRROR_IMAGE else cx) / float(max(1, width - 1))
    direction = "left" if nx < LEFT_TH else "right" if nx > RIGHT_TH else "center"
    area_pixels = int(round(xs.size * inv * inv))
    return {
        "centroid": {"x": cx, "y": cy},
        "centroid_norm_x": round(nx, 4),
        "direction": direction,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "bbox_h_ratio": round((y2 - y1) / float(height), 6),
        "area_pixels": area_pixels,
        "area_ratio": round(area_pixels / float(width * height), 6),
    }


def _write_debug(frame: np.ndarray, name: str, geometry: Dict[str, Any],
                 mask_data: Any, scale: float, tag: str) -> Optional[str]:
    if not WRITE_DEBUG_IMAGE:
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{tag}_detection.jpg")
    draw = frame.copy()
    mask = np.asarray(mask_data, dtype=np.uint8)
    if scale != 1.0:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    selected = mask.astype(bool)
    if selected.shape == draw.shape[:2]:
        draw[selected] = (draw[selected] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    box, center = geometry["bbox"], geometry["centroid"]
    cv2.rectangle(draw, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
    cv2.circle(draw, (center["x"], center["y"]), 6, (0, 0, 255), -1)
    cv2.putText(draw, f"{name} {geometry['direction']} {geometry['area_ratio']:.3f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(path, draw)
    return path


def detect_with_context(robot: RobotNode, object_name: str, tag: str = "robot"
                        ) -> Tuple[Detection, Optional[np.ndarray], Any, float]:
    """One SAM detection, returning the frame and mask as well.

    The largest mask is always selected, in every architecture, so that the
    perception stage cannot explain any difference between architectures.
    The extra return values exist only so that the optional visualization can
    draw the mask without running SAM a second time.
    """
    object_name = str(object_name).strip().lower()
    if not object_name:
        return Detection(False, object_name, error="object_name is empty"), None, None, 1.0

    frame, frame_time, frame_seq = robot.latest_frame()
    if frame is None:
        return (Detection(False, object_name, error="No camera frame received yet"),
                None, None, 1.0)

    age = None if frame_time is None else round(time.time() - frame_time, 3)
    if age is not None and age > STALE_FRAME_SEC:
        return (Detection(False, object_name, error=f"Stale frame: {age:.2f}s old",
                          frame_seq=frame_seq, frame_age_sec=age), frame, None, 1.0)

    sam_frame, scale = _resize_for_sam(frame)
    started = time.monotonic()
    try:
        masks = _call_sam(sam_frame, object_name)
    except Exception as exc:
        return (Detection(False, object_name, error=f"SAM service error: {exc}",
                          frame_seq=frame_seq, frame_age_sec=age,
                          sam_latency_sec=round(time.monotonic() - started, 3)),
                frame, None, scale)
    latency = round(time.monotonic() - started, 3)

    height, width = frame.shape[:2]
    best: Optional[Dict[str, Any]] = None
    best_mask: Any = None
    for mask in masks:
        geometry = _mask_geometry(mask, width, height, scale)
        if geometry and (best is None or geometry["area_pixels"] > best["area_pixels"]):
            best, best_mask = geometry, mask

    if best is None:
        return (Detection(False, object_name, frame_seq=frame_seq,
                          frame_age_sec=age, sam_latency_sec=latency),
                frame, None, scale)

    detection = Detection(
        found=True,
        object=object_name,
        direction=best["direction"],
        centroid_norm_x=best["centroid_norm_x"],
        area_ratio=best["area_ratio"],
        bbox_h_ratio=best["bbox_h_ratio"],
        bbox=best["bbox"],
        centroid=best["centroid"],
        debug_image_path=_write_debug(frame, object_name, best, best_mask, scale, tag),
        frame_seq=frame_seq,
        frame_age_sec=age,
        sam_latency_sec=latency,
    )
    return detection, frame, best_mask, scale


def detect_once(robot: RobotNode, object_name: str, tag: str = "robot") -> Detection:
    """One SAM detection on the newest camera frame."""
    return detect_with_context(robot, object_name, tag)[0]


def detect_stable(robot: RobotNode, object_name: str, samples: int = 1,
                  delay_sec: float = 0.08, tag: str = "robot",
                  state: str = "DETECT") -> Dict[str, Any]:
    """Aggregated detection used by the primitive tool ``detect_object``.

    Publishes the annotated view for the primitive architectures too, so that
    A, A1 and A2 can produce exactly the same figures as B and C.
    """
    samples = int(clamp(samples, 1, 5, 1))

    def one() -> Detection:
        if not VISUALIZATION_ENABLED:
            return detect_once(robot, object_name, tag)
        detection, frame, mask, scale = detect_with_context(robot, object_name, tag)
        if frame is not None and robot.visualization_due():
            robot.publish_visualization(
                render_visualization(frame, object_name, state, detection, mask, scale,
                                     detection.error or ""), force=True)
        return detection

    detections = [one()]
    for _ in range(samples - 1):
        time.sleep(max(0.0, delay_sec))
        detections.append(one())

    valid = [d for d in detections if d.found]
    if not valid:
        error = next((d.error for d in detections if d.error), None)
        return {
            "found": False, "object": object_name, "confidence": 0.0,
            "samples_taken": samples, "error": error,
            "camera_frame_age_sec": detections[-1].frame_age_sec,
            "sam_latency_sec": detections[-1].sam_latency_sec,
        }

    directions = [d.direction for d in valid if d.direction]
    best = max(valid, key=lambda d: float(d.area_ratio or 0.0))
    mean = lambda values: round(sum(values) / len(values), 6)  # noqa: E731
    return {
        "found": True,
        "object": object_name,
        "direction": max(set(directions), key=directions.count) if directions else None,
        "centroid_norm_x": round(mean([float(d.centroid_norm_x or 0.5) for d in valid]), 4),
        "area_ratio": mean([float(d.area_ratio or 0.0) for d in valid]),
        "bbox_h_ratio": mean([float(d.bbox_h_ratio or 0.0) for d in valid]),
        "bbox": best.bbox,
        "confidence": round(len(valid) / float(samples), 3),
        "samples_taken": samples,
        "camera_frame_age_sec": best.frame_age_sec,
        "sam_latency_sec": best.sam_latency_sec,
        "debug_image_path": best.debug_image_path,
        "error": None,
    }


# --------------------------------------------------------------------------- #
# perception worker (decouples SAM latency from the control loop)
# --------------------------------------------------------------------------- #
@dataclass
class PerceptionSample:
    detection: Detection
    timestamp: float = field(default_factory=time.monotonic)


class PerceptionWorker(threading.Thread):
    """Continuously detect one object in a background thread.

    The skill architectures (B and C) must keep publishing velocity commands at
    a fixed rate.  Calling SAM inline would freeze the control loop for the
    whole inference time, which starves the driver watchdog and produces jerky
    motion, so perception runs here instead.
    """

    def __init__(self, robot: RobotNode, object_name: str, tag: str = "robot",
                 period: float = SAM_EVERY_SEC, state: str = "DETECT") -> None:
        super().__init__(daemon=True)
        self.robot = robot
        self.object_name = object_name
        self.tag = tag
        self.period = period
        self._lock = threading.Lock()
        self._sample: Optional[PerceptionSample] = None
        self._stop = threading.Event()
        self._state = state
        self._message = ""
        self.detections = 0
        self.hits = 0

    def set_state(self, state: str, message: str = "") -> None:
        """Label shown in the visualization banner (SEARCH / ALIGN / APPROACH)."""
        with self._lock:
            self._state, self._message = state, message

    def run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            if VISUALIZATION_ENABLED:
                detection, frame, mask, scale = detect_with_context(
                    self.robot, self.object_name, self.tag)
            else:
                detection = detect_once(self.robot, self.object_name, self.tag)
                frame = mask = None
                scale = 1.0
            with self._lock:
                self._sample = PerceptionSample(detection)
                self.detections += 1
                if detection.found:
                    self.hits += 1
                state, message = self._state, self._message
            if VISUALIZATION_ENABLED and frame is not None:
                # Rate-limited inside publish_visualization; rendering only
                # happens when a frame will actually be published.
                if self.robot.viz_pub is not None and self.robot.visualization_due():
                    self.robot.publish_visualization(
                        render_visualization(frame, self.object_name, state, detection,
                                             mask, scale,
                                             message or (detection.error or "")),
                        force=True)
            remaining = self.period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def latest(self) -> Optional[PerceptionSample]:
        with self._lock:
            return self._sample

    def wait_first(self, timeout: float = 5.0) -> Optional[PerceptionSample]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.latest()
            if sample is not None:
                return sample
            time.sleep(0.02)
        return None

    def stop(self) -> None:
        self._stop.set()
