#!/usr/bin/env python3
"""Standalone live view of what the robot perceives.

Run it in its own terminal, next to any of the five architectures:

    source experiment_env.sh
    VISUALIZATION_ENABLED=true python3 camera_viewer.py

It subscribes to the annotated stream published by the server
(``VISUALIZATION_TOPIC``) and falls back to the raw camera topic whenever the
annotated stream is older than ``VISUALIZATION_FRESH_SEC``.

Why a separate process instead of a thread inside the agent client:

* the OpenCV window lives in the main thread, where it is actually supported;
* the agent client keeps working over SSH with no ``DISPLAY``;
* the same viewer serves A, A1, A2, B and C, so no architecture gets a
  visualization advantage over another.

Press ESC or q to close. Closing the viewer never affects the robot.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

CAMERA_TOPIC = os.getenv("CAMERA_TOPIC", "camera/image")
VISUALIZATION_TOPIC = os.getenv("VISUALIZATION_TOPIC", "/robot_llm/visualization")
VISUALIZATION_COMPRESSED = os.getenv("VISUALIZATION_COMPRESSED", "true").lower() in {"1", "true", "yes"}
VISUALIZATION_FRESH_SEC = float(os.getenv("VISUALIZATION_FRESH_SEC", "1.0"))
WINDOW = os.getenv("CAMERA_WINDOW", "robot-llm live view")
MIRROR = os.getenv("CAMERA_MIRROR", "false").lower() in {"1", "true", "yes"}
RECORD_PATH = os.getenv("VIEWER_RECORD_PATH", "")     # e.g. /tmp/run.mp4
RECORD_FPS = float(os.getenv("VIEWER_RECORD_FPS", "10"))

# Encodings handled without cv_bridge, which avoids the NumPy 1.x/2.x binary
# incompatibility of cv_bridge on ROS 2 Humble.
_ENCODINGS = {
    "bgr8": (3, "bgr"), "rgb8": (3, "rgb"), "8uc3": (3, "bgr"),
    "mono8": (1, "mono"), "8uc1": (1, "mono"),
    "bgra8": (4, "bgra"), "rgba8": (4, "rgba"),
}


def image_to_bgr(msg: Image) -> np.ndarray:
    encoding = str(msg.encoding or "").lower()
    spec = _ENCODINGS.get(encoding)
    if spec is None:
        raise ValueError(f"unsupported encoding: {msg.encoding!r}")
    channels, layout = spec
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid size {width}x{height}")
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(f"invalid step {step}, expected at least {row_bytes}")
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("short image buffer")
    rows = raw[:height * step].reshape(height, step)[:, :row_bytes]
    image = rows.reshape(height, width) if channels == 1 else rows.reshape(height, width, channels)
    if layout == "bgr":
        return image.copy()
    conversions = {"rgb": cv2.COLOR_RGB2BGR, "mono": cv2.COLOR_GRAY2BGR,
                   "bgra": cv2.COLOR_BGRA2BGR, "rgba": cv2.COLOR_RGBA2BGR}
    return cv2.cvtColor(image, conversions[layout])


class Viewer(Node):
    def __init__(self) -> None:
        super().__init__("robot_llm_camera_viewer")
        self._lock = threading.Lock()
        self.raw: Optional[np.ndarray] = None
        self.annotated: Optional[np.ndarray] = None
        self.annotated_time = 0.0
        self._warned: set[str] = set()

        self.create_subscription(Image, CAMERA_TOPIC, self._raw_callback, 10)
        if VISUALIZATION_COMPRESSED:
            self.create_subscription(CompressedImage, VISUALIZATION_TOPIC + "/compressed",
                                     self._compressed_callback, 5)
            shown = VISUALIZATION_TOPIC + "/compressed"
        else:
            self.create_subscription(Image, VISUALIZATION_TOPIC, self._annotated_callback, 5)
            shown = VISUALIZATION_TOPIC
        self.get_logger().info(f"raw: {CAMERA_TOPIC} | annotated: {shown}")

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warning(message)

    def _raw_callback(self, msg: Image) -> None:
        try:
            frame = image_to_bgr(msg)
            with self._lock:
                self.raw = frame
        except Exception as exc:
            self._warn_once(f"raw:{msg.encoding}", f"raw conversion failed: {exc}")

    def _annotated_callback(self, msg: Image) -> None:
        try:
            frame = image_to_bgr(msg)
            with self._lock:
                self.annotated, self.annotated_time = frame, time.monotonic()
        except Exception as exc:
            self._warn_once(f"viz:{msg.encoding}", f"annotated conversion failed: {exc}")

    def _compressed_callback(self, msg: CompressedImage) -> None:
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG decode failed")
            with self._lock:
                self.annotated, self.annotated_time = frame, time.monotonic()
        except Exception as exc:
            self._warn_once("viz:compressed", f"annotated decode failed: {exc}")

    def current(self) -> Optional[np.ndarray]:
        with self._lock:
            fresh = (self.annotated is not None
                     and time.monotonic() - self.annotated_time <= VISUALIZATION_FRESH_SEC)
            if fresh:
                return self.annotated.copy()
            return None if self.raw is None else self.raw.copy()


def main() -> None:
    rclpy.init()
    node = Viewer()
    writer = None
    print("Viewer running. Press ESC or q to close (the robot keeps working).")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            frame = node.current()
            if frame is None:
                continue
            if MIRROR:
                frame = cv2.flip(frame, 1)
            if RECORD_PATH:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(RECORD_PATH, fourcc, RECORD_FPS,
                                             (frame.shape[1], frame.shape[0]))
                    print(f"Recording to {RECORD_PATH}")
                writer.write(frame)
            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) in {27, ord("q")}:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
