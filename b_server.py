#!/usr/bin/env python3
"""Architecture B server: ONE camera-only high-level skill, ``go_to_object``.

Role of the LLM: it only understands the command and names the target object.
Search, visual servoing, approach and the stop decision are deterministic and
run entirely inside this server.

Two structural properties matter for the comparison:

* ``go_to_object`` is an ``async`` tool that offloads its control loop to a
  worker thread, so ``stop_robot`` is served within milliseconds while the
  skill is running.
* Perception runs in its own thread (``PerceptionWorker``). The control loop
  therefore keeps a fixed rate and keeps refreshing ``/cmd_vel`` even while SAM
  is busy, instead of freezing for the whole inference time.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

import robot_common as rc

mcp = FastMCP("robot-control-server-architecture-b")

ARCH = "B"
TAG = "arch_b"

robot = rc.start_robot("mcp_robot_controller_architecture_b")
metrics = rc.MetricsLogger(ARCH, "server")

# Skill parameters (shared kinematics; only the skill logic lives here).
MAX_GO_TO_OBJECT_SEC = rc.env_float("MAX_GO_TO_OBJECT_SEC", "90")
SEEN_RECENTLY_SEC = rc.env_float("SEEN_RECENTLY_SEC", "1.0")
MISS_GRACE_SEC = rc.env_float("MISS_GRACE_SEC", "1.0")
CENTER_GO_TH = rc.env_float("CENTER_GO_TH", "0.15")
KP_ANG = rc.env_float("KP_ANG_OBJ", "0.60")
MAX_ANG = rc.env_float("MAX_ANG", "0.30")
APPROACH_SPEED = rc.env_float("APPROACH_SPEED", "0.10")
SEARCH_TURN_TIME = rc.env_float("SEARCH_TURN_TIME", "1.20")
SEARCH_FWD_TIME = rc.env_float("SEARCH_FWD_TIME", "0.80")
CONTROL_DT = 1.0 / max(rc.CONTROL_HZ, 1.0)


def _go_to_object(object_name: str) -> Dict[str, Any]:
    object_name = str(object_name).strip().lower()
    if not object_name:
        return {"ok": False, "behavior": "go_to_object",
                "error": "object_name must not be empty"}

    robot.clear_stop()
    worker = rc.PerceptionWorker(robot, object_name, TAG, state="GO_TO_OBJECT")
    worker.start()

    start = time.monotonic()
    phase_start = start
    phase = "TURN"
    last_seen = 0.0
    last_error = 0.0
    last_area = 0.0
    last_bbox_h = 0.0
    last_detection: Optional[Dict[str, Any]] = None
    seen_once = False
    metrics.log("skill_start", behavior="go_to_object", object=object_name)

    def result(ok: bool, arrived: bool, reason: str) -> Dict[str, Any]:
        payload = {
            "ok": ok, "behavior": "go_to_object", "object": object_name,
            "arrived": arrived, "reason": reason,
            "elapsed_sec": round(time.monotonic() - start, 2),
            "detections": worker.detections, "hits": worker.hits,
            "last_detection": last_detection,
        }
        metrics.log("skill_end", **payload)
        return payload

    try:
        while time.monotonic() - start <= MAX_GO_TO_OBJECT_SEC:
            if robot.stop_requested():
                return result(False, False, "interrupted_by_stop")

            now = time.monotonic()
            sample = worker.latest()
            if sample is not None:
                last_detection = sample.detection.compact()
                fresh = (now - sample.timestamp) <= SEEN_RECENTLY_SEC
                if sample.detection.found and sample.detection.centroid_norm_x is not None and fresh:
                    seen_once = True
                    last_seen = sample.timestamp
                    last_error = sample.detection.centroid_norm_x - 0.5
                    last_area = float(sample.detection.area_ratio or 0.0)
                    last_bbox_h = float(sample.detection.bbox_h_ratio or 0.0)

            # A short grace period prevents a single missed SAM frame from
            # dropping the robot back into search mode.
            visible = seen_once and (now - last_seen) <= max(SEEN_RECENTLY_SEC, MISS_GRACE_SEC)

            if visible:
                phase, phase_start = "TURN", now
                worker.set_state("APPROACH", object_name)
                if last_area >= rc.STOP_AREA_RATIO or last_bbox_h >= rc.STOP_BBOX_H_RATIO:
                    robot.stop(reason="arrived")
                    robot.publish_banner("TARGET REACHED", f"Arrived at {object_name}")
                    return result(True, True, "camera_stop_condition_reached")
                angular = rc.clamp(-KP_ANG * last_error, -MAX_ANG, MAX_ANG)
                linear = 0.0
                if abs(last_error) < CENTER_GO_TH:
                    slow = (last_area >= rc.STOP_AREA_RATIO * 0.6
                            or last_bbox_h >= rc.STOP_BBOX_H_RATIO * 0.6)
                    linear = APPROACH_SPEED * (0.5 if slow else 1.0)
                robot.drive(linear, angular, "go_to_object")
            else:
                worker.set_state("SEARCH", object_name)
                elapsed = now - phase_start
                if phase == "TURN":
                    robot.drive(0.0, rc.ANGULAR_SPEED, "search_turn")
                    if elapsed >= SEARCH_TURN_TIME:
                        phase, phase_start = "FWD", now
                else:
                    robot.drive(rc.LINEAR_SPEED, 0.0, "search_forward")
                    if elapsed >= SEARCH_FWD_TIME:
                        phase, phase_start = "TURN", now

            time.sleep(CONTROL_DT)

        return result(False, False, "timeout")
    finally:
        worker.stop()
        robot.stop(reason="skill_end")


@mcp.tool()
def stop_robot() -> Dict[str, Any]:
    """Stop the robot immediately and interrupt go_to_object."""
    robot.stop(reason="tool")
    robot.publish_banner("STOPPED", "Emergency stop")
    metrics.log("stop_robot")
    return {"ok": True, "status": "stopped"}


@mcp.tool()
async def go_to_object(object_name: str) -> Dict[str, Any]:
    """Search, visually track, approach and stop at the requested object."""
    return await rc.run_blocking(_go_to_object, object_name)


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        rc.shutdown_robot(robot)
