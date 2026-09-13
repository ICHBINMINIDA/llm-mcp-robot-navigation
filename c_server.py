#!/usr/bin/env python3
"""Architecture C server: three composable camera-only skills.

Role of the LLM: high-level planner. It decides WHICH skill to run and, above
all, how to recover when a skill fails. The skills themselves are deterministic:

* ``search_target``  : alternate rotation and short forward motion until seen
* ``align_to_target``: rotate with visual servoing until centred
* ``approach_target``: drive with visual servoing until the camera stop condition

Design points that differ from the first version:

1. Every skill is an ``async`` tool whose control loop runs in a worker thread,
   so ``stop_robot`` is always served immediately.
2. Perception runs in a separate thread, so the control loop keeps a fixed rate
   and keeps refreshing ``/cmd_vel`` while SAM is busy.
3. A single missed detection no longer aborts a skill: alignment and approach
   tolerate ``MISS_GRACE_SEC`` and ``MAX_CONSECUTIVE_MISSES`` before reporting
   ``target_not_visible``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

import robot_common as rc

mcp = FastMCP("robot-control-server-architecture-c-skills")

ARCH = "C"
TAG = "arch_c"

robot = rc.start_robot("mcp_robot_controller_architecture_c")
metrics = rc.MetricsLogger(ARCH, "server")

SEARCH_MAX_SEC = rc.env_float("SEARCH_MAX_SEC", "45")
ALIGN_MAX_SEC = rc.env_float("ALIGN_MAX_SEC", "15")
APPROACH_MAX_SEC = rc.env_float("APPROACH_MAX_SEC", "45")
SEEN_RECENTLY_SEC = rc.env_float("SEEN_RECENTLY_SEC", "1.0")
MISS_GRACE_SEC = rc.env_float("MISS_GRACE_SEC", "1.5")
MAX_CONSECUTIVE_MISSES = int(rc.env_float("MAX_CONSECUTIVE_MISSES", "3"))
ALIGN_CENTER_TH = rc.env_float("ALIGN_CENTER_TH", "0.08")
CENTER_GO_TH = rc.env_float("CENTER_GO_TH", "0.15")
KP_ANG = rc.env_float("KP_ANG_OBJ", "0.60")
MAX_ANG = rc.env_float("MAX_ANG", "0.30")
APPROACH_SPEED = rc.env_float("APPROACH_SPEED", "0.10")
SEARCH_TURN_TIME = rc.env_float("SEARCH_TURN_TIME", "1.20")
SEARCH_FWD_TIME = rc.env_float("SEARCH_FWD_TIME", "0.80")
CONTROL_DT = 1.0 / max(rc.CONTROL_HZ, 1.0)


class SkillContext:
    """Shared bookkeeping for one skill execution."""

    def __init__(self, behavior: str, object_name: str) -> None:
        self.behavior = behavior
        self.object = object_name
        self.worker = rc.PerceptionWorker(robot, object_name, TAG)
        self.start = time.monotonic()
        self.last_seen = 0.0
        self.seen_once = False
        self.misses = 0
        self.error = 0.0
        self.area = 0.0
        self.bbox_h = 0.0
        self.last_detection: Optional[Dict[str, Any]] = None
        self._last_sample_ts = 0.0

    def __enter__(self) -> "SkillContext":
        robot.clear_stop()
        self.worker.set_state(self.behavior.replace("_target", "").upper(), self.object)
        self.worker.start()
        metrics.log("skill_start", behavior=self.behavior, object=self.object)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.worker.stop()
        robot.stop(reason=f"{self.behavior}_end")

    # ---------------------------------------------------------------- #
    def update(self) -> None:
        """Consume the newest perception sample, if it is a new one."""
        sample = self.worker.latest()
        if sample is None or sample.timestamp == self._last_sample_ts:
            return
        self._last_sample_ts = sample.timestamp
        detection = sample.detection
        self.last_detection = detection.compact()
        if detection.found and detection.centroid_norm_x is not None:
            self.seen_once = True
            self.misses = 0
            self.last_seen = sample.timestamp
            self.error = detection.centroid_norm_x - 0.5
            self.area = float(detection.area_ratio or 0.0)
            self.bbox_h = float(detection.bbox_h_ratio or 0.0)
        else:
            self.misses += 1

    @property
    def visible(self) -> bool:
        if not self.seen_once:
            return False
        age = time.monotonic() - self.last_seen
        return age <= max(SEEN_RECENTLY_SEC, MISS_GRACE_SEC)

    @property
    def lost(self) -> bool:
        """A skill gives up only after several consecutive misses AND a grace period."""
        if not self.seen_once:
            return self.misses >= MAX_CONSECUTIVE_MISSES
        return (not self.visible) and self.misses >= MAX_CONSECUTIVE_MISSES

    @property
    def close_enough(self) -> bool:
        return self.area >= rc.STOP_AREA_RATIO or self.bbox_h >= rc.STOP_BBOX_H_RATIO

    def result(self, ok: bool, reason: str, **extra: Any) -> Dict[str, Any]:
        payload = {
            "ok": ok, "behavior": self.behavior, "object": self.object, "reason": reason,
            "elapsed_sec": round(time.monotonic() - self.start, 2),
            "detections": self.worker.detections, "hits": self.worker.hits,
            "last_detection": self.last_detection,
        }
        payload.update(extra)
        metrics.log("skill_end", **payload)
        return payload


def _guard(object_name: str, behavior: str) -> Optional[Dict[str, Any]]:
    if not object_name:
        return {"ok": False, "behavior": behavior, "error": "object_name must not be empty"}
    return None


# --------------------------------------------------------------------------- #
# skills (blocking implementations, executed in a worker thread)
# --------------------------------------------------------------------------- #
def _search_target(object_name: str) -> Dict[str, Any]:
    object_name = str(object_name).strip().lower()
    guard = _guard(object_name, "search_target")
    if guard:
        return guard

    with SkillContext("search_target", object_name) as ctx:
        phase, phase_start = "TURN", time.monotonic()
        while time.monotonic() - ctx.start <= SEARCH_MAX_SEC:
            if robot.stop_requested():
                return ctx.result(False, "interrupted_by_stop", found=False)
            ctx.update()
            if ctx.visible:
                robot.stop(reason="target_found")
                return ctx.result(True, "target_found", found=True)

            now = time.monotonic()
            if phase == "TURN":
                robot.drive(0.0, rc.ANGULAR_SPEED, "search_turn")
                if now - phase_start >= SEARCH_TURN_TIME:
                    phase, phase_start = "FWD", now
            else:
                robot.drive(rc.LINEAR_SPEED, 0.0, "search_forward")
                if now - phase_start >= SEARCH_FWD_TIME:
                    phase, phase_start = "TURN", now
            time.sleep(CONTROL_DT)
        return ctx.result(False, "timeout", found=False)


def _align_to_target(object_name: str) -> Dict[str, Any]:
    object_name = str(object_name).strip().lower()
    guard = _guard(object_name, "align_to_target")
    if guard:
        return guard

    with SkillContext("align_to_target", object_name) as ctx:
        while time.monotonic() - ctx.start <= ALIGN_MAX_SEC:
            if robot.stop_requested():
                return ctx.result(False, "interrupted_by_stop", aligned=False)
            ctx.update()

            if ctx.visible and abs(ctx.error) <= ALIGN_CENTER_TH:
                robot.stop(reason="aligned")
                return ctx.result(True, "target_centered", aligned=True)
            if ctx.lost:
                return ctx.result(False, "target_not_visible", aligned=False)

            if ctx.visible:
                robot.drive(0.0, rc.clamp(-KP_ANG * ctx.error, -MAX_ANG, MAX_ANG), "align")
            else:
                robot.drive(0.0, 0.0, "align_wait")   # wait out a short SAM miss
            time.sleep(CONTROL_DT)
        return ctx.result(False, "timeout", aligned=False)


def _approach_target(object_name: str) -> Dict[str, Any]:
    object_name = str(object_name).strip().lower()
    guard = _guard(object_name, "approach_target")
    if guard:
        return guard

    with SkillContext("approach_target", object_name) as ctx:
        while time.monotonic() - ctx.start <= APPROACH_MAX_SEC:
            if robot.stop_requested():
                return ctx.result(False, "interrupted_by_stop", arrived=False)
            ctx.update()

            if ctx.visible and ctx.close_enough:
                robot.stop(reason="arrived")
                robot.publish_banner("TARGET REACHED", f"Arrived at {object_name}")
                return ctx.result(True, "camera_stop_condition_reached", arrived=True)
            if ctx.lost:
                return ctx.result(False, "target_not_visible", arrived=False)

            if ctx.visible:
                angular = rc.clamp(-KP_ANG * ctx.error, -MAX_ANG, MAX_ANG)
                linear = 0.0
                if abs(ctx.error) < CENTER_GO_TH:
                    slow = (ctx.area >= rc.STOP_AREA_RATIO * 0.6
                            or ctx.bbox_h >= rc.STOP_BBOX_H_RATIO * 0.6)
                    linear = APPROACH_SPEED * (0.5 if slow else 1.0)
                robot.drive(linear, angular, "approach")
            else:
                robot.drive(0.0, 0.0, "approach_wait")
            time.sleep(CONTROL_DT)
        return ctx.result(False, "timeout", arrived=False)


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def stop_robot() -> Dict[str, Any]:
    """Stop immediately and interrupt the active skill."""
    robot.stop(reason="tool")
    robot.publish_banner("STOPPED", "Emergency stop")
    metrics.log("stop_robot")
    return {"ok": True, "status": "stopped"}


@mcp.tool()
async def search_target(object_name: str) -> Dict[str, Any]:
    """Search by alternating turn and forward motion until the target is visible."""
    return await rc.run_blocking(_search_target, object_name)


@mcp.tool()
async def align_to_target(object_name: str) -> Dict[str, Any]:
    """Rotate with visual servoing until the target is centred."""
    return await rc.run_blocking(_align_to_target, object_name)


@mcp.tool()
async def approach_target(object_name: str) -> Dict[str, Any]:
    """Approach with visual servoing and stop at the visual distance threshold."""
    return await rc.run_blocking(_approach_target, object_name)


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        rc.shutdown_robot(robot)
