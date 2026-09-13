#!/usr/bin/env python3
"""Architecture A server: primitive, TIMED camera-only robot tools.

Role of the LLM in this architecture: full control authority. The server offers
no behaviour at all, only discrete perception and motion primitives whose
duration is derived from the requested distance/angle.

All tools are ``async def`` and offload their blocking work to a worker thread,
so ``stop_robot`` is always served immediately, even in the middle of a motion.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

import robot_common as rc

mcp = FastMCP("robot-control-server-architecture-a")

ARCH = "A"
TAG = "arch_a"

robot = rc.start_robot("mcp_robot_controller_architecture_a")
metrics = rc.MetricsLogger(ARCH, "server")


# --------------------------------------------------------------------------- #
# blocking implementations (executed in a worker thread)
# --------------------------------------------------------------------------- #
def _detect(object_name: str, samples: int) -> Dict[str, Any]:
    object_name = str(object_name).strip().lower()
    if not object_name:
        return {"found": False, "object": object_name, "error": "object_name must not be empty"}
    # Never reason about a frame captured before the previous motion ended.
    fresh = robot.wait_settled()
    result = rc.detect_stable(robot, object_name, samples, tag=TAG)
    result["fresh_frame"] = fresh
    metrics.log("detect_object", object=object_name, samples=samples,
                found=result.get("found"), direction=result.get("direction"),
                area_ratio=result.get("area_ratio"), fresh_frame=fresh,
                sam_latency_sec=result.get("sam_latency_sec"))
    return result


def _timed(linear: float, angular: float, duration: float, motion: str,
           key: str, value: float) -> Dict[str, Any]:
    outcome = robot.run_timed(linear, angular, duration, motion)
    metrics.log("motion", motion=motion, **{key: round(value, 3)}, **outcome)
    return {"ok": True, "motion": motion, key: round(value, 3), **outcome}


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def stop_robot() -> Dict[str, Any]:
    """Stop the robot immediately and interrupt any running motion."""
    robot.stop(reason="tool")
    robot.publish_banner("STOPPED", "Emergency stop")
    metrics.log("stop_robot")
    return {"ok": True, "status": "stopped"}


@mcp.tool()
async def detect_object(object_name: str, samples: int = 1) -> Dict[str, Any]:
    """Detect an object with the camera. Returns direction and size cues."""
    return await rc.run_blocking(_detect, object_name, samples)


@mcp.tool()
async def move_forward(distance_m: float) -> Dict[str, Any]:
    """Move forward by an approximate distance in meters. Safety-clamped."""
    distance = rc.clamp(distance_m, rc.MIN_FORWARD_M, rc.MAX_FORWARD_M, rc.MIN_FORWARD_M)
    duration = distance / rc.LINEAR_SPEED if rc.LINEAR_SPEED > 0 else 0.0
    return await rc.run_blocking(_timed, rc.LINEAR_SPEED, 0.0, duration,
                                 "move_forward", "distance_m", distance)


@mcp.tool()
async def move_backward(distance_m: float) -> Dict[str, Any]:
    """Move backward by an approximate distance in meters. Safety-clamped."""
    distance = rc.clamp(distance_m, rc.MIN_FORWARD_M, rc.MAX_BACKWARD_M, rc.MIN_FORWARD_M)
    duration = distance / rc.LINEAR_SPEED if rc.LINEAR_SPEED > 0 else 0.0
    return await rc.run_blocking(_timed, -rc.LINEAR_SPEED, 0.0, duration,
                                 "move_backward", "distance_m", distance)


@mcp.tool()
async def turn_left(angle_deg: float) -> Dict[str, Any]:
    """Turn left by an approximate angle in degrees. Safety-clamped."""
    angle = rc.clamp(angle_deg, rc.MIN_TURN_DEG, rc.MAX_TURN_DEG, rc.MIN_TURN_DEG)
    duration = math.radians(angle) / rc.ANGULAR_SPEED if rc.ANGULAR_SPEED > 0 else 0.0
    return await rc.run_blocking(_timed, 0.0, rc.ANGULAR_SPEED, duration,
                                 "turn_left", "angle_deg", angle)


@mcp.tool()
async def turn_right(angle_deg: float) -> Dict[str, Any]:
    """Turn right by an approximate angle in degrees. Safety-clamped."""
    angle = rc.clamp(angle_deg, rc.MIN_TURN_DEG, rc.MAX_TURN_DEG, rc.MIN_TURN_DEG)
    duration = math.radians(angle) / rc.ANGULAR_SPEED if rc.ANGULAR_SPEED > 0 else 0.0
    return await rc.run_blocking(_timed, 0.0, -rc.ANGULAR_SPEED, duration,
                                 "turn_right", "angle_deg", angle)


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        rc.shutdown_robot(robot)
