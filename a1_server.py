#!/usr/bin/env python3
"""Architecture A1 server: primitive, CONTINUOUS camera-only robot tools.

Role of the LLM: full control authority, as in architecture A, but the motion
primitives are continuous. A motion command returns immediately and the robot
keeps moving until another command or ``stop_robot`` arrives.

This single server is used by BOTH architecture A1 clients (plain and
decision-cached), so the only difference between those two experiments is the
client-side decision policy, never the robot or perception stack.

Safety: continuous motion is protected by the dead-man watchdog implemented in
``robot_common.RobotNode``. If no command is refreshed within
``MAX_CONTINUOUS_MOTION_SEC`` the node stops the robot by itself.
"""

from __future__ import annotations

from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

import robot_common as rc

mcp = FastMCP("robot-control-server-architecture-a1")

ARCH = "A1"
TAG = "arch_a1"

robot = rc.start_robot("mcp_robot_controller_architecture_a1")
metrics = rc.MetricsLogger(ARCH, "server")


def _detect(object_name: str, samples: int, settle: bool) -> Dict[str, Any]:
    object_name = str(object_name).strip().lower()
    if not object_name:
        return {"found": False, "object": object_name, "error": "object_name must not be empty"}
    # While a continuous motion is active the robot must NOT wait for the
    # chassis to settle, otherwise perception would lag behind the motion.
    if settle:
        robot.wait_settled()
    result = rc.detect_stable(robot, object_name, samples, tag=TAG)
    metrics.log("detect_object", object=object_name, samples=samples,
                found=result.get("found"), direction=result.get("direction"),
                area_ratio=result.get("area_ratio"),
                sam_latency_sec=result.get("sam_latency_sec"))
    return result


@mcp.tool()
def stop_robot() -> Dict[str, Any]:
    """Stop every robot motion immediately."""
    robot.stop(reason="tool")
    robot.publish_banner("STOPPED", "Emergency stop")
    metrics.log("stop_robot")
    return {"ok": True, "status": "stopped"}


@mcp.tool()
async def detect_object(object_name: str, samples: int = 1) -> Dict[str, Any]:
    """Detect an object with the camera. Returns direction and size cues."""
    return await rc.run_blocking(_detect, object_name, samples, not robot.is_moving())


@mcp.tool()
def keep_moving() -> Dict[str, Any]:
    """Keep the current continuous motion alive (refreshes the safety watchdog).

    Cheap, non-blocking, and it never changes the velocity. A client that keeps
    a motion active for longer than MAX_CONTINUOUS_MOTION_SEC must call this,
    otherwise the watchdog stops the robot on purpose.
    """
    return robot.refresh_watchdog()


@mcp.tool()
def start_moving_forward() -> Dict[str, Any]:
    """Start continuous forward motion until another command or stop_robot."""
    result = robot.start_continuous("moving_forward", rc.LINEAR_SPEED, 0.0)
    metrics.log("start_motion", **result)
    return result


@mcp.tool()
def start_moving_backward() -> Dict[str, Any]:
    """Start continuous backward motion until another command or stop_robot."""
    result = robot.start_continuous("moving_backward", -rc.LINEAR_SPEED, 0.0)
    metrics.log("start_motion", **result)
    return result


@mcp.tool()
def start_turning_left() -> Dict[str, Any]:
    """Start continuous left rotation until another command or stop_robot."""
    result = robot.start_continuous("turning_left", 0.0, rc.ANGULAR_SPEED)
    metrics.log("start_motion", **result)
    return result


@mcp.tool()
def start_turning_right() -> Dict[str, Any]:
    """Start continuous right rotation until another command or stop_robot."""
    result = robot.start_continuous("turning_right", 0.0, -rc.ANGULAR_SPEED)
    metrics.log("start_motion", **result)
    return result


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        rc.shutdown_robot(robot)
