#!/usr/bin/env python3
"""Architecture A2 client: LLM decisions with action persistence.

Same server, same tools, same prompt and same thresholds as ``a1_client.py``.
The single independent variable is the decision cache: after the LLM has chosen
a continuous motion, the client keeps that motion active and only re-detects,
instead of asking the model again at every control step.

The cache is NOT a planner. It only answers one question: "is the previous LLM
decision still consistent with the newest observation?". As soon as the answer
is no, the motion is paused and control goes back to the model.

Two bounds keep the cached motion safe:
* a distance/rotation budget (the cache never runs longer than the robot needs
  to travel ``MAX_CACHED_DISTANCE_M`` or to rotate ``MAX_CACHED_TURN_DEG``);
* the server-side watchdog, which stops the robot if this client dies.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Set

from mcp import ClientSession
from mcp.client.stdio import stdio_client

import llm_common as lc
from a1_client import TOOLS, build_prompt, validate

# ``keep_moving`` is a safety keep-alive, not a decision tool: the LLM never
# sees it, but the client must be allowed to call it.
REQUIRED_TOOLS = TOOLS | {"keep_moving"}

ARCH = "A2"
SERVER_SCRIPT = "a1_server.py"

MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "45"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "5"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))

# Cache bounds.
DETECT_INTERVAL_SEC = float(os.getenv("DETECT_INTERVAL_SEC", "0.12"))
MAX_PERSIST_DETECTIONS = int(os.getenv("MAX_PERSIST_DETECTIONS", "8"))
MAX_CACHED_DISTANCE_M = float(os.getenv("MAX_CACHED_DISTANCE_M", "0.35"))
MAX_CACHED_TURN_DEG = float(os.getenv("MAX_CACHED_TURN_DEG", "45"))
MAX_LOST_WHILE_FORWARD = int(os.getenv("MAX_LOST_WHILE_FORWARD", "1"))
LINEAR_SPEED = float(os.getenv("ROBOT_LINEAR_SPEED", "0.10"))
ANGULAR_SPEED = float(os.getenv("ROBOT_ANGULAR_SPEED", "0.30"))

# Break the rotation slightly before the formal centre band to avoid overshoot,
# and break the forward motion slightly before the stop condition.
PRE_CENTER_LEFT_X = float(os.getenv("PRE_CENTER_LEFT_X", "0.35"))
PRE_CENTER_RIGHT_X = float(os.getenv("PRE_CENTER_RIGHT_X", "0.65"))
PRE_STOP_FACTOR = float(os.getenv("PRE_STOP_FACTOR", "0.80"))


async def decide(target: str, history: Deque[Dict[str, Any]], step: int,
                 available: Set[str], metrics: lc.RunMetrics) -> lc.ToolAction:
    prompt = build_prompt(target, history, step)
    raw: Dict[str, Any] = {}
    for attempt in range(MAX_REPAIRS + 1):
        try:
            raw = await lc.ask_ollama(prompt, metrics)
            return validate(raw, target, available)
        except Exception as error:
            metrics.invalid_actions += 1
            metrics.event("invalid_action", attempt=attempt, error=str(error))
            if attempt >= MAX_REPAIRS:
                break
            prompt = (f"{build_prompt(target, history, step)}\n\n"
                      f"Your previous answer was invalid: {error}\n"
                      f"Invalid answer: {json.dumps(raw, ensure_ascii=False)}\n"
                      f"Allowed tools: {sorted(TOOLS & available)}\n"
                      f"Return corrected JSON only.")
    metrics.event("fallback_action", tool="detect_object")
    return lc.ToolAction("detect_object", {"object_name": target, "samples": 1},
                         "Fallback: re-check the target.", False)


def budget_seconds(motion_tool: str) -> float:
    """How long the cached motion may run before the LLM is consulted again."""
    if motion_tool == "start_moving_forward":
        return MAX_CACHED_DISTANCE_M / max(LINEAR_SPEED, 1e-3)
    if motion_tool in {"start_turning_left", "start_turning_right"}:
        return math.radians(MAX_CACHED_TURN_DEG) / max(ANGULAR_SPEED, 1e-3)
    return 0.0     # backward motion is never cached


def cache_still_valid(motion_tool: str, detection: Any, lost_count: int) -> bool:
    """Is the previous LLM decision still consistent with this observation?"""
    if not isinstance(detection, dict) or not motion_tool.startswith("start_"):
        return False

    found = bool(detection.get("found"))
    try:
        x = float(detection.get("centroid_norm_x"))
    except (TypeError, ValueError):
        x = None
    area = float(detection.get("area_ratio") or 0.0)
    bbox_h = float(detection.get("bbox_h_ratio") or 0.0)

    if motion_tool == "start_moving_forward":
        if found and (area >= lc.STOP_AREA_RATIO * PRE_STOP_FACTOR
                      or bbox_h >= lc.STOP_BBOX_H_RATIO * PRE_STOP_FACTOR):
            return False                      # nearly there: let the LLM decide
        if found and x is not None:
            return lc.LEFT_TH <= x <= lc.RIGHT_TH
        if found:
            return detection.get("direction") == "center"
        return lost_count <= MAX_LOST_WHILE_FORWARD

    if motion_tool == "start_turning_left":
        if not found:
            return True                       # still searching
        if x is None:
            return detection.get("direction") == "left"
        return x < PRE_CENTER_LEFT_X

    if motion_tool == "start_turning_right":
        if not found:
            return True
        if x is None:
            return detection.get("direction") == "right"
        return x > PRE_CENTER_RIGHT_X

    return False


async def run_task(session: ClientSession, command: str, available: Set[str],
                   metrics: lc.RunMetrics, stop_event: asyncio.Event) -> str:
    target = await lc.infer_target(command, metrics)
    metrics.new_run(target)
    print(f"Target object: {target}")
    print("Type 'stop' and press Enter to interrupt the robot.")

    history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
    cached_detections = 0

    for step in range(1, MAX_STEPS + 1):
        if stop_event.is_set():
            return "user_stop"
        metrics.steps = step

        action = await decide(target, history, step, available, metrics)
        if stop_event.is_set():
            return "user_stop"

        print(f"Robot feedback: {action.feedback}")
        result = await lc.call_tool(session, action.tool, action.arguments, metrics)
        observation = lc.compact_result(result)
        print("Tool result: " + json.dumps(observation, ensure_ascii=False))
        history.append({"step": step, "source": "llm", "tool": action.tool,
                        "observation": observation})

        if action.tool == "stop_robot" or action.done:
            return "llm_stop"
        if action.tool == "detect_object" and lc.reached_from_detection(result):
            await lc.call_tool(session, "stop_robot", {}, metrics)
            return "success"
        if not action.tool.startswith("start_"):
            continue

        # ------------------------- decision cache ------------------------- #
        deadline = time.monotonic() + budget_seconds(action.tool)
        lost_count = 0
        reason = "budget_exhausted"

        for index in range(1, MAX_PERSIST_DETECTIONS + 1):
            if stop_event.is_set():
                await lc.call_tool(session, "stop_robot", {}, metrics)
                return "user_stop"
            if time.monotonic() >= deadline:
                reason = "motion_budget"
                break

            await asyncio.sleep(DETECT_INTERVAL_SEC)
            detection = await lc.call_tool(
                session, "detect_object", {"object_name": target, "samples": 1}, metrics)
            compact = lc.compact_result(detection)
            cached_detections += 1
            print(f"Cached-motion detection {index}: "
                  + json.dumps(compact, ensure_ascii=False))
            history.append({"step": step, "source": "cache", "motion": action.tool,
                            "observation": compact})

            if lc.reached_from_detection(detection):
                await lc.call_tool(session, "stop_robot", {}, metrics)
                metrics.event("cache_end", motion=action.tool, reason="reached",
                              detections=index)
                return "success"

            lost_count = lost_count + 1 if not compact.get("found") else 0
            if not cache_still_valid(action.tool, compact, lost_count):
                reason = "observation_changed"
                break
            # Keep-alive: the cached motion stays active only while this client
            # is demonstrably supervising it.
            await lc.call_tool(session, "keep_moving", {}, metrics)
            metrics.event("cache_keep", motion=action.tool, detection=index)
        else:
            reason = "max_detections"

        # Pause before consulting the model again: the robot must not drive
        # blindly during the LLM latency.
        await lc.call_tool(session, "stop_robot", {}, metrics)
        metrics.event("cache_end", motion=action.tool, reason=reason)

    await lc.call_tool(session, "stop_robot", {}, metrics)
    metrics.event("cache_summary", cached_detections=cached_detections)
    return "max_steps"


async def supervise(session: ClientSession, command: str, available: Set[str],
                    metrics: lc.RunMetrics, terminal: lc.Terminal) -> bool:
    stop_event = asyncio.Event()
    task = asyncio.create_task(run_task(session, command, available, metrics, stop_event))
    keep_running = True

    while not task.done():
        reader = asyncio.create_task(terminal.read())
        done, _ = await asyncio.wait({task, reader}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            reader.cancel()
            break
        text = reader.result().strip().lower()
        if text in {"stop", "exit", "quit"}:
            stop_event.set()
            await lc.emergency_stop(session, metrics)
            task.cancel()
            keep_running = text == "stop"
            try:
                await task
            except asyncio.CancelledError:
                pass
            lc.print_summary(metrics.finish("user_stop"))
            print("Robot answer: stopped by user.")
            return keep_running
        if text:
            print("A task is running. Type 'stop' to interrupt it.")

    try:
        outcome = await task
    except Exception:
        await lc.call_tool(session, "stop_robot", {}, metrics)
        lc.print_summary(metrics.finish("error"))
        raise

    messages = {
        "success": "Target reached, robot stopped.",
        "llm_stop": "The agent decided the task is finished.",
        "max_steps": "Maximum number of steps reached, robot stopped.",
        "user_stop": "Stopped by user.",
    }
    print(f"Robot answer: {messages.get(outcome, outcome)}")
    lc.print_summary(metrics.finish(outcome))
    return keep_running


async def main() -> None:
    metrics = lc.RunMetrics(ARCH)
    terminal = lc.Terminal(asyncio.get_running_loop())

    async with stdio_client(lc.server_params(SERVER_SCRIPT)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available = {tool.name for tool in (await session.list_tools()).tools}
            lc.check_tools(available, REQUIRED_TOOLS, ARCH)

            print("Architecture A2 ready: LLM decisions with action persistence.")
            print("Type an object ('red ball'), 'stop' or 'exit'.")

            while True:
                command = (await terminal.read("\nGoal/Object> ")).strip()
                if not command:
                    continue
                if command.lower() in {"exit", "quit"}:
                    await lc.call_tool(session, "stop_robot", {}, metrics)
                    break
                if command.lower() == "stop":
                    await lc.emergency_stop(session, metrics)
                    print("Robot answer: robot stopped.")
                    continue
                try:
                    if not await supervise(session, command, available, metrics, terminal):
                        break
                except Exception as exc:
                    print(f"Agent error: {exc}")
                    try:
                        await lc.call_tool(session, "stop_robot", {}, metrics)
                    except Exception:
                        pass


if __name__ == "__main__":
    asyncio.run(main())
