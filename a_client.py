#!/usr/bin/env python3
"""Architecture A client: the LLM sequences TIMED primitive tools.

Research role: maximum LLM authority. Deterministic Python only starts the
server, validates the JSON action, clamps the arguments, applies the camera
stop condition and records metrics. It never plans.

Compared with the first version this client can be interrupted at any moment:
a single permanent stdin reader runs next to the agent loop, so typing ``stop``
works while the robot is moving.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from typing import Any, Deque, Dict, Set

from mcp import ClientSession
from mcp.client.stdio import stdio_client

import llm_common as lc

ARCH = "A"
SERVER_SCRIPT = "a_server.py"

MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "45"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "5"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))

MAX_TURN_DEG = float(os.getenv("MAX_TURN_DEG", "45"))
MIN_TURN_DEG = float(os.getenv("MIN_TURN_DEG", "8"))
MAX_FORWARD_M = float(os.getenv("MAX_FORWARD_M", "0.30"))
MAX_BACKWARD_M = float(os.getenv("MAX_BACKWARD_M", "0.15"))
MIN_FORWARD_M = float(os.getenv("MIN_FORWARD_M", "0.02"))
DEFAULT_TURN_DEG = float(os.getenv("DEFAULT_TURN_DEG", "15"))
DEFAULT_FORWARD_M = float(os.getenv("DEFAULT_FORWARD_M", "0.10"))
DEFAULT_BACKWARD_M = float(os.getenv("DEFAULT_BACKWARD_M", "0.05"))

TOOLS: Set[str] = {"detect_object", "turn_left", "turn_right",
                   "move_forward", "move_backward", "stop_robot"}

LEFT, RIGHT = lc.LEFT_TH, lc.RIGHT_TH

SYSTEM_PROMPT = f"""
You control a real camera-only mobile robot by choosing ONE primitive tool call
at a time. There is no LiDAR and no obstacle sensor.

Task: find the target object and drive close to it.

Tools:
- detect_object(object_name, samples)
- turn_left(angle_deg), turn_right(angle_deg)
- move_forward(distance_m), move_backward(distance_m)
- stop_robot()

Loop: detect, then move or turn, then detect again. Never move twice without a
detection in between.

If found = false (search):
- turn_left(30), then detect.
- After four turns without success, move_forward(0.10), then detect.
- Never move forward more than 0.10 while the target is not visible.

If found = true, use centroid_norm_x (0 = far left, 1 = far right):
- below {LEFT}: turn_left(15)
- between {LEFT} and {RIGHT}: the target is centred, move forward
- above {RIGHT}: turn_right(15)

When the target is centred, choose the distance from area_ratio:
- area_ratio < 0.010: move_forward(0.30)
- 0.010 to 0.020: move_forward(0.20)
- 0.020 to 0.040: move_forward(0.10)
- 0.040 to {lc.STOP_AREA_RATIO}: move_forward(0.04)

Stop condition:
- If area_ratio >= {lc.STOP_AREA_RATIO} or bbox_h_ratio >= {lc.STOP_BBOX_H_RATIO},
  call stop_robot with done = true.

Limits: angle_deg <= {MAX_TURN_DEG}, forward <= {MAX_FORWARD_M},
backward <= {MAX_BACKWARD_M}.

Answer with JSON only, no other text:
{{"tool":"tool_name","arguments":{{}},"feedback":"max 8 words","done":false}}

Examples:
{{"tool":"detect_object","arguments":{{"object_name":"red ball","samples":1}},"feedback":"Check target position.","done":false}}
{{"tool":"move_forward","arguments":{{"distance_m":0.30}},"feedback":"Centred and far, drive fast.","done":false}}
{{"tool":"turn_left","arguments":{{"angle_deg":15}},"feedback":"Target on the left.","done":false}}
{{"tool":"stop_robot","arguments":{{}},"feedback":"Target close enough.","done":true}}
""".strip()


# --------------------------------------------------------------------------- #
# action handling
# --------------------------------------------------------------------------- #
def validate(raw: Dict[str, Any], target: str, available: Set[str],
             metrics: lc.RunMetrics) -> lc.ToolAction:
    if not isinstance(raw, dict):
        raise ValueError("The action must be a JSON object")
    tool = raw.get("tool")
    args = raw.get("arguments") or {}
    feedback = str(raw.get("feedback", "")).strip()
    done = bool(raw.get("done", False))

    if not isinstance(tool, str) or tool not in TOOLS:
        raise ValueError(f"Architecture A forbids this tool: {tool}")
    if tool not in available:
        raise ValueError(f"Tool not available on the server: {tool}")
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")

    if done:
        return lc.ToolAction("stop_robot", {}, feedback or "Task complete.", True)

    # Arguments are rebuilt from scratch: any extra key invented by the model
    # would be rejected by the server tool signature.
    if tool == "detect_object":
        clean = {"object_name": target, "samples": 1}
    elif tool in {"turn_left", "turn_right"}:
        clean = {"angle_deg": _clamp(args.get("angle_deg"), DEFAULT_TURN_DEG,
                                     MIN_TURN_DEG, MAX_TURN_DEG)}
    elif tool == "move_forward":
        clean = {"distance_m": _clamp(args.get("distance_m"), DEFAULT_FORWARD_M,
                                      MIN_FORWARD_M, MAX_FORWARD_M)}
    elif tool == "move_backward":
        clean = {"distance_m": _clamp(args.get("distance_m"), DEFAULT_BACKWARD_M,
                                      MIN_FORWARD_M, MAX_BACKWARD_M)}
    else:
        clean = {}
    return lc.ToolAction(tool, clean, feedback, False)


def _clamp(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(low, min(high, number)), 3)


def build_prompt(target: str, history: Deque[Dict[str, Any]], step: int) -> str:
    return (f"{SYSTEM_PROMPT}\n\nTarget object: {target}\nStep: {step}/{MAX_STEPS}\n"
            f"Recent tool calls and observations:\n"
            f"{json.dumps(list(history), ensure_ascii=False)}\n"
            f"Choose the next single primitive tool call. JSON only.")


async def decide(target: str, history: Deque[Dict[str, Any]], step: int,
                 available: Set[str], metrics: lc.RunMetrics) -> lc.ToolAction:
    prompt = build_prompt(target, history, step)
    raw: Dict[str, Any] = {}
    for attempt in range(MAX_REPAIRS + 1):
        try:
            raw = await lc.ask_ollama(prompt, metrics)
            return validate(raw, target, available, metrics)
        except Exception as error:
            metrics.invalid_actions += 1
            metrics.event("invalid_action", attempt=attempt, error=str(error))
            if attempt >= MAX_REPAIRS:
                break
            # The repair prompt keeps the full context, otherwise the model
            # chooses blindly.
            prompt = (f"{build_prompt(target, history, step)}\n\n"
                      f"Your previous answer was invalid: {error}\n"
                      f"Invalid answer: {json.dumps(raw, ensure_ascii=False)}\n"
                      f"Allowed tools: {sorted(TOOLS & available)}\n"
                      f"Return corrected JSON only.")
    # Deterministic fallback keeps the experiment running instead of aborting.
    metrics.event("fallback_action", tool="detect_object")
    return lc.ToolAction("detect_object", {"object_name": target, "samples": 1},
                         "Fallback: re-check the target.", False)


# --------------------------------------------------------------------------- #
# task loop
# --------------------------------------------------------------------------- #
async def run_task(session: ClientSession, command: str, available: Set[str],
                   metrics: lc.RunMetrics, stop_event: asyncio.Event) -> str:
    target = await lc.infer_target(command, metrics)
    metrics.new_run(target)
    print(f"Target object: {target}")
    print("Type 'stop' and press Enter to interrupt the robot.")

    history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)

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
        history.append({"step": step, "tool": action.tool,
                        "arguments": action.arguments, "observation": observation})

        if action.tool == "stop_robot" or action.done:
            return "llm_stop"
        if action.tool == "detect_object" and lc.reached_from_detection(result):
            await lc.call_tool(session, "stop_robot", {}, metrics)
            return "success"

    await lc.call_tool(session, "stop_robot", {}, metrics)
    return "max_steps"


async def supervise(session: ClientSession, command: str, available: Set[str],
                    metrics: lc.RunMetrics, terminal: lc.Terminal) -> bool:
    """Run the task while watching stdin for 'stop' / 'exit'."""
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
    except Exception as exc:
        await lc.call_tool(session, "stop_robot", {}, metrics)
        lc.print_summary(metrics.finish("error"))
        raise exc

    summary = metrics.finish(outcome)
    messages = {
        "success": "Target reached, robot stopped.",
        "llm_stop": "The agent decided the task is finished.",
        "max_steps": "Maximum number of steps reached, robot stopped.",
        "user_stop": "Stopped by user.",
    }
    print(f"Robot answer: {messages.get(outcome, outcome)}")
    lc.print_summary(summary)
    return keep_running


async def main() -> None:
    metrics = lc.RunMetrics(ARCH)
    terminal = lc.Terminal(asyncio.get_running_loop())

    async with stdio_client(lc.server_params(SERVER_SCRIPT)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available = {tool.name for tool in (await session.list_tools()).tools}
            lc.check_tools(available, TOOLS, ARCH)

            print("Architecture A ready: the LLM sequences timed primitives.")
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
