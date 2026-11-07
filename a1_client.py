#!/usr/bin/env python3
"""Architecture A1 client: the LLM drives CONTINUOUS primitives, one call per step.

Research role: the LLM keeps full authority, but each decision now stays active
until the next decision, so the LLM latency is part of the control loop.
This client is the baseline that architecture A2 is compared against; both
run against the very same ``a1_server.py``.

Safety: the robot never keeps moving while the client is waiting for the model.
The motion is paused before every new LLM call, and the server watchdog stops
the robot anyway if this client dies.
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

ARCH = "A1"
SERVER_SCRIPT = "a1_server.py"

MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "45"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "5"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))
PAUSE_BEFORE_THINKING = os.getenv("PAUSE_BEFORE_THINKING", "false").lower() in {"1", "true", "yes"}

TOOLS: Set[str] = {"detect_object", "start_moving_forward", "start_moving_backward",
                   "start_turning_left", "start_turning_right", "stop_robot"}

SYSTEM_PROMPT = f"""
You control a real camera-only mobile robot with continuous motion tools.
A motion command keeps running until you send another command or stop_robot.
There is no LiDAR and no obstacle sensor.

Task: find the target object and drive close to it.

Tools:
- detect_object(object_name, samples)
- start_moving_forward(), start_moving_backward()
- start_turning_left(), start_turning_right()
- stop_robot()

Rules:
- Call detect_object(samples=1) often; never move twice without detecting.
- found = false: start_turning_left, then detect again.
- found = true, use centroid_norm_x (0 = far left, 1 = far right):
  below {lc.LEFT_TH}: start_turning_left
  between {lc.LEFT_TH} and {lc.RIGHT_TH}: start_moving_forward
  above {lc.RIGHT_TH}: start_turning_right
- Stop the search rotation as soon as found = true.
- If area_ratio >= {lc.STOP_AREA_RATIO} or bbox_h_ratio >= {lc.STOP_BBOX_H_RATIO},
  call stop_robot with done = true.

Answer with JSON only, no other text:
{{"tool":"tool_name","arguments":{{}},"feedback":"max 8 words","done":false}}

Examples:
{{"tool":"detect_object","arguments":{{"object_name":"red ball","samples":1}},"feedback":"Check target position.","done":false}}
{{"tool":"start_moving_forward","arguments":{{}},"feedback":"Centred, drive forward.","done":false}}
{{"tool":"start_turning_left","arguments":{{}},"feedback":"Target on the left.","done":false}}
{{"tool":"stop_robot","arguments":{{}},"feedback":"Target close enough.","done":true}}
""".strip()


def validate(raw: Dict[str, Any], target: str, available: Set[str]) -> lc.ToolAction:
    if not isinstance(raw, dict):
        raise ValueError("The action must be a JSON object")
    tool = raw.get("tool")
    feedback = str(raw.get("feedback", "")).strip()
    done = bool(raw.get("done", False))

    if not isinstance(tool, str) or tool not in TOOLS:
        raise ValueError(f"Architecture A1 forbids this tool: {tool}")
    if tool not in available:
        raise ValueError(f"Tool not available on the server: {tool}")

    if done:
        return lc.ToolAction("stop_robot", {}, feedback or "Task complete.", True)
    args = {"object_name": target, "samples": 1} if tool == "detect_object" else {}
    return lc.ToolAction(tool, args, feedback, False)


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


async def run_task(session: ClientSession, command: str, available: Set[str],
                   metrics: lc.RunMetrics, stop_event: asyncio.Event) -> str:
    target = await lc.start_run(command, metrics)
    print(f"Target object: {target}")
    print("Type 'stop' and press Enter to interrupt the robot.")

    history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)

    for step in range(1, MAX_STEPS + 1):
        if stop_event.is_set():
            return "user_stop"
        metrics.steps = step

        # The robot must not drive blindly during the model latency.
        if PAUSE_BEFORE_THINKING and step > 1:
            await lc.call_tool(session, "stop_robot", {}, metrics)

        action = await decide(target, history, step, available, metrics)
        if stop_event.is_set():
            return "user_stop"

        print(f"Robot feedback: {action.feedback}")
        result = await lc.call_tool(session, action.tool, action.arguments, metrics)
        if lc.stop_flag_set():          # latched emergency stop from any source
            return "emergency_stop"
        observation = lc.compact_result(result)
        print("Tool result: " + json.dumps(observation, ensure_ascii=False))
        history.append({"step": step, "tool": action.tool, "observation": observation})

        if action.tool == "stop_robot" or action.done:
            return "llm_stop"
        if action.tool == "detect_object" and lc.reached_from_detection(result):
            await lc.call_tool(session, "stop_robot", {}, metrics)
            return "success"

    await lc.call_tool(session, "stop_robot", {}, metrics)
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
            lc.check_tools(available, TOOLS, ARCH)

            print("Architecture A1 ready: the LLM drives continuous primitives.")
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
