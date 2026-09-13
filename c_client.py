#!/usr/bin/env python3
"""Architecture C client: the LLM plans and repairs a sequence of skills.

Research role: the LLM is a high-level planner. The interesting behaviour is
not the happy path (search, align, approach) but recovery: what does the model
do when a skill reports ``target_not_visible`` or ``timeout``? The prompt makes
that decision space explicit so it can actually be measured.
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
from voice_input import WAKE_WORD, VoiceInput

ARCH = "C"
SERVER_SCRIPT = "c_server.py"

MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "6"))
MAX_REPAIRS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))

SKILLS: Set[str] = {"search_target", "align_to_target", "approach_target"}
TOOLS: Set[str] = SKILLS | {"stop_robot"}

SYSTEM_PROMPT = """
You are the skill planner of a real camera-only mobile robot.
The task is always: go to the target object.

Skills (one per answer):
- search_target(object_name): rotates and drives until the object is seen.
  Success: found = true.
- align_to_target(object_name): rotates until the object is centred.
  Success: aligned = true.
- approach_target(object_name): drives to the object and stops close to it.
  Success: arrived = true.
- stop_robot(): ends the task.

Nominal plan: search_target, then align_to_target, then approach_target,
then stop_robot with done = true.

Recovery rules:
- reason = target_not_visible after align or approach: call search_target again.
- search_target returns found = false (timeout): call stop_robot with done = true,
  the object is not reachable.
- align_to_target fails twice in a row: call search_target instead.
- approach_target returns arrived = true: call stop_robot with done = true.
- Never repeat the same failing skill more than twice in a row.

Answer with JSON only, no other text:
{"tool":"tool_name","arguments":{},"feedback":"max 8 words","done":false}
""".strip()


def validate(raw: Dict[str, Any], target: str, available: Set[str]) -> lc.ToolAction:
    if not isinstance(raw, dict):
        raise ValueError("The action must be a JSON object")
    tool = raw.get("tool")
    feedback = str(raw.get("feedback", "")).strip()
    done = bool(raw.get("done", False))

    if not isinstance(tool, str) or tool not in TOOLS:
        raise ValueError(f"Architecture C forbids this tool: {tool}")
    if tool not in available:
        raise ValueError(f"Tool not available on the server: {tool}")

    if done:
        return lc.ToolAction("stop_robot", {}, feedback or "Task complete.", True)
    args = {"object_name": target} if tool in SKILLS else {}
    return lc.ToolAction(tool, args, feedback, False)


def build_prompt(target: str, history: Deque[Dict[str, Any]], step: int) -> str:
    return (f"{SYSTEM_PROMPT}\n\nTarget object: {target}\nStep: {step}/{MAX_STEPS}\n"
            f"Skill results so far:\n{json.dumps(list(history), ensure_ascii=False)}\n"
            f"Choose the next single skill. JSON only.")


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
            # Keep the full context in the repair prompt, otherwise the model
            # has to choose a skill without knowing the target or the history.
            prompt = (f"{build_prompt(target, history, step)}\n\n"
                      f"Your previous answer was invalid: {error}\n"
                      f"Invalid answer: {json.dumps(raw, ensure_ascii=False)}\n"
                      f"Allowed tools: {sorted(TOOLS & available)}\n"
                      f"Return corrected JSON only.")
    metrics.event("fallback_action", tool="stop_robot")
    return lc.ToolAction("stop_robot",{},
                         "Fallback: stop safely.", True)


async def run_skill(session: ClientSession, action: lc.ToolAction, terminal: lc.Terminal,
                    voice: VoiceInput, metrics: lc.RunMetrics) -> Any:
    """Run one skill while keyboard and microphone stay responsive."""
    if action.tool not in SKILLS:
        return await lc.call_tool(session, action.tool, action.arguments, metrics)

    hint = f"Type 'stop'{f" or say '{WAKE_WORD}'" if voice.available else ''} to interrupt."
    print(f"Robot is moving. {hint}")
    task = asyncio.create_task(lc.call_tool(session, action.tool, action.arguments, metrics))

    while not task.done():
        reader = asyncio.create_task(terminal.read())
        listener = asyncio.create_task(voice.read())
        done, pending = await asyncio.wait({task, reader, listener},
                                           return_when=asyncio.FIRST_COMPLETED)
        for pending_task in pending:
            if pending_task is not task:
                pending_task.cancel()

        if task in done:
            return task.result()

        if listener in done:
            # Barge-in: a new spoken command replaces the current goal.
            spoken = listener.result().strip()
            print("Robot answer: new spoken command received; stopping the active skill.")
            await lc.emergency_stop(session, metrics)
            try:
                await asyncio.wait_for(task, 5.0)
            except asyncio.TimeoutError:
                task.cancel()
            return {"ok": False, "behavior": action.tool,
                    "reason": "new_voice_command", "next_command": spoken}

        text = reader.result().strip().lower()
        if text in {"stop", "exit", "quit"}:
            await lc.emergency_stop(session, metrics)
            try:
                return await asyncio.wait_for(task, 5.0)
            except asyncio.TimeoutError:
                task.cancel()
                return {"ok": False, "behavior": action.tool, "reason": "user_stop_timeout"}
        if text:
            print(f"A skill is running. {hint}")
    return task.result()


async def read_goal(terminal: lc.Terminal, voice: VoiceInput) -> tuple[str, str]:
    """Next goal, from the keyboard or the microphone, whichever comes first."""
    typed = asyncio.create_task(terminal.read("\nGoal/Object> "))
    spoken = asyncio.create_task(voice.read())
    done, pending = await asyncio.wait({typed, spoken}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if spoken in done:
        return spoken.result().strip(), "voice"
    return typed.result().strip(), "keyboard"


async def run_task(session: ClientSession, command: str, available: Set[str],
                   metrics: lc.RunMetrics, terminal: lc.Terminal,
                   voice: VoiceInput, pending: Dict[str, Any]) -> str:
    lc.clear_stop_flag()
    target = await lc.infer_target(command, metrics)
    metrics.new_run(target)
    print(f"Target object: {target}")

    history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)

    for step in range(1, MAX_STEPS + 1):
        metrics.steps = step
        action = await decide(target, history, step, available, metrics)
        print(f"Robot feedback: {action.feedback}")

        try:
            result = await run_skill(session, action, terminal, voice, metrics)
        except Exception:
            await lc.call_tool(session, "stop_robot", {}, metrics)
            raise

        observation = lc.compact_result(result)
        print("Tool result: " + json.dumps(observation, ensure_ascii=False))
        history.append({"step": step, "tool": action.tool, "result": observation})

        if action.tool == "stop_robot":
            return "llm_stop"
        if isinstance(result, dict):
            if result.get("reason") == "new_voice_command":
                pending["command"] = str(result.get("next_command", "")).strip()
                return "new_voice_command"
            if result.get("reason") in {"interrupted_by_stop", "user_stop_timeout"}:
                return "user_stop"
            if result.get("behavior") == "approach_target" and result.get("arrived"):
                await lc.call_tool(session, "stop_robot", {}, metrics)
                return "success"

    await lc.call_tool(session, "stop_robot", {}, metrics)
    return "max_steps"


async def main() -> None:
    metrics = lc.RunMetrics(ARCH)
    loop = asyncio.get_running_loop()
    terminal = lc.Terminal(loop)
    voice = VoiceInput(loop)
    pending: Dict[str, Any] = {}

    try:
        async with stdio_client(lc.server_params(SERVER_SCRIPT)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                available = {tool.name for tool in (await session.list_tools()).tools}
                lc.check_tools(available, TOOLS, ARCH)

                print("Architecture C ready: search_target, align_to_target, approach_target.")
                if voice.available:
                    print(f"Say '{WAKE_WORD}' then the goal, or type it. 'stop' / 'exit' also work.")
                else:
                    print("Type an object ('red ball'), 'stop' or 'exit'.")
                    print(f"Voice input off: {voice.error}")

                while True:
                    if pending.get("command"):
                        command, source = pending.pop("command"), "voice"
                    else:
                        command, source = await read_goal(terminal, voice)
                    command = command.strip()
                    if not command:
                        continue
                    if source == "voice":
                        print(f"Using spoken goal: {command}")
                    if command.lower() in {"exit", "quit"}:
                        await lc.emergency_stop(session, metrics)
                        break
                    if command.lower() in {"stop", "stop robot"}:
                        await lc.emergency_stop(session, metrics)
                        print("Robot answer: robot stopped.")
                        continue
                    try:
                        outcome = await run_task(session, command, available, metrics,
                                                 terminal, voice, pending)
                        messages = {
                            "success": "Target reached, robot stopped.",
                            "llm_stop": "The planner ended the task.",
                            "max_steps": "Maximum number of skills reached, robot stopped.",
                            "user_stop": "Stopped by user.",
                            "new_voice_command": "Interrupted by a new spoken command.",
                        }
                        print(f"Robot answer: {messages.get(outcome, outcome)}")
                        lc.print_summary(metrics.finish(outcome))
                    except Exception as exc:
                        print(f"Agent error: {exc}")
                        lc.print_summary(metrics.finish("error"))
                        try:
                            await lc.call_tool(session, "stop_robot", {}, metrics)
                        except Exception:
                            pass
    finally:
        voice.close()


if __name__ == "__main__":
    asyncio.run(main())
