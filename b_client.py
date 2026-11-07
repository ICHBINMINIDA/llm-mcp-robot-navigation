#!/usr/bin/env python3
"""Architecture B client: the LLM only names the target; the server navigates.

Research role: minimum LLM authority. The model performs language understanding
(target extraction) and nothing else, so this architecture is the lower bound
of "how much autonomy does the LLM actually need?".

The stop path is real here: ``go_to_object`` runs in a worker thread on the
server, so ``stop_robot`` is answered immediately instead of queueing behind
the skill.
"""

from __future__ import annotations

import asyncio
import json
from typing import Set

from mcp import ClientSession
from mcp.client.stdio import stdio_client

import llm_common as lc

ARCH = "B"
SERVER_SCRIPT = "b_server.py"
TOOLS: Set[str] = {"go_to_object", "stop_robot"}


async def run_task(session: ClientSession, command: str, metrics: lc.RunMetrics,
                   terminal: lc.Terminal) -> str:
    target = await lc.start_run(command, metrics)
    if not target:
        print("Robot answer: I could not identify the target object.")
        return "no_target"

    print(f"Target object: {target}")
    print("Robot feedback: starting go_to_object. Type 'stop' to interrupt.")
    metrics.steps = 1

    skill = asyncio.create_task(
        lc.call_tool(session, "go_to_object", {"object_name": target}, metrics))

    while not skill.done():
        reader = asyncio.create_task(terminal.read())
        done, _ = await asyncio.wait({skill, reader}, return_when=asyncio.FIRST_COMPLETED)
        if skill in done:
            reader.cancel()
            break
        text = reader.result().strip().lower()
        if text in {"stop", "exit", "quit"}:
            await lc.emergency_stop(session, metrics)
            try:
                await asyncio.wait_for(skill, 5.0)
            except asyncio.TimeoutError:
                skill.cancel()
            print("Robot answer: stopped by user.")
            return "user_stop"
        if text:
            print("The skill is running. Type 'stop' to interrupt it.")

    result = skill.result()
    print("Tool result: " + json.dumps(lc.compact_result(result), ensure_ascii=False))
    if isinstance(result, dict) and result.get("arrived"):
        print(f"Robot answer: I reached the {target} and stopped.")
        return "success"
    reason = result.get("reason", "unknown") if isinstance(result, dict) else "unknown"
    print(f"Robot answer: I stopped before reaching the target. reason={reason}")
    return reason


async def main() -> None:
    metrics = lc.RunMetrics(ARCH)
    terminal = lc.Terminal(asyncio.get_running_loop())

    async with stdio_client(lc.server_params(SERVER_SCRIPT)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available = {tool.name for tool in (await session.list_tools()).tools}
            lc.check_tools(available, TOOLS, ARCH)

            print("Architecture B ready: the LLM extracts the object, the server navigates.")
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
                    outcome = await run_task(session, command, metrics, terminal)
                    lc.print_summary(metrics.finish(outcome))
                except Exception as exc:
                    print(f"Agent error: {exc}")
                    lc.print_summary(metrics.finish("error"))
                    try:
                        await lc.call_tool(session, "stop_robot", {}, metrics)
                    except Exception:
                        pass


if __name__ == "__main__":
    asyncio.run(main())
