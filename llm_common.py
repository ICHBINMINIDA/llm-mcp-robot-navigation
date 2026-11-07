#!/usr/bin/env python3
"""Shared client layer for architectures A, A1, A2, B and C.

Everything that must be identical on the client side lives here:

* the Ollama call and its sampling options (same model, same context window,
  same token budget in all five architectures)
* robust JSON extraction with a repair counter, so truncated answers are
  measured instead of silently patched
* target-object extraction with a deterministic fallback
* a single permanent stdin reader (``Terminal``) so the user can always type
  ``stop`` while the robot is moving, without leaking blocked input threads
* the run metrics written to the same JSONL file as the server metrics
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import requests
from mcp import ClientSession, StdioServerParameters

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:18080/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Identical sampling options everywhere.
# num_ctx must be large enough for the system prompt AND the history, otherwise
# Ollama silently truncates the beginning of the prompt (i.e. the rules).
# num_predict must be large enough for a complete JSON action, otherwise the
# model returns truncated JSON.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "120"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))

ROS_SETUP = os.getenv("ROS_SETUP", "/opt/ros/humble/setup.bash")
ROS_DOMAIN_ID = os.getenv("ROS_DOMAIN_ID", "34")
WORKDIR = os.getenv("ROBOT_WORKDIR", "/home/kilab/thb/ros2/ws2425/opencv_start")

TOOL_TIMEOUT_SEC = float(os.getenv("TOOL_TIMEOUT_SEC", "20"))
DETECT_TIMEOUT_SEC = float(os.getenv("DETECT_TIMEOUT_SEC", "90"))
SKILL_TIMEOUT_SEC = float(os.getenv("BEHAVIOR_TIMEOUT_SEC", "120"))

STOP_AREA_RATIO = float(os.getenv("STOP_AREA_RATIO", "0.05"))
STOP_BBOX_H_RATIO = float(os.getenv("STOP_BBOX_H_RATIO", "0.55"))
LEFT_TH = float(os.getenv("LEFT_TH", "0.40"))
RIGHT_TH = float(os.getenv("RIGHT_TH", "0.60"))

STOP_FLAG = os.getenv("ROBOT_STOP_FLAG", "/tmp/robot_llm_stop.flag")

METRICS_PATH = os.getenv("METRICS_PATH",
                         os.path.join(os.getenv("MCP_ROBOT_OUTPUT_DIR", "/tmp/mcp_robot"),
                                      "metrics.jsonl"))

TARGET_PROMPT = """
You extract the target object of a robot navigation task.
Return JSON only: {"target_object":"short object phrase"}
Rules:
- Keep the colour and the noun, e.g. "red ball", "blue bottle", "chair".
- If the user typed only an object name, use it as is.
- Correct obvious typing mistakes, e.g. "bred ball" -> "red ball".
""".strip()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@dataclass
class RunMetrics:
    """Per-run counters. The same fields are produced by every architecture."""

    architecture: str
    path: str = METRICS_PATH
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    target: str = ""
    started: float = field(default_factory=time.time)
    llm_calls: int = 0
    llm_seconds: float = 0.0
    tool_calls: int = 0
    detect_calls: int = 0
    json_repairs: int = 0
    json_failures: int = 0
    invalid_actions: int = 0
    steps: int = 0
    outcome: str = "unknown"

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()

    def new_run(self, command: str = "") -> None:
        """Start a new run. Call it BEFORE target extraction, so that the
        extraction LLM call is counted in (and logged under) this run."""
        self.run_id = uuid.uuid4().hex[:12]
        self.target = ""
        self.started = time.time()
        self.llm_calls = self.tool_calls = self.detect_calls = 0
        self.llm_seconds = 0.0
        self.json_repairs = self.json_failures = self.invalid_actions = self.steps = 0
        self.outcome = "unknown"
        self.event("run_start", command=command)

    def set_target(self, target: str) -> None:
        self.target = target
        self.event("target", target=target)

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.started, 2)

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            "architecture": self.architecture,
            "side": "client",
            "event": event,
        }
        record.update(fields)
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def finish(self, outcome: str) -> Dict[str, Any]:
        self.outcome = outcome
        summary = {
            "outcome": outcome,
            "target": self.target,
            "elapsed_sec": self.elapsed,
            "llm_calls": self.llm_calls,
            "llm_seconds": round(self.llm_seconds, 2),
            "tool_calls": self.tool_calls,
            "detect_calls": self.detect_calls,
            "steps": self.steps,
            "json_repairs": self.json_repairs,
            "json_failures": self.json_failures,
            "invalid_actions": self.invalid_actions,
        }
        self.event("run_end", **summary)
        return summary


# --------------------------------------------------------------------------- #
# JSON handling
# --------------------------------------------------------------------------- #
def extract_json(text: str, metrics: Optional[RunMetrics] = None) -> Dict[str, Any]:
    """Parse the first JSON object in ``text``, repairing a truncated tail.

    Repairs are counted rather than hidden: a high repair rate means
    ``OLLAMA_NUM_PREDICT`` is too small for the architecture's prompt.
    """
    text = (text or "").strip()
    start = text.find("{")
    if start < 0:
        if metrics:
            metrics.json_failures += 1
        raise ValueError(f"No JSON found in LLM response: {text[:300]}")

    end = text.rfind("}") + 1
    candidate = text[start:end] if end > start else text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = candidate.rstrip().rstrip(",")
        missing = repaired.count("{") - repaired.count("}")
        if missing > 0:
            repaired += "}" * missing
        try:
            value = json.loads(repaired)
            if metrics:
                metrics.json_repairs += 1
            return value
        except json.JSONDecodeError as exc:
            if metrics:
                metrics.json_failures += 1
            raise ValueError(f"Invalid JSON from LLM: {exc}. Raw: {candidate[:300]}") from exc


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
async def ask_ollama(prompt: str, metrics: Optional[RunMetrics] = None) -> Dict[str, Any]:
    def request() -> Dict[str, Any]:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": OLLAMA_TEMPERATURE,
                    "top_p": OLLAMA_TOP_P,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            },
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    started = time.monotonic()
    raw = await asyncio.to_thread(request)
    duration = time.monotonic() - started
    if metrics:
        metrics.llm_calls += 1
        metrics.llm_seconds += duration
        metrics.event("llm_call", duration_sec=round(duration, 3))
    return extract_json(raw, metrics)


def fallback_target(command: str) -> str:
    goal = re.sub(r"\s+", " ", command.strip().lower())
    for prefix in ("go to the ", "go to ", "find the ", "find ", "approach the ",
                   "approach ", "search for the ", "search for ", "look for the ",
                   "look for ", "drive to the ", "drive to "):
        if goal.startswith(prefix):
            return goal[len(prefix):].strip()
    return goal


async def start_run(command: str, metrics: RunMetrics) -> str:
    """Begin a task: re-arm the emergency stop, open the run, extract the target.

    Typing a new goal is the explicit operator action that clears a latched
    emergency stop flag; nothing else on the client or server clears it.
    """
    clear_stop_flag()
    metrics.new_run(command)
    target = await infer_target(command, metrics)
    metrics.set_target(target)
    return target


async def infer_target(command: str, metrics: Optional[RunMetrics] = None) -> str:
    """LLM target extraction with a deterministic fallback (same in all clients)."""
    try:
        raw = await ask_ollama(f"{TARGET_PROMPT}\n\nUser command:\n{command}", metrics)
        target = str(raw.get("target_object", "")).strip().lower()
        return target or fallback_target(command)
    except Exception as exc:
        if metrics:
            metrics.event("target_extraction_failed", error=str(exc))
        return fallback_target(command)


# --------------------------------------------------------------------------- #
# MCP helpers
# --------------------------------------------------------------------------- #
def server_params(script: str) -> StdioServerParameters:
    command = (f"source {ROS_SETUP} && export ROS_DOMAIN_ID={ROS_DOMAIN_ID} && "
               f"cd {WORKDIR} && python3 {script}")
    return StdioServerParameters(command="bash", args=["-lc", command])


def tool_result_to_python(result: Any) -> Any:
    content = getattr(result, "content", result)
    if isinstance(content, list) and content:
        content = getattr(content[0], "text", content[0])
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            return content
    return content


async def call_tool(session: ClientSession, tool: str, args: Dict[str, Any],
                    metrics: Optional[RunMetrics] = None,
                    timeout: Optional[float] = None) -> Any:
    if timeout is None:
        if tool == "detect_object":
            timeout = DETECT_TIMEOUT_SEC
        elif tool in {"go_to_object", "search_target", "align_to_target", "approach_target"}:
            timeout = SKILL_TIMEOUT_SEC
        else:
            timeout = TOOL_TIMEOUT_SEC

    started = time.monotonic()
    result = await asyncio.wait_for(session.call_tool(tool, args), timeout=timeout)
    parsed = tool_result_to_python(result)
    duration = round(time.monotonic() - started, 3)

    if metrics:
        metrics.tool_calls += 1
        if tool == "detect_object":
            metrics.detect_calls += 1
        metrics.event("tool_call", tool=tool, arguments=args, duration_sec=duration,
                      result=compact_result(parsed))

    if isinstance(parsed, str) and "Error executing tool" in parsed:
        raise RuntimeError(parsed)
    return parsed


def set_stop_flag() -> None:
    """Raise the emergency flag the servers poll in their control loops.

    This is the stop layer that does not depend on the MCP transport: it works
    even if the server is busy inside a skill, and it survives a broken pipe.
    """
    try:
        os.makedirs(os.path.dirname(STOP_FLAG) or ".", exist_ok=True)
        with open(STOP_FLAG, "w", encoding="utf-8") as handle:
            handle.write("stop\n")
    except Exception:
        pass


def clear_stop_flag() -> None:
    try:
        os.remove(STOP_FLAG)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def stop_flag_set() -> bool:
    """True while the emergency stop is latched (raised by anyone, any process)."""
    return os.path.exists(STOP_FLAG)


async def emergency_stop(session: ClientSession, metrics: Optional[RunMetrics] = None) -> None:
    """Stop the robot through both client-side layers.

    1. the flag file: instantaneous, independent of the transport;
    2. ``stop_robot`` over MCP: clears the server state and confirms.
    """
    set_stop_flag()
    if metrics:
        metrics.event("emergency_stop")
    try:
        await call_tool(session, "stop_robot", {}, metrics, timeout=5.0)
    except Exception as exc:
        if metrics:
            metrics.event("stop_robot_failed", error=str(exc))
        print(f"NOTE: stop_robot over MCP failed ({exc}); the stop flag is raised.")


def compact_result(result: Any) -> Any:
    """Small, uniform observation handed back to the LLM."""
    if not isinstance(result, dict):
        return result
    if "found" in result:
        return {key: result.get(key) for key in
                ("found", "object", "direction", "centroid_norm_x", "area_ratio",
                 "bbox_h_ratio", "confidence", "error")}
    if "behavior" in result:
        out = {key: result[key] for key in
               ("ok", "behavior", "object", "found", "aligned", "arrived", "reason",
                "elapsed_sec", "detections") if key in result}
        detection = result.get("result") or result.get("last_detection")
        if isinstance(detection, dict):
            out["detection"] = {key: detection.get(key) for key in
                                ("found", "direction", "centroid_norm_x",
                                 "area_ratio", "bbox_h_ratio", "error")
                                if key in detection}
        return out
    return result


def reached_from_detection(result: Any) -> bool:
    """Camera-only completion check. Identical threshold in every architecture."""
    if not isinstance(result, dict) or not result.get("found"):
        return False
    return (float(result.get("area_ratio") or 0.0) >= STOP_AREA_RATIO
            or float(result.get("bbox_h_ratio") or 0.0) >= STOP_BBOX_H_RATIO)


# --------------------------------------------------------------------------- #
# terminal
# --------------------------------------------------------------------------- #
class Terminal:
    """One permanent stdin reader shared by the whole client.

    ``input()`` inside a thread cannot be cancelled; creating one thread per
    read leaks blocked threads that later swallow the user's commands.  A
    single reader thread feeding an asyncio queue avoids that entirely and
    keeps ``stop`` usable while a task is running.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":
                self.loop.call_soon_threadsafe(self.queue.put_nowait, "exit")
                return
            self.loop.call_soon_threadsafe(self.queue.put_nowait, line.rstrip("\n"))

    async def read(self, prompt: str = "") -> str:
        if prompt:
            print(prompt, end="", flush=True)
        return await self.queue.get()

    def drain(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: Dict[str, Any]
    feedback: str = ""
    done: bool = False


def check_tools(available: Set[str], required: Set[str], architecture: str) -> None:
    missing = required - available
    if missing:
        raise RuntimeError(f"Architecture {architecture}: server is missing {sorted(missing)}")
    forbidden = ({"go_to_object", "search_target", "align_to_target", "approach_target",
                  "move_forward", "move_backward", "turn_left", "turn_right",
                  "start_moving_forward", "start_turning_left"} - required) & available
    if forbidden:
        print(f"NOTE: the server also exposes {sorted(forbidden)}; "
              f"architecture {architecture} will not use them.")


def print_summary(summary: Dict[str, Any]) -> None:
    print("Run summary: " + json.dumps(summary, ensure_ascii=False))
