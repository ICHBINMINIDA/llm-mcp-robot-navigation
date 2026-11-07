"""Regression tests for the stop layers, the run metrics and JSON handling."""

import ast
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import llm_common as lc
import robot_common as rc

ROOT = Path(__file__).resolve().parent.parent


# ── latched emergency stop ────────────────────────────────────────────────────

def test_motion_works_without_the_flag(robot):
    result = robot.start_continuous("moving_forward", 0.1, 0.0)
    assert result["ok"] is True and robot.is_moving()


def test_continuous_motion_is_refused_while_flag_is_set(robot, stop_flag):
    stop_flag.write_text("stop\n")
    result = robot.start_continuous("moving_forward", 0.1, 0.0)
    assert result == rc.emergency_stop_refusal("moving_forward")
    assert not robot.is_moving()
    assert stop_flag.exists(), "a motion command must never clear the emergency stop"


def test_timed_motion_is_refused_while_flag_is_set(robot, stop_flag):
    stop_flag.write_text("stop\n")
    result = robot.run_timed(0.1, 0.0, 2.0, "move_forward")
    assert result["ok"] is False and result["interrupted"] is True
    assert result["elapsed_sec"] == 0.0
    assert stop_flag.exists()


def test_flag_raised_during_timed_motion_interrupts_it(robot, stop_flag):
    import threading

    threading.Timer(0.2, lambda: stop_flag.write_text("stop\n")).start()
    result = robot.run_timed(0.1, 0.0, 3.0, "move_forward")
    assert result["interrupted"] is True and result["elapsed_sec"] < 1.0
    assert not robot.is_moving()
    assert stop_flag.exists()


def test_skill_style_clear_stop_keeps_the_flag(robot, stop_flag):
    stop_flag.write_text("stop\n")
    robot.clear_stop()
    assert robot.stop_requested() is True
    assert stop_flag.exists()


def test_only_a_new_task_rearms_the_robot(stop_flag, monkeypatch, tmp_path):
    stop_flag.write_text("stop\n")

    async def fake_ollama(prompt, metrics=None):
        return {"target_object": "red ball"}

    monkeypatch.setattr(lc, "ask_ollama", fake_ollama)
    metrics = lc.RunMetrics("T", path=str(tmp_path / "m.jsonl"))
    asyncio.run(lc.start_run("go to the red ball", metrics))
    assert not stop_flag.exists()


# ── run metrics ───────────────────────────────────────────────────────────────

class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": '{"target_object": "red ball"}'}


def test_target_extraction_is_counted_in_the_new_run(stop_flag, monkeypatch, tmp_path):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(lc.requests, "post", lambda *a, **k: _FakeResponse())
    metrics = lc.RunMetrics("B", path=str(path))

    target = asyncio.run(lc.start_run("go to the red ball", metrics))
    summary = metrics.finish("success")

    assert target == "red ball"
    assert summary["llm_calls"] == 1, "the extraction call must belong to this run"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["run_id"] for r in records} == {metrics.run_id}
    assert [r["event"] for r in records] == ["run_start", "llm_call", "target", "run_end"]


# ── JSON handling ─────────────────────────────────────────────────────────────

def test_truncated_json_is_repaired_and_counted():
    metrics = lc.RunMetrics.__new__(lc.RunMetrics)
    metrics.json_repairs = metrics.json_failures = 0
    assert lc.extract_json('{"tool":"turn_left","arguments":{"angle_deg":15}', metrics) == {
        "tool": "turn_left", "arguments": {"angle_deg": 15}}
    assert metrics.json_repairs == 1


def test_missing_json_is_a_counted_failure():
    metrics = lc.RunMetrics.__new__(lc.RunMetrics)
    metrics.json_repairs = metrics.json_failures = 0
    with pytest.raises(ValueError):
        lc.extract_json("I will turn left now.", metrics)
    assert metrics.json_failures == 1


@pytest.mark.parametrize("command,target", [
    ("go to the red ball", "red ball"),
    ("Find   the Blue Bottle", "blue bottle"),
    ("chair", "chair"),
])
def test_fallback_target(command, target):
    assert lc.fallback_target(command) == target


# ── configuration and syntax ──────────────────────────────────────────────────

@pytest.mark.parametrize("path", sorted(ROOT.glob("*.py")), ids=lambda p: p.name)
def test_every_module_parses_on_this_python(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_architecture_c_keeps_its_own_step_budget():
    """experiment_env.sh exports MAX_AGENT_STEPS=45 for the primitive architectures."""
    env = {**os.environ, "MAX_AGENT_STEPS": "45", "HISTORY_LIMIT": "5", "PYTHONPATH": str(ROOT)}
    env.pop("C_MAX_AGENT_STEPS", None)
    code = ("import sys, types; sys.modules['voice_input'] = types.SimpleNamespace("
            "WAKE_WORD='hi', VoiceInput=object); import c_client; "
            "print(c_client.MAX_STEPS, c_client.HISTORY_LIMIT)")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["10", "6"]
