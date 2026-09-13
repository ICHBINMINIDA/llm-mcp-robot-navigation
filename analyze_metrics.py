#!/usr/bin/env python3
"""Aggregate the JSONL traces into the per-architecture comparison table.

Usage:
    python3 analyze_metrics.py [/tmp/mcp_robot/metrics.jsonl] [--csv results.csv]

Every client writes one ``run_start`` and one ``run_end`` record per trial, plus
one record per tool call and per LLM call. This script turns that into the
numbers the evaluation chapter needs:

    success rate, time to target, LLM calls, LLM latency share, tool calls,
    detections, JSON repair rate and invalid-action rate.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List

SUCCESS = {"success"}


def load(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    runs = [r for r in records if r.get("event") == "run_end"]
    by_arch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_arch[run.get("architecture", "?")].append(run)

    table: Dict[str, Dict[str, Any]] = {}
    for architecture, items in sorted(by_arch.items()):
        successes = [r for r in items if r.get("outcome") in SUCCESS]
        times = [float(r.get("elapsed_sec") or 0.0) for r in successes]
        llm_calls = [int(r.get("llm_calls") or 0) for r in items]
        llm_seconds = [float(r.get("llm_seconds") or 0.0) for r in items]
        tool_calls = [int(r.get("tool_calls") or 0) for r in items]
        detect_calls = [int(r.get("detect_calls") or 0) for r in items]
        elapsed = [float(r.get("elapsed_sec") or 0.0) for r in items]
        repairs = sum(int(r.get("json_repairs") or 0) for r in items)
        invalid = sum(int(r.get("invalid_actions") or 0) for r in items)

        mean = lambda xs: round(statistics.mean(xs), 2) if xs else 0.0      # noqa: E731
        table[architecture] = {
            "runs": len(items),
            "success_rate": round(len(successes) / len(items), 3) if items else 0.0,
            "median_time_to_target_sec": round(statistics.median(times), 2) if times else None,
            "mean_time_to_target_sec": mean(times),
            "mean_llm_calls": mean(llm_calls),
            "mean_tool_calls": mean(tool_calls),
            "mean_detect_calls": mean(detect_calls),
            "llm_time_share": (round(sum(llm_seconds) / sum(elapsed), 3)
                               if sum(elapsed) > 0 else 0.0),
            "json_repairs_per_run": mean([int(r.get("json_repairs") or 0) for r in items]),
            "total_json_repairs": repairs,
            "total_invalid_actions": invalid,
            "outcomes": dict(sorted(
                (outcome, sum(1 for r in items if r.get("outcome") == outcome))
                for outcome in {str(r.get("outcome")) for r in items})),
        }
    return table


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "/tmp/mcp_robot/metrics.jsonl"
    table = summarise(load(path))

    print(json.dumps(table, ensure_ascii=False, indent=2))

    if "--csv" in sys.argv:
        target = sys.argv[sys.argv.index("--csv") + 1]
        fields = ["architecture", "runs", "success_rate", "median_time_to_target_sec",
                  "mean_time_to_target_sec", "mean_llm_calls", "mean_tool_calls",
                  "mean_detect_calls", "llm_time_share", "json_repairs_per_run"]
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for architecture, row in table.items():
                writer.writerow({"architecture": architecture, **row})
        print(f"CSV written to {target}")


if __name__ == "__main__":
    main()
