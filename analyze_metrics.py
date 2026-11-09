#!/usr/bin/env python3
"""Evaluation of the experiment traces for chapter 6 (architecture C).

Usage:
    python3 analyze_metrics.py [FILE_OR_DIR ...] [--protocol Versuchsprotokoll.csv]
                               [--arch C] [--csv DIR] [--latex]
    python3 analyze_metrics.py FILE --by-architecture                  # pre-study (table 4)

Without a path, every *.jsonl in the folder of $METRICS_PATH is read, i.e. all
sessions of the study. Identical lines are counted once.

Runs marked "nein" in the column "gueltig" of the protocol are excluded.

What is computed (see section 6.1):
  * outcomes per scenario, success rate with a 95 % Wilson interval
  * time to target (median, IQR), model time share, model calls per run
  * skill calls per successful run
  * decision behaviour: target behaviour in S8/S10, correctness of the
    declared outcome, next decision after every failed skill result
  * robustness of the model interface (JSON repairs, invalid actions, fallbacks)
  * perception hit rate (context metric)

Only client-side events are evaluated: server events carry their own process
run_id and cannot be linked to a client run. The skill results (including
``hits``) reach the client log through the ``tool_call`` events.

Note on timing: the target extraction call happens before ``new_run()``, so it
is NOT included in ``llm_calls``, ``llm_seconds`` or ``elapsed_sec`` of a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

SKILLS = {"search_target", "align_to_target", "approach_target"}
OUTCOMES = ["success", "wrong_object", "llm_stop", "fallback_stop", "max_steps",
            "user_stop", "error"]
# Manual correction from the protocol column "Ausgang_manuell": the camera stop
# condition was reached, but in front of the wrong object (observed by the operator).
MANUAL_OUTCOMES = {"falsches_objekt": "wrong_object"}
# The object is not reachable at the end of the run: absent from the start (S8)
# or removed permanently during the run (S11). Success is impossible there.
ABSENT_SCENARIOS = {"S8", "S11"}
LOSS_REASONS = {"target_not_visible", "target_not_found"}
Z = 1.96                       # 95 % interval
PLANNED = {f"S{i}": 10 for i in range(1, 12)}     # ten rounds, each scenario once per round

FAILURE_ORDER = [
    "target_not_visible",
    "target_not_found",
    "timeout (align/approach)",
    "timeout (search, hits=0)",
    "timeout (search, hits>0)",
    "timeout (search, hits n/a)",
]
DECISION_ORDER = ["search_target", "align_to_target", "approach_target",
                  "stop_robot", "fallback"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_paths(paths: List[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files += sorted(os.path.join(path, f) for f in os.listdir(path)
                            if f.endswith(".jsonl"))
        elif os.path.exists(path):
            files.append(path)
    return files


def load(files: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set = set()
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line in seen:
                    continue
                seen.add(line)
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def read_protocol(path: str) -> Dict[str, Dict[str, str]]:
    """run_id -> row of the experiment protocol (semicolon separated)."""
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            run_id = (row.get("run_id") or "").strip()
            if run_id:
                rows[run_id] = {k: (v or "").strip() for k, v in row.items() if k}
    return rows


def check_sessions(records: List[Dict[str, Any]], arch: Optional[str]) -> None:
    configs: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r.get("event") != "session_start" or (arch and r.get("architecture") != arch):
            continue
        key = json.dumps({k: r.get(k) for k in ("prompt_sha", "code_sha", "params")},
                         sort_keys=True)
        configs[key].append(time_text(r.get("ts")))
    print(f"\n=== Sitzungen: {sum(len(v) for v in configs.values())}, "
          f"Konfigurationen: {len(configs)} ===")
    for key, starts in configs.items():
        c = json.loads(key)
        print(f"prompt {c['prompt_sha']} | code {c['code_sha']} | Sitzungen: {', '.join(starts)}")
    if len(configs) > 1:
        print("WARNUNG: Die Sitzungen wurden mit unterschiedlichem Code, Prompt oder "
              "unterschiedlichen Parametern durchgeführt. Die Läufe sind nicht vergleichbar.")
        base = json.loads(next(iter(configs)))["params"]
        for key in list(configs)[1:]:
            other = json.loads(key)["params"]
            diff = {k: (base.get(k), other.get(k)) for k in set(base) | set(other)
                    if base.get(k) != other.get(k)}
            if diff:
                print(f"  abweichende Parameter: {diff}")


def time_text(ts: Any) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "?"


def wilson(k: int, n: int, z: float = Z) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def median_iqr(values: List[float]) -> Optional[Tuple[float, float, float]]:
    if not values:
        return None
    med = statistics.median(values)
    if len(values) < 2:
        return med, med, med
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return med, q1, q3


def scenario_sort_key(name: str) -> Tuple[int, int]:
    if name.upper().startswith("S") and name[1:].isdigit():
        return 0, int(name[1:])
    return 1, 0


def de(value: Optional[float], digits: int = 1) -> str:
    """German decimal comma for the thesis."""
    if value is None:
        return "–"
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: Optional[float]) -> str:
    return "–" if value is None else de(100 * value, 0) + " %"


# --------------------------------------------------------------------------- #
# run extraction
# --------------------------------------------------------------------------- #
def collect_runs(records: List[Dict[str, Any]], arch: Optional[str]) -> List[Dict[str, Any]]:
    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:                       # file order = time order
        if record.get("side") != "client":
            continue
        if arch and record.get("architecture") != arch:
            continue
        by_run[str(record.get("run_id"))].append(record)

    runs: List[Dict[str, Any]] = []
    for run_id, events in by_run.items():
        end = next((e for e in events if e.get("event") == "run_end"), None)
        if end is None:
            continue                               # incomplete run
        scenario = "UNSET"
        declared = None
        for event in events:
            if event.get("event") == "scenario":
                scenario = str(event.get("scenario") or "UNSET").upper()
            elif event.get("event") == "llm_declared_outcome":
                declared = event.get("declared")
        runs.append({"run_id": run_id, "scenario": scenario, "end": end,
                     "events": events, "declared": declared})
    return runs


def failure_label(result: Dict[str, Any]) -> Optional[str]:
    reason = result.get("reason")
    if reason in LOSS_REASONS:
        return reason
    if reason == "timeout":
        if result.get("behavior") == "search_target":
            hits = result.get("hits")
            if hits is None:
                return "timeout (search, hits n/a)"
            return "timeout (search, hits=0)" if int(hits) == 0 else "timeout (search, hits>0)"
        return "timeout (align/approach)"
    return None


def transitions(run: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(failed result, next MODEL decision). Client-forced stops are skipped."""
    events = run["events"]
    out: List[Tuple[str, str]] = []
    for index, event in enumerate(events):
        if event.get("event") != "tool_call" or event.get("tool") not in SKILLS:
            continue
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        label = failure_label(result)
        if label is None:
            continue
        saw_llm = fallback = False
        next_tool = None
        for later in events[index + 1:]:
            kind = later.get("event")
            if kind == "llm_call":
                saw_llm = True
            elif kind == "fallback_action":
                fallback = True
            elif kind == "tool_call":
                next_tool = later.get("tool")
                break
            elif kind == "run_end":
                break
        if not saw_llm:
            continue                               # e.g. forced stop after max_steps
        out.append((label, "fallback" if fallback else (next_tool or "none")))
    return out


def skill_results(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e["result"] for e in run["events"]
            if e.get("event") == "tool_call" and e.get("tool") in SKILLS
            and isinstance(e.get("result"), dict)]


def judge_declared(run: Dict[str, Any]) -> Optional[str]:
    if run["end"].get("outcome") != "llm_stop":
        return None
    declared = run["declared"] or "unspecified"
    if run["scenario"] == "UNSET":
        return "unbekannt (kein Szenario)"
    if declared == "unreachable":
        return "zutreffend" if run["scenario"] in ABSENT_SCENARIOS else "vorzeitige Aufgabe"
    if declared == "reached":
        return "unzutreffende Erfolgsmeldung"
    return "keine Angabe"


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_scenario[run["scenario"]].append(run)

    scenarios: Dict[str, Dict[str, Any]] = {}
    all_transitions: List[Tuple[str, str, str, str]] = []   # label, decision, outcome, scenario
    declared_counts: Counter = Counter()

    for name in sorted(by_scenario, key=scenario_sort_key):
        items = by_scenario[name]
        n = len(items)
        outcomes = Counter(r["end"].get("outcome", "unknown") for r in items)
        successes = [r for r in items if r["end"].get("outcome") == "success"]
        k = len(successes)

        rate = ci_low = ci_high = None
        if name not in ABSENT_SCENARIOS and n:
            rate = k / n
            ci_low, ci_high = wilson(k, n)

        times = [float(r["end"].get("elapsed_sec") or 0.0) for r in successes]
        skill_calls = [sum(1 for e in r["events"] if e.get("event") == "tool_call"
                           and e.get("tool") in SKILLS) for r in successes]
        llm_calls = [int(r["end"].get("llm_calls") or 0) for r in items]
        llm_sec = sum(float(r["end"].get("llm_seconds") or 0.0) for r in items)
        elapsed = sum(float(r["end"].get("elapsed_sec") or 0.0) for r in items)

        hits = detections = 0
        hits_known = True
        per_skill: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for run in items:
            for result in skill_results(run):
                got = int(result.get("detections") or 0)
                detections += got
                if result.get("hits") is None:
                    hits_known = False
                    continue
                hits += int(result["hits"])
                slot = per_skill[str(result.get("behavior") or "?")]
                slot[0] += int(result["hits"])
                slot[1] += got

        loss_runs = sum(1 for r in items
                        if any(res.get("reason") in LOSS_REASONS for res in skill_results(r)))

        # target behaviour
        target_ok = target_n = None
        searches_after_loss: List[float] = []
        if name in ABSENT_SCENARIOS:
            target_n = n
            target_ok = sum(1 for r in items if r["end"].get("outcome") == "llm_stop"
                            and r["declared"] == "unreachable")
        if name == "S11":
            # persistence: how many searches the model runs after the first loss report
            for run in items:
                calls = [e for e in run["events"] if e.get("event") == "tool_call"
                         and e.get("tool") in SKILLS and isinstance(e.get("result"), dict)]
                first = next((i for i, e in enumerate(calls)
                              if e["result"].get("reason") in LOSS_REASONS), None)
                if first is not None:
                    searches_after_loss.append(float(sum(
                        1 for e in calls[first + 1:] if e.get("tool") == "search_target")))
        elif name == "S10":
            target_n = target_ok = 0
            for run in items:
                first = next((d for label, d in transitions(run)
                              if label == "target_not_visible"), None)
                if first is None:
                    continue                       # occlusion did not trigger a loss report
                target_n += 1
                target_ok += int(first in SKILLS)

        for run in items:
            outcome = run["end"].get("outcome", "unknown")
            for label, decision in transitions(run):
                all_transitions.append((label, decision, outcome, name))
            verdict = judge_declared(run)
            if verdict:
                declared_counts[verdict] += 1

        scenarios[name] = {
            "runs": n,
            "outcomes": {o: outcomes.get(o, 0) for o in OUTCOMES},
            "other_outcomes": {o: c for o, c in outcomes.items() if o not in OUTCOMES},
            "success_rate": rate, "ci_low": ci_low, "ci_high": ci_high,
            "time_to_target": median_iqr(times),
            "skill_calls_per_success": median_iqr([float(x) for x in skill_calls]),
            "llm_calls_per_run": median_iqr([float(x) for x in llm_calls]),
            "llm_time_share": (llm_sec / elapsed) if elapsed > 0 else None,
            "json_repairs": sum(int(r["end"].get("json_repairs") or 0) for r in items),
            "invalid_actions": sum(int(r["end"].get("invalid_actions") or 0) for r in items),
            "fallbacks": sum(1 for r in items for e in r["events"]
                             if e.get("event") == "fallback_action"),
            "hit_rate": (hits / detections) if detections and hits_known else None,
            "hit_rate_by_skill": {name: (h / d) for name, (h, d) in per_skill.items() if d},
            "hits_logged": hits_known,
            "runs_with_loss_report": loss_runs,
            "target_behaviour": ({"conform": target_ok, "of": target_n}
                                 if target_n is not None else None),
            "searches_after_loss": median_iqr(searches_after_loss),
            "runs_with_loss_after_sighting": len(searches_after_loss),
        }

    # transition table
    table: Dict[str, Dict[str, Any]] = {}
    for label in FAILURE_ORDER:
        rows = [t for t in all_transitions if t[0] == label]
        if not rows:
            continue
        decisions = Counter(d for _, d, _, _ in rows)
        # Recovery is only meaningful where a success is possible at all.
        solvable = [t for t in rows if t[3] not in ABSENT_SCENARIOS]
        recovered = sum(1 for _, d, o, _ in solvable if o == "success")
        table[label] = {
            "n": len(rows),
            "decisions": {d: decisions.get(d, 0) for d in DECISION_ORDER},
            "n_solvable": len(solvable),
            "recovery_success": (recovered / len(solvable)) if solvable else None,
        }

    return {"scenarios": scenarios, "transitions": table,
            "declared_outcome": dict(declared_counts)}


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def print_report(result: Dict[str, Any]) -> None:
    scenarios = result["scenarios"]
    print("\n=== Ausgänge je Szenario (Tabelle 12) ===")
    header = ["Szen.", "n"] + OUTCOMES + ["Erfolg [95-%-KI]"]
    print(" | ".join(f"{h:>13}" for h in header))
    for name, s in scenarios.items():
        ci = ("–" if s["success_rate"] is None else
              f"{pct(s['success_rate'])} [{pct(s['ci_low'])}; {pct(s['ci_high'])}]")
        cells = [name, s["runs"]] + [s["outcomes"][o] for o in OUTCOMES] + [ci]
        print(" | ".join(f"{str(c):>13}" for c in cells))
        if s["other_outcomes"]:
            print(f"{'':>13}   weitere Ausgänge: {s['other_outcomes']}")

    print("\n=== Zeitverhalten und Modellschnittstelle ===")
    for name, s in scenarios.items():
        t = s["time_to_target"]
        sk = s["skill_calls_per_success"]
        lc = s["llm_calls_per_run"]
        time_text = f"Median {de(t[0])} s (IQR {de(t[1])}–{de(t[2])} s)" if t else "–"
        print(f"{name}: Zeit bis Ziel {time_text} | "
              f"Skills/Erfolg {de(sk[0], 1) if sk else '–'} | "
              f"LLM-Aufrufe/Lauf {de(lc[0], 1) if lc else '–'} | "
              f"Modellzeitanteil {pct(s['llm_time_share'])} | "
              f"JSON-Rep. {s['json_repairs']}, ungültig {s['invalid_actions']}, "
              f"Ersatz {s['fallbacks']} | Trefferquote "
              f"{pct(s['hit_rate']) if s['hits_logged'] else 'n/a (hits fehlt, K1)'}"
              + ("" if not s["hit_rate_by_skill"] else " (" + ", ".join(
                  f"{k.split('_')[0]} {pct(v)}" for k, v in sorted(s["hit_rate_by_skill"].items()))
                  + ")"))

    print("\n=== Sollverhalten ===")
    for name in ("S8", "S9", "S10", "S11"):
        s = scenarios.get(name)
        if not s:
            continue
        if name == "S9":
            bridged = s["runs"] - s["runs_with_loss_report"]
            print(f"S9: Verdeckung ohne Verlustmeldung überbrückt in {bridged} von {s['runs']} Läufen")
        elif s["target_behaviour"]:
            tb = s["target_behaviour"]
            text = ("Fortsetzung nach target_not_visible" if name == "S10"
                    else "Abbruch mit 'unreachable'")
            print(f"{name}: {text} in {tb['conform']} von {tb['of']} Fällen")
            if name == "S11":
                sa = s["searches_after_loss"]
                print(f"S11: Objekt vor der Entfernung gesehen in "
                      f"{s['runs_with_loss_after_sighting']} von {s['runs']} Läufen; "
                      f"Suchen nach dem Verlust bis zum Abbruch: Median "
                      f"{de(sa[0], 1) if sa else '–'}")

    print("\n=== Endeinschätzung des Modells (nur llm_stop) ===")
    for verdict, count in sorted(result["declared_outcome"].items()):
        print(f"{verdict}: {count}")

    print("\n=== Folgeentscheidungen nach Fehlschlägen (Tabelle 13) ===")
    for label, row in result["transitions"].items():
        decisions = ", ".join(f"{d}={c}" for d, c in row["decisions"].items())
        print(f"{label} (n={row['n']}): {decisions} | Wiederherstellungserfolg "
              f"{pct(row['recovery_success'])} (n={row['n_solvable']}, ohne S8/S11)")

    unset = scenarios.get("UNSET")
    if unset:
        print(f"\nWARNUNG: {unset['runs']} Läufe ohne Szenario-Präfix (UNSET).")


def write_csv(result: Dict[str, Any], directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "scenarios.csv"), "w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["scenario", "runs", *OUTCOMES, "success_rate", "ci_low", "ci_high",
                         "time_median", "time_q1", "time_q3", "skills_per_success_median",
                         "llm_calls_median", "llm_time_share", "json_repairs",
                         "invalid_actions", "fallbacks", "hit_rate", "runs_with_loss_report",
                         "target_conform", "target_of"])
        for name, s in result["scenarios"].items():
            t = s["time_to_target"] or (None, None, None)
            sk = s["skill_calls_per_success"] or (None,)
            lc = s["llm_calls_per_run"] or (None,)
            tb = s["target_behaviour"] or {"conform": None, "of": None}
            writer.writerow([name, s["runs"], *[s["outcomes"][o] for o in OUTCOMES],
                             s["success_rate"], s["ci_low"], s["ci_high"], *t, sk[0], lc[0],
                             s["llm_time_share"], s["json_repairs"], s["invalid_actions"],
                             s["fallbacks"], s["hit_rate"], s["runs_with_loss_report"],
                             tb["conform"], tb["of"]])
    with open(os.path.join(directory, "transitions.csv"), "w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["failure", "n", *DECISION_ORDER, "n_solvable", "recovery_success"])
        for label, row in result["transitions"].items():
            writer.writerow([label, row["n"], *[row["decisions"][d] for d in DECISION_ORDER],
                             row["n_solvable"], row["recovery_success"]])
    print(f"CSV-Dateien geschrieben nach {directory}")


def tex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def print_latex(result: Dict[str, Any]) -> None:
    print("\n% --- Tabelle 12 (benötigt \\usepackage{booktabs}) ---")
    print(r"\begin{tabular}{lrrrrrrrrl}")
    print(r"\toprule")
    print(r"Szenario & $n$ & success & wrong\_object & llm\_stop & fallback\_stop & max\_steps & "
          r"user\_stop & error & Erfolgsquote [95-\%-KI] \\")
    print(r"\midrule")
    for name, s in result["scenarios"].items():
        if name == "UNSET":
            continue
        ci = ("--" if s["success_rate"] is None else
              tex(f"{pct(s['success_rate'])} [{pct(s['ci_low'])}; {pct(s['ci_high'])}]"))
        cells = [name, s["runs"]] + [s["outcomes"][o] for o in OUTCOMES]
        print(" & ".join(str(c) for c in cells) + f" & {ci} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n% --- Tabelle 13 ---")
    print(r"\begin{tabular}{lrrrrrrl}")
    print(r"\toprule")
    print(r"Ergebnis & $n$ & search & align & approach & stop & Ersatz & "
          r"Wiederherstellung \\")
    print(r"\midrule")
    for label, row in result["transitions"].items():
        d = row["decisions"]
        print(f"{tex(label)} & {row['n']} & {d['search_target']} & {d['align_to_target']} & "
              f"{d['approach_target']} & {d['stop_robot']} & {d['fallback']} & "
              f"{tex(pct(row['recovery_success']))} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# --------------------------------------------------------------------------- #
# pre-study comparison (table 4), kept from the previous version
# --------------------------------------------------------------------------- #
def by_architecture(records: List[Dict[str, Any]]) -> None:
    ends = [r for r in records if r.get("event") == "run_end"]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in ends:
        grouped[run.get("architecture", "?")].append(run)
    for arch, items in sorted(grouped.items()):
        wins = [r for r in items if r.get("outcome") == "success"]
        times = [float(r.get("elapsed_sec") or 0.0) for r in wins]
        med = statistics.median(times) if times else None
        print(f"{arch}: Läufe {len(items)}, Erfolgsquote {pct(len(wins) / len(items))}, "
              f"Median Zeit bis Ziel {de(med)} s")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--protocol", metavar="CSV",
                        help="experiment protocol; rows with gueltig=nein are excluded")
    parser.add_argument("--arch", default="C")
    parser.add_argument("--csv", metavar="DIR")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--json", action="store_true", help="full result as JSON")
    parser.add_argument("--by-architecture", action="store_true")
    args = parser.parse_args()

    paths = args.paths
    if not paths and os.getenv("METRICS_PATH"):
        paths = [os.path.dirname(os.getenv("METRICS_PATH")) or "."]
    files = resolve_paths(paths)
    if not files:
        sys.exit(f"No metrics files found in {paths!r} (pass a path or set METRICS_PATH)")

    records = load(files)
    if args.by_architecture:
        by_architecture(records)
        return

    runs = collect_runs(records, args.arch)
    if not runs:
        sys.exit(f"No completed runs for architecture {args.arch} in {files}")

    print(f"Dateien: {', '.join(os.path.basename(f) for f in files)} | Architektur {args.arch}")
    check_sessions(records, args.arch)

    excluded: Counter = Counter()
    if args.protocol:
        protocol = read_protocol(args.protocol)
        kept = []
        for run in runs:
            row = protocol.get(run["run_id"])
            if row is None:
                if run["scenario"] != "UNSET":
                    print(f"HINWEIS: Lauf {run['run_id']} ({run['scenario']}) fehlt im Protokoll.")
                kept.append(run)
                continue
            if row.get("Szenario") and row["Szenario"].upper() != run["scenario"]:
                # The prefix is only an annotation; it is stripped before the command
                # reaches the system. The protocol, written during the run, decides.
                print(f"KORREKTUR: Lauf {run['run_id']}: Szenario {run['scenario']} "
                      f"(Log) -> {row['Szenario'].upper()} (Protokoll).")
                run["scenario"] = row["Szenario"].upper()
            if row.get("gueltig", "").lower() == "nein":
                excluded[row.get("Grund") or "ohne Angabe"] += 1
                continue
            manual = row.get("Ausgang_manuell", "").lower()
            if manual:
                if manual not in MANUAL_OUTCOMES:
                    print(f"WARNUNG: Lauf {run['run_id']}: unbekannter manueller Ausgang "
                          f"'{manual}' (erlaubt: {', '.join(MANUAL_OUTCOMES)}).")
                else:
                    before = run["end"].get("outcome")
                    run["end"] = dict(run["end"], outcome=MANUAL_OUTCOMES[manual])
                    print(f"KORREKTUR: Lauf {run['run_id']} ({run['scenario']}): "
                          f"{before} -> {MANUAL_OUTCOMES[manual]}")
            kept.append(run)
        runs = kept
    unset = sum(1 for r in runs if r["scenario"] == "UNSET")
    runs = [r for r in runs if r["scenario"] != "UNSET"]

    print(f"\nGültige Läufe: {len(runs)} | ohne Szenario (z. B. Aufwärmlauf): {unset} | "
          f"laut Protokoll ausgeschlossen: {sum(excluded.values())}")
    for reason, count in excluded.items():
        print(f"  ausgeschlossen – {reason}: {count}")

    counts = Counter(r["scenario"] for r in runs)
    print("\n=== Fortschritt (gültig / geplant) ===")
    print("  ".join(f"{name}: {counts.get(name, 0)}/{need}"
                    + (" ✓" if counts.get(name, 0) >= need else "")
                    for name, need in PLANNED.items()))
    if not runs:
        return
    result = evaluate(runs)
    print_report(result)
    if args.latex:
        print_latex(result)
    if args.csv:
        write_csv(result, args.csv)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
