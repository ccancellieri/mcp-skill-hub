"""Native Codex runner for paired synthetic-development task evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from context_value import (
    _run_condition,
    build_cases,
    build_memory_questions,
    seed_store,
    sha256_json,
)
from context_value_tasks import (
    build_task_specs,
    counterbalanced_slots,
    fixture_snapshot,
    materialize_fixture,
)

CODEX_PATH = "/Applications/ChatGPT.app/Contents/Resources/codex"
MODEL = "gpt-6-astra"
REASONING_EFFORT = "high"
MAX_TIMEOUT = 600


def _usage_dict(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("usage") or value.get("token_usage") or value
    if not isinstance(raw, dict):
        return None
    input_tokens = raw.get("input_tokens")
    output_tokens = raw.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    cached = raw.get("cached_input_tokens", raw.get("cache_read_input_tokens",
                     input_details.get("cached_tokens")))
    reasoning = raw.get("reasoning_tokens", output_details.get("reasoning_tokens"))
    if cached is not None and not isinstance(cached, int):
        return None
    if reasoning is not None and not isinstance(reasoning, int):
        return None
    # Native input includes cached input; native output includes reasoning.
    # These are breakdown fields and must never be added again.
    total = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "cached_input_tokens": cached,
            "output_tokens": output_tokens, "reasoning_tokens": reasoning,
            "main_tokens": total, "auxiliary_tokens": 0,
            "token_source": "native", "total_tokens": total,
            "cached_input_inclusive": True, "reasoning_output_inclusive": True}


def parse_native_usage(payload: dict) -> dict | None:
    """Compatibility parser for a final event or direct usage payload."""
    return _usage_dict(payload)


def parse_codex_jsonl(text: str) -> tuple[list[dict], dict | None]:
    events = []
    usage = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")  # noqa: TRY004 - malformed serialized input
        events.append(value)
        candidates = [value, value.get("turn"), value.get("response"), value.get("data")]
        for candidate in candidates:
            parsed = _usage_dict(candidate)
            if parsed is not None:
                usage = parsed
    return events, usage


def codex_argv(codex_path: str, workspace: Path) -> list[str]:
    return [codex_path, "-a", "never", "exec", "--json", "--ephemeral", "--ignore-user-config",
            "--skip-git-repo-check", "--sandbox", "workspace-write",
            "--model", MODEL,
            "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
            "-C", str(workspace), "-"]


def acceptance(rows: list[dict]) -> dict:
    if not rows or any(row.get("status") != "complete" for row in rows):
        return {"decision": "unknown", "reason": "one or more required paired runs are incomplete"}
    paired = {}
    for row in rows:
        paired.setdefault((row["task_id"], row["repeat"]), {})[row["condition"]] = row
    required = {"A_no_hub", "B_build_context", "C_context_composer"}
    if any(set(value) != required for value in paired.values()):
        return {"decision": "unknown", "reason": "paired condition coverage is incomplete"}
    if any(row.get("usage") is None or row.get("correct") is None or
           row.get("critical_failure") is None for row in rows):
        return {"decision": "unknown", "reason": "native usage or correctness evidence is missing"}
    reductions = [
        (value["A_no_hub"]["usage"]["total_tokens"]
         - value["C_context_composer"]["usage"]["total_tokens"])
        / value["A_no_hub"]["usage"]["total_tokens"]
        for value in paired.values()
    ]
    reduction = statistics.median(reductions)
    correct = all(value["C_context_composer"]["correct"] >= value["A_no_hub"]["correct"]
                  for value in paired.values())
    no_critical = not any(value["C_context_composer"]["critical_failure"] for value in paired.values())
    return {"decision": "accept" if reduction >= .15 and correct and no_critical else "reject",
            "median_task_token_reduction": reduction, "correctness_no_regression": correct,
            "no_critical_failures": no_critical, "threshold": .15}


def project_variability(rows: list[dict]) -> dict:
    paired: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        if row.get("status") == "complete":
            paired.setdefault((row["task_id"], row["repeat"]), {})[row["condition"]] = row
    output = {}
    for project in ("skill-hub", "tellurion"):
        reductions = []
        baseline_success = []
        candidate_success = []
        for conditions in paired.values():
            if not {"A_no_hub", "C_context_composer"}.issubset(conditions):
                continue
            baseline, candidate = conditions["A_no_hub"], conditions["C_context_composer"]
            if baseline.get("project") != project or baseline["usage"]["total_tokens"] <= 0:
                continue
            reductions.append((baseline["usage"]["total_tokens"]
                               - candidate["usage"]["total_tokens"])
                              / baseline["usage"]["total_tokens"])
            baseline_success.append(bool(baseline["correct"]))
            candidate_success.append(bool(candidate["correct"]))
        output[project] = {
            "paired_repetitions": len(reductions),
            "median_token_reduction": statistics.median(reductions) if reductions else None,
            "min_token_reduction": min(reductions) if reductions else None,
            "max_token_reduction": max(reductions) if reductions else None,
            "baseline_success_rate": statistics.fmean(baseline_success) if baseline_success else None,
            "candidate_success_rate": statistics.fmean(candidate_success) if candidate_success else None,
        }
    return output


def _dataset_hash() -> str:
    return sha256_json({"retrieval_cases": build_cases(), "memory_questions": build_memory_questions(),
                        "synthetic_development_tasks": build_task_specs()})


def _harness_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1]
    paths = (
        "benchmarks/context_value.py",
        "benchmarks/context_value_tasks.py",
        "src/skill_hub/context_composer.py",
        "src/skill_hub/context_service.py",
        "src/skill_hub/context_learning.py",
        "src/skill_hub/store.py",
        "src/skill_hub/compression/digest.py",
    )
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _base_report(candidate_version: str | None, dataset_hash: str | None,
                 runs: list[dict], reason: str) -> dict:
    experimental_completed = sum(row.get("status") == "complete" for row in runs)
    promotion_runs = []
    for row in runs:
        if row.get("condition") not in {"A_no_hub", "C_context_composer"}:
            continue
        normalized = dict(row)
        normalized["raw_condition"] = normalized["condition"]
        normalized["condition"] = normalized["condition_role"]
        normalized["repetition"] = normalized["repeat"]
        normalized["success"] = normalized.get("correct")
        normalized["critical_error"] = normalized.get("critical_failure")
        usage = normalized.get("usage") if isinstance(normalized.get("usage"), dict) else {}
        normalized["token_source"] = usage.get("token_source")
        normalized["main_tokens"] = usage.get("main_tokens")
        normalized["auxiliary_tokens"] = usage.get("auxiliary_tokens")
        promotion_runs.append(normalized)
    completed = sum(row.get("status") == "complete" for row in promotion_runs)
    complete = completed == len(promotion_runs) == 96 and experimental_completed == 144
    return {"source": "evaluation_harness", "complete": complete,
            "status": "complete" if complete else "incomplete",
            "candidate_version": candidate_version, "selector_version": candidate_version,
            "dataset_hash": dataset_hash, "evaluation_dataset_hash": _dataset_hash(),
            "harness_sha256": _harness_sha256(),
            "runtime_source_hashes": _runtime_source_hashes(),
            "estimated": False,
            "model": MODEL, "reasoning_effort": REASONING_EFFORT,
            "unavailable_metrics": {
                "rereads": None,
                "user_corrections": None,
                "client_rss_after_bytes": None,
                "client_peak_rss_bytes": None,
                "reason": "native JSONL adapter does not expose these measurements",
            },
            "condition_roles": {"baseline": "A_no_hub", "control": "B_build_context",
                                "candidate": "C_context_composer"},
            "reason": None if complete else reason, "expected_runs": 96,
            "completed_runs": completed, "runs": promotion_runs,
            "experimental_expected_runs": 144,
            "experimental_completed_runs": experimental_completed,
            "experimental_runs": runs,
            "project_variability": project_variability(runs),
            "acceptance": acceptance(runs) if complete else
                {"decision": "unknown", "reason": "required paired runs are incomplete"}}


def pending_report(reason: str = "native Codex execution was not requested") -> dict:
    runs = [{**slot, "status": "pending", "pending_reason": reason, "usage": None,
             "correct": None, "critical_failure": None}
            for slot in counterbalanced_slots(build_task_specs())]
    return _base_report(None, None, runs, reason)


def _run_test(task: dict, workspace: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(task["test_argv"], cwd=workspace, capture_output=True,
                          text=True, timeout=timeout, check=False)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(cmd="whole synthetic-development run", timeout=0)
    return remaining


def _slot_key(row: dict) -> tuple[str, int, str]:
    return row["task_id"], row["repeat"], row["condition"]


def balanced_qualification_slots(all_slots: list[dict], tasks: list[dict]) -> list[dict]:
    selected_ids = []
    for project in ("skill-hub", "tellurion"):
        selected_ids.extend([task["id"] for task in tasks if task["project"] == project][:2])
    selected = set(selected_ids)
    return [slot for slot in all_slots if slot["task_id"] in selected and slot["repeat"] == 1]


def _write_checkpoint(output_path: Path, report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output_path)


def _journal_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".rows.jsonl")


def _append_row_journal(output_path: Path, row: dict) -> None:
    path = _journal_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _journal_rows(output_path: Path) -> list[dict]:
    path = _journal_path(output_path)
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"row journal line {line_number} is not an object")  # noqa: TRY004 - malformed serialized input
        rows.append(value)
    return rows


def _record_row(output_path: Path, candidate_version: str, dataset_hash: str,
                all_slots: list[dict], row_map: dict, row: dict,
                max_runs: int, qualification: bool) -> None:
    row_map[_slot_key(row)] = row
    _append_row_journal(output_path, row)
    _checkpoint_current(output_path, candidate_version, dataset_hash,
                        all_slots, row_map, max_runs, qualification)


def _validate_resume(report: dict, candidate_version: str, dataset_hash: str) -> None:
    expected = {
        "source": "evaluation_harness", "estimated": False,
        "selector_version": candidate_version, "dataset_hash": dataset_hash,
        "evaluation_dataset_hash": _dataset_hash(),
        "harness_sha256": _harness_sha256(), "model": MODEL,
        "runtime_source_hashes": _runtime_source_hashes(),
        "reasoning_effort": REASONING_EFFORT,
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise ValueError("resume checkpoint mismatch: " + ", ".join(mismatches))


def _condition_context(task: dict, condition: str, store: Any) -> str:
    case = next(row for row in build_cases() if row["topic"] == task["topic"])
    evaluation_case = {**case, "prompt": task["objective"]}
    return _run_condition(condition, evaluation_case, store)["context"]


def _prompt(task: dict, condition: str, context: str) -> str:
    evidence = context or "(No supplemental Skill Hub context in this condition.)"
    return f"""Complete this synthetic-development task in the current isolated fixture repository.

Objective: {task['objective']}
Condition: {condition}
Editable implementation files are listed in FIXTURE.json. Do not modify tests or FIXTURE.json.
Run the declared test command and finish only after it passes. Keep the change minimal.

Supplemental evidence:
{evidence}
"""


def run(*, output_path: Path, candidate_version: str, dataset_hash: str,
        codex_path: str = CODEX_PATH,
        timeout: int = MAX_TIMEOUT, max_runs: int = 0, resume: bool = False,
        qualification: bool = False) -> dict:
    task_list = build_task_specs()
    tasks = {task["id"]: task for task in task_list}
    all_slots = counterbalanced_slots(list(tasks.values()))
    previous: dict = {}
    if not resume and _journal_path(output_path).is_file():
        _journal_path(output_path).unlink()
    if resume and output_path.is_file():
        previous = json.loads(output_path.read_text())
        _validate_resume(previous, candidate_version, dataset_hash)
    previous_rows = previous.get("experimental_runs", []) if isinstance(previous, dict) else []
    recovered_rows = [*previous_rows, *_journal_rows(output_path)] if resume else previous_rows
    row_map = {_slot_key(row): row for row in recovered_rows
               if isinstance(row, dict) and row.get("status") == "complete"}
    eligible_slots = (balanced_qualification_slots(all_slots, task_list)
                      if qualification else all_slots)
    remaining_slots = [slot for slot in eligible_slots if _slot_key(slot) not in row_map]
    limit = len(remaining_slots) if max_runs <= 0 else min(max_runs, len(remaining_slots))
    selected_slots = remaining_slots[:limit]
    traces = output_path.parent / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    cases, questions = build_cases(), build_memory_questions()
    with tempfile.TemporaryDirectory(prefix="context-value-store-") as store_dir:
        store = seed_store(Path(store_dir) / "fixture.sqlite", cases, questions)
        try:
            for run_index, slot in enumerate(selected_slots):
                task = tasks[slot["task_id"]]
                with tempfile.TemporaryDirectory(prefix="context-value-task-") as temp_dir:
                    workspace = Path(temp_dir)
                    deadline = time.monotonic() + timeout
                    fixture = materialize_fixture(task, workspace)
                    fixture_contract = {"task_id": task["id"], "kind": "synthetic-development",
                                        "editable_files": fixture["editable_files"],
                                        "test_argv": task["test_argv"],
                                        "initial_snapshot": fixture["initial_snapshot"]}
                    (workspace / "FIXTURE.json").write_text(json.dumps(fixture_contract, indent=2) + "\n")
                    initial_snapshot = fixture_snapshot(workspace)
                    baseline = _run_test(task, workspace, _remaining_seconds(deadline))
                    if baseline.returncode == 0:
                        row = {
                            **slot, "status": "fixture_invalid",
                            "reason": "baseline test unexpectedly passed; Codex was not invoked",
                            "initial_snapshot": initial_snapshot, "usage": None,
                            "correct": None, "critical_failure": True,
                        }
                        _record_row(output_path, candidate_version, dataset_hash,
                                    all_slots, row_map, row, max_runs, qualification)
                        continue
                    context = _condition_context(task, slot["condition"], store)
                    started = time.monotonic()
                    try:
                        completed = subprocess.run(
                            codex_argv(codex_path, workspace), input=_prompt(task, slot["condition"], context),
                            cwd=workspace,
                            capture_output=True, text=True,
                            timeout=_remaining_seconds(deadline), check=False,
                        )
                        trace_name = f"{run_index + 1:03d}-{task['id']}-r{slot['repeat']}-{slot['condition']}.jsonl"
                        trace_path = traces / trace_name
                        trace_path.write_text(completed.stdout)
                        events, usage = parse_codex_jsonl(completed.stdout)
                        after = _run_test(task, workspace, _remaining_seconds(deadline))
                        final_snapshot = fixture_snapshot(workspace)
                        changed_paths = sorted(
                            path for path in
                            set(initial_snapshot["files"]) | set(final_snapshot["files"])
                            if initial_snapshot["files"].get(path) != final_snapshot["files"].get(path)
                        )
                        protected_files_unchanged = set(changed_paths) <= set(fixture["editable_files"])
                        _remaining_seconds(deadline)
                        correct = after.returncode == 0 and protected_files_unchanged
                        status = "complete" if completed.returncode == 0 and usage is not None else "incomplete"
                        row = {**slot, "status": status, "elapsed_seconds": time.monotonic() - started,
                                     "returncode": completed.returncode, "usage": usage,
                                     "correct": correct,
                                     "critical_failure": not protected_files_unchanged,
                                     "baseline_test_returncode": baseline.returncode,
                                     "test_returncode": after.returncode,
                                     "protected_files_unchanged": protected_files_unchanged,
                                     "changed_paths": changed_paths,
                                     "initial_snapshot": initial_snapshot, "final_snapshot": final_snapshot,
                                     "context_sha256": sha256_json(context), "context_chars": len(context),
                                     "jsonl_event_count": len(events), "trace": str(trace_path),
                                     "test_stderr": after.stderr[-2000:], "stderr": completed.stderr[-2000:]}
                    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, ValueError) as exc:
                        row = {**slot, "status": "incomplete", "reason": type(exc).__name__,
                                     "detail": str(exc)[:1000], "initial_snapshot": initial_snapshot,
                                     "usage": None, "correct": None, "critical_failure": None}
                    _record_row(output_path, candidate_version, dataset_hash,
                                all_slots, row_map, row, max_runs, qualification)
        finally:
            store.close()
    pending_reason = ("balanced 12-run qualification; full matrix not requested"
                      if qualification else
                      (f"qualification limit --max-runs={max_runs}"
                       if limit < len(all_slots) else "run did not complete"))
    rows = [row_map.get(_slot_key(slot), {**slot, "status": "pending",
            "pending_reason": pending_reason, "usage": None, "correct": None,
            "critical_failure": None}) for slot in all_slots]
    report = _base_report(candidate_version, dataset_hash, rows, pending_reason)
    _write_checkpoint(output_path, report)
    return report


def _checkpoint_current(output_path: Path, candidate_version: str, dataset_hash: str,
                        all_slots: list[dict], row_map: dict, max_runs: int,
                        qualification: bool = False) -> None:
    reason = ("balanced 12-run qualification; full matrix not requested" if qualification
              else f"qualification limit --max-runs={max_runs}" if max_runs
              else "run interrupted or incomplete")
    rows = [row_map.get(_slot_key(slot), {**slot, "status": "pending",
            "pending_reason": reason, "usage": None, "correct": None,
            "critical_failure": None}) for slot in all_slots]
    _write_checkpoint(output_path, _base_report(candidate_version, dataset_hash, rows, reason))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/results/context-value-latest/e2e-results.json"))
    parser.add_argument("--run", action="store_true", help="invoke the native Codex adapter")
    parser.add_argument("--candidate-version", help="selector version under evaluation")
    parser.add_argument("--dataset-hash", help="training snapshot hash bound to the selector")
    parser.add_argument("--codex-path", default=CODEX_PATH)
    parser.add_argument("--timeout", type=int, default=MAX_TIMEOUT)
    parser.add_argument("--max-runs", type=int, default=0,
                        help="qualification cap; zero runs all 144 slots")
    parser.add_argument("--resume", action="store_true",
                        help="keep completed rows from a matching checkpoint and run remaining slots")
    parser.add_argument("--qualification", action="store_true",
                        help="run one repeat of two Python and two Rust tasks across A/B/C (12 runs)")
    args = parser.parse_args()
    if not args.run:
        report = pending_report()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        if not args.candidate_version or not args.dataset_hash:
            parser.error("--candidate-version and --dataset-hash are required with --run")
        report = run(output_path=args.output, candidate_version=args.candidate_version,
                     dataset_hash=args.dataset_hash,
                     codex_path=args.codex_path, timeout=min(max(args.timeout, 1), MAX_TIMEOUT),
                     max_runs=max(args.max_runs, 0), resume=args.resume,
                     qualification=args.qualification)
    print(json.dumps({key: report[key] for key in
                      ("status", "expected_runs", "completed_runs", "reason", "acceptance")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
