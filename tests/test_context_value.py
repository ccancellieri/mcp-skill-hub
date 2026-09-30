"""Acceptance and reproducibility tests for the context-value benchmark."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


def _load(name: str):
    path = ROOT / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


context_value = _load("context_value")
context_value_tasks = _load("context_value_tasks")
context_value_e2e = _load("context_value_e2e")


def test_frozen_case_counts_splits_languages_and_unique_prompts():
    cases = context_value.build_cases()
    questions = context_value.build_memory_questions()

    assert len(cases) == 120
    assert sum(case["split"] == "calibration" for case in cases) == 40
    assert sum(case["split"] == "holdout" for case in cases) == 80
    assert len({case["prompt"] for case in cases}) == 120
    assert len(questions) == 24
    assert {q["language"] for q in questions} == {"en", "it"}
    assert {q["category"] for q in questions} == {"updated", "conflicting", "unanswerable"}
    assert all("synthetic" in task["source"] for task in context_value_tasks.build_task_specs())


def test_dev_matrix_has_24_tasks_two_repeats_and_counterbalanced_conditions():
    tasks = context_value_tasks.build_task_specs()
    slots = context_value_tasks.counterbalanced_slots(tasks)

    assert len(tasks) == 24
    assert sum(task["project"] == "skill-hub" for task in tasks) == 12
    assert sum(task["project"] == "tellurion" for task in tasks) == 12
    assert len(slots) == 144
    for task in tasks:
        own = [slot for slot in slots if slot["task_id"] == task["id"]]
        assert len(own) == 6
        assert {slot["repeat"] for slot in own} == {1, 2}
        assert {slot["condition"] for slot in own} == {
            "A_no_hub", "B_build_context", "C_context_composer"
        }
    qualification = context_value_e2e.balanced_qualification_slots(slots, tasks)
    assert len(qualification) == 12
    assert {slot["project"] for slot in qualification} == {"skill-hub", "tellurion"}
    assert all(slot["repeat"] == 1 for slot in qualification)
    assert len({slot["task_id"] for slot in qualification}) == 4
    for task_id in {slot["task_id"] for slot in qualification}:
        assert {slot["condition"] for slot in qualification if slot["task_id"] == task_id} == {
            "A_no_hub", "B_build_context", "C_context_composer"
        }


def test_offline_execution_is_scoped_and_writes_hashed_manifest(tmp_path):
    report = context_value.run(tmp_path)

    assert report["manifest"]["counts"] == {
        "retrieval": 120, "calibration": 40, "holdout": 80, "memory_qa": 24
    }
    assert len(report["rows"]) == 432
    build_rows = [row for row in report["rows"] if row["condition"] == "B_build_context"]
    assert all(row["elapsed_ms"] >= 0 for row in report["rows"])
    assert all(row["process_rss_after_bytes"] > 0 for row in report["rows"])
    assert report["summary"]["B_build_context"]["latency_ms_p95"] >= 0
    assert report["summary"]["B_build_context"]["process_rss_after_bytes_p50"] > 0
    assert all(row["constraint_adherent"] for row in build_rows)
    assert not any(row["scope_leakage"] for row in build_rows)
    corpus = json.loads((tmp_path / "corpus.json").read_text())
    assert report["manifest"]["corpus_sha256"] == context_value.sha256_json(corpus)
    assert len(report["manifest"]["source_hashes"]) == 30
    composer_rows = [row for row in report["rows"] if row["condition"] == "C_context_composer"]
    assert all(row["compression_ratio_estimate"] is None or row["compression_ratio_estimate"] >= 0
               for row in composer_rows)
    assert all("selector_version" in row["raw"] for row in composer_rows)
    assert (tmp_path / "offline-results.json").is_file()


def test_native_usage_does_not_double_count_cached_input():
    usage = context_value_e2e.parse_native_usage({"usage": {
        "input_tokens": 100, "cached_input_tokens": 80,
        "output_tokens": 20, "reasoning_tokens": 5,
    }})
    assert usage["total_tokens"] == 120
    assert usage["main_tokens"] == 120
    assert usage["auxiliary_tokens"] == 0
    assert usage["token_source"] == "native"
    assert usage["cached_input_inclusive"] is True
    assert usage["reasoning_output_inclusive"] is True


def test_codex_jsonl_parser_uses_native_final_turn_usage_without_double_counting():
    jsonl = "\n".join((
        json.dumps({"type": "thread.started", "thread_id": "fixture"}),
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 200, "cached_input_tokens": 120,
            "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30},
        }}),
    ))
    events, usage = context_value_e2e.parse_codex_jsonl(jsonl)
    assert len(events) == 2
    assert usage["total_tokens"] == 250
    assert usage["reasoning_tokens"] == 30

    _, absent = context_value_e2e.parse_codex_jsonl(json.dumps({
        "type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}
    }))
    assert absent["cached_input_tokens"] is None
    assert absent["reasoning_tokens"] is None


def test_codex_adapter_is_fixed_bounded_and_preserves_execpolicy(tmp_path):
    argv = context_value_e2e.codex_argv("/native/codex", tmp_path)
    assert argv[:5] == ["/native/codex", "-a", "never", "exec", "--json"]
    assert "--ephemeral" in argv and "--ignore-user-config" in argv
    assert ["--sandbox", "workspace-write"] == argv[argv.index("--sandbox"):argv.index("--sandbox") + 2]
    assert "--ask-for-approval" not in argv
    assert "--ignore-rules" not in argv
    assert not any("bypass" in value for value in argv)


def test_every_synthetic_development_fixture_has_a_failing_baseline():
    for task in context_value_tasks.build_task_specs():
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = context_value_tasks.materialize_fixture(task, root)
            before = fixture["initial_snapshot"]
            completed = subprocess.run(task["test_argv"], cwd=root, capture_output=True,
                                       text=True, timeout=60, check=False)
            assert completed.returncode != 0, task["id"]
            assert before["files"]
            if task["language"] == "rust":
                assert "tests/contract.rs" in before["files"]
                assert "tests/contract.rs" not in fixture["editable_files"]


def test_fixture_snapshot_excludes_generated_caches(tmp_path):
    (tmp_path / "fixture.py").write_text("pass\n")
    (tmp_path / "Cargo.lock").write_text("generated\n")
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "artifact").write_text("cache")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "fixture.pyc").write_bytes(b"cache")
    snapshot = context_value_tasks.fixture_snapshot(tmp_path)
    assert snapshot["files"] == {"fixture.py": context_value_tasks.hashlib.sha256(b"pass\n").hexdigest()}


def test_acceptance_uses_median_of_paired_reductions():
    rows = []
    for task_id, baseline, candidate in (("a", 10, 0), ("b", 100, 90), ("c", 1000, 800)):
        for condition, total in (("A_no_hub", baseline), ("B_build_context", baseline),
                                 ("C_context_composer", candidate)):
            rows.append({"task_id": task_id, "repeat": 1, "condition": condition,
                         "status": "complete", "usage": {"total_tokens": total},
                         "correct": True, "critical_failure": False})
    result = context_value_e2e.acceptance(rows)
    assert result["median_task_token_reduction"] == 0.2
    assert result["decision"] == "accept"


def test_promotion_denies_unknown_and_requires_15_percent_without_regression():
    assert context_value_e2e.acceptance([])["decision"] == "unknown"
    rows = []
    for repeat in (1, 2):
        for condition, total in (("A_no_hub", 100), ("B_build_context", 95),
                                 ("C_context_composer", 84)):
            rows.append({"task_id": "task", "repeat": repeat, "condition": condition,
                         "status": "complete", "usage": {"total_tokens": total},
                         "correct": True, "critical_failure": False})
    assert context_value_e2e.acceptance(rows)["decision"] == "accept"
    rows[-1]["correct"] = False
    assert context_value_e2e.acceptance(rows)["decision"] == "reject"


def test_pending_e2e_report_never_counts_unexecuted_slots():
    report = context_value_e2e.pending_report()
    assert report["expected_runs"] == 96
    assert report["completed_runs"] == 0
    assert len(report["runs"]) == 96
    assert {run["condition"] for run in report["runs"]} == {"baseline", "candidate"}
    assert report["experimental_expected_runs"] == 144
    assert len(report["experimental_runs"]) == 144
    assert report["source"] == "evaluation_harness"
    assert report["complete"] is False
    assert report["dataset_hash"] is None
    assert report["evaluation_dataset_hash"]
    assert report["estimated"] is False
    assert report["selector_version"] is None
    assert report["unavailable_metrics"] == {
        "rereads": None, "user_corrections": None,
        "client_rss_after_bytes": None, "client_peak_rss_bytes": None,
        "reason": "native JSONL adapter does not expose these measurements",
    }
    assert report["acceptance"]["decision"] == "unknown"
    assert all(run["status"] == "pending" and run["usage"] is None for run in report["runs"])


def test_complete_report_exposes_strict_selector_promotion_pairs():
    rows = []
    token_by_condition = {"A_no_hub": 100, "B_build_context": 90,
                          "C_context_composer": 80}
    for slot in context_value_tasks.counterbalanced_slots(context_value_tasks.build_task_specs()):
        total = token_by_condition[slot["condition"]]
        rows.append({**slot, "status": "complete", "correct": True,
                     "critical_failure": False, "usage": {
                         "token_source": "native", "main_tokens": total,
                         "auxiliary_tokens": 0, "total_tokens": total,
                     }})
    report = context_value_e2e._base_report("selector-v1", "training-hash", rows, "")
    assert report["complete"] is True
    assert report["expected_runs"] == report["completed_runs"] == len(report["runs"]) == 96
    assert report["experimental_completed_runs"] == len(report["experimental_runs"]) == 144
    assert {row["condition"] for row in report["runs"]} == {"baseline", "candidate"}
    assert {row["project"] for row in report["runs"]} == {"skill-hub", "tellurion"}
    assert all(row["token_source"] == "native" and isinstance(row["success"], bool)
               for row in report["runs"])
    from skill_hub.context_learning import _evidence_passes
    passed, failures, metrics = _evidence_passes(
        report, version="selector-v1", dataset_hash="training-hash"
    )
    assert passed is True and failures == []
    assert metrics["paired_runs"] == 48
    assert report["project_variability"]["skill-hub"]["paired_repetitions"] == 24
    assert report["project_variability"]["tellurion"]["paired_repetitions"] == 24


def test_atomic_checkpoint_preserves_completed_rows_and_marks_remaining_pending(tmp_path):
    slots = context_value_tasks.counterbalanced_slots(context_value_tasks.build_task_specs())
    complete = {**slots[0], "status": "complete", "correct": True,
                "critical_failure": False, "usage": {"token_source": "native",
                "main_tokens": 10, "auxiliary_tokens": 0, "total_tokens": 10}}
    path = tmp_path / "checkpoint.json"
    context_value_e2e._checkpoint_current(
        path, "selector-v1", "dataset-v1", slots,
        {context_value_e2e._slot_key(complete): complete}, 1,
    )
    report = json.loads(path.read_text())
    assert report["experimental_completed_runs"] == 1
    assert report["experimental_runs"][0]["status"] == "complete"
    assert all(row["status"] == "pending" for row in report["experimental_runs"][1:])
    assert not path.with_suffix(".json.tmp").exists()


def test_project_variability_tolerates_baseline_and_control_only():
    rows = [
        {"task_id": "task", "project": "skill-hub", "repeat": 1,
         "condition": condition, "status": "complete", "correct": True,
         "usage": {"total_tokens": 10}}
        for condition in ("A_no_hub", "B_build_context")
    ]
    result = context_value_e2e.project_variability(rows)
    assert result["skill-hub"]["paired_repetitions"] == 0
    assert result["skill-hub"]["median_token_reduction"] is None


def test_resume_binding_rejects_changed_harness_configuration():
    report = {
        "source": "evaluation_harness", "estimated": False,
        "selector_version": "selector-v1", "dataset_hash": "training-v1",
        "evaluation_dataset_hash": context_value_e2e._dataset_hash(),
        "harness_sha256": context_value_e2e._harness_sha256(),
        "runtime_source_hashes": context_value_e2e._runtime_source_hashes(),
        "model": context_value_e2e.MODEL,
        "reasoning_effort": context_value_e2e.REASONING_EFFORT,
    }
    context_value_e2e._validate_resume(report, "selector-v1", "training-v1")
    for field in ("evaluation_dataset_hash", "harness_sha256", "runtime_source_hashes",
                  "model", "reasoning_effort"):
        changed = dict(report)
        changed[field] = "changed"
        try:
            context_value_e2e._validate_resume(changed, "selector-v1", "training-v1")
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"resume accepted changed {field}")


def test_row_journal_round_trips_completed_measurement(tmp_path):
    output = tmp_path / "report.json"
    row = {"task_id": "task", "repeat": 1, "condition": "A_no_hub",
           "status": "complete", "usage": {"total_tokens": 12}}
    context_value_e2e._append_row_journal(output, row)
    assert context_value_e2e._journal_rows(output) == [row]
