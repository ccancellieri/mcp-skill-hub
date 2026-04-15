"""Tests for plan_executor.validator — schema, kind enum, file existence, cycle detection."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.plan_executor import (  # noqa: E402
    PlanValidationError,
    TIER_MAP,
    VALID_KINDS,
    validate_plan,
    validate_plan_file,
)


def _minimal_plan(**overrides):
    plan = {
        "plan_id": "test",
        "goal": "do a thing",
        "steps": [
            {
                "id": "T1",
                "kind": "boilerplate",
                "files": ["src/foo.py"],
                "acceptance": "pytest",
            }
        ],
    }
    plan.update(overrides)
    return plan


def test_tier_map_covers_all_kinds():
    assert set(TIER_MAP.keys()) == VALID_KINDS


def test_minimal_plan_passes():
    out = validate_plan(_minimal_plan(), check_files=False)
    assert out["plan_id"] == "test"


def test_yaml_string_input():
    yaml_str = """
plan_id: s1
goal: hi
steps:
  - id: A
    kind: tests
    files: [t.py]
    acceptance: pytest
"""
    out = validate_plan(yaml_str, check_files=False)
    assert out["steps"][0]["id"] == "A"


def test_missing_plan_id():
    plan = _minimal_plan()
    plan.pop("plan_id")
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("plan_id" in e for e in exc.value.errors)


def test_invalid_kind():
    plan = _minimal_plan()
    plan["steps"][0]["kind"] = "refactor"
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("kind" in e for e in exc.value.errors)


def test_empty_files():
    plan = _minimal_plan()
    plan["steps"][0]["files"] = []
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("files" in e for e in exc.value.errors)


def test_duplicate_step_id():
    plan = _minimal_plan()
    plan["steps"].append(dict(plan["steps"][0]))
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("duplicate" in e for e in exc.value.errors)


def test_unknown_depends_on():
    plan = _minimal_plan()
    plan["steps"][0]["depends_on"] = ["ghost"]
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("unknown id" in e for e in exc.value.errors)


def test_depends_on_cycle():
    plan = _minimal_plan()
    plan["steps"] = [
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "x", "depends_on": ["B"]},
        {"id": "B", "kind": "tests", "files": ["b.py"], "acceptance": "x", "depends_on": ["A"]},
    ]
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("cycle" in e for e in exc.value.errors)


def test_model_hint_validation():
    plan = _minimal_plan()
    plan["steps"][0]["model_hint"] = "tier_bogus"
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert any("model_hint" in e for e in exc.value.errors)


def test_model_hint_valid():
    plan = _minimal_plan()
    plan["steps"][0]["model_hint"] = "tier_smart"
    validate_plan(plan, check_files=False)


def test_file_existence_check(tmp_path):
    plan = _minimal_plan()
    plan["steps"][0]["protocols_ref"] = ["does/not/exist.py"]
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, repo_path=tmp_path, check_files=True)
    assert any("file not found" in e for e in exc.value.errors)


def test_file_existence_passes_when_present(tmp_path):
    (tmp_path / "real.py").write_text("")
    plan = _minimal_plan()
    plan["steps"][0]["protocols_ref"] = ["real.py"]
    validate_plan(plan, repo_path=tmp_path, check_files=True)


def test_files_field_not_checked_for_existence(tmp_path):
    # 'files' are outputs (may not exist yet) — existence check must skip them.
    plan = _minimal_plan()
    plan["steps"][0]["files"] = ["src/not_yet_created.py"]
    validate_plan(plan, repo_path=tmp_path, check_files=True)


def test_bad_yaml_string():
    with pytest.raises(PlanValidationError) as exc:
        validate_plan("not: valid: yaml: [", check_files=False)
    assert any("YAML" in e for e in exc.value.errors)


def test_validate_plan_file_missing(tmp_path):
    with pytest.raises(PlanValidationError) as exc:
        validate_plan_file(tmp_path / "nope.yaml", check_files=False)
    assert any("not found" in e for e in exc.value.errors)


def test_validate_plan_file_roundtrip(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "plan_id: rt\ngoal: x\nsteps:\n  - id: A\n    kind: docs\n    files: [r.md]\n    acceptance: review\n"
    )
    out = validate_plan_file(path, check_files=False)
    assert out["plan_id"] == "rt"


def test_multiple_errors_accumulated():
    plan = {"plan_id": "", "goal": "", "steps": []}
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, check_files=False)
    assert len(exc.value.errors) >= 3
