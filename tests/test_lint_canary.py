"""Tests for skill_hub.lint_canary -- M3 issue #17.

Covers:
- Cursor rotation advances on each run and wraps mod len.
- Each invocation appends one JSONL record to the witness log.
- Custom selector lists are honored and persisted.
- ``ruff`` not installed (or failing) still advances the rotation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest

from skill_hub import lint_canary as lc


@pytest.fixture()
def paths(tmp_path: Path) -> tuple[Path, Path]:
    """Hermetic state + witness-log file pair."""
    return tmp_path / "lint_canary.json", tmp_path / "witness_log.jsonl"


def _fake_runner(count: int = 0, sample=None, error: str | None = None):
    """Return a ruff_runner injection that ignores selector/target."""
    def runner(_selector: str, _target: str):
        return (count, list(sample or []), error)
    return runner


# ---------------------------------------------------------------------------
# Rotation behavior
# ---------------------------------------------------------------------------


def test_rotation_advances_on_each_run(paths):
    state_file, witness_file = paths
    selectors = ["A", "B", "C"]
    seen = []
    for _ in range(5):
        run = lc.run_lint_canary(
            target=".",
            selectors=selectors,
            state_file=state_file,
            witness_file=witness_file,
            ruff_runner=_fake_runner(count=0),
        )
        seen.append(run.selector)
    # Wraps mod len: A, B, C, A, B
    assert seen == ["A", "B", "C", "A", "B"]


def test_rotation_wraps_via_persisted_cursor(paths):
    state_file, witness_file = paths
    selectors = ["X", "Y"]
    lc.run_lint_canary(
        target=".",
        selectors=selectors,
        state_file=state_file,
        witness_file=witness_file,
        ruff_runner=_fake_runner(),
    )
    run = lc.run_lint_canary(
        target=".",
        selectors=selectors,
        state_file=state_file,
        witness_file=witness_file,
        ruff_runner=_fake_runner(),
    )
    assert run.selector == "Y"
    persisted = json.loads(state_file.read_text())
    assert persisted["cursor"] == 0  # wrapped back
    assert persisted["last_selector"] == "Y"
    assert persisted["selectors"] == selectors


def test_default_rotation_used_when_no_override(paths):
    state_file, witness_file = paths
    run = lc.run_lint_canary(
        state_file=state_file,
        witness_file=witness_file,
        ruff_runner=_fake_runner(),
    )
    assert run.selector == lc.DEFAULT_SELECTORS[0]
    # Default list is not persisted to state unless caller passes an override.
    persisted = json.loads(state_file.read_text())
    assert "selectors" not in persisted


# ---------------------------------------------------------------------------
# Witness-log capture
# ---------------------------------------------------------------------------


def test_findings_captured_to_witness_log(paths):
    state_file, witness_file = paths
    sample = [{"code": "F841", "message": "unused", "filename": "x.py", "location": {"row": 1}}]
    lc.run_lint_canary(
        selectors=["F841"],
        state_file=state_file,
        witness_file=witness_file,
        ruff_runner=_fake_runner(count=3, sample=sample),
    )
    assert witness_file.exists()
    lines = witness_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "lint_canary"
    assert record["selector"] == "F841"
    assert record["findings"] == 3
    assert record["findings_sample"] == sample
    assert record["cursor_before"] == 0
    assert record["cursor_after"] == 0  # single-selector list wraps immediately


def test_each_run_appends_one_line(paths):
    state_file, witness_file = paths
    for _ in range(4):
        lc.run_lint_canary(
            selectors=["A", "B"],
            state_file=state_file,
            witness_file=witness_file,
            ruff_runner=_fake_runner(count=1),
        )
    lines = witness_file.read_text().strip().splitlines()
    assert len(lines) == 4
    selectors = [json.loads(line)["selector"] for line in lines]
    assert selectors == ["A", "B", "A", "B"]


# ---------------------------------------------------------------------------
# Failure modes still advance the cursor
# ---------------------------------------------------------------------------


def test_ruff_missing_still_advances_rotation(paths):
    state_file, witness_file = paths
    run = lc.run_lint_canary(
        selectors=["A", "B"],
        state_file=state_file,
        witness_file=witness_file,
        ruff_runner=_fake_runner(error="ruff binary not found on PATH"),
    )
    assert run.ruff_available is False
    assert run.error == "ruff binary not found on PATH"
    # Cursor advanced even on failure.
    persisted = json.loads(state_file.read_text())
    assert persisted["cursor"] == 1
    # Record still landed in the witness log.
    record = json.loads(witness_file.read_text().strip())
    assert record["error"] == "ruff binary not found on PATH"
    assert record["ruff_available"] is False


def test_format_run_includes_selector_and_cursor():
    run = lc.CanaryRun(
        selector="F821",
        findings=2,
        findings_sample=[],
        cursor_before=1,
        cursor_after=2,
        ruff_available=True,
        error=None,
        target=".",
    )
    out = lc.format_run(run)
    assert "F821" in out
    assert "findings=2" in out
    assert "1->2" in out


def test_format_run_surfaces_error():
    run = lc.CanaryRun(
        selector="F821",
        findings=0,
        findings_sample=[],
        cursor_before=0,
        cursor_after=1,
        ruff_available=False,
        error="ruff binary not found on PATH",
        target=".",
    )
    out = lc.format_run(run)
    assert "error: ruff binary not found on PATH" in out
