"""Regression tests for the retired generic proceed hook."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


HOOK = Path(__file__).resolve().parent.parent / "hooks" / "auto_proceed.py"


def _run_hook(home: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ | {"HOME": str(home)} | (env or {})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"session_id": "legacy-session"}),
        text=True,
        capture_output=True,
        env=run_env,
        check=False,
    )


def test_legacy_enabled_config_cannot_emit_proceed_or_read_task_db(tmp_path: Path):
    config_dir = tmp_path / ".claude" / "mcp-skill-hub"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps({"auto_proceed": True, "auto_proceed_max": 20})
    )
    plans = tmp_path / ".claude" / "plans"
    plans.mkdir()
    (plans / "active.md").write_text("- [ ] legacy work remains\n")

    result = _run_hook(tmp_path)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_legacy_environment_cannot_repeat_proceed(tmp_path: Path):
    plans = tmp_path / ".claude" / "plans"
    plans.mkdir(parents=True)
    (plans / "active.md").write_text("- [ ] legacy work remains\n")

    first = _run_hook(tmp_path, {"SKILL_HUB_AUTO_PROCEED": "1"})
    second = _run_hook(tmp_path, {"SKILL_HUB_AUTO_PROCEED": "1"})

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout == ""
