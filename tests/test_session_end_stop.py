"""Regression tests for retired per-turn Stop hooks."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(HOOKS))

import session_end as stop_hook  # noqa: E402


def _stop_input() -> str:
    return json.dumps({
        "session_id": "foreground-session",
        "last_assistant_message": "completed work",
        "transcript_path": "/path/that/must/not/be/read.jsonl",
        "stop_hook_active": False,
        "session_end_enabled": True,
        "memory_maintenance_enabled": True,
    })


def test_python_stop_hook_never_spawns_or_emits_for_legacy_enabled_input(monkeypatch, capsys):
    if hasattr(stop_hook, "subprocess"):
        monkeypatch.setattr(
            stop_hook.subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("Stop hook must not spawn the CLI"),
        )
    monkeypatch.setattr(sys, "stdin", io.StringIO(_stop_input()))

    assert stop_hook.main() == 0
    assert capsys.readouterr().out == ""


def test_shell_stop_hook_is_a_noop_for_legacy_enabled_input(tmp_path: Path):
    hook = HOOKS / "session-end.sh"
    source = hook.read_text()

    result = subprocess.run(
        ["/bin/bash", str(hook)],
        input=_stop_input(),
        text=True,
        capture_output=True,
        env={"HOME": str(tmp_path), "PATH": ""},
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""
    assert "skill-hub-cli" not in source
    assert "python3" not in source


def test_legacy_session_end_does_not_journal_unscoped_activity(monkeypatch):
    from skill_hub import cli, config

    updates = []

    class Store:
        def get_session_context(self, session_id):
            return {"recent_messages": ["Review alpha beta changes"], "message_count": 3}

        def get_interception_totals(self):
            return {}

        def list_tasks(self, **kwargs):
            return [{"id": 1, "title": "Alpha beta work", "tags": "", "summary": "Original"}]

        def update_task(self, task_id, **kwargs):
            updates.append((task_id, kwargs))

        def close(self):
            pass

    monkeypatch.setattr(cli, "SkillStore", Store)
    monkeypatch.setattr(cli, "smart_memory_write", lambda **kwargs: {
        "quality": 0, "escalate": True, "reason": "test", "directive": "",
    })
    monkeypatch.setattr(cli, "embed_available", lambda: False)
    monkeypatch.setattr(config, "get", lambda key: key == "task_journal_enabled")

    cli._cmd_session_end("session-alpha", "", "")

    assert updates == []
