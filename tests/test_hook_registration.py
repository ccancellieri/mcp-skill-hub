"""Tests for install.py's step_install_hooks — registration + upgrade-in-place.

These cover the behaviour change introduced when the project moved from
"only-add" to "add-or-update" hook registration: re-running install.py must
patch existing entries with new fields (e.g. the `if` filter from Claude
Code 2.1.83) without duplicating them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_install_module(monkeypatch, tmp_settings: Path):
    """Import install.py with its SETTINGS constant pointed at tmp_settings."""
    spec = importlib.util.spec_from_file_location("install_under_test",
                                                   ROOT / "install.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["install_under_test"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SETTINGS", tmp_settings)
    return mod


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def settings_path(tmp_path):
    return tmp_path / "settings.json"


def test_fresh_install_registers_all_events(monkeypatch, settings_path):
    """First run must register all known event types into a missing settings file."""
    mod = _load_install_module(monkeypatch, settings_path)
    mod.step_install_hooks(1, 1)

    data = _read(settings_path)
    hooks = data["hooks"]

    # Every event we care about must be present.
    for event in (
        "UserPromptSubmit", "PreToolUse", "PostToolUse",
        "PostToolUseFailure", "Stop", "StopFailure", "SessionEnd",
        "PreCompact", "PostCompact", "SubagentStart", "SubagentStop",
    ):
        assert event in hooks, f"event {event} not registered"
        assert any(h.get("hooks") for h in hooks[event]), f"{event} has no commands"


def test_re_run_does_not_duplicate(monkeypatch, settings_path):
    """Second invocation must be a no-op — same command, same event, no duplicates."""
    mod = _load_install_module(monkeypatch, settings_path)
    mod.step_install_hooks(1, 1)
    first = _read(settings_path)
    mod.step_install_hooks(1, 1)
    second = _read(settings_path)

    assert first == second, "re-run produced different settings"
    for event, blocks in second["hooks"].items():
        commands = [h["command"] for entry in blocks for h in entry.get("hooks", [])]
        assert len(commands) == len(set(commands)), \
            f"event {event} has duplicate commands: {commands}"


def test_re_run_upgrades_existing_entry_with_if_filter(monkeypatch, settings_path):
    """Old entries (no `if` field) must be upgraded in place on re-run.

    The shallow-merge upgrade also preserves user customizations: if the user
    edited ``statusMessage`` or ``timeout``, those changes survive the upgrade.
    Only NEW keys we ship in this release are added.
    """
    mod = _load_install_module(monkeypatch, settings_path)

    # Seed an OLD-style entry (no `if`) for auto-approve.sh, with the user
    # having customized ``statusMessage`` and ``timeout`` to non-defaults.
    old_cmd = mod._hook_command("auto-approve.sh")
    settings_path.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [{"hooks": [{
                "type": "command",
                "command": old_cmd,
                "timeout": 99,                          # user override
                "statusMessage": "USER CUSTOM STATUS",  # user override
            }]}]
        }
    }))

    mod.step_install_hooks(1, 1)

    data = _read(settings_path)
    pre_hooks = [h for entry in data["hooks"]["PreToolUse"] for h in entry["hooks"]]
    matching = [h for h in pre_hooks if h["command"] == old_cmd]
    assert len(matching) == 1, "duplicate entry created instead of in-place upgrade"
    assert matching[0].get("if") == "Bash(*)", \
        "existing entry was not upgraded with the `if` filter"
    # User customizations preserved.
    assert matching[0]["timeout"] == 99
    assert matching[0]["statusMessage"] == "USER CUSTOM STATUS"


def test_post_tool_use_failure_uses_same_observer_script(monkeypatch, settings_path):
    """PostToolUseFailure must reuse post-tool-observer.sh (negative reinforcement)."""
    mod = _load_install_module(monkeypatch, settings_path)
    mod.step_install_hooks(1, 1)

    data = _read(settings_path)
    failure_hooks = [
        h for entry in data["hooks"]["PostToolUseFailure"]
        for h in entry["hooks"]
    ]
    assert any(
        h["command"].endswith("post-tool-observer.sh") or
        h["command"].endswith("post_tool_observer.py")
        for h in failure_hooks
    ), f"expected post-tool-observer.sh in PostToolUseFailure: {failure_hooks}"


def test_subagent_observer_handles_both_events(monkeypatch, settings_path):
    """SubagentStart and SubagentStop should both invoke subagent-observer.sh."""
    mod = _load_install_module(monkeypatch, settings_path)
    mod.step_install_hooks(1, 1)

    data = _read(settings_path)
    for event in ("SubagentStart", "SubagentStop"):
        cmds = [h["command"] for entry in data["hooks"][event] for h in entry["hooks"]]
        assert any("subagent-observer" in c or "subagent_observer" in c for c in cmds), \
            f"{event} missing subagent observer hook: {cmds}"
