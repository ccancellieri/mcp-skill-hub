"""Tests for the base hook config + idempotent settings merger (base_config)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from skill_hub import base_config as bc  # noqa: E402


def test_base_hooks_cover_all_events():
    hooks = bc.base_hooks()
    for event in (
        "PreCompact", "UserPromptSubmit", "Stop", "PreToolUse", "PostToolUse",
        "PostToolUseFailure", "StopFailure", "SessionEnd", "PostCompact",
        "SubagentStart", "SubagentStop",
    ):
        assert event in hooks, f"missing event {event}"
    # The three UserPromptSubmit hooks (enforcer, intercept, router) must be there.
    ups = [h["hooks"][0]["command"] for h in hooks["UserPromptSubmit"]]
    assert any("session-start-enforcer.sh" in c for c in ups)
    assert any("prompt-router.sh" in c for c in ups)
    assert any("intercept-task-commands.sh" in c for c in ups)


def test_commands_are_absolute_paths():
    for groups in bc.base_hooks().values():
        for g in groups:
            cmd = g["hooks"][0]["command"]
            assert cmd.startswith("/"), f"not absolute: {cmd}"
            assert cmd.endswith(".sh")


def test_check_on_empty_settings_lists_everything():
    missing = bc.check({})
    # one label per hook spec (subagent-observer appears twice: start + stop)
    assert len(missing) == len(bc._HOOKS)
    assert "UserPromptSubmit:prompt-router.sh" in missing


def test_merge_into_empty_adds_all_then_is_idempotent():
    settings: dict = {}
    merged, added = bc.merge(settings)
    assert len(added) == len(bc._HOOKS)
    assert bc.check(merged) == []  # nothing missing now
    # Second run adds nothing.
    merged2, added2 = bc.merge(merged)
    assert added2 == []


def test_merge_preserves_existing_unrelated_hooks():
    settings = {
        "hooks": {
            "Stop": [
                {"matcher": ".*", "hooks": [{"type": "command", "command": "codegraph sync-if-dirty"}]}
            ],
            "PostToolUse": [
                {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "codegraph mark-dirty", "async": True}]}
            ],
        },
        "model": "sonnet",
    }
    merged, added = bc.merge(settings)
    # Unrelated entries survive.
    assert merged["model"] == "sonnet"
    cmds_stop = [h["command"] for g in merged["hooks"]["Stop"] for h in g["hooks"]]
    assert "codegraph sync-if-dirty" in cmds_stop
    assert any("session-end.sh" in c for c in cmds_stop)
    cmds_ptu = [h["command"] for g in merged["hooks"]["PostToolUse"] for h in g["hooks"]]
    assert "codegraph mark-dirty" in cmds_ptu
    assert any("post-tool-observer.sh" in c for c in cmds_ptu)


def test_merge_matches_by_basename_not_full_path(tmp_path, monkeypatch):
    """A skill-hub hook already present under a DIFFERENT absolute path must not
    be re-added (clobber-repair must not duplicate after a checkout move)."""
    settings = {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command",
                            "command": "/some/old/path/prompt-router.sh", "timeout": 20}]}
            ]
        }
    }
    merged, added = bc.merge(settings)
    assert "UserPromptSubmit:prompt-router.sh" not in added
    routers = [h["command"] for g in merged["hooks"]["UserPromptSubmit"]
               for h in g["hooks"] if "prompt-router.sh" in h["command"]]
    assert len(routers) == 1  # not duplicated


def test_pretooluse_hook_keeps_if_matcher():
    hooks = bc.base_hooks()
    auto = hooks["PreToolUse"][0]["hooks"][0]
    assert auto.get("if") == "Bash(*)"


def test_install_writes_and_backs_up(tmp_path):
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({"model": "sonnet", "hooks": {}}), encoding="utf-8")
    report = bc.install(sp)
    assert report["added"], "should add hooks to an empty hooks block"
    assert report["backup_path"], "must back up existing settings"
    assert (tmp_path / "settings.json.bak").exists()
    written = json.loads(sp.read_text(encoding="utf-8"))
    assert written["model"] == "sonnet"  # preserved
    assert bc.check(written) == []  # all present


def test_install_creates_missing_settings(tmp_path):
    sp = tmp_path / "nope" / "settings.json"
    report = bc.install(sp, backup=False)
    assert report["existed"] is False
    assert sp.exists()
    assert bc.check(json.loads(sp.read_text(encoding="utf-8"))) == []


def test_install_dry_run_does_not_write(tmp_path):
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({"hooks": {}}), encoding="utf-8")
    report = bc.install(sp, dry_run=True)
    assert report["dry_run"] is True
    assert report["added"]
    # File unchanged (still no skill-hub hooks).
    assert bc.check(json.loads(sp.read_text(encoding="utf-8")))


def test_install_idempotent_second_run_adds_nothing(tmp_path):
    sp = tmp_path / "settings.json"
    sp.write_text("{}", encoding="utf-8")
    bc.install(sp, backup=False)
    report2 = bc.install(sp, backup=False)
    assert report2["added"] == []


def test_hooks_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILL_HUB_HOOKS_DIR", str(tmp_path))
    assert bc.hooks_dir() == tmp_path
    cmd = bc.base_hooks()["PostCompact"][0]["hooks"][0]["command"]
    assert cmd == str(tmp_path / "postcompact.sh")
