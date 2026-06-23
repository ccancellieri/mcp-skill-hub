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
    _, added2 = bc.merge(merged)
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
    merged, _ = bc.merge(settings)
    # Unrelated entries survive.
    assert merged["model"] == "sonnet"
    cmds_stop = [h["command"] for g in merged["hooks"]["Stop"] for h in g["hooks"]]
    assert "codegraph sync-if-dirty" in cmds_stop
    assert any("session-end.sh" in c for c in cmds_stop)
    cmds_ptu = [h["command"] for g in merged["hooks"]["PostToolUse"] for h in g["hooks"]]
    assert "codegraph mark-dirty" in cmds_ptu
    assert any("post-tool-observer.sh" in c for c in cmds_ptu)


def test_merge_matches_by_basename_not_full_path(tmp_path):
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


# ---------------------------------------------------------------------------
# MCP registration tests
# ---------------------------------------------------------------------------

def test_check_mcp_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_HUB_CLAUDE_JSON", str(tmp_path / "nonexistent.json"))
    issues = bc.check_mcp()
    assert issues  # file not found → issue reported


def test_check_mcp_empty_json(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text("{}", encoding="utf-8")
    issues = bc.check_mcp(p)
    assert any("skill-hub" in i for i in issues)


def test_merge_mcp_into_empty(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text("{}", encoding="utf-8")
    data, added = bc.merge_mcp(p)
    assert "mcpServers:skill-hub" in added
    assert "skill-hub" in data["mcpServers"]
    entry = data["mcpServers"]["skill-hub"]
    assert entry.get("command")  # command is set
    assert entry.get("type") == "stdio"


def test_merge_mcp_preserves_other_servers(tmp_path):
    p = tmp_path / "claude.json"
    existing = {
        "mcpServers": {
            "codegraph": {"type": "stdio", "command": "codegraph", "args": ["serve", "--mcp"]}
        }
    }
    p.write_text(json.dumps(existing), encoding="utf-8")
    data, added = bc.merge_mcp(p)
    assert "codegraph" in data["mcpServers"]  # preserved
    assert "skill-hub" in data["mcpServers"]   # added
    assert "mcpServers:skill-hub" in added


def test_merge_mcp_idempotent(tmp_path):
    p = tmp_path / "claude.json"
    existing = {
        "mcpServers": {
            "skill-hub": {"type": "stdio", "command": "/some/path/skill-hub"}
        }
    }
    p.write_text(json.dumps(existing), encoding="utf-8")
    data, added = bc.merge_mcp(p)
    assert added == []  # already present
    # Original command preserved (not overwritten)
    assert data["mcpServers"]["skill-hub"]["command"] == "/some/path/skill-hub"


def test_install_mcp_writes_and_backs_up(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text(json.dumps({"numStartups": 5}), encoding="utf-8")
    report = bc.install_mcp(p)
    assert report["added"]
    assert report["backup_path"]
    bak = Path(report["backup_path"])
    assert bak.exists()
    written = json.loads(p.read_text(encoding="utf-8"))
    assert "skill-hub" in written["mcpServers"]
    assert written["numStartups"] == 5  # other keys preserved


def test_install_mcp_dry_run_does_not_write(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text("{}", encoding="utf-8")
    report = bc.install_mcp(p, dry_run=True)
    assert report["dry_run"] is True
    assert report["added"]
    assert bc.check_mcp(p)  # still missing after dry-run


def test_install_mcp_idempotent(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text("{}", encoding="utf-8")
    bc.install_mcp(p, backup=False)
    report2 = bc.install_mcp(p, backup=False)
    assert report2["added"] == []


# ---------------------------------------------------------------------------
# CLAUDE.md base-roles block tests
# ---------------------------------------------------------------------------

def test_check_roles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_HUB_CLAUDE_MD", str(tmp_path / "nonexistent.md"))
    issues = bc.check_roles()
    assert issues  # file not found


def test_check_roles_no_sentinel(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text("# My Config\n\nSome content.\n", encoding="utf-8")
    issues = bc.check_roles(p)
    assert any("sentinel" in i or "not present" in i for i in issues)


def test_merge_roles_insert_into_empty_file(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text("# My Config\n\n", encoding="utf-8")
    new_text, added = bc.merge_roles(p)
    assert "CLAUDE.md:base-roles:inserted" in added
    assert bc._ROLES_START in new_text
    assert bc._ROLES_END in new_text
    assert "# My Config" in new_text  # surrounding content preserved


def test_merge_roles_refresh_existing_block(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text(
        f"# My Config\n\n{bc._ROLES_START}\nOLD CONTENT\n{bc._ROLES_END}\n\n## After\n",
        encoding="utf-8",
    )
    new_text, added = bc.merge_roles(p)
    assert added  # something changed
    assert "OLD CONTENT" not in new_text
    assert bc._BASE_ROLES_CONTENT.strip() in new_text
    assert "# My Config" in new_text   # content before preserved
    assert "## After" in new_text      # content after preserved


def test_merge_roles_idempotent(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text("", encoding="utf-8")
    new_text1, _ = bc.merge_roles(p)
    # Write the result to disk, then run again.
    p.write_text(new_text1, encoding="utf-8")
    new_text2, added2 = bc.merge_roles(p)
    assert added2 == []  # nothing changed on second run
    assert new_text1 == new_text2


def test_install_roles_writes_and_backs_up(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text("# Existing\n", encoding="utf-8")
    report = bc.install_roles(p)
    assert report["added"]
    assert report["backup_path"]
    assert Path(report["backup_path"]).exists()
    written = p.read_text(encoding="utf-8")
    assert bc._ROLES_START in written
    assert "# Existing" in written


def test_install_roles_dry_run_does_not_write(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text("# Existing\n", encoding="utf-8")
    report = bc.install_roles(p, dry_run=True)
    assert report["dry_run"] is True
    assert report["added"]
    assert bc.check_roles(p)  # still missing after dry-run


# ---------------------------------------------------------------------------
# check_all / restore_all tests
# ---------------------------------------------------------------------------

def test_check_all_empty(tmp_path):
    sp = tmp_path / "settings.json"
    cj = tmp_path / "claude.json"
    cm = tmp_path / "CLAUDE.md"
    # None of the files exist yet.
    status = bc.check_all(settings_path=sp, claude_json_path=cj, claude_md_path=cm)
    assert status["hooks"]   # all hooks missing
    assert status["mcp"]     # entry missing
    assert status["roles"]   # block missing


def test_check_all_present(tmp_path):
    sp = tmp_path / "settings.json"
    cj = tmp_path / "claude.json"
    cm = tmp_path / "CLAUDE.md"
    # Install everything.
    bc.install(sp, backup=False)
    bc.install_mcp(cj, backup=False)
    bc.install_roles(cm, backup=False)
    status = bc.check_all(settings_path=sp, claude_json_path=cj, claude_md_path=cm)
    assert status["hooks"] == []
    assert status["mcp"] == []
    assert status["roles"] == []


def test_restore_all_applies_everything(tmp_path):
    sp = tmp_path / "settings.json"
    cj = tmp_path / "claude.json"
    cm = tmp_path / "CLAUDE.md"
    report = bc.restore_all(
        settings_path=sp,
        claude_json_path=cj,
        claude_md_path=cm,
        dry_run=False,
        backup=False,
    )
    assert report["hooks"]["added"]
    assert report["mcp"]["added"]
    assert report["roles"]["added"]
    # Idempotent second run.
    report2 = bc.restore_all(
        settings_path=sp,
        claude_json_path=cj,
        claude_md_path=cm,
        dry_run=False,
        backup=False,
    )
    assert report2["hooks"]["added"] == []
    assert report2["mcp"]["added"] == []
    assert report2["roles"]["added"] == []


def test_restore_all_dry_run(tmp_path):
    sp = tmp_path / "settings.json"
    cj = tmp_path / "claude.json"
    cm = tmp_path / "CLAUDE.md"
    report = bc.restore_all(
        settings_path=sp,
        claude_json_path=cj,
        claude_md_path=cm,
        dry_run=True,
        backup=False,
    )
    assert report["hooks"]["dry_run"] is True
    assert report["hooks"]["added"]   # would add
    assert not sp.exists()            # nothing written


def test_format_report_unified(tmp_path):
    sp = tmp_path / "settings.json"
    cj = tmp_path / "claude.json"
    cm = tmp_path / "CLAUDE.md"
    report = bc.restore_all(
        settings_path=sp, claude_json_path=cj, claude_md_path=cm,
        dry_run=False, backup=False,
    )
    text = bc.format_report(report)
    assert "hook install" in text
    assert "MCP server" in text
    assert "Base roles" in text
