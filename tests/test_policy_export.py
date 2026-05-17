"""Unit tests for policy_export — feedback_*.md → per-repo POLICY.md."""
from __future__ import annotations

import datetime as _dt
import os as _os
import sys
import time as _time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.policy_export import (
    _has_ai_path_leak,
    _list_feedback_files,
    _parse_frontmatter,
    _render_policy_md,
    _render_rule_block,
    _scrub_ai_paths,
    _slug_to_title,
    export_policies,
)


# ---------------------------------------------------------------------------
# _scrub_ai_paths — the load-bearing safety net
# ---------------------------------------------------------------------------

def test_scrub_strips_tilde_claude_path():
    out = _scrub_ai_paths("See ~/.claude/plans/foo.md for context.")
    assert "~/.claude" not in out
    assert "[memory]" in out


def test_scrub_strips_relative_claude_path():
    out = _scrub_ai_paths("Files under .claude/projects/-Users-x/memory/ change.")
    assert ".claude/" not in out
    assert "[memory]" in out


def test_scrub_strips_absolute_user_claude_path():
    out = _scrub_ai_paths("Saved at /Users/alice/.claude/projects/foo/memory/x.md.")
    assert "/.claude/" not in out
    assert "[memory]" in out


def test_scrub_idempotent():
    s = "see ~/.claude/foo and .claude/bar"
    once = _scrub_ai_paths(s)
    twice = _scrub_ai_paths(once)
    assert once == twice
    assert "~/.claude" not in twice
    assert ".claude/" not in twice


def test_scrub_passes_unrelated_text_through():
    s = "Use Literal['catalog', 'obfuscated']. Never 'stac'."
    assert _scrub_ai_paths(s) == s


# ---------------------------------------------------------------------------
# _has_ai_path_leak — the final-line-of-defense check
# ---------------------------------------------------------------------------

def test_has_ai_path_leak_detects_tilde():
    assert _has_ai_path_leak("a ~/.claude/x b")


def test_has_ai_path_leak_detects_relative():
    assert _has_ai_path_leak("path: .claude/projects/x")


def test_has_ai_path_leak_clean():
    assert not _has_ai_path_leak("Use Literal['catalog']. No leaks here.")


# ---------------------------------------------------------------------------
# _parse_frontmatter
# ---------------------------------------------------------------------------

def test_parse_frontmatter_extracts_name_and_description():
    raw = (
        "---\n"
        "name: Use catalog not stac\n"
        "description: Mode values should be catalog/obfuscated\n"
        "type: feedback\n"
        "---\n"
        "body line 1\nbody line 2\n"
    )
    fm, body = _parse_frontmatter(raw)
    assert fm["name"] == "Use catalog not stac"
    assert fm["description"] == "Mode values should be catalog/obfuscated"
    assert fm["type"] == "feedback"
    assert body.startswith("body line 1")


def test_parse_frontmatter_no_yaml_block():
    raw = "# Heading\n\nplain body\n"
    fm, body = _parse_frontmatter(raw)
    assert fm == {}
    assert body == raw


# ---------------------------------------------------------------------------
# _slug_to_title
# ---------------------------------------------------------------------------

def test_slug_to_title_strips_prefix_and_titles():
    assert _slug_to_title("feedback_no_intermodule_deps") == "No Intermodule Deps"


def test_slug_to_title_handles_dashes():
    assert _slug_to_title("feedback_three-repo-sync") == "Three Repo Sync"


def test_slug_to_title_passthrough_without_prefix():
    assert _slug_to_title("anything") == "Anything"


# ---------------------------------------------------------------------------
# _render_rule_block + _render_policy_md
# ---------------------------------------------------------------------------

def test_render_rule_block_uses_frontmatter_name(tmp_path: Path):
    p = tmp_path / "feedback_x.md"
    p.write_text(
        "---\nname: My Rule\ndescription: Short desc\n---\n\n"
        "Detailed body text.\n",
        encoding="utf-8",
    )
    block = _render_rule_block(p)
    assert "### My Rule" in block
    assert "_Short desc_" in block
    assert "Detailed body text." in block


def test_render_rule_block_falls_back_to_slug_title(tmp_path: Path):
    p = tmp_path / "feedback_no_test_in_prod.md"
    p.write_text("Just a body with no frontmatter.\n", encoding="utf-8")
    block = _render_rule_block(p)
    assert "### No Test In Prod" in block
    assert "Just a body" in block


def test_render_rule_block_scrubs_paths_in_body(tmp_path: Path):
    p = tmp_path / "feedback_leaky.md"
    p.write_text(
        "---\nname: Leaky\ndescription: foo\n---\n\n"
        "See ~/.claude/projects/x/memory/feedback_leaky.md for details.\n",
        encoding="utf-8",
    )
    block = _render_rule_block(p)
    assert "~/.claude" not in block
    assert ".claude/" not in block


def test_render_policy_md_empty_renders_placeholder():
    out = _render_policy_md("myproj", "2026-05-17", [])
    assert "# POLICY — myproj" in out
    assert "No feedback rules found" in out


def test_render_policy_md_full():
    blocks = ["### One\n\nbody one\n", "### Two\n\nbody two\n"]
    out = _render_policy_md("myproj", "2026-05-17", blocks)
    assert out.startswith("# POLICY — myproj")
    assert "### One" in out
    assert "### Two" in out
    assert "Generated 2026-05-17" in out


# ---------------------------------------------------------------------------
# _list_feedback_files
# ---------------------------------------------------------------------------

def test_list_feedback_files_filters_and_sorts(tmp_path: Path):
    (tmp_path / "feedback_b.md").write_text("b", encoding="utf-8")
    (tmp_path / "feedback_a.md").write_text("a", encoding="utf-8")
    (tmp_path / "project_x.md").write_text("p", encoding="utf-8")  # excluded
    (tmp_path / "feedback_other.txt").write_text("nope", encoding="utf-8")  # excluded
    out = _list_feedback_files(tmp_path)
    assert [p.name for p in out] == ["feedback_a.md", "feedback_b.md"]


# ---------------------------------------------------------------------------
# export_policies — end-to-end
# ---------------------------------------------------------------------------

def _setup_project_with_memory(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Stand up a project + a matching auto-memory dir under fake HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    slug = "-" + str(project.resolve()).strip("/").replace("/", "-")
    mem_dir = tmp_path / ".claude" / "projects" / slug / "memory"
    mem_dir.mkdir(parents=True)
    return project, mem_dir


def test_export_policies_skipped_when_no_memory_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    result = export_policies(str(project), dry_run=True)
    assert result["status"] == "skipped"


def test_export_policies_dry_run(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    (mem_dir / "feedback_one.md").write_text(
        "---\nname: Rule One\ndescription: be good\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )
    result = export_policies(str(project), dry_run=True)
    assert result["status"] == "dry_run"
    assert "### Rule One" in result["rendered"]
    # No file written
    assert not (project / ".skill-hub" / "POLICY.md").exists()


def test_export_policies_writes_file(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    (mem_dir / "feedback_one.md").write_text(
        "---\nname: Rule One\ndescription: be good\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )
    (mem_dir / "feedback_two.md").write_text(
        "Plain body without frontmatter.\n",
        encoding="utf-8",
    )
    result = export_policies(str(project), dry_run=False)
    assert result["status"] == "written"
    policy = project / ".skill-hub" / "POLICY.md"
    assert policy.exists()
    text = policy.read_text(encoding="utf-8")
    assert "### Rule One" in text
    assert "### Two" in text  # slug-derived title from feedback_two.md
    # Acceptance: zero ~/.claude references
    assert "~/.claude" not in text
    assert ".claude/" not in text


def test_export_policies_noop_when_policy_fresher(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    fb = mem_dir / "feedback_old.md"
    fb.write_text("body\n", encoding="utf-8")
    # Backdate the feedback file
    long_ago = _dt.datetime.now() - _dt.timedelta(hours=1)
    _os.utime(fb, (long_ago.timestamp(), long_ago.timestamp()))
    # Write a fresh POLICY.md
    policy = project / ".skill-hub" / "POLICY.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("# already here\n", encoding="utf-8")

    result = export_policies(str(project), dry_run=False)
    assert result["status"] == "noop"
    # Content unchanged
    assert policy.read_text(encoding="utf-8") == "# already here\n"


def test_export_policies_force_overrides_noop(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    fb = mem_dir / "feedback_x.md"
    fb.write_text("---\nname: X\n---\nbody\n", encoding="utf-8")
    long_ago = _dt.datetime.now() - _dt.timedelta(hours=1)
    _os.utime(fb, (long_ago.timestamp(), long_ago.timestamp()))
    policy = project / ".skill-hub" / "POLICY.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("# stale\n", encoding="utf-8")

    result = export_policies(str(project), dry_run=False, force=True)
    assert result["status"] == "written"
    assert "### X" in policy.read_text(encoding="utf-8")


def test_export_policies_empty_when_no_feedback_files(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    # No feedback_*.md present; project_*.md is ignored.
    (mem_dir / "project_x.md").write_text("p", encoding="utf-8")
    result = export_policies(str(project), dry_run=False)
    assert result["status"] == "empty"
    policy = project / ".skill-hub" / "POLICY.md"
    assert policy.exists()
    assert "No feedback rules found" in policy.read_text(encoding="utf-8")


def test_export_policies_backs_up_existing(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    fb = mem_dir / "feedback_x.md"
    fb.write_text("---\nname: X\n---\nbody\n", encoding="utf-8")
    policy = project / ".skill-hub" / "POLICY.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("# previous\n", encoding="utf-8")
    # Make policy older so it doesn't trip the noop guard.
    long_ago = _dt.datetime.now() - _dt.timedelta(hours=1)
    _os.utime(policy, (long_ago.timestamp(), long_ago.timestamp()))

    result = export_policies(str(project), dry_run=False)
    assert result["status"] == "written"
    backups = list((policy.parent / ".backups").glob("POLICY-*.md"))
    assert len(backups) == 1
    assert "previous" in backups[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Round-trip property: idempotent re-export over its own output
# ---------------------------------------------------------------------------

def test_export_policies_idempotent_round_trip(tmp_path: Path, monkeypatch):
    """Round-trip lossless modulo formatting: a re-export after no source change
    produces a noop (same content). After touching source, re-export updates."""
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    fb = mem_dir / "feedback_x.md"
    fb.write_text("---\nname: X\n---\nbody one\n", encoding="utf-8")
    r1 = export_policies(str(project), dry_run=False)
    assert r1["status"] == "written"
    policy = project / ".skill-hub" / "POLICY.md"
    snapshot = policy.read_text(encoding="utf-8")

    r2 = export_policies(str(project), dry_run=False)
    assert r2["status"] == "noop"
    assert policy.read_text(encoding="utf-8") == snapshot

    # Touch the source — re-export must rewrite.
    _time.sleep(0.05)
    fb.write_text("---\nname: X\n---\nbody two\n", encoding="utf-8")
    r3 = export_policies(str(project), dry_run=False)
    assert r3["status"] == "written"
    assert "body two" in policy.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Acceptance criterion: rendered POLICY.md has zero ~/.claude refs
# ---------------------------------------------------------------------------

def test_export_policies_strips_path_leaks_from_source(tmp_path: Path, monkeypatch):
    project, mem_dir = _setup_project_with_memory(tmp_path, monkeypatch)
    (mem_dir / "feedback_leaky.md").write_text(
        "---\nname: Leaky Rule\ndescription: refers to ~/.claude/foo.md\n---\n\n"
        "See ~/.claude/projects/-Users-x-foo/memory/feedback_leaky.md for source.\n"
        "Also .claude/plans/some-plan.md.\n",
        encoding="utf-8",
    )
    result = export_policies(str(project), dry_run=True)
    assert result["status"] == "dry_run"
    rendered = result["rendered"]
    assert "~/.claude" not in rendered
    assert ".claude/" not in rendered
    # Substantive content survives.
    assert "Leaky Rule" in rendered
