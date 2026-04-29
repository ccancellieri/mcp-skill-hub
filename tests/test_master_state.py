"""Unit tests for master_state compaction (master_state.py)."""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.master_state import (
    _atomic_write,
    _list_recent_memory,
    _prune_backups,
    _read_existing_section,
    _render_markdown,
    _summarize_memory_entries,
    _upsert_section,
    compact_to_master_state,
)


# ---------------------------------------------------------------------------
# _read_existing_section
# ---------------------------------------------------------------------------

def test_read_existing_section_present(tmp_path: Path):
    f = tmp_path / "decisions.md"
    f.write_text(
        "# Decisions\n\n"
        "## Master Project State\n\nbody1\n\n"
        "## Other\n\nbody2\n",
        encoding="utf-8",
    )
    out = _read_existing_section(f, "Master Project State")
    assert out.startswith("## Master Project State")
    assert "body1" in out
    assert "body2" not in out


def test_read_existing_section_missing(tmp_path: Path):
    f = tmp_path / "decisions.md"
    f.write_text("# Decisions\n\n## Other\n", encoding="utf-8")
    assert _read_existing_section(f, "Master Project State") == ""


def test_read_existing_section_no_file(tmp_path: Path):
    assert _read_existing_section(tmp_path / "nope.md", "X") == ""


# ---------------------------------------------------------------------------
# _list_recent_memory
# ---------------------------------------------------------------------------

def test_list_recent_memory_filters_by_mtime(tmp_path: Path):
    old = tmp_path / "project_old.md"
    new = tmp_path / "project_new.md"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    # Backdate `old` by 30 days
    long_ago = _dt.datetime.now() - _dt.timedelta(days=30)
    import os as _os
    _os.utime(old, (long_ago.timestamp(), long_ago.timestamp()))

    since = _dt.datetime.now() - _dt.timedelta(days=7)
    found = _list_recent_memory(tmp_path, since)
    names = [p.name for p in found]
    assert "project_new.md" in names
    assert "project_old.md" not in names


def test_list_recent_memory_includes_feedback(tmp_path: Path):
    (tmp_path / "feedback_x.md").write_text("fb", encoding="utf-8")
    (tmp_path / "project_y.md").write_text("pr", encoding="utf-8")
    (tmp_path / "user_z.md").write_text("usr", encoding="utf-8")  # not matched
    since = _dt.datetime.now() - _dt.timedelta(days=1)
    found = _list_recent_memory(tmp_path, since)
    names = [p.name for p in found]
    assert set(names) == {"feedback_x.md", "project_y.md"}


# ---------------------------------------------------------------------------
# _summarize_memory_entries
# ---------------------------------------------------------------------------

def test_summarize_respects_char_budget(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"project_{i}.md").write_text("# Big\n" + "X" * 5000, encoding="utf-8")
    paths = list(tmp_path.glob("project_*.md"))
    summary = _summarize_memory_entries(paths, char_budget=200)
    assert len(summary) <= 250  # budget + small overhead


# ---------------------------------------------------------------------------
# _render_markdown
# ---------------------------------------------------------------------------

def test_render_markdown_full_payload():
    payload = {
        "architecture": "**X**\n\nfoo bar",
        "invariants": ["NEVER do X.", "2. ALWAYS do Y."],
        "active_modules": [
            {"name": "modules/a", "paragraph": "p1"},
            {"name": "modules/b", "paragraph": "p2"},
            {"name": "modules/c", "paragraph": "p3"},
        ],
        "recent_pivots": [
            {"date": "2026-04-29", "title": "Pivot1", "trigger": "T", "decision": "D", "why": "W"},
        ],
    }
    md = _render_markdown(payload, "Master Project State", "2026-04-29")
    assert md.startswith("## Master Project State — 2026-04-29")
    assert "### Current Architecture" in md
    assert "### Global Invariants" in md
    assert "1. NEVER do X." in md  # auto-numbered when not pre-numbered
    assert "2. ALWAYS do Y." in md  # preserves pre-numbered
    assert "### Active Working Set" in md
    assert "**modules/a**" in md
    assert "### Recent Pivot Log" in md
    assert "**2026-04-29 — Pivot1**" in md
    assert "Trigger: T" in md


def test_render_markdown_fallback_marker():
    payload = {
        "architecture": "",
        "invariants": [],
        "active_modules": [],
        "recent_pivots": [],
        "_fallback": True,
    }
    md = _render_markdown(payload, "Master Project State", "2026-04-29")
    assert "LLM compaction returned an empty payload" in md


# ---------------------------------------------------------------------------
# _upsert_section
# ---------------------------------------------------------------------------

def test_upsert_section_replaces_existing(tmp_path: Path):
    f = tmp_path / "decisions.md"
    f.write_text(
        "# Decisions\n\n## Master Project State\n\nold\n\n## Other\n\nkept\n",
        encoding="utf-8",
    )
    new_section = "## Master Project State\n\nnew\n"
    result = _upsert_section(f, "Master Project State", new_section)
    text = f.read_text(encoding="utf-8")
    assert "new" in text
    assert "old" not in text
    assert "## Other" in text  # untouched
    assert result["backup"] is not None
    assert Path(result["backup"]).exists()


def test_upsert_section_prepends_when_missing(tmp_path: Path):
    f = tmp_path / "decisions.md"
    f.write_text("# Decisions\n\n## Other\n\nbody\n", encoding="utf-8")
    new_section = "## Master Project State\n\nnew\n"
    _upsert_section(f, "Master Project State", new_section)
    text = f.read_text(encoding="utf-8")
    assert text.index("## Master Project State") < text.index("## Other")


def test_upsert_section_creates_file(tmp_path: Path):
    f = tmp_path / "subdir" / "decisions.md"
    new_section = "## Master Project State\n\nnew\n"
    result = _upsert_section(f, "Master Project State", new_section)
    assert f.exists()
    assert "new" in f.read_text(encoding="utf-8")
    assert result["backup"] is None  # no backup on first write


def test_upsert_section_idempotent(tmp_path: Path):
    f = tmp_path / "decisions.md"
    f.write_text("# Decisions\n\n## Master Project State\n\nbody\n", encoding="utf-8")
    new_section = "## Master Project State\n\nbody\n"
    r1 = _upsert_section(f, "Master Project State", new_section)
    r2 = _upsert_section(f, "Master Project State", new_section)
    # Second write produces a backup of the first; content same.
    assert f.read_text(encoding="utf-8").count("## Master Project State") == 1
    assert r1["backup"] is not None  # had pre-existing file
    assert r2["backup"] is not None  # had pre-existing file again


# ---------------------------------------------------------------------------
# compact_to_master_state — end-to-end with mocked LLM
# ---------------------------------------------------------------------------

def test_compact_to_master_state_skipped_when_no_memory_dir(tmp_path: Path, monkeypatch):
    # Force no auto-memory dir lookup: point HOME to a tmp dir without ~/.claude/projects.
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    result = compact_to_master_state(str(project), dry_run=True)
    assert result["status"] == "skipped"


def test_compact_to_master_state_dry_run(tmp_path: Path, monkeypatch):
    # Stand up a fake auto-memory dir matching what _project_to_memory_dir expects.
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    # Slug must mirror the resolved project path: -<tmp>-myproj
    slug = "-" + str(project.resolve()).strip("/").replace("/", "-")
    mem_dir = tmp_path / ".claude" / "projects" / slug / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "project_recent.md").write_text("# Recent\nstuff happened today\n", encoding="utf-8")

    fake_payload = {
        "architecture": "**Test**\n\nbody",
        "invariants": ["NEVER X."],
        "active_modules": [{"name": "modules/x", "paragraph": "para"}],
        "recent_pivots": [{"date": "2026-04-29", "title": "T", "trigger": "tg", "decision": "d", "why": "w"}],
    }
    with patch("skill_hub.embeddings.compact_master_state", return_value=fake_payload):
        result = compact_to_master_state(str(project), dry_run=True)
    assert result["status"] == "dry_run"
    assert "## Master Project State" in result["rendered"]
    # File should NOT have been written
    assert not (project / ".memory" / "decisions.md").exists()


# ---------------------------------------------------------------------------
# Atomic write + backup retention
# ---------------------------------------------------------------------------

def test_atomic_write_creates_file(tmp_path: Path):
    target = tmp_path / "sub" / "out.md"
    _atomic_write(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_overwrites_existing(tmp_path: Path):
    target = tmp_path / "out.md"
    target.write_text("old", encoding="utf-8")
    _atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    # Confirm no orphan tempfiles in the dir.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.md.")]
    assert leftovers == []


def test_prune_backups_retains_n_newest(tmp_path: Path):
    backup_dir = tmp_path / ".backups"
    backup_dir.mkdir()
    # Create 15 backup files with monotonic mtimes.
    import time as _time
    for i in range(15):
        f = backup_dir / f"decisions-2026-04-{i:02d}-000000.md"
        f.write_text(f"v{i}", encoding="utf-8")
        _time.sleep(0.01)  # ensure distinct mtime
    deleted = _prune_backups(backup_dir, "decisions", retain=10)
    assert deleted == 5
    remaining = sorted(backup_dir.glob("decisions-*"))
    assert len(remaining) == 10
    # The 5 oldest should be gone, newest 10 retained.
    survived_names = {p.name for p in remaining}
    for i in range(5, 15):
        assert f"decisions-2026-04-{i:02d}-000000.md" in survived_names


def test_prune_backups_noop_when_under_threshold(tmp_path: Path):
    backup_dir = tmp_path / ".backups"
    backup_dir.mkdir()
    for i in range(3):
        (backup_dir / f"decisions-2026-04-{i:02d}-000000.md").write_text("x", encoding="utf-8")
    deleted = _prune_backups(backup_dir, "decisions", retain=10)
    assert deleted == 0


def test_upsert_section_prunes_backups(tmp_path: Path):
    f = tmp_path / "decisions.md"
    # Pre-seed 12 stale backups.
    backup_dir = tmp_path / ".backups"
    backup_dir.mkdir()
    import time as _time
    for i in range(12):
        (backup_dir / f"decisions-2026-04-{i:02d}-000000.md").write_text("x", encoding="utf-8")
        _time.sleep(0.005)
    # Existing file forces a fresh backup on upsert; combined with prune, count must end at retain=10.
    f.write_text("# Decisions\n\n## Master Project State\n\nold\n", encoding="utf-8")
    _upsert_section(f, "Master Project State", "## Master Project State\n\nnew\n", retain_backups=10)
    backups = list(backup_dir.glob("decisions-*"))
    assert len(backups) == 10  # 12 pre-seeded + 1 new = 13, prune to 10


# ---------------------------------------------------------------------------
# mtime-based no-op
# ---------------------------------------------------------------------------

def test_compact_to_master_state_noop_when_snapshot_fresher(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    slug = "-" + str(project.resolve()).strip("/").replace("/", "-")
    mem_dir = tmp_path / ".claude" / "projects" / slug / "memory"
    mem_dir.mkdir(parents=True)
    mem_file = mem_dir / "project_recent.md"
    mem_file.write_text("# Recent\n", encoding="utf-8")

    decisions = project / ".memory" / "decisions.md"
    decisions.parent.mkdir(parents=True)
    decisions.write_text("# Decisions\n\n## Master Project State\n\nfresh\n", encoding="utf-8")

    # Snapshot is brand-new; memory was made before it.
    long_ago = _dt.datetime.now() - _dt.timedelta(hours=1)
    import os as _os
    _os.utime(mem_file, (long_ago.timestamp(), long_ago.timestamp()))

    with patch("skill_hub.embeddings.compact_master_state") as mock_llm:
        result = compact_to_master_state(str(project), dry_run=False)
    assert result["status"] == "noop"
    assert "fresher than newest" in result["reason"]
    mock_llm.assert_not_called()  # critical: no LLM call when no-op


def test_compact_to_master_state_dry_run_bypasses_noop(tmp_path: Path, monkeypatch):
    """Dry run must always render even if snapshot is fresh — user explicitly asked."""
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    slug = "-" + str(project.resolve()).strip("/").replace("/", "-")
    mem_dir = tmp_path / ".claude" / "projects" / slug / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "project_recent.md").write_text("x", encoding="utf-8")
    decisions = project / ".memory" / "decisions.md"
    decisions.parent.mkdir(parents=True)
    decisions.write_text("# Decisions\n\n## Master Project State\n\nfresh\n", encoding="utf-8")

    fake_payload = {
        "architecture": "x", "invariants": [], "active_modules": [], "recent_pivots": [],
    }
    with patch("skill_hub.embeddings.compact_master_state", return_value=fake_payload):
        result = compact_to_master_state(str(project), dry_run=True)
    assert result["status"] == "dry_run"


# ---------------------------------------------------------------------------
# Slug walk-up logic against a layout shaped like the real one
# ---------------------------------------------------------------------------

def test_project_to_memory_dir_walks_up_to_parent_slug(tmp_path: Path, monkeypatch):
    """Mimics real layout: ~/.claude/projects/-Users-X-work-code/memory/ exists,
    but ~/.claude/projects/-Users-X-work-code-geoid/ exists too WITHOUT a memory subdir.
    The lookup must find the parent's memory dir, not the leaf's missing one."""
    from skill_hub.master_state import _project_to_memory_dir
    monkeypatch.setenv("HOME", str(tmp_path))
    parent = tmp_path / "work" / "code"
    geoid = parent / "geoid"
    geoid.mkdir(parents=True)
    # Parent slug HAS memory; leaf slug exists but no memory subdir.
    parent_slug = "-" + str(parent.resolve()).strip("/").replace("/", "-")
    geoid_slug = "-" + str(geoid.resolve()).strip("/").replace("/", "-")
    (tmp_path / ".claude" / "projects" / parent_slug / "memory").mkdir(parents=True)
    (tmp_path / ".claude" / "projects" / geoid_slug).mkdir(parents=True)  # no memory/

    found = _project_to_memory_dir(geoid)
    assert found is not None
    assert found.name == "memory"
    # Should be the parent's, not the leaf's
    assert parent_slug in str(found)


def test_compact_to_master_state_writes_and_backs_up(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    slug = "-" + str(project.resolve()).strip("/").replace("/", "-")
    mem_dir = tmp_path / ".claude" / "projects" / slug / "memory"
    mem_dir.mkdir(parents=True)
    mem_file = mem_dir / "project_recent.md"
    mem_file.write_text("# Recent\n", encoding="utf-8")

    # Pre-existing decisions.md with a Master Project State section.
    decisions = project / ".memory" / "decisions.md"
    decisions.parent.mkdir(parents=True)
    decisions.write_text(
        "# Architectural Decisions\n\n## Master Project State — old\n\nold body\n\n## Other\n\nkept\n",
        encoding="utf-8",
    )
    # Backdate decisions.md so memory file is newer (forces non-noop path).
    import os as _os2
    long_ago = _dt.datetime.now() - _dt.timedelta(days=2)
    _os2.utime(decisions, (long_ago.timestamp(), long_ago.timestamp()))

    fake_payload = {
        "architecture": "**New**\n\nbody",
        "invariants": ["NEVER X."],
        "active_modules": [{"name": "modules/x", "paragraph": "para"}],
        "recent_pivots": [{"date": "2026-04-29", "title": "T", "trigger": "tg", "decision": "d", "why": "w"}],
    }
    with patch("skill_hub.embeddings.compact_master_state", return_value=fake_payload):
        result = compact_to_master_state(str(project), dry_run=False)
    assert result["status"] == "written"
    text = decisions.read_text(encoding="utf-8")
    assert "**New**" in text
    assert "old body" not in text
    assert "## Other" in text  # preserved
    # Backup file exists
    backups = list((decisions.parent / ".backups").glob("*.md"))
    assert len(backups) == 1
    assert "old body" in backups[0].read_text(encoding="utf-8")
