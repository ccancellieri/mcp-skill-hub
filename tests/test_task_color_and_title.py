"""Tests for the task color column + title rewrite + auto-derive flow.

Covers the patch that lets ``update_task`` rewrite the title column and assign
a colour, plus the session_start_enforcer's auto-derivation of title (from
memory frontmatter) and colour (from description text).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
HOOKS_DIR = ROOT / "hooks"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(HOOKS_DIR))


@pytest.fixture
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    db_path = tmp_path / "skill_hub.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    s = SkillStore(db_path=db_path)
    yield s
    s.close()


def test_color_column_present(store):
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(tasks)")}
    assert "color" in cols, f"color column missing from tasks table: {cols}"


def test_save_task_persists_color(store):
    tid = store.save_task(title="t1", summary="s1", vector=[],
                          color="yellow")
    row = store._conn.execute(
        "SELECT title, color FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["title"] == "t1"
    assert row["color"] == "yellow"


def test_save_task_color_optional(store):
    """Color must be NULL when omitted, not coerced to empty string."""
    tid = store.save_task(title="t2", summary="s2", vector=[])
    row = store._conn.execute(
        "SELECT color FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["color"] is None


def test_update_task_rewrites_title(store):
    tid = store.save_task(title="old-title", summary="s", vector=[])
    assert store.update_task(tid, title="New Human Title")
    row = store._conn.execute(
        "SELECT title FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["title"] == "New Human Title"


def test_update_task_assigns_color(store):
    tid = store.save_task(title="t", summary="s", vector=[])
    assert store.update_task(tid, color="red")
    row = store._conn.execute(
        "SELECT color FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["color"] == "red"


def test_update_task_preserves_other_fields(store):
    """Setting color must not wipe summary/title/tags."""
    tid = store.save_task(title="orig-title", summary="orig-summary",
                          vector=[], tags="a,b")
    assert store.update_task(tid, color="green")
    row = store._conn.execute(
        "SELECT title, summary, tags, color FROM tasks WHERE id = ?",
        (tid,)).fetchone()
    assert row["title"] == "orig-title"
    assert row["summary"] == "orig-summary"
    assert row["tags"] == "a,b"
    assert row["color"] == "green"


def test_list_tasks_includes_color(store):
    tid = store.save_task(title="t", summary="s", vector=[], color="cyan")
    rows = store.list_tasks(status="open")
    matching = [r for r in rows if r["id"] == tid]
    assert len(matching) == 1
    assert matching[0]["color"] == "cyan"


# -----------------------------------------------------------------------------
# session_start_enforcer auto-derive: classify_color + frontmatter parse
# -----------------------------------------------------------------------------


def test_classify_color_deferred_beats_shipped():
    from session_start_enforcer import _classify_color
    assert _classify_color(
        "Phase A SHIPPED 2026-04-14; Phase B DEFERRED to own branch") == "red"


def test_classify_color_partial_beats_shipped():
    from session_start_enforcer import _classify_color
    assert _classify_color(
        "SHIPPED v0.4.58 + follow-on (PARTIAL: items 5b/5c remain)") == "yellow"


def test_classify_color_in_progress():
    from session_start_enforcer import _classify_color
    assert _classify_color("IN PROGRESS 2026-04-27 PR-1 done locally") == "yellow"


def test_classify_color_open_pr():
    from session_start_enforcer import _classify_color
    assert _classify_color("OPEN PR #82 awaiting review") == "cyan"


def test_classify_color_shipped():
    from session_start_enforcer import _classify_color
    assert _classify_color("SHIPPED 2026-04-23 v0.5.38") == "green"


def test_classify_color_default_blue():
    from session_start_enforcer import _classify_color
    assert _classify_color("some neutral note about a topic") == "blue"


def test_read_memory_frontmatter_basic(tmp_path):
    from session_start_enforcer import _read_memory_frontmatter
    p = tmp_path / "mem.md"
    p.write_text(
        "---\n"
        "name: GeoID — proper title\n"
        "description: SHIPPED 2026-04-15 v0.3.30\n"
        "type: project\n"
        "---\n"
        "Body content here.\n"
    )
    fm = _read_memory_frontmatter(p)
    assert fm["name"] == "GeoID — proper title"
    assert fm["description"] == "SHIPPED 2026-04-15 v0.3.30"
    assert fm["type"] == "project"


def test_read_memory_frontmatter_missing_file(tmp_path):
    from session_start_enforcer import _read_memory_frontmatter
    assert _read_memory_frontmatter(tmp_path / "nope.md") == {}


def test_read_memory_frontmatter_no_frontmatter(tmp_path):
    """Plain markdown with no YAML block returns empty dict (graceful)."""
    from session_start_enforcer import _read_memory_frontmatter
    p = tmp_path / "plain.md"
    p.write_text("Just a heading\n\nSome content.\n")
    assert _read_memory_frontmatter(p) == {}


def test_list_tasks_renders_glyph(monkeypatch, store):
    """The list_tasks MCP tool prefixes each row with a colour glyph."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    store.save_task(title="Red task", summary="x", vector=[], color="red")
    store.save_task(title="Green task", summary="x", vector=[], color="green")
    store.save_task(title="No-colour task", summary="x", vector=[])

    out = srv.list_tasks(status="open")
    assert "✗ #" in out, "red glyph missing"
    assert "● #" in out, "green glyph missing"
    # No-colour rows render with a leading space placeholder so columns align.
    assert "No-colour task" in out
