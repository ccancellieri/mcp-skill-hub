"""M3 — cross-project task federation.

Tests that every task carries a ``repo`` column (auto-captured from the
worktree/cwd when possible), and that the list_tasks tool + dashboard
metrics expose a per-repo grouping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    db_path = tmp_path / "skill_hub.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    s = SkillStore(db_path=db_path)
    yield s
    s.close()


# ----- schema migration -------------------------------------------------------


def test_repo_column_present(store):
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(tasks)")}
    assert "repo" in cols, f"repo column missing from tasks: {cols}"


def test_repo_index_present(store):
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='tasks'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "idx_tasks_repo" in names, f"idx_tasks_repo missing: {names}"


# ----- save_task + list_tasks filter -----------------------------------------


def test_save_task_persists_repo(store):
    tid = store.save_task(title="t1", summary="s1", vector=[],
                          repo="geoid")
    row = store._conn.execute(
        "SELECT repo FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["repo"] == "geoid"


def test_save_task_repo_optional(store):
    """repo must be NULL when omitted, not coerced to empty string."""
    tid = store.save_task(title="t2", summary="s2", vector=[])
    row = store._conn.execute(
        "SELECT repo FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["repo"] is None


def test_list_tasks_repo_filter_basic(store):
    tid_a = store.save_task(title="A", summary="x", vector=[], repo="geoid")
    tid_b = store.save_task(title="B", summary="y", vector=[],
                            repo="fao-aip-catalog")
    tid_c = store.save_task(title="C", summary="z", vector=[])  # no repo

    geoid_rows = store.list_tasks(status="open", repo="geoid")
    assert {r["id"] for r in geoid_rows} == {tid_a}

    catalog_rows = store.list_tasks(status="open", repo="fao-aip-catalog")
    assert {r["id"] for r in catalog_rows} == {tid_b}

    # repo=None / "" returns everything
    all_rows = store.list_tasks(status="open")
    assert {r["id"] for r in all_rows} == {tid_a, tid_b, tid_c}


def test_list_tasks_repo_filter_no_match(store):
    store.save_task(title="A", summary="x", vector=[], repo="geoid")
    rows = store.list_tasks(status="open", repo="does-not-exist")
    assert rows == []


def test_list_tasks_includes_repo_column(store):
    tid = store.save_task(title="A", summary="x", vector=[],
                          repo="mcp-skill-hub")
    rows = store.list_tasks(status="open")
    match = [r for r in rows if r["id"] == tid]
    assert len(match) == 1
    assert match[0]["repo"] == "mcp-skill-hub"


def test_list_tasks_by_repo_groups(store):
    a = store.save_task(title="A", summary="x", vector=[], repo="geoid")
    b = store.save_task(title="B", summary="y", vector=[], repo="geoid")
    c = store.save_task(title="C", summary="z", vector=[],
                        repo="fao-aip-catalog")
    d = store.save_task(title="D", summary="q", vector=[])  # no repo

    groups = store.list_tasks_by_repo(status="open")
    assert {r["id"] for r in groups["geoid"]} == {a, b}
    assert {r["id"] for r in groups["fao-aip-catalog"]} == {c}
    assert {r["id"] for r in groups[""]} == {d}


def test_list_tasks_repo_and_tag_combine(store):
    """repo + tag filters must AND together, not replace each other."""
    a = store.save_task(title="A", summary="x", vector=[], repo="geoid",
                        tags="fanout:abc123")
    store.save_task(title="B", summary="y", vector=[], repo="geoid",
                    tags="other")
    store.save_task(title="C", summary="z", vector=[],
                    repo="fao-aip-catalog", tags="fanout:abc123")
    rows = store.list_tasks(status="open", repo="geoid",
                            tag="fanout:abc123")
    assert {r["id"] for r in rows} == {a}


# ----- MCP list_tasks tool ---------------------------------------------------


def test_mcp_list_tasks_repo_filter(monkeypatch, store):
    """The list_tasks MCP tool respects the ``repo`` kwarg."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    store.save_task(title="GeoID task", summary="x", vector=[], repo="geoid")
    store.save_task(title="Catalog task", summary="y", vector=[],
                    repo="fao-aip-catalog")

    out_geoid = srv.list_tasks(status="open", repo="geoid")
    assert "GeoID task" in out_geoid
    assert "Catalog task" not in out_geoid
    assert "repo=geoid" in out_geoid

    out_none = srv.list_tasks(status="open", repo="does-not-exist")
    assert "No open tasks" in out_none


def test_mcp_list_tasks_group_by_repo(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    store.save_task(title="A", summary="x", vector=[], repo="geoid")
    store.save_task(title="B", summary="y", vector=[],
                    repo="fao-aip-catalog")
    store.save_task(title="C", summary="z", vector=[])  # unassigned

    out = srv.list_tasks(status="open", group_by_repo=True)
    assert "## geoid" in out
    assert "## fao-aip-catalog" in out
    assert "## (unassigned)" in out
    assert "across 3 repo(s)" in out


# ----- dashboard aggregator --------------------------------------------------


def test_dashboard_tasks_by_repo(store):
    from skill_hub import dashboard as dash

    store.save_task(title="A", summary="x", vector=[], repo="geoid")
    store.save_task(title="B", summary="y", vector=[], repo="geoid")
    store.save_task(title="C", summary="z", vector=[],
                    repo="fao-aip-catalog")
    # Close one of the geoid tasks so we exercise both open and closed counts.
    rows = store.list_tasks(status="open", repo="geoid")
    store.close_task(rows[0]["id"], compact="done")

    metrics = dash._db_metrics(store)
    by_repo = {r["repo"]: r for r in metrics["tasks_by_repo"]}
    assert by_repo["geoid"]["open"] == 1
    assert by_repo["geoid"]["closed"] == 1
    assert by_repo["fao-aip-catalog"]["open"] == 1
    assert by_repo["fao-aip-catalog"]["closed"] == 0


def test_dashboard_tasks_by_repo_unassigned(store):
    from skill_hub import dashboard as dash
    store.save_task(title="A", summary="x", vector=[])  # no repo
    metrics = dash._db_metrics(store)
    repos = [r["repo"] for r in metrics["tasks_by_repo"]]
    assert "(unassigned)" in repos


# ----- save_task MCP tool — auto-derive repo --------------------------------


def test_mcp_save_task_explicit_repo(monkeypatch, store):
    """Explicit ``repo=`` kwarg lands in the database."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    # Avoid real embedding; the tool tolerates RuntimeError but we short-circuit
    # to keep the test offline.
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    out = srv.save_task(title="My task", summary="body",
                        repo="explicit-repo")
    # Find the row by title (server returns "Task #N saved").
    row = store._conn.execute(
        "SELECT repo FROM tasks WHERE title = ?", ("My task",)
    ).fetchone()
    assert row["repo"] == "explicit-repo"
    assert "saved" in out


def test_mcp_save_task_derives_repo_from_cwd(monkeypatch, store, tmp_path):
    """When ``cwd`` resolves under a configured root, repo is auto-captured."""
    from skill_hub import server as srv
    from skill_hub import worktree as _wt

    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    # Stub detect_project_from_cwd to return a known project name.
    monkeypatch.setattr(_wt, "detect_project_from_cwd",
                        lambda *_a, **_kw: "auto-derived-repo")

    srv.save_task(title="Derived", summary="body", cwd=str(tmp_path))
    row = store._conn.execute(
        "SELECT repo FROM tasks WHERE title = ?", ("Derived",)
    ).fetchone()
    assert row["repo"] == "auto-derived-repo"
