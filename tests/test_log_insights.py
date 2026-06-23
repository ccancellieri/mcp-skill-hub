"""Tests for log_insights module and the logs vector namespace (issue #90).

Covers:
(a) The ``logs`` namespace is present in ``_DEFAULT_VECTOR_INDEXES``.
(b) ``index_recent_logs(dry_run=True)`` scans events but writes zero vectors.
(c) After appending events, ``index_recent_logs()`` indexes them and the
    summary counts + by_kind dict are correct.
(d) doc_id / metadata shape is stable.

Constraints:
- Uses only skill_hub.store and skill_hub.log_insights — never skill_hub.server.
- Every test uses a fresh tmp_path DB, never DB_PATH.
- No embedding model loaded (skill_hub.embeddings.embed is monkeypatched).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Guard: ensure this module's imports never drag in skill_hub.server.
def test_server_not_imported(assert_server_not_imported):  # noqa: PT019
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """A fresh DB-backed SkillStore isolated from the live DB."""
    from skill_hub.store import SkillStore

    db_path = tmp_path / "test_log_insights.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture()
def patched_store(isolated_store, monkeypatch):
    """isolated_store with embeddings.embed stubbed out (no ML model needed)."""
    monkeypatch.setattr("skill_hub.embeddings.embed", lambda text, **kw: [0.1, 0.2, 0.3])
    # Also point log_insights.get_store() at our isolated store.
    monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
    yield isolated_store


# ---------------------------------------------------------------------------
# (a) logs namespace in _DEFAULT_VECTOR_INDEXES
# ---------------------------------------------------------------------------

def test_logs_namespace_in_default_vector_indexes():
    """The 'logs' namespace must be seeded in _DEFAULT_VECTOR_INDEXES."""
    from skill_hub.store import _DEFAULT_VECTOR_INDEXES

    assert "logs" in _DEFAULT_VECTOR_INDEXES, (
        f"'logs' missing from _DEFAULT_VECTOR_INDEXES; keys: {list(_DEFAULT_VECTOR_INDEXES)}"
    )
    cfg = _DEFAULT_VECTOR_INDEXES["logs"]
    assert "default_level" in cfg, "logs entry must have 'default_level'"
    assert "half_life_days" in cfg, "logs entry must have 'half_life_days'"


def test_logs_namespace_has_short_half_life():
    """logs namespace should have a shorter half-life than permanent namespaces."""
    from skill_hub.store import _DEFAULT_VECTOR_INDEXES

    logs_half_life = _DEFAULT_VECTOR_INDEXES["logs"]["half_life_days"]
    skills_half_life = _DEFAULT_VECTOR_INDEXES["skills"]["half_life_days"]
    assert logs_half_life < skills_half_life, (
        f"logs half_life {logs_half_life} should be less than skills {skills_half_life}"
    )


# ---------------------------------------------------------------------------
# (b) dry_run=True: events scanned, zero vectors written
# ---------------------------------------------------------------------------

def test_dry_run_scans_but_writes_nothing(patched_store):
    """dry_run=True returns scanned>0 / indexed>0 but writes nothing to vectors."""
    t0 = time.time() - 60  # within the last hour
    patched_store.append_event("sess-dry", "tool_invoke", {"q": "hello"}, tool_name="search_skills", ts=t0)
    patched_store.append_event("sess-dry", "tool_result", {"ok": True}, tool_name="search_skills", ts=t0 + 1)

    from skill_hub import log_insights

    report = log_insights.index_recent_logs(hours=1, dry_run=True)

    assert report["dry_run"] is True
    assert report["namespace"] == "logs"
    assert report["scanned"] >= 2
    assert report["indexed"] >= 2

    # No rows in vectors table.
    count = patched_store._conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace = 'logs'"
    ).fetchone()[0]
    assert count == 0, f"dry_run must write nothing, got {count} rows"


# ---------------------------------------------------------------------------
# (c) Live run: events indexed, counts / by_kind correct
# ---------------------------------------------------------------------------

def test_live_run_indexes_events(patched_store):
    """After appending events, index_recent_logs() indexes them correctly."""
    t0 = time.time() - 60
    patched_store.append_event("sess-live", "tool_invoke", {"q": "a"}, tool_name="teach", ts=t0)
    patched_store.append_event("sess-live", "tool_invoke", {"q": "b"}, tool_name="teach", ts=t0 + 1)
    patched_store.append_event("sess-live", "tool_result", {"ok": True}, tool_name="teach", ts=t0 + 2)

    from skill_hub import log_insights

    report = log_insights.index_recent_logs(hours=1, dry_run=False)

    assert report["dry_run"] is False
    assert report["scanned"] >= 3
    assert report["indexed"] >= 3
    assert "tool_invoke" in report["by_kind"]
    assert "tool_result" in report["by_kind"]
    assert report["by_kind"]["tool_invoke"] >= 2
    assert report["by_kind"]["tool_result"] >= 1

    # Rows must exist in the vectors table.
    count = patched_store._conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace = 'logs'"
    ).fetchone()[0]
    assert count >= 3, f"Expected >=3 rows in logs namespace, got {count}"


def test_by_kind_counts_match_event_kinds(patched_store):
    """by_kind values must match the actual distribution of event kinds."""
    t0 = time.time() - 60
    patched_store.append_event("sess-kinds", "session_start", {}, ts=t0)
    patched_store.append_event("sess-kinds", "tool_invoke", {"x": 1}, tool_name="t1", ts=t0 + 1)
    patched_store.append_event("sess-kinds", "tool_invoke", {"x": 2}, tool_name="t2", ts=t0 + 2)
    patched_store.append_event("sess-kinds", "session_end", {}, ts=t0 + 3)

    from skill_hub import log_insights

    report = log_insights.index_recent_logs(hours=1, dry_run=True)

    bk = report["by_kind"]
    assert bk.get("session_start", 0) >= 1
    assert bk.get("tool_invoke", 0) >= 2
    assert bk.get("session_end", 0) >= 1


# ---------------------------------------------------------------------------
# (d) doc_id and metadata shape are stable
# ---------------------------------------------------------------------------

def test_doc_id_and_metadata_shape(patched_store):
    """Vectors written to the logs namespace must have the expected doc_id and metadata."""
    t0 = time.time() - 60
    patched_store.append_event(
        "sess-shape", "tool_invoke", {"tool": "search_skills"},
        tool_name="search_skills", ts=t0,
    )

    from skill_hub import log_insights

    report = log_insights.index_recent_logs(hours=1, dry_run=False)
    assert report["indexed"] >= 1

    rows = patched_store._conn.execute(
        "SELECT doc_id, metadata FROM vectors WHERE namespace = 'logs'"
    ).fetchall()
    assert len(rows) >= 1

    row = rows[0]
    doc_id: str = row["doc_id"]
    # doc_id format: event:<session_id>:<ts>:<kind>
    assert doc_id.startswith("event:"), f"doc_id {doc_id!r} must start with 'event:'"
    parts = doc_id.split(":")
    assert len(parts) >= 4, f"doc_id {doc_id!r} must have at least 4 colon-separated parts"

    meta = json.loads(row["metadata"])
    for field in ("kind", "tool", "session_id", "ts", "source"):
        assert field in meta, f"metadata missing field {field!r}"
    assert meta["source"] == "event_log"


def test_old_events_excluded_by_hours_filter(patched_store):
    """Events older than 'hours' must not appear in the scan."""
    t_old = time.time() - 3 * 3600 - 60  # 3 h + 1 min ago
    t_recent = time.time() - 60            # 1 min ago

    patched_store.append_event("sess-filter", "tool_invoke", {"x": "old"}, ts=t_old)
    patched_store.append_event("sess-filter", "tool_invoke", {"x": "new"}, ts=t_recent)

    from skill_hub import log_insights

    # Only look 1 hour back — should pick up only the recent event.
    report = log_insights.index_recent_logs(hours=1, dry_run=True)

    assert report["scanned"] == 1, (
        f"Expected 1 event scanned (the recent one), got {report['scanned']}"
    )


def test_limit_respected(patched_store):
    """limit parameter caps the number of events scanned."""
    t0 = time.time() - 60
    for i in range(10):
        patched_store.append_event("sess-lim", "tool_invoke", {"i": i}, ts=t0 + i)

    from skill_hub import log_insights

    report = log_insights.index_recent_logs(hours=1, limit=5, dry_run=True)

    assert report["scanned"] <= 5, (
        f"Expected at most 5 events scanned with limit=5, got {report['scanned']}"
    )
