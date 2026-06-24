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


# ===========================================================================
# Part B — cluster_failures (Phase 2)
# ===========================================================================

class TestClusterFailures:
    """Tests for log_insights.cluster_failures (deterministic, no LLM)."""

    def test_groups_identical_errors(self, isolated_store, monkeypatch):
        """Two identical errors for the same tool → single cluster with count=2."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        payload = json.dumps({"ok": False, "error": "connection refused", "elapsed_ms": 5})
        isolated_store.append_event("s", "tool_result", payload, tool_name="search_skills", ts=t0)
        isolated_store.append_event("s", "tool_result", payload, tool_name="search_skills", ts=t0 + 1)

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=2)

        assert result["scanned"] >= 2
        assert len(result["clusters"]) == 1
        c = result["clusters"][0]
        assert c["tool"] == "search_skills"
        assert c["count"] == 2
        assert "connection refused" in c["pattern"]

    def test_respects_min_count(self, isolated_store, monkeypatch):
        """Clusters with count < min_count are excluded."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        payload = json.dumps({"ok": False, "error": "timeout", "elapsed_ms": 10})
        # Only one occurrence
        isolated_store.append_event("s", "tool_result", payload, tool_name="teach", ts=t0)

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=2)

        assert result["scanned"] >= 1
        # min_count=2 means this single error must NOT appear
        assert len(result["clusters"]) == 0

    def test_normalisation_merges_near_identical(self, isolated_store, monkeypatch):
        """Errors differing only in numeric IDs / addresses merge into one cluster."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        err1 = "sqlite error: UNIQUE constraint failed row 12345"
        err2 = "sqlite error: UNIQUE constraint failed row 67890"
        isolated_store.append_event(
            "s", "tool_result",
            json.dumps({"ok": False, "error": err1}),
            tool_name="save_task", ts=t0,
        )
        isolated_store.append_event(
            "s", "tool_result",
            json.dumps({"ok": False, "error": err2}),
            tool_name="save_task", ts=t0 + 1,
        )

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=2)

        # After normalisation, both errors should collapse to one cluster
        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["count"] == 2

    def test_ok_true_not_included(self, isolated_store, monkeypatch):
        """Successful tool_result events (ok=True) must not appear in clusters."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        ok_payload = json.dumps({"ok": True, "elapsed_ms": 5})
        for _ in range(3):
            isolated_store.append_event(
                "s", "tool_result", ok_payload, tool_name="search_skills", ts=t0,
            )

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=1)

        assert len(result["clusters"]) == 0, "ok=True events must not cluster"

    def test_clusters_sorted_by_count_desc(self, isolated_store, monkeypatch):
        """cluster list must be sorted by count descending."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        err_a = json.dumps({"ok": False, "error": "error alpha"})
        err_b = json.dumps({"ok": False, "error": "error beta"})
        # 3 of alpha, 2 of beta
        for _ in range(3):
            isolated_store.append_event("s", "tool_result", err_a, tool_name="t1", ts=t0)
        for _ in range(2):
            isolated_store.append_event("s", "tool_result", err_b, tool_name="t2", ts=t0)

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=1)

        assert len(result["clusters"]) >= 2
        counts = [c["count"] for c in result["clusters"]]
        assert counts == sorted(counts, reverse=True)

    def test_result_shape(self, isolated_store, monkeypatch):
        """Each cluster dict must have all required keys."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        payload = json.dumps({"ok": False, "error": "test error"})
        for _ in range(2):
            isolated_store.append_event("s", "tool_result", payload, tool_name="t", ts=t0)

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=2)

        assert "scanned" in result
        assert "clusters" in result
        assert len(result["clusters"]) == 1
        c = result["clusters"][0]
        for key in ("tool", "pattern", "count", "example", "first_ts", "last_ts", "suggested_action"):
            assert key in c, f"cluster missing key '{key}'"

    def test_suggested_action_contains_tool(self, isolated_store, monkeypatch):
        """suggested_action must reference the tool name."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        t0 = time.time() - 60
        payload = json.dumps({"ok": False, "error": "some error"})
        for _ in range(2):
            isolated_store.append_event("s", "tool_result", payload, tool_name="my_tool", ts=t0)

        from skill_hub import log_insights
        result = log_insights.cluster_failures(hours=1, min_count=2)

        assert len(result["clusters"]) == 1
        assert "my_tool" in result["clusters"][0]["suggested_action"]


# ===========================================================================
# Part C — skill_selection_stats (Phase 2, feasible subset)
# ===========================================================================

class TestSkillSelectionStats:
    """Tests for log_insights.skill_selection_stats."""

    def test_injection_counts_per_skill(self, isolated_store, monkeypatch):
        """Injection counts must reflect log_skill_injection calls."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-A", query="q1", session_id="s1")
        isolated_store.log_skill_injection("skill-A", query="q2", session_id="s1")
        isolated_store.log_skill_injection("skill-B", query="q3", session_id="s2")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        assert by_id["skill-A"]["injections"] == 2
        assert by_id["skill-B"]["injections"] == 1
        assert stats["total_injections"] == 3

    def test_feedback_helpful_rate(self, isolated_store, monkeypatch):
        """helpful_rate is the fraction of feedback rows where helpful=1."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        # Need a skill in the skills table before adding feedback (FK not enforced on SQLite
        # without PRAGMA foreign_keys=ON, so we can insert directly).
        isolated_store.log_skill_injection("skill-X", query="q", session_id="s")

        conn = isolated_store._conn
        # Insert feedback directly (record_feedback requires query_vector)
        conn.execute(
            "INSERT INTO feedback (query, query_vector, skill_id, helpful) VALUES (?,?,?,?)",
            ("q", "[]", "skill-X", 1),
        )
        conn.execute(
            "INSERT INTO feedback (query, query_vector, skill_id, helpful) VALUES (?,?,?,?)",
            ("q", "[]", "skill-X", 0),
        )
        conn.commit()

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        assert "skill-X" in by_id
        r = by_id["skill-X"]
        assert r["feedback_n"] == 2
        # 1 helpful out of 2 → 0.5
        assert abs(r["helpful_rate"] - 0.5) < 1e-9

    def test_never_helpful_status(self, isolated_store, monkeypatch):
        """A skill with only unhelpful feedback should get status 'review: never-helpful'."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-Z", query="q", session_id="s")
        conn = isolated_store._conn
        for _ in range(2):
            conn.execute(
                "INSERT INTO feedback (query, query_vector, skill_id, helpful) VALUES (?,?,?,?)",
                ("q", "[]", "skill-Z", 0),
            )
        conn.commit()

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        assert by_id["skill-Z"]["status"] == "review: never-helpful"

    def test_no_feedback_skill_status(self, isolated_store, monkeypatch):
        """A frequently injected skill with no feedback should say 'no feedback yet'."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        for i in range(5):
            isolated_store.log_skill_injection("skill-NF", query=f"q{i}", session_id="s")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        assert "skill-NF" in by_id
        assert by_id["skill-NF"]["status"] == "no feedback yet"

    def test_result_shape(self, isolated_store, monkeypatch):
        """Each skill row must have all required keys."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-shape", query="q", session_id="s")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        assert "skills" in stats
        assert "total_injections" in stats
        assert "total_feedback" in stats
        assert len(stats["skills"]) >= 1
        for key in ("skill_id", "injections", "feedback_n", "helpful_rate", "status"):
            assert key in stats["skills"][0], f"skill row missing key '{key}'"

    def test_skills_sorted_by_injections_desc(self, isolated_store, monkeypatch):
        """Skills must be sorted by injection count descending."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        for _ in range(3):
            isolated_store.log_skill_injection("skill-hi", query="q", session_id="s")
        isolated_store.log_skill_injection("skill-lo", query="q", session_id="s")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        injection_counts = [r["injections"] for r in stats["skills"]]
        assert injection_counts == sorted(injection_counts, reverse=True)

    def test_used_rate_is_zero_when_no_skill_used_events(self, isolated_store, monkeypatch):
        """used_rate is 0/injections when no skill.used events exist; status is NOT
        'injected-but-unused' because total_used==0 means the hook has no data yet."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-noused", query="q", session_id="s1")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        r = by_id["skill-noused"]
        assert r["used"] == 0
        assert r["used_rate"] == 0.0
        # No hook data yet → do NOT flag as injected-but-unused
        assert r["status"] != "injected-but-unused"
        assert stats["total_used"] == 0

    def test_injected_but_unused_requires_hook_data(self, isolated_store, monkeypatch):
        """'injected-but-unused' is NOT assigned when total_used==0 (no hook data),
        and IS assigned when total_used>0 and that specific skill has zero uses."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)

        # Two skills injected; skill-live gets a used event, skill-dead does not.
        isolated_store.log_skill_injection("skill-dead2", query="q", session_id="s1")
        isolated_store.log_skill_injection("skill-live2", query="q", session_id="s1")

        from skill_hub import log_insights

        # Before any skill.used events: neither skill is flagged.
        stats_before = log_insights.skill_selection_stats(limit=100)
        by_id_before = {r["skill_id"]: r for r in stats_before["skills"]}
        assert stats_before["total_used"] == 0
        assert by_id_before["skill-dead2"]["status"] != "injected-but-unused"
        assert by_id_before["skill-live2"]["status"] != "injected-but-unused"

        # Emit a skill.used event only for skill-live2.
        isolated_store.record_skill_used("skill-live2", "s1")

        stats_after = log_insights.skill_selection_stats(limit=100)
        by_id_after = {r["skill_id"]: r for r in stats_after["skills"]}
        assert stats_after["total_used"] == 1
        # skill-dead2 has injections but zero used events → flagged
        assert by_id_after["skill-dead2"]["status"] == "injected-but-unused"
        # skill-live2 was used → not flagged
        assert by_id_after["skill-live2"]["status"] != "injected-but-unused"

    def test_used_rate_computed_from_skill_used_events(self, isolated_store, monkeypatch):
        """used_rate = used / injections when skill.used events are present."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-used", query="q", session_id="s1")
        isolated_store.log_skill_injection("skill-used", query="q", session_id="s1")
        # Emit one skill.used event (used=1, injections=2 → used_rate=0.5)
        isolated_store.record_skill_used("skill-used", "s1")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        r = by_id["skill-used"]
        assert r["used"] == 1
        assert r["injections"] == 2
        assert abs(r["used_rate"] - 0.5) < 1e-9
        assert r["status"] != "injected-but-unused"
        assert stats["total_used"] == 1

    def test_injected_but_unused_flag(self, isolated_store, monkeypatch):
        """A skill injected but never used must be flagged 'injected-but-unused'."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-dead", query="q", session_id="s1")
        isolated_store.log_skill_injection("skill-live", query="q", session_id="s1")
        # Only skill-live gets a used event
        isolated_store.record_skill_used("skill-live", "s1")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        by_id = {r["skill_id"]: r for r in stats["skills"]}
        assert by_id["skill-dead"]["status"] == "injected-but-unused"
        assert by_id["skill-live"]["status"] != "injected-but-unused"

    def test_result_shape_includes_used_fields(self, isolated_store, monkeypatch):
        """Each skill row must include 'used', 'used_rate', and 'total_used'."""
        monkeypatch.setattr("skill_hub.store._default_store", isolated_store)
        isolated_store.log_skill_injection("skill-shape2", query="q", session_id="s")

        from skill_hub import log_insights
        stats = log_insights.skill_selection_stats(limit=100)

        assert "total_used" in stats
        for key in ("skill_id", "injections", "used", "used_rate",
                    "feedback_n", "helpful_rate", "status"):
            assert key in stats["skills"][0], f"skill row missing key '{key}'"


# ===========================================================================
# Part D — record_skill_used store method
# ===========================================================================


class TestRecordSkillUsed:
    """Tests for SkillStore.record_skill_used."""

    @pytest.fixture()
    def store(self, tmp_path):
        from skill_hub.store import SkillStore
        return SkillStore(db_path=tmp_path / "skill_used.db")

    def test_emits_skill_used_event(self, store):
        """record_skill_used writes a skill.used event row."""
        store.log_skill_injection("sk-a", query="q", session_id="sess-1")
        eid = store.record_skill_used("sk-a", "sess-1")
        assert eid is not None and eid > 0

        rows = store.get_events(session_id="sess-1", kind="skill.used")
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["skill_id"] == "sk-a"
        assert payload["session_id"] == "sess-1"
        assert payload["matched"] is True
        assert payload["injection_id"] is not None

    def test_resolves_injection_id_automatically(self, store):
        """injection_id is the most recent skill_injections row for skill+session."""
        store.log_skill_injection("sk-b", query="q1", session_id="sess-2")
        store.log_skill_injection("sk-b", query="q2", session_id="sess-2")
        # Get the last injection id
        last_inj = store._conn.execute(
            "SELECT id FROM skill_injections WHERE skill_id='sk-b' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert last_inj is not None

        store.record_skill_used("sk-b", "sess-2")
        rows = store.get_events(session_id="sess-2", kind="skill.used")
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["injection_id"] == last_inj["id"]

    def test_unmatched_when_no_prior_injection(self, store):
        """skill.used is still written even with no matching injection (matched=False)."""
        eid = store.record_skill_used("sk-orphan", "sess-3")
        assert eid is not None

        rows = store.get_events(session_id="sess-3", kind="skill.used")
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["matched"] is False
        assert payload["injection_id"] is None

    def test_explicit_injection_id_bypasses_lookup(self, store):
        """Passing injection_id=99 stores it directly without querying the DB."""
        eid = store.record_skill_used("sk-c", "sess-4", injection_id=99)
        assert eid is not None

        rows = store.get_events(session_id="sess-4", kind="skill.used")
        payload = json.loads(rows[0]["payload"])
        assert payload["injection_id"] == 99
        assert payload["matched"] is True
