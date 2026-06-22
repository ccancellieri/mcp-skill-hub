"""Tests for M2 W1 — event log.

Covers issue #27 acceptance criteria:
- append_event + get_events round-trip (session_id, kind, since, limit filters).
- events_prune(dry_run=True) reports candidates, deletes nothing.
- Closed session older than cutoff is coalesced to session_snapshot; re-run is
  a no-op.
- Open session (no session_end) is never pruned regardless of age.
- retention_days=0 → prune is a no-op.
- The envelope emit hook populates events_emitted on ToolResult.
- Envelope emit failures are non-fatal.
- session_start / session_end kinds round-trip.

Constraints:
- Uses only skill_hub.store and skill_hub.envelope — never skill_hub.server.
- Every test uses a fresh tmp_path DB, never DB_PATH.
- No embedding model loaded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Guard: ensure this module's imports never drag in skill_hub.server (which
# builds a module-level SkillStore against the live DB).  The fixture checks
# against the sys.modules snapshot taken before any test ran, so it is immune
# to other test files that legitimately import the server later in the suite.
def test_server_not_imported(assert_server_not_imported):  # noqa: PT019
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    from skill_hub.store import SkillStore
    return SkillStore(db_path=tmp_path / "test_events.db")


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_events_table_exists(store):
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchone()
    assert row is not None


def test_events_table_has_required_columns(store):
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(events)")}
    for col in ("id", "session_id", "ts", "kind", "tool_name", "payload", "node_id"):
        assert col in cols, f"events.{col} missing"


# ---------------------------------------------------------------------------
# append_event + get_events round-trip
# ---------------------------------------------------------------------------


def test_append_and_retrieve_by_session_id(store):
    eid = store.append_event("sess-a", "tool_invoke", {"q": "hello"}, tool_name="my_tool")
    assert eid is not None and eid > 0
    rows = store.get_events(session_id="sess-a")
    assert len(rows) == 1
    assert rows[0]["kind"] == "tool_invoke"
    assert rows[0]["tool_name"] == "my_tool"
    assert "hello" in rows[0]["payload"]


def test_session_id_filter_isolates_sessions(store):
    store.append_event("sess-1", "tool_invoke", {"x": 1})
    store.append_event("sess-2", "tool_invoke", {"x": 2})
    rows_1 = store.get_events(session_id="sess-1")
    rows_2 = store.get_events(session_id="sess-2")
    assert len(rows_1) == 1 and rows_1[0]["session_id"] == "sess-1"
    assert len(rows_2) == 1 and rows_2[0]["session_id"] == "sess-2"


def test_get_events_empty_session_id_returns_all(store):
    store.append_event("sess-a", "tool_invoke", {})
    store.append_event("sess-b", "session_start", {})
    rows = store.get_events(session_id="")
    assert len(rows) == 2


def test_kind_filter(store):
    store.append_event("sess-x", "tool_invoke", {})
    store.append_event("sess-x", "tool_result", {})
    rows = store.get_events(session_id="sess-x", kind="tool_result")
    assert len(rows) == 1
    assert rows[0]["kind"] == "tool_result"


def test_since_filter(store):
    t0 = time.time()
    store.append_event("sess-s", "tool_invoke", {}, ts=t0 - 100)
    store.append_event("sess-s", "tool_result", {}, ts=t0 - 50)
    store.append_event("sess-s", "session_end", {}, ts=t0)
    # Since 60 seconds ago — should exclude the oldest row.
    rows = store.get_events(session_id="sess-s", since=t0 - 75)
    assert len(rows) == 2
    for r in rows:
        assert r["ts"] >= t0 - 75


def test_limit_filter(store):
    for i in range(10):
        store.append_event("sess-lim", "tool_invoke", {"i": i})
    rows = store.get_events(session_id="sess-lim", limit=3)
    assert len(rows) == 3


def test_results_ordered_by_ts_asc(store):
    t0 = time.time()
    store.append_event("sess-ord", "tool_invoke", {}, ts=t0 + 2)
    store.append_event("sess-ord", "tool_result", {}, ts=t0 + 1)
    store.append_event("sess-ord", "session_start", {}, ts=t0)
    rows = store.get_events(session_id="sess-ord")
    ts_list = [r["ts"] for r in rows]
    assert ts_list == sorted(ts_list)


def test_payload_json_encoded(store):
    store.append_event("sess-p", "config_change", {"key": "val", "num": 42})
    rows = store.get_events(session_id="sess-p")
    parsed = json.loads(rows[0]["payload"])
    assert parsed == {"key": "val", "num": 42}


def test_append_event_returns_none_on_error(store):
    """Close the connection and confirm append_event is non-fatal."""
    store._conn.close()
    result = store.append_event("sess-err", "tool_invoke", {})
    assert result is None


def test_source_parameter_overrides_node_id(store):
    store.append_event("sess-src", "tool_invoke", {}, source="custom-node")
    rows = store.get_events(session_id="sess-src")
    assert rows[0]["node_id"] == "custom-node"


def test_session_start_and_end_kinds_round_trip(store):
    store.append_event("sess-lifecycle", "session_start", {"source": "test"})
    store.append_event("sess-lifecycle", "session_end", {"summary": "done"})
    rows = store.get_events(session_id="sess-lifecycle")
    kinds = [r["kind"] for r in rows]
    assert "session_start" in kinds
    assert "session_end" in kinds


# ---------------------------------------------------------------------------
# events_prune — dry_run
# ---------------------------------------------------------------------------


def test_events_prune_dry_run_reports_candidates_no_delete(store):
    t_old = time.time() - 40 * 86400  # 40 days ago
    store.append_event("sess-old", "session_start", {}, ts=t_old - 100)
    store.append_event("sess-old", "tool_invoke", {}, ts=t_old - 50)
    store.append_event("sess-old", "session_end", {}, ts=t_old)

    before_count = store._conn.execute("SELECT count(*) FROM events").fetchone()[0]
    result = store.events_prune(dry_run=True)

    after_count = store._conn.execute("SELECT count(*) FROM events").fetchone()[0]
    assert after_count == before_count, "dry_run must not delete rows"
    assert result["candidates"] >= 1
    assert result["dry_run"] is True
    assert result["snapshots_written"] == 0


# ---------------------------------------------------------------------------
# events_prune — closed session older than cutoff
# ---------------------------------------------------------------------------


def test_prune_closed_old_session_coalesces_to_snapshot(store):
    t_old = time.time() - 40 * 86400
    store.append_event("sess-close", "session_start", {}, ts=t_old - 200)
    store.append_event("sess-close", "tool_invoke", {}, ts=t_old - 100)
    store.append_event("sess-close", "tool_result", {}, ts=t_old - 50)
    store.append_event("sess-close", "session_end", {}, ts=t_old)

    result = store.events_prune(dry_run=False)

    assert result["candidates"] >= 1
    assert result["rows_deleted"] >= 4
    assert result["snapshots_written"] >= 1

    remaining = store.get_events(session_id="sess-close")
    assert len(remaining) == 1
    assert remaining[0]["kind"] == "session_snapshot"
    payload = json.loads(remaining[0]["payload"])
    assert "kind_counts" in payload
    assert payload["raw_row_count"] == 4


def test_prune_reruns_are_noop_for_already_coalesced_session(store):
    t_old = time.time() - 40 * 86400
    store.append_event("sess-noop", "session_start", {}, ts=t_old - 100)
    store.append_event("sess-noop", "session_end", {}, ts=t_old)

    result1 = store.events_prune(dry_run=False)
    assert result1["candidates"] >= 1

    # Second run: session only has session_snapshot — must be a no-op.
    result2 = store.events_prune(dry_run=False)
    remaining = store.get_events(session_id="sess-noop")
    assert len(remaining) == 1
    assert remaining[0]["kind"] == "session_snapshot"
    # candidates should be 0 now (snapshot-only sessions are skipped)
    assert result2["candidates"] == 0


# ---------------------------------------------------------------------------
# events_prune — open session never pruned
# ---------------------------------------------------------------------------


def test_prune_leaves_open_session_untouched(store):
    t_old = time.time() - 40 * 86400
    # Open session: has session_start but NO session_end.
    store.append_event("sess-open", "session_start", {}, ts=t_old - 100)
    store.append_event("sess-open", "tool_invoke", {}, ts=t_old - 50)
    # No session_end

    result = store.events_prune(dry_run=False)

    remaining = store.get_events(session_id="sess-open")
    assert len(remaining) == 2, "open session must not be pruned"
    assert result["candidates"] == 0


def test_prune_recent_closed_session_not_pruned(store):
    """A closed session whose events are within retention window stays intact."""
    t_recent = time.time() - 5 * 86400  # 5 days ago, within default 30-day window
    store.append_event("sess-recent", "session_start", {}, ts=t_recent - 100)
    store.append_event("sess-recent", "session_end", {}, ts=t_recent)

    store.events_prune(dry_run=False)

    remaining = store.get_events(session_id="sess-recent")
    assert len(remaining) == 2, "recent closed session must not be pruned"


# ---------------------------------------------------------------------------
# events_prune — retention_days=0 is a no-op
# ---------------------------------------------------------------------------


def test_prune_retention_days_zero_is_noop(store):
    t_old = time.time() - 100 * 86400
    store.append_event("sess-z", "session_start", {}, ts=t_old - 10)
    store.append_event("sess-z", "session_end", {}, ts=t_old)

    result = store.events_prune(dry_run=False, retention_days=0)

    assert result["candidates"] == 0
    assert result["rows_deleted"] == 0
    remaining = store.get_events(session_id="sess-z")
    assert len(remaining) == 2


# ---------------------------------------------------------------------------
# Envelope emit hook wiring
# ---------------------------------------------------------------------------


def test_emit_hook_populates_events_emitted(tmp_path):
    """set_emit_hook causes events_emitted to be non-empty on ToolResult."""
    from skill_hub.store import SkillStore
    from skill_hub.envelope import tool_envelope, set_emit_hook

    db = tmp_path / "emit_hook.db"
    s = SkillStore(db_path=db)
    emitted: list[tuple] = []

    def _hook(kind: str, tool_name, payload: dict):
        emitted.append((kind, tool_name, payload))
        return s.append_event("test-session", kind, payload, tool_name=tool_name)

    set_emit_hook(_hook)

    @tool_envelope
    def sample_tool(x: int) -> str:
        return f"got {x}"

    result = sample_tool.envelope(42)

    assert len(result.events_emitted) == 2  # tool_invoke + tool_result
    assert emitted[0][0] == "tool_invoke"
    assert emitted[1][0] == "tool_result"
    assert emitted[1][2]["ok"] is True

    # Verify rows landed in the DB.
    rows = s.get_events(session_id="test-session")
    assert len(rows) == 2

    # Restore no-op hook so other tests aren't affected.
    from skill_hub.envelope import _noop_emit
    set_emit_hook(_noop_emit)  # type: ignore[arg-type]


def test_emit_hook_failure_is_nonfatal(tmp_path):
    """A raising emit hook must not break the tool call."""
    from skill_hub.envelope import tool_envelope, set_emit_hook, _noop_emit

    def _bad_hook(kind, tool_name, payload):
        raise RuntimeError("hook exploded")

    set_emit_hook(_bad_hook)  # type: ignore[arg-type]

    @tool_envelope
    def safe_tool() -> str:
        return "ok"

    result = safe_tool.envelope()
    assert result.stdout == "ok"
    assert result.error is None

    set_emit_hook(_noop_emit)  # type: ignore[arg-type]


def test_envelope_events_emitted_empty_with_noop_hook():
    """Default no-op hook: events_emitted is empty (pre-W1 contract preserved)."""
    from skill_hub.envelope import tool_envelope, set_emit_hook, _noop_emit

    set_emit_hook(_noop_emit)  # type: ignore[arg-type]

    @tool_envelope
    def t() -> str:
        return "x"

    result = t.envelope()
    assert result.events_emitted == []


def test_emit_hook_error_path_also_emits_tool_result(tmp_path):
    """Even when the tool raises, tool_result with ok=False is emitted."""
    from skill_hub.store import SkillStore
    from skill_hub.envelope import tool_envelope, set_emit_hook, _noop_emit

    s = SkillStore(db_path=tmp_path / "err.db")
    emitted_kinds: list[str] = []

    def _hook(kind, tool_name, payload):
        emitted_kinds.append(kind)
        return s.append_event("err-session", kind, payload, tool_name=tool_name)

    set_emit_hook(_hook)  # type: ignore[arg-type]

    @tool_envelope
    def broken_tool() -> str:
        raise ValueError("oops")

    result = broken_tool.envelope()
    assert result.error is not None
    assert "tool_invoke" in emitted_kinds
    assert "tool_result" in emitted_kinds
    # Find the tool_result event and check ok=False.
    rows = s.get_events(session_id="err-session", kind="tool_result")
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload"])
    assert payload["ok"] is False
    assert "oops" in payload["error"]

    set_emit_hook(_noop_emit)  # type: ignore[arg-type]
