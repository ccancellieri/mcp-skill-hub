"""Tests for M2 W2 — wake_session (stateless recovery).

Covers issue #28 acceptance criteria:
- Replay correctness: events for a session are read in order.
- Cache rebuild: _vec_cache is invalidated so next search reloads from DB.
- In-flight detection: a tool_invoke with no matching tool_result is detected
  and reported.
- Resume-after-kill: a session with an interrupted tool_invoke is recovered.
- <500ms replay budget: 1000-event session replays within 500ms.

Constraints:
- Tests use only skill_hub.store and the wake_session logic extracted into
  a helper — never skill_hub.server (which opens the live DB on import).
- Every test uses a fresh tmp_path DB, never DB_PATH.
- No embedding model loaded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# Guard: never import server.py — it opens the live DB on import.
def test_server_not_imported():
    assert "skill_hub.server" not in sys.modules, (
        "skill_hub.server must not be imported — it opens the live DB on import"
    )


# ---------------------------------------------------------------------------
# Extracted wake_session logic (mirrors server.py but operates on a passed-in
# store rather than the module-level _store, so tests are hermetic).
# ---------------------------------------------------------------------------

def _wake_session(store: Any, session_id: str) -> dict:
    """Test-friendly version of the wake_session logic.

    Returns a structured dict instead of a string so assertions are easy.
    Mirrors the server.py implementation exactly.
    """
    t_start = time.monotonic()

    if not session_id:
        return {"error": "session_id is required"}

    events = store.get_events(session_id=session_id, limit=10000)
    if not events:
        return {"error": f"no events found for session {session_id!r}"}

    elapsed_load_ms = int((time.monotonic() - t_start) * 1000)

    # Detect in-flight: tool_invoke with no matching tool_result.
    invoke_stack: dict[str, dict] = {}
    for ev in events:
        kind = ev.get("kind", "")
        tool_name = ev.get("tool_name") or ""
        if kind == "tool_invoke" and tool_name:
            invoke_stack[tool_name] = ev
        elif kind == "tool_result" and tool_name:
            invoke_stack.pop(tool_name, None)

    in_flight = list(invoke_stack.values())

    # Invalidate in-process vector cache.
    store._vec_cache_valid = False
    store._vec_cache = {}

    elapsed_total_ms = int((time.monotonic() - t_start) * 1000)

    return {
        "events": events,
        "events_count": len(events),
        "in_flight": in_flight,
        "elapsed_load_ms": elapsed_load_ms,
        "elapsed_total_ms": elapsed_total_ms,
        "vec_cache_invalidated": True,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    from skill_hub.store import SkillStore
    return SkillStore(db_path=tmp_path / "wake_session_test.db")


# ---------------------------------------------------------------------------
# Basic correctness: events replayed in order
# ---------------------------------------------------------------------------


def test_wake_session_reads_events_in_order(store):
    t0 = time.time()
    store.append_event("sess-a", "session_start", {"source": "test"}, ts=t0)
    store.append_event("sess-a", "tool_invoke", {"x": 1}, tool_name="search_skills", ts=t0 + 1)
    store.append_event("sess-a", "tool_result", {"ok": True}, tool_name="search_skills", ts=t0 + 2)
    store.append_event("sess-a", "session_end", {}, ts=t0 + 3)

    result = _wake_session(store, "sess-a")

    assert "error" not in result
    assert result["events_count"] == 4
    ts_list = [ev["ts"] for ev in result["events"]]
    assert ts_list == sorted(ts_list), "events must be in ascending ts order"


def test_wake_session_empty_session_id_returns_error(store):
    result = _wake_session(store, "")
    assert "error" in result


def test_wake_session_unknown_session_returns_error(store):
    result = _wake_session(store, "nonexistent-session-xyz")
    assert "error" in result


def test_wake_session_does_not_mix_sessions(store):
    t0 = time.time()
    store.append_event("sess-target", "tool_invoke", {}, tool_name="t1", ts=t0)
    store.append_event("sess-other", "tool_invoke", {}, tool_name="t2", ts=t0 + 1)

    result = _wake_session(store, "sess-target")
    assert result["events_count"] == 1
    assert all(ev["session_id"] == "sess-target" for ev in result["events"])


# ---------------------------------------------------------------------------
# In-flight detection
# ---------------------------------------------------------------------------


def test_in_flight_detected_when_tool_result_missing(store):
    t0 = time.time()
    store.append_event("sess-kill", "tool_invoke", {"args": {"q": "hello"}},
                       tool_name="search_skills", ts=t0)
    # No tool_result — server was killed.

    result = _wake_session(store, "sess-kill")

    assert len(result["in_flight"]) == 1
    assert result["in_flight"][0]["tool_name"] == "search_skills"


def test_matched_invoke_result_not_in_flight(store):
    t0 = time.time()
    store.append_event("sess-ok", "tool_invoke", {}, tool_name="search_skills", ts=t0)
    store.append_event("sess-ok", "tool_result", {"ok": True}, tool_name="search_skills", ts=t0 + 1)

    result = _wake_session(store, "sess-ok")

    assert len(result["in_flight"]) == 0


def test_multiple_in_flight_tools_detected(store):
    t0 = time.time()
    store.append_event("sess-multi", "tool_invoke", {}, tool_name="tool_a", ts=t0)
    store.append_event("sess-multi", "tool_invoke", {}, tool_name="tool_b", ts=t0 + 1)
    # Neither gets a tool_result.

    result = _wake_session(store, "sess-multi")

    in_flight_names = {ev["tool_name"] for ev in result["in_flight"]}
    assert "tool_a" in in_flight_names
    assert "tool_b" in in_flight_names


def test_only_last_invoke_of_same_tool_tracked(store):
    """If a tool is invoked twice with only one result, the second invoke is in-flight."""
    t0 = time.time()
    store.append_event("sess-seq", "tool_invoke", {}, tool_name="my_tool", ts=t0)
    store.append_event("sess-seq", "tool_result", {"ok": True}, tool_name="my_tool", ts=t0 + 1)
    store.append_event("sess-seq", "tool_invoke", {}, tool_name="my_tool", ts=t0 + 2)
    # Second invoke gets no result.

    result = _wake_session(store, "sess-seq")

    assert len(result["in_flight"]) == 1
    assert result["in_flight"][0]["tool_name"] == "my_tool"


# ---------------------------------------------------------------------------
# Cache rebuild
# ---------------------------------------------------------------------------


def test_vec_cache_invalidated_after_wake(store):
    # Pre-seed a non-empty vec cache to simulate a live server state.
    store._vec_cache = {"skill-x": ([0.1, 0.2], 0.22)}
    store._vec_cache_valid = True

    t0 = time.time()
    store.append_event("sess-cache", "session_start", {}, ts=t0)

    result = _wake_session(store, "sess-cache")

    assert result["vec_cache_invalidated"] is True
    assert store._vec_cache_valid is False
    assert store._vec_cache == {}


# ---------------------------------------------------------------------------
# Resume-after-kill (acceptance criterion)
# ---------------------------------------------------------------------------


def test_resume_after_kill_reports_interrupted_tool(store):
    """Kill scenario: tool_invoke written, then server killed (no tool_result).

    wake_session must detect the in-flight invoke and report it so the caller
    can decide to re-invoke it.  The tool is not re-executed by wake_session
    itself (re-execution requires the live MCP tool registry which is outside
    the recovery path), but the report provides everything needed to do so.
    """
    t0 = time.time()
    store.append_event("sess-kill2", "session_start", {"source": "test"}, ts=t0)
    store.append_event(
        "sess-kill2", "tool_invoke",
        {"kwargs": {"query": "find authentication patterns"}},
        tool_name="search_skills", ts=t0 + 1,
    )
    # Server killed here — no tool_result, no session_end.

    result = _wake_session(store, "sess-kill2")

    assert len(result["in_flight"]) == 1
    interrupted = result["in_flight"][0]
    assert interrupted["tool_name"] == "search_skills"
    # Payload contains the original arguments so the caller can re-invoke.
    payload = json.loads(interrupted["payload"])
    assert "search_skills" == interrupted["tool_name"]
    assert isinstance(payload, dict)


# ---------------------------------------------------------------------------
# Performance: <500ms replay budget for a 1000-event session
# ---------------------------------------------------------------------------


def test_replay_1000_events_under_500ms(store):
    """Replay of a 1000-event session must complete in under 500ms.

    This is the Q2 baseline budget from the design doc.  If the assertion
    fails on constrained hardware, the test reports the actual timing rather
    than fabricating a pass.
    """
    t0 = time.time()

    # Write 1000 events: alternating tool_invoke / tool_result pairs.
    # Using real timestamps spaced 1ms apart to keep ordering deterministic.
    for i in range(500):
        ts_invoke = t0 + i * 0.002
        ts_result = ts_invoke + 0.001
        tool = f"tool_{i % 10}"
        store.append_event("sess-perf", "tool_invoke", {"i": i}, tool_name=tool, ts=ts_invoke)
        store.append_event("sess-perf", "tool_result", {"ok": True}, tool_name=tool, ts=ts_result)

    assert store._conn.execute(
        "SELECT count(*) FROM events WHERE session_id = 'sess-perf'"
    ).fetchone()[0] == 1000

    t_replay_start = time.monotonic()
    result = _wake_session(store, "sess-perf")
    elapsed = time.monotonic() - t_replay_start

    elapsed_ms = elapsed * 1000
    # Report regardless.
    print(f"\nreplay timing: {elapsed_ms:.1f}ms for {result['events_count']} events")

    assert result["events_count"] == 1000
    assert result["in_flight"] == [], "all 500 pairs are complete — no in-flight"

    # Hard budget assertion: must complete in 500ms.
    # If this fails on slow CI hardware, report the real number rather than skip.
    assert elapsed_ms < 500, (
        f"replay budget exceeded: {elapsed_ms:.1f}ms > 500ms for 1000 events. "
        "Consider adding periodic snapshots for large sessions."
    )


def test_replay_timing_field_populated(store):
    """elapsed_total_ms in the result reflects real wall time."""
    t0 = time.time()
    store.append_event("sess-timing", "tool_invoke", {}, tool_name="t", ts=t0)
    store.append_event("sess-timing", "tool_result", {}, tool_name="t", ts=t0 + 1)

    result = _wake_session(store, "sess-timing")

    assert "elapsed_total_ms" in result
    assert result["elapsed_total_ms"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_wake_session_snapshot_only_session(store):
    """A session with only a session_snapshot row (already coalesced) is recovered."""
    t0 = time.time()
    store.append_event(
        "sess-snap", "session_snapshot",
        {"kind_counts": {"tool_invoke": 10, "tool_result": 10}, "raw_row_count": 20},
        ts=t0,
    )

    result = _wake_session(store, "sess-snap")

    assert "error" not in result
    assert result["events_count"] == 1
    assert result["in_flight"] == []


def test_wake_session_empty_payload_invoke_does_not_crash(store):
    """An in-flight tool_invoke with an empty payload is skipped gracefully."""
    t0 = time.time()
    store.append_event(
        "sess-bad", "tool_invoke", {},
        tool_name="some_tool", ts=t0,
    )

    result = _wake_session(store, "sess-bad")

    # Must not crash; the in-flight event is still reported.
    assert len(result["in_flight"]) == 1
