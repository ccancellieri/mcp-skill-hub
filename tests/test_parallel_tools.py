"""Tests for MCP tool parallel-safety.

FastMCP runs on an asyncio event loop (single-threaded), so "parallel" tool
calls are sequential coroutines that share one SkillStore connection — no
actual OS threading. The tests here verify the *logical* safety properties
that matter for parallel tool use:

  - save_task      — repeated calls produce distinct rows, no silent merge
  - update_task    — sequential updates on the same task preserve final state
  - close_task     — idempotent: a second close on an already-closed task is
                     a no-op (not an error and not an overwrite)
  - record_model_reward — repeated reward upserts accumulate, not overwrite
  - get/rebuild_session_memory — pure reads or file writes, always safe

The bandit upsert is also verified to tolerate rapid-fire calls correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures

@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from skill_hub.store import SkillStore

    db = SkillStore(db_path=tmp_path / "test.db")
    yield db
    db.close()


# ---------------------------------------------------------------------------
# save_task: repeated inserts must never silently merge

def test_repeated_save_task_produces_distinct_ids(store):
    """N sequential saves (simulating rapid async calls) → N distinct task IDs."""
    ids = [
        store.save_task(title=f"Task {i}", summary=f"Summary {i}", vector=[float(i)] * 8)
        for i in range(5)
    ]
    assert len(ids) == 5
    assert len(set(ids)) == 5, f"duplicate IDs: {ids}"
    for tid in ids:
        assert store.get_task(tid) is not None


def test_repeated_save_same_title_produces_separate_rows(store):
    """Saving the same title N times creates N independent rows."""
    ids = [
        store.save_task(title="Dup", summary="Same", vector=[0.1] * 8)
        for _ in range(3)
    ]
    assert len(set(ids)) == 3, "identical saves should produce separate rows"


# ---------------------------------------------------------------------------
# update_task: final state is always the last write

def test_sequential_updates_preserve_final_state(store):
    tid = store.save_task(title="Base", summary="original", vector=[0.0] * 8)

    for i in range(4):
        ok = store.update_task(tid, summary=f"updated-{i}", vector=[float(i)] * 8)
        assert ok, f"update {i} failed"

    task = store.get_task(tid)
    assert task is not None
    assert task["summary"] == "updated-3"


# ---------------------------------------------------------------------------
# close_task: idempotent

def test_close_task_idempotent(store):
    """Closing a task twice: first call succeeds, second is a no-op."""
    tid = store.save_task(title="CloseMe", summary="to close", vector=[0.0] * 8)

    first = store.close_task(tid, compact="done", compact_vector=[0.0] * 8)
    assert first is True, "first close should return True"

    second = store.close_task(tid, compact="overwrite?", compact_vector=[1.0] * 8)
    assert second is False, "second close should return False (already closed)"

    # The compacted text from the FIRST close must survive.
    task = store.get_task(tid)
    assert task["status"] == "closed"
    assert "done" in task["compact"]
    assert "overwrite?" not in task["compact"]


# ---------------------------------------------------------------------------
# record_model_reward: upsert accumulates

def test_reward_upsert_accumulates(store):
    """N sequential reward records accumulate trials/successes correctly."""
    from skill_hub.router import bandit as _bandit

    for _ in range(6):
        _bandit.record_reward(store, "tier_cheap", "trivial", "_none", 1.0)

    rows = _bandit.summary(store)
    row = next(
        (r for r in rows if r["task_class"] == "trivial" and r["tier"] == "tier_cheap"),
        None,
    )
    assert row is not None
    assert row["trials"] == 6
    assert row["successes"] >= 5.5  # Laplace-smoothed but close to 6


def test_reward_mixed_success_rates(store):
    """Alternating success/failure tracks partial rewards correctly."""
    from skill_hub.router import bandit as _bandit

    for i in range(4):
        _bandit.record_reward(store, "tier_mid", "simple", "testing", float(i % 2))

    rows = _bandit.summary(store)
    row = next(
        (r for r in rows if r["task_class"] == "simple" and r["tier"] == "tier_mid"),
        None,
    )
    assert row is not None
    assert row["trials"] == 4
    assert 1.0 <= row["successes"] <= 3.0  # 2 successes + Laplace prior


# ---------------------------------------------------------------------------
# session_memory: write then read is safe under repeated calls

def test_session_memory_write_read_stable(tmp_path, monkeypatch):
    """Multiple writes to the same session file converge to last value."""
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "memory_dir", lambda: tmp_path / "mem")
    monkeypatch.setattr(
        sm, "memory_path",
        lambda sid: tmp_path / "mem" / f"{sid}.md",
    )

    sid = "stable-sess"
    for i in range(3):
        sm.write_memory(sid, f"## User Intent\nVersion {i}.")

    final = sm.read_memory(sid)
    assert "Version 2" in final


# ---------------------------------------------------------------------------
# Parallel-safe documentation: verify docstrings contain the audit note

def test_server_tools_have_parallel_safety_notes():
    """Spot-check that write-path server tools document their parallel semantics."""
    from skill_hub import server

    for fn_name in ("save_task", "close_task", "update_task", "record_model_reward"):
        fn = getattr(server, fn_name, None)
        assert fn is not None, f"{fn_name} not found in server module"
        doc = fn.__doc__ or ""
        assert "parallel" in doc.lower() or "safe" in doc.lower(), (
            f"{fn_name} missing parallel-safety note in docstring"
        )
