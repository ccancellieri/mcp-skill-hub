"""M1 — claims-board: claim / handoff / steal / release on tasks (no LLM).

Validates the SQLite-only ownership transitions that let multiple Claude
Code sessions (or future swarm-lite subprocesses) coordinate work-item
ownership without an LLM.

Covers schema, store-level atomic transitions, and the MCP tool wrappers.
"""
from __future__ import annotations

import sys
import time
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


# ─────────────────────── schema migration ────────────────────────────────────


def test_claims_columns_present(store):
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(tasks)")}
    for col in ("claimed_by", "claim_token", "claimed_at", "stealable_at"):
        assert col in cols, f"{col!r} missing from tasks: {cols}"


def test_claimed_by_index_present(store):
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND tbl_name='tasks'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "idx_tasks_claimed_by" in names, names


def test_save_task_starts_unclaimed(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    claim = store.get_task_claim(tid)
    assert claim == {
        "claimed_by":   None,
        "claim_token":  None,
        "claimed_at":   None,
        "stealable_at": None,
    }


# ─────────────────────── claim_task ──────────────────────────────────────────


def test_claim_task_sets_owner_and_returns_token(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    token = store.claim_task(tid, "agent-A")
    assert token and isinstance(token, str)
    claim = store.get_task_claim(tid)
    assert claim["claimed_by"] == "agent-A"
    assert claim["claim_token"] == token
    assert claim["claimed_at"] is not None
    assert claim["stealable_at"] is None  # non-stealable by default


def test_double_claim_rejected(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    assert store.claim_task(tid, "agent-A")
    # Second claim by a different agent must fail (returns None).
    assert store.claim_task(tid, "agent-B") is None
    claim = store.get_task_claim(tid)
    assert claim["claimed_by"] == "agent-A"


def test_claim_requires_agent_id(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    assert store.claim_task(tid, "") is None


def test_claim_closed_task_rejected(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.close_task(tid, compact="done")
    assert store.claim_task(tid, "agent-A") is None


def test_claim_with_stealable_window(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    token = store.claim_task(tid, "agent-A", stealable_after_sec=60)
    assert token
    claim = store.get_task_claim(tid)
    assert claim["stealable_at"] is not None


# ─────────────────────── handoff_task ────────────────────────────────────────


def test_handoff_transfers_ownership_and_rotates_token(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    t1 = store.claim_task(tid, "agent-A")
    t2 = store.handoff_task(tid, "agent-B", from_agent="agent-A")
    assert t2 and t2 != t1
    claim = store.get_task_claim(tid)
    assert claim["claimed_by"] == "agent-B"
    assert claim["claim_token"] == t2


def test_handoff_rejects_wrong_from_agent(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A")
    assert store.handoff_task(tid, "agent-C", from_agent="agent-B") is None
    assert store.get_task_claim(tid)["claimed_by"] == "agent-A"


def test_handoff_without_from_agent_works(store):
    """Admin-style handoff (no from_agent) succeeds regardless of holder."""
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A")
    token = store.handoff_task(tid, "agent-B")
    assert token
    assert store.get_task_claim(tid)["claimed_by"] == "agent-B"


def test_handoff_unclaimed_rejected(store):
    """Cannot hand off a task that nobody owns — use claim_task instead."""
    tid = store.save_task(title="t1", summary="s1", vector=[])
    assert store.handoff_task(tid, "agent-B") is None


# ─────────────────────── steal_task ──────────────────────────────────────────


def test_steal_before_window_rejected(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    # Long window so steal must fail.
    store.claim_task(tid, "agent-A", stealable_after_sec=3600)
    assert store.steal_task(tid, "agent-B") is None
    assert store.get_task_claim(tid)["claimed_by"] == "agent-A"


def test_steal_unclaimed_rejected(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    assert store.steal_task(tid, "agent-B") is None


def test_steal_non_stealable_rejected(store):
    """Claims without a stealable_at can never be stolen."""
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A")  # stealable_after_sec=None
    assert store.steal_task(tid, "agent-B") is None
    assert store.get_task_claim(tid)["claimed_by"] == "agent-A"


def test_steal_after_window_succeeds(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    # Backdate stealable_at directly so we don't need to sleep in tests.
    store.claim_task(tid, "agent-A", stealable_after_sec=10)
    store._conn.execute(
        "UPDATE tasks SET stealable_at = datetime('now', '-1 seconds') "
        "WHERE id = ?",
        (tid,),
    )
    store._conn.commit()
    token = store.steal_task(tid, "agent-B")
    assert token
    claim = store.get_task_claim(tid)
    assert claim["claimed_by"] == "agent-B"
    # New owner inherits a fresh (non-stealable by default) window.
    assert claim["stealable_at"] is None


# ─────────────────────── release_task ────────────────────────────────────────


def test_release_clears_all_claim_fields(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A", stealable_after_sec=60)
    assert store.release_task(tid, agent_id="agent-A") is True
    claim = store.get_task_claim(tid)
    assert claim == {
        "claimed_by":   None,
        "claim_token":  None,
        "claimed_at":   None,
        "stealable_at": None,
    }


def test_release_wrong_agent_rejected(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A")
    assert store.release_task(tid, agent_id="agent-B") is False
    assert store.get_task_claim(tid)["claimed_by"] == "agent-A"


def test_release_already_unclaimed(store):
    tid = store.save_task(title="t1", summary="s1", vector=[])
    assert store.release_task(tid) is False


def test_release_admin_force(store):
    """Empty agent_id allows force-release regardless of holder."""
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A")
    assert store.release_task(tid) is True
    assert store.get_task_claim(tid)["claimed_by"] is None


def test_release_after_steal_uses_new_owner(store):
    """Token rotation: after steal, the original owner can no longer release."""
    tid = store.save_task(title="t1", summary="s1", vector=[])
    store.claim_task(tid, "agent-A", stealable_after_sec=10)
    store._conn.execute(
        "UPDATE tasks SET stealable_at = datetime('now', '-1 seconds') "
        "WHERE id = ?",
        (tid,),
    )
    store._conn.commit()
    store.steal_task(tid, "agent-B")
    # agent-A no longer holds the claim → release must fail.
    assert store.release_task(tid, agent_id="agent-A") is False
    assert store.release_task(tid, agent_id="agent-B") is True


# ─────────────────────── single-session flow unaffected ──────────────────────


def test_existing_task_flow_unchanged_when_unclaimed(store):
    """Acceptance criterion: when claimed_by IS NULL, nothing else changes."""
    tid = store.save_task(title="single", summary="hello", vector=[],
                          repo="my-repo")
    # All existing listings + lookups still work and ignore claim fields.
    rows = store.list_tasks(status="open")
    assert any(r["id"] == tid for r in rows)
    rows_repo = store.list_tasks(status="open", repo="my-repo")
    assert any(r["id"] == tid for r in rows_repo)
    assert store.update_task(tid, summary="updated") is True
    assert store.close_task(tid, compact="done") is True
    assert store.reopen_task(tid) is True


# ─────────────────────── MCP tool wrappers ───────────────────────────────────


def test_mcp_claim_task_happy_path(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    out = srv.claim_task(tid, "agent-A")
    assert "claimed by agent-A" in out
    assert store.get_task_claim(tid)["claimed_by"] == "agent-A"


def test_mcp_claim_task_double_claim_rejected(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A")
    out = srv.claim_task(tid, "agent-B")
    assert "already claimed" in out
    assert "agent-A" in out


def test_mcp_claim_task_missing(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    out = srv.claim_task(99999, "agent-A")
    assert "not found" in out


def test_mcp_claim_requires_agent_id(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    tid = store.save_task(title="t1", summary="s1", vector=[])
    out = srv.claim_task(tid, "")
    assert "agent_id required" in out


def test_mcp_handoff_task(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A")
    out = srv.handoff_task(tid, "agent-B", from_agent="agent-A")
    assert "handed off to agent-B" in out
    assert store.get_task_claim(tid)["claimed_by"] == "agent-B"


def test_mcp_handoff_wrong_from(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A")
    out = srv.handoff_task(tid, "agent-C", from_agent="agent-B")
    assert "rejected" in out
    assert store.get_task_claim(tid)["claimed_by"] == "agent-A"


def test_mcp_handoff_unclaimed(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    out = srv.handoff_task(tid, "agent-B")
    assert "unclaimed" in out


def test_mcp_steal_before_window(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A", stealable_after_sec=3600)
    out = srv.steal_task(tid, "agent-B")
    assert "not yet stealable" in out


def test_mcp_steal_non_stealable_claim(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A")  # no stealable window
    out = srv.steal_task(tid, "agent-B")
    assert "non-stealable" in out


def test_mcp_steal_after_window(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A", stealable_after_sec=10)
    # Backdate so steal is allowed.
    store._conn.execute(
        "UPDATE tasks SET stealable_at = datetime('now', '-1 seconds') "
        "WHERE id = ?",
        (tid,),
    )
    store._conn.commit()

    out = srv.steal_task(tid, "agent-B")
    assert "stolen by agent-B" in out
    assert store.get_task_claim(tid)["claimed_by"] == "agent-B"


def test_mcp_release_task(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A")
    out = srv.release_task(tid, agent_id="agent-A")
    assert "released" in out
    assert store.get_task_claim(tid)["claimed_by"] is None


def test_mcp_release_wrong_agent(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    srv.claim_task(tid, "agent-A")
    out = srv.release_task(tid, agent_id="agent-B")
    assert "rejected" in out


def test_mcp_release_already_unclaimed(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    tid = store.save_task(title="t1", summary="s1", vector=[])
    out = srv.release_task(tid)
    assert "already unclaimed" in out


# ─────────────────────── capabilities registry ───────────────────────────────


def test_capabilities_registry_includes_claim_tools():
    from skill_hub.capabilities import TOOLS
    names = {t.name for t in TOOLS}
    for tool in ("claim_task", "handoff_task", "steal_task", "release_task"):
        assert tool in names, f"{tool} missing from TOOLS"
