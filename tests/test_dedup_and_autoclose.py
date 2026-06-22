"""Tests for issue #38 remainder: stable-key dedup, auto-close, and session re-bind.

Constraints (hard)
------------------
- Uses skill_hub.store, skill_hub.claude_tasks, skill_hub.session_binding only.
  NEVER imports skill_hub.server (it opens the live DB at module level).
- Every test uses a fresh tmp_path DB, NEVER the live DB_PATH.
- No embedding model loaded — vectors are injected as plain float lists.
- All gh subprocess calls are mocked (no real GitHub I/O).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

def test_server_not_imported(assert_server_not_imported):  # noqa: PT019
    """skill_hub.server must not be imported at collection time — it opens the live DB."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    from skill_hub.store import SkillStore
    s = SkillStore(db_path=tmp_path / "dedup_test.db")
    yield s
    s.close()


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """session_binding tests need the marker redirected to tmp_path."""
    from skill_hub import session_binding
    marker = tmp_path / "active_task.json"
    monkeypatch.setattr(session_binding, "ACTIVE_TASK_MARKER", marker)
    from skill_hub.store import SkillStore
    s = SkillStore(db_path=tmp_path / "sb_test.db")
    yield s, marker
    s.close()


# ---------------------------------------------------------------------------
# A) memory_stable_key determinism
# ---------------------------------------------------------------------------

class TestMemoryStableKey:
    def test_same_path_gives_same_key(self):
        from skill_hub.claude_tasks import memory_stable_key
        k1 = memory_stable_key("/home/user/.claude/projects/x/memory/proj.md")
        k2 = memory_stable_key("/home/user/.claude/projects/x/memory/proj.md")
        assert k1 == k2

    def test_different_paths_give_different_keys(self):
        from skill_hub.claude_tasks import memory_stable_key
        k1 = memory_stable_key("/path/to/alpha.md")
        k2 = memory_stable_key("/path/to/beta.md")
        assert k1 != k2

    def test_key_has_mem_prefix(self):
        from skill_hub.claude_tasks import memory_stable_key
        k = memory_stable_key("/some/path.md")
        assert k.startswith("mem:")
        assert len(k) == len("mem:") + 16

    def test_whitespace_stripped(self):
        from skill_hub.claude_tasks import memory_stable_key
        k1 = memory_stable_key("  /path/to/file.md  ")
        k2 = memory_stable_key("/path/to/file.md")
        assert k1 == k2


# ---------------------------------------------------------------------------
# B) MEMORY_DONE_PATTERN
# ---------------------------------------------------------------------------

class TestMemoryDonePattern:
    def test_shipped_matches(self):
        from skill_hub.claude_tasks import MEMORY_DONE_PATTERN
        assert MEMORY_DONE_PATTERN.search("PR #123 SHIPPED — all tests green")

    def test_done_matches(self):
        from skill_hub.claude_tasks import MEMORY_DONE_PATTERN
        assert MEMORY_DONE_PATTERN.search("DONE as of 2026-05-01")

    def test_complete_matches(self):
        from skill_hub.claude_tasks import MEMORY_DONE_PATTERN
        assert MEMORY_DONE_PATTERN.search("COMPLETE")

    def test_partial_does_not_match(self):
        from skill_hub.claude_tasks import MEMORY_DONE_PATTERN
        assert not MEMORY_DONE_PATTERN.search("PARTIAL — wave-2 files remain")

    def test_deferred_does_not_match(self):
        from skill_hub.claude_tasks import MEMORY_DONE_PATTERN
        assert not MEMORY_DONE_PATTERN.search("DEFERRED pending decision")


# ---------------------------------------------------------------------------
# C) project_memory_task — stable-key dedup
# ---------------------------------------------------------------------------

class TestProjectMemoryTaskDedup:
    def test_create_new_task(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/alpha.md")
        result = store.project_memory_task(
            key=key, title="Alpha work", summary="desc", tags="src:memory"
        )
        assert result["action"] == "created"
        assert result["task_id"] is not None

    def test_idempotent_on_same_key(self, store):
        """Same memory-entry key → update, never duplicate."""
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/beta.md")
        r1 = store.project_memory_task(key=key, title="Beta", summary="v1")
        r2 = store.project_memory_task(key=key, title="Beta", summary="v1")
        assert r1["action"] == "created"
        assert r2["action"] == "updated"
        count = store._conn.execute(
            "SELECT count(*) FROM tasks WHERE claude_task_key = ?", (key,)
        ).fetchone()[0]
        assert count == 1

    def test_claude_task_plus_memory_entry_yield_one_task(self, store):
        """A Claude /task (project_claude_task) and a MEMORY.md entry for the
        same work must converge to exactly ONE skill-hub task.

        Scenario: project_claude_task creates a task via the PostToolUse hook.
        Later, _ensure_open_tasks scans MEMORY.md and calls project_memory_task
        with the same title but different key format.  The vector-similarity
        fallback (threshold=0.85) detects the match and re-uses the existing row.
        """
        from skill_hub.claude_tasks import stable_key, memory_stable_key

        # Step 1: Claude tool creates the task (high-dim identical vector).
        vec = [0.6, 0.8, 0.0]
        claude_key = stable_key("Implement DGGS tiling", cwd="/proj", branch="main")
        r1 = store.project_claude_task(
            key=claude_key,
            title="Implement DGGS tiling",
            status="open",
            cwd="/proj",
            branch="main",
        )
        # Manually set the vector so similarity search can find it.
        store.update_task(r1["task_id"], vector=vec)
        assert r1["action"] == "created"
        original_id = r1["task_id"]

        # Step 2: Memory auto-create for the same work, same vector, different key.
        mem_key = memory_stable_key("/mem/dggs.md")
        r2 = store.project_memory_task(
            key=mem_key,
            title="Implement DGGS tiling",
            summary="PARTIAL — H3 done, S2 pending",
            vector=vec,
        )
        # Only one row must exist (similarity matched the existing task).
        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 1, f"Expected 1 task, got {count}"
        assert r2["action"] in ("updated", "created")
        assert r2["task_id"] == original_id

    def test_different_memory_entries_create_distinct_tasks(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key1 = memory_stable_key("/mem/task_a.md")
        key2 = memory_stable_key("/mem/task_b.md")
        store.project_memory_task(key=key1, title="Work A", summary="WIP A")
        store.project_memory_task(key=key2, title="Work B", summary="WIP B")
        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 2

    def test_vector_fallback_adopts_pre_existing_task(self, store):
        """A task without a stable key is adopted when vector similarity >= 0.85."""
        vec = [0.0, 1.0, 0.0]
        tid = store.save_task(
            title="Old task no key",
            summary="old",
            vector=vec,
        )
        # Confirm no claude_task_key yet.
        row = store._conn.execute("SELECT claude_task_key FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["claude_task_key"] is None

        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/old.md")
        result = store.project_memory_task(key=key, title="Old task no key", vector=vec)

        # Must re-use the existing task, not create a second one.
        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 1
        assert result["task_id"] == tid

        # The stable key is now stamped on the existing row.
        row = store._conn.execute("SELECT claude_task_key FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["claude_task_key"] == key

    def test_vector_below_threshold_creates_new_task(self, store):
        """Low-similarity vector must NOT match — a new task is created."""
        vec_existing = [1.0, 0.0, 0.0]
        store.save_task(title="Unrelated task", summary="", vector=vec_existing)

        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/different.md")
        vec_new = [0.0, 0.0, 1.0]  # orthogonal → cosine = 0.0
        result = store.project_memory_task(key=key, title="Different work", vector=vec_new)
        assert result["action"] == "created"
        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# D) Auto-close when memory entry is SHIPPED/DONE
# ---------------------------------------------------------------------------

class TestAutoCloseShippedTasks:
    def test_close_on_done_flag(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/shipped.md")
        r_create = store.project_memory_task(key=key, title="Feature shipped", summary="WIP")
        assert r_create["action"] == "created"
        tid = r_create["task_id"]

        r_close = store.project_memory_task(key=key, title="Feature shipped", close=True)
        assert r_close["action"] == "closed"
        row = store._conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "closed"

    def test_close_noop_when_already_closed(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/already_done.md")
        store.project_memory_task(key=key, title="Done task", summary="")
        store.project_memory_task(key=key, title="Done task", close=True)
        result = store.project_memory_task(key=key, title="Done task", close=True)
        assert result["action"] == "noop"

    def test_close_noop_when_task_missing(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/phantom.md")
        result = store.project_memory_task(key=key, title="", close=True)
        assert result["action"] == "noop"
        assert result["task_id"] is None


# ---------------------------------------------------------------------------
# E) find_open_task_by_stable_key
# ---------------------------------------------------------------------------

class TestFindOpenTaskByStableKey:
    def test_returns_none_for_unknown_key(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        assert store.find_open_task_by_stable_key(memory_stable_key("/no/such.md")) is None

    def test_returns_none_after_task_closed(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/closeable.md")
        store.project_memory_task(key=key, title="Open task", summary="")
        store.project_memory_task(key=key, title="Open task", close=True)
        assert store.find_open_task_by_stable_key(key) is None

    def test_returns_row_for_open_task(self, store):
        from skill_hub.claude_tasks import memory_stable_key
        key = memory_stable_key("/mem/open.md")
        r = store.project_memory_task(key=key, title="Open task", summary="")
        row = store.find_open_task_by_stable_key(key)
        assert row is not None
        assert int(row["id"]) == r["task_id"]


# ---------------------------------------------------------------------------
# F) session_binding stable-key tier
# ---------------------------------------------------------------------------

class TestSessionBindingStableKey:
    def test_stable_key_tier_resumes_without_cwd_match(self, tmp_store, monkeypatch):
        """Same first-120-chars + cwd + branch re-binds the existing task even
        when the cwd/branch tier is bypassed (strategy=semantic) and embeddings
        are unavailable."""
        from skill_hub import session_binding
        store, marker = tmp_store

        # Disable all tiers except stable-key.
        monkeypatch.setattr(session_binding, "_get_config", lambda: {
            "enabled": True, "strategy": "semantic",
            "window_days": 7, "semantic_threshold": 0.75,
        })
        monkeypatch.setattr(session_binding, "_embed_message", lambda _m: [])

        message = "Implement the DGGS H3 tiling pipeline"
        cwd = "/proj/geoid"
        branch = "feat/dggs"

        # First session: creates task + stamps stable key.
        action1, tid1, title1, reason1 = session_binding.bind_session_to_task(
            session_id="sess-A", message=message,
            cwd=cwd, branch=branch, store=store,
        )
        assert action1 == "created"

        # Second session: same message + cwd + branch → stable_key hit.
        action2, tid2, title2, reason2 = session_binding.bind_session_to_task(
            session_id="sess-B", message=message,
            cwd=cwd, branch=branch, store=store,
        )
        assert action2 == "resumed"
        assert tid2 == tid1
        assert reason2 == "stable_key"

    def test_stable_key_tier_fires_before_cwd_branch(self, tmp_store, monkeypatch):
        """When both stable_key and cwd+branch could match, stable_key wins."""
        from skill_hub import session_binding
        store, _marker = tmp_store

        message = "Fix the proxy CORS header bug"
        cwd = "/proj/api"
        branch = "fix/cors"

        # Session A: creates task via stable_key route (strategy=cwd_branch so
        # tier-1 should kick in too, but stable_key fires first on second call).
        monkeypatch.setattr(session_binding, "_get_config", lambda: {
            "enabled": True, "strategy": "hybrid",
            "window_days": 7, "semantic_threshold": 0.75,
        })
        monkeypatch.setattr(session_binding, "_embed_message", lambda _m: [])

        action1, tid1, _, _ = session_binding.bind_session_to_task(
            session_id="sess-A", message=message,
            cwd=cwd, branch=branch, store=store,
        )
        assert action1 == "created"

        action2, tid2, _, reason2 = session_binding.bind_session_to_task(
            session_id="sess-B", message=message,
            cwd=cwd, branch=branch, store=store,
        )
        assert action2 == "resumed"
        assert tid2 == tid1
        assert reason2 == "stable_key"  # not "cwd+branch"

    def test_distinct_messages_create_distinct_tasks(self, tmp_store, monkeypatch):
        """Different first-120-chars → different stable keys → separate tasks."""
        from skill_hub import session_binding
        store, _marker = tmp_store
        monkeypatch.setattr(session_binding, "_get_config", lambda: {
            "enabled": True, "strategy": "semantic",
            "window_days": 7, "semantic_threshold": 0.75,
        })
        monkeypatch.setattr(session_binding, "_embed_message", lambda _m: [])

        action1, tid1, _, _ = session_binding.bind_session_to_task(
            session_id="sid-1", message="Work on feature A",
            cwd="/proj", branch="main", store=store,
        )
        action2, tid2, _, _ = session_binding.bind_session_to_task(
            session_id="sid-2", message="Work on feature B",
            cwd="/proj", branch="main", store=store,
        )
        assert action1 == "created"
        assert action2 == "created"
        assert tid1 != tid2

    def test_stable_key_not_stamped_on_empty_message(self, tmp_store, monkeypatch):
        """Empty / whitespace-only message → no stable key stamped (no crash)."""
        from skill_hub import session_binding
        store, _marker = tmp_store
        monkeypatch.setattr(session_binding, "_get_config", lambda: {
            "enabled": True, "strategy": "cwd_branch",
            "window_days": 7, "semantic_threshold": 0.75,
        })
        action, tid, _, _ = session_binding.bind_session_to_task(
            session_id="sid-empty", message="   ",
            cwd="/proj", branch="main", store=store,
        )
        assert action == "created"
        row = store._conn.execute(
            "SELECT claude_task_key FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert row["claude_task_key"] is None  # no stable key for empty message


# ---------------------------------------------------------------------------
# G) Integration: memory entry lifecycle end-to-end
# ---------------------------------------------------------------------------

class TestMemoryEntryLifecycle:
    def test_memory_entry_becomes_one_task_across_sessions(self, tmp_store, monkeypatch):
        """Deliverable (a): a MEMORY.md entry projected twice → exactly one task."""
        from skill_hub.claude_tasks import memory_stable_key
        store, _marker = tmp_store

        key = memory_stable_key("/mem/lifecycle.md")

        # Session 1: first scan creates the task.
        r1 = store.project_memory_task(
            key=key, title="Lifecycle feature", summary="PARTIAL — in progress"
        )
        assert r1["action"] == "created"

        # Session 2: second scan hits the same key → update, not create.
        r2 = store.project_memory_task(
            key=key, title="Lifecycle feature", summary="PARTIAL — in progress"
        )
        assert r2["action"] == "updated"
        assert r2["task_id"] == r1["task_id"]

        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 1

    def test_shipped_memory_entry_closes_task(self, tmp_store, monkeypatch):
        """Deliverable (b): MEMORY.md entry flipping to SHIPPED closes the task."""
        from skill_hub.claude_tasks import memory_stable_key
        store, _marker = tmp_store

        key = memory_stable_key("/mem/done_feature.md")
        r_create = store.project_memory_task(
            key=key, title="Feature to ship", summary="WIP"
        )
        assert r_create["action"] == "created"
        tid = r_create["task_id"]

        # Memory entry is now SHIPPED → close the task.
        r_close = store.project_memory_task(key=key, title="Feature to ship", close=True)
        assert r_close["action"] == "closed"

        row = store._conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "closed"

    def test_same_work_new_session_rebinds(self, tmp_store, monkeypatch):
        """Deliverable (c): same work resumed in a new session re-binds the task."""
        from skill_hub import session_binding
        store, _marker = tmp_store

        monkeypatch.setattr(session_binding, "_get_config", lambda: {
            "enabled": True, "strategy": "semantic",
            "window_days": 7, "semantic_threshold": 0.75,
        })
        monkeypatch.setattr(session_binding, "_embed_message", lambda _m: [])

        msg = "Refactor the assets upload pipeline to support GCS"
        cwd, branch = "/geoid", "refactor/assets"

        a1, tid1, _, _ = session_binding.bind_session_to_task(
            session_id="original-session", message=msg,
            cwd=cwd, branch=branch, store=store,
        )
        assert a1 == "created"

        a2, tid2, _, reason = session_binding.bind_session_to_task(
            session_id="resumed-session", message=msg,
            cwd=cwd, branch=branch, store=store,
        )
        assert a2 == "resumed"
        assert tid2 == tid1
        assert reason == "stable_key"

        # Confirm session_id was updated.
        row = store._conn.execute(
            "SELECT session_id FROM tasks WHERE id=?", (tid1,)
        ).fetchone()
        assert row["session_id"] == "resumed-session"
