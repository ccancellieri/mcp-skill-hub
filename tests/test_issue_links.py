"""Tests for issue #37 — typed task↔issue links + bidirectional sync.

Constraints (hard)
------------------
- Uses only skill_hub.store + skill_hub.issue_sync — never skill_hub.server.
- Every test uses a fresh tmp_path DB, NEVER DB_PATH.
- All gh helpers (_gh_view, _gh_comment, _gh_close) are mocked — no real
  subprocess calls.
- No embedding model loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure we import from the worktree src, not any installed copy.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Guard: server.py must never be imported (it opens the live DB at module level).
def test_server_not_imported():
    assert "skill_hub.server" not in sys.modules, (
        "skill_hub.server must not be imported — it opens the live DB on import"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    from skill_hub.store import SkillStore
    return SkillStore(db_path=tmp_path / "test_issuelinks.db")


@pytest.fixture()
def store2(tmp_path):
    """Second store opened on the same DB — simulates re-open for migration test."""
    return tmp_path / "test_migr.db"


def _make_open_task(store, title="Test task", tags=""):
    """Insert an open task and return its id."""
    return store.save_task(
        title=title,
        summary="summary",
        vector=[],
        tags=tags,
        session_id="test-session",
    )


# ---------------------------------------------------------------------------
# A) Schema: link_task_issue + get_issue_links round-trip
# ---------------------------------------------------------------------------

class TestLinkRoundTrip:
    def test_link_and_retrieve(self, store):
        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 42, repo="owner/repo", url="https://gh/issues/42")

        links = store.get_issue_links(task_id)
        assert len(links) == 1
        lnk = links[0]
        assert lnk["task_id"] == task_id
        assert lnk["issue_number"] == 42
        assert lnk["repo"] == "owner/repo"
        assert lnk["url"] == "https://gh/issues/42"
        assert lnk["state"] is None
        assert lnk["writeback_done"] == 0

    def test_unique_prevents_dupes(self, store):
        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 42, repo="owner/repo")
        store.link_task_issue(task_id, 42, repo="owner/repo")  # INSERT OR IGNORE
        assert len(store.get_issue_links(task_id)) == 1

    def test_different_repos_are_distinct(self, store):
        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 42, repo="owner/repo-a")
        store.link_task_issue(task_id, 42, repo="owner/repo-b")
        assert len(store.get_issue_links(task_id)) == 2

    def test_list_all_issue_links_repo_filter(self, store):
        t1 = _make_open_task(store, title="T1")
        t2 = _make_open_task(store, title="T2")
        store.link_task_issue(t1, 10, repo="org/a")
        store.link_task_issue(t2, 20, repo="org/b")

        all_links = store.list_all_issue_links()
        assert len(all_links) == 2

        filtered = store.list_all_issue_links(repo="org/a")
        assert len(filtered) == 1
        assert filtered[0]["issue_number"] == 10

    def test_update_link_state(self, store):
        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 7, repo="")
        link = store.get_issue_links(task_id)[0]
        store.update_link_state(link["id"], "closed")

        updated = store.get_issue_links(task_id)[0]
        assert updated["state"] == "closed"
        assert updated["writeback_done"] == 0
        assert updated["last_synced_at"] is not None

    def test_update_link_state_with_writeback_done(self, store):
        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 99, repo="")
        link = store.get_issue_links(task_id)[0]
        store.update_link_state(link["id"], "open", writeback_done=1)

        updated = store.get_issue_links(task_id)[0]
        assert updated["writeback_done"] == 1


# ---------------------------------------------------------------------------
# B) Tag migration: tasks with 'issue:<n>' tags get a link row on re-open
# ---------------------------------------------------------------------------

class TestTagMigration:
    def test_migration_creates_link_for_issue_tag(self, store2):
        """Tags like 'issue:42' in tasks → link row created during _migrate."""
        from skill_hub.store import SkillStore

        db_path = store2  # type: Path

        # Open store and create a task with an issue tag (simulates old data).
        st = SkillStore(db_path=db_path)
        task_id = st.save_task(
            title="Old fanout task",
            summary="body",
            vector=[],
            tags="fanout:abc src:gh issue:42",
            session_id="s1",
        )
        # Manually remove any link row that save_task might have created
        # (it doesn't, but be defensive) so we test _migrate alone.
        st._conn.execute("DELETE FROM task_issue_links")
        st._conn.commit()
        st._conn.close()

        # Re-open store: _migrate() should detect the tag and insert the link.
        st2 = SkillStore(db_path=db_path)
        links = st2.get_issue_links(task_id)
        assert len(links) == 1
        assert links[0]["issue_number"] == 42

    def test_migration_parses_source_prefixed_tag(self, store2):
        """Real fanout tags carry a source prefix ('issue:gh:123') — the
        migration must extract the trailing number, not choke on 'gh:'."""
        from skill_hub.store import SkillStore

        db_path = store2
        st = SkillStore(db_path=db_path)
        task_id = st.save_task(
            title="Real fanout task",
            summary="body",
            vector=[],
            tags="fanout:abc src:gh issue:gh:123",
            session_id="s1",
        )
        st._conn.execute("DELETE FROM task_issue_links")
        st._conn.commit()
        st._conn.close()

        st2 = SkillStore(db_path=db_path)
        links = st2.get_issue_links(task_id)
        assert len(links) == 1
        assert links[0]["issue_number"] == 123

    def test_migration_idempotent(self, store2):
        """Running _migrate multiple times (via reopen) does not create dupe links."""
        from skill_hub.store import SkillStore

        db_path = store2

        st = SkillStore(db_path=db_path)
        task_id = st.save_task(
            title="T", summary="s", vector=[],
            tags="issue:7", session_id="s",
        )
        st._conn.execute("DELETE FROM task_issue_links")
        st._conn.commit()
        st._conn.close()

        # Open twice — migration runs both times, should not duplicate.
        SkillStore(db_path=db_path)._conn.close()
        st3 = SkillStore(db_path=db_path)
        links = st3.get_issue_links(task_id)
        assert len(links) == 1


# ---------------------------------------------------------------------------
# C) Reconcile: issue→task (issue closed, task open → task closed)
# ---------------------------------------------------------------------------

class TestReconcileIssueWins:
    def test_closed_issue_closes_open_task(self, store):
        from skill_hub import issue_sync

        task_id = _make_open_task(store, title="Linked open task")
        store.link_task_issue(task_id, 55, repo="org/r")

        emit_calls: list = []

        with patch.object(issue_sync, "_gh_view", return_value={"state": "CLOSED", "title": "Bug", "url": ""}):
            report = issue_sync.reconcile(
                store, repo="org/r", dry_run=False, writeback="off",
                emit=lambda k, t, p: emit_calls.append((k, t, p)),
            )

        assert report["tasks_closed"] == 1
        assert report["dry_run"] is False

        # Task should now be closed in DB.
        task = store.get_task(task_id)
        assert task["status"] == "closed"

        # Event emitted.
        assert any(k == "task.closed" for k, _, _ in emit_calls)
        evt = next(k for k, _, p in emit_calls if k == "task.closed")
        assert evt == "task.closed"

        # Link state updated.
        link = store.get_issue_links(task_id)[0]
        assert link["state"] == "closed"

    def test_dry_run_does_not_close_task(self, store):
        from skill_hub import issue_sync

        task_id = _make_open_task(store, title="DryRun task")
        store.link_task_issue(task_id, 1, repo="")

        with patch.object(issue_sync, "_gh_view", return_value={"state": "closed", "title": "", "url": ""}):
            report = issue_sync.reconcile(store, dry_run=True, writeback="off")

        assert report["tasks_closed"] == 0
        assert report["dry_run"] is True

        task = store.get_task(task_id)
        assert task["status"] == "open"  # unchanged

    def test_emit_called_with_correct_payload(self, store):
        from skill_hub import issue_sync

        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 77, repo="org/q")

        captured: list = []

        with patch.object(issue_sync, "_gh_view", return_value={"state": "closed", "title": "", "url": ""}):
            issue_sync.reconcile(
                store, repo="org/q", dry_run=False,
                emit=lambda k, t, p: captured.append({"kind": k, "payload": p}),
            )

        assert any(e["kind"] == "task.closed" for e in captured)
        evt = next(e for e in captured if e["kind"] == "task.closed")
        assert evt["payload"]["task_id"] == task_id
        assert evt["payload"]["issue_number"] == 77
        assert evt["payload"]["reason"] == "issue_closed"


# ---------------------------------------------------------------------------
# D) Reconcile: task→issue writeback='off' (default) — no gh writes
# ---------------------------------------------------------------------------

class TestWritebackOff:
    def test_off_makes_no_gh_calls(self, store):
        from skill_hub import issue_sync

        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 88, repo="org/x")
        # Close the task locally.
        store.close_task(task_id, compact="done")

        mock_comment = MagicMock()
        mock_close = MagicMock()

        with (
            patch.object(issue_sync, "_gh_view", return_value={"state": "open", "title": "", "url": ""}),
            patch.object(issue_sync, "_gh_comment", mock_comment),
            patch.object(issue_sync, "_gh_close", mock_close),
        ):
            report = issue_sync.reconcile(store, writeback="off")

        mock_comment.assert_not_called()
        mock_close.assert_not_called()
        assert report["issues_commented"] == 0
        assert report["issues_closed"] == 0


# ---------------------------------------------------------------------------
# E) Reconcile: writeback='comment' — idempotency via writeback_done
# ---------------------------------------------------------------------------

class TestWritebackComment:
    def test_comment_posted_once(self, store):
        from skill_hub import issue_sync

        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 33, repo="org/y")
        store.close_task(task_id, compact="done")

        mock_comment = MagicMock(return_value=True)
        mock_close = MagicMock()

        with (
            patch.object(issue_sync, "_gh_view", return_value={"state": "open", "title": "", "url": ""}),
            patch.object(issue_sync, "_gh_comment", mock_comment),
            patch.object(issue_sync, "_gh_close", mock_close),
        ):
            report = issue_sync.reconcile(store, writeback="comment")
            assert report["issues_commented"] == 1
            mock_comment.assert_called_once()
            mock_close.assert_not_called()

            # Second reconcile run — writeback_done=1 → no second comment.
            mock_comment.reset_mock()
            report2 = issue_sync.reconcile(store, writeback="comment")

        assert report2["issues_commented"] == 0
        mock_comment.assert_not_called()

    def test_writeback_done_flag_persisted(self, store):
        from skill_hub import issue_sync

        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 44, repo="")
        store.close_task(task_id, compact="done")
        link = store.get_issue_links(task_id)[0]
        assert link["writeback_done"] == 0

        with (
            patch.object(issue_sync, "_gh_view", return_value={"state": "open", "title": "", "url": ""}),
            patch.object(issue_sync, "_gh_comment", return_value=True),
        ):
            issue_sync.reconcile(store, writeback="comment")

        link_after = store.get_issue_links(task_id)[0]
        assert link_after["writeback_done"] == 1


# ---------------------------------------------------------------------------
# F) Safety: no real subprocess ever runs in these tests
# ---------------------------------------------------------------------------

class TestNoRealSubprocess:
    """Belt-and-suspenders: confirm the real subprocess is never called."""

    def test_all_gh_helpers_mocked(self, store, monkeypatch):
        """Confirm _gh_view is never reached via real subprocess in reconcile."""
        from skill_hub import issue_sync
        import subprocess as _sp

        task_id = _make_open_task(store)
        store.link_task_issue(task_id, 1, repo="")

        real_run = _sp.run
        called_real = []

        def fake_run(args, **kwargs):
            if args and "gh" in args[0]:
                called_real.append(args)
                raise AssertionError(f"Real subprocess.run called with gh: {args}")
            return real_run(args, **kwargs)

        monkeypatch.setattr(_sp, "run", fake_run)

        with patch.object(issue_sync, "_gh_view", return_value={"state": "open", "title": "", "url": ""}):
            issue_sync.reconcile(store, writeback="off")

        assert not called_real, "Real gh subprocess was invoked"
