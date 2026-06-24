"""Tests for the wiki vault watcher (_WikiVaultHandler)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest

from skill_hub.watcher import _WikiVaultHandler, _MIN_REINDEX_INTERVAL, _IGNORE_PATH_PARTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal watchdog-style filesystem event."""

    def __init__(self, src_path: str, event_type: str = "modified") -> None:
        self.src_path = src_path
        self.event_type = event_type
        self.dest_path = ""


class _FakeMovedEvent:
    def __init__(self, src_path: str, dest_path: str) -> None:
        self.src_path = src_path
        self.event_type = "moved"
        self.dest_path = dest_path


# ---------------------------------------------------------------------------
# dispatch() gating
# ---------------------------------------------------------------------------


def test_dispatch_ignores_non_md_files():
    handler = _WikiVaultHandler(delay=0.01)
    handler.dispatch(_FakeEvent("/wiki/pages/entity/some-page.json"))
    # Timer must NOT be set for a non-.md file.
    assert handler._timer is None


def test_dispatch_ignores_git_paths():
    handler = _WikiVaultHandler(delay=0.01)
    handler.dispatch(_FakeEvent("/wiki/.git/HEAD"))
    assert handler._timer is None


def test_dispatch_sets_timer_for_md_file():
    handler = _WikiVaultHandler(delay=0.05)
    handler.dispatch(_FakeEvent("/wiki/pages/entity/page.md"))
    assert handler._timer is not None
    if handler._timer:
        handler._timer.cancel()


def test_dispatch_skips_while_busy():
    handler = _WikiVaultHandler(delay=0.05)
    handler._busy = True
    handler.dispatch(_FakeEvent("/wiki/pages/entity/page.md"))
    assert handler._timer is None


def test_dispatch_skips_during_cooldown():
    handler = _WikiVaultHandler(delay=0.05)
    # Pretend a reindex just finished.
    handler._last_done = time.monotonic()
    handler.dispatch(_FakeEvent("/wiki/pages/entity/page.md"))
    assert handler._timer is None


def test_dispatch_routes_deleted_event_to_pending_deleted():
    handler = _WikiVaultHandler(delay=0.05)
    handler.dispatch(_FakeEvent("/wiki/pages/entity/gone.md", event_type="deleted"))
    assert handler._pending_deleted, "deleted event must populate _pending_deleted"
    assert not handler._pending_changed, "deleted event must NOT populate _pending_changed"
    if handler._timer:
        handler._timer.cancel()


def test_dispatch_routes_modified_event_to_pending_changed():
    handler = _WikiVaultHandler(delay=0.05)
    handler.dispatch(_FakeEvent("/wiki/pages/entity/page.md", event_type="modified"))
    assert handler._pending_changed, "modified event must populate _pending_changed"
    assert not handler._pending_deleted, "modified event must NOT populate _pending_deleted"
    if handler._timer:
        handler._timer.cancel()


def test_dispatch_accumulates_multiple_paths():
    handler = _WikiVaultHandler(delay=0.5)
    handler.dispatch(_FakeEvent("/wiki/pages/entity/alpha.md"))
    handler.dispatch(_FakeEvent("/wiki/pages/entity/beta.md"))
    handler.dispatch(_FakeEvent("/wiki/pages/entity/gamma.md", event_type="deleted"))
    assert len(handler._pending_changed) == 2
    assert len(handler._pending_deleted) == 1
    if handler._timer:
        handler._timer.cancel()


# ---------------------------------------------------------------------------
# _do_reindex() — patched to avoid real IO
# ---------------------------------------------------------------------------


def test_do_reindex_calls_wiki_reindex_paths(tmp_path):
    """_do_reindex must call wiki.reindex_paths (not wiki.reindex) and log the result."""
    handler = _WikiVaultHandler(delay=0.01)
    changed_path = tmp_path / "page.md"
    handler._pending_changed.add(changed_path)
    handler._busy = False

    log_lines: list[str] = []
    reindex_paths_calls: list[dict] = []

    def _fake_log(tag, msg):
        log_lines.append(msg)

    def _fake_reindex_paths(store, wiki_root, *, changed=None, deleted=None):
        reindex_paths_calls.append({"changed": changed, "deleted": deleted})
        return {"pages_updated": 1, "pages_deleted": 0, "vectors": 3, "fallback": False}

    with patch("skill_hub.activity_log.log_event", _fake_log):
        with patch("skill_hub.wiki.reindex_paths", _fake_reindex_paths):
            with patch("skill_hub.store.SkillStore") as MockStore:
                instance = MagicMock()
                instance.close = MagicMock()
                MockStore.return_value = instance
                handler._do_reindex()

    assert not handler._busy, "_busy must be False after reindex"
    assert reindex_paths_calls, "wiki.reindex_paths must have been called"
    call = reindex_paths_calls[0]
    assert changed_path in (call["changed"] or set()), (
        "the changed path must be forwarded to reindex_paths"
    )
    assert any("wiki incremental reindex complete" in l for l in log_lines), (
        f"expected completion log; got: {log_lines}"
    )


def test_do_reindex_passes_deleted_set(tmp_path):
    """Deleted paths must be forwarded to reindex_paths, not changed."""
    handler = _WikiVaultHandler(delay=0.01)
    del_path = tmp_path / "old-page.md"
    handler._pending_deleted.add(del_path)
    handler._busy = False

    calls: list[dict] = []

    def _fake_reindex_paths(store, wiki_root, *, changed=None, deleted=None):
        calls.append({"changed": changed, "deleted": deleted})
        return {"pages_updated": 0, "pages_deleted": 1, "vectors": 0, "fallback": False}

    with patch("skill_hub.activity_log.log_event", lambda *a: None):
        with patch("skill_hub.wiki.reindex_paths", _fake_reindex_paths):
            with patch("skill_hub.store.SkillStore") as MockStore:
                instance = MagicMock()
                instance.close = MagicMock()
                MockStore.return_value = instance
                handler._do_reindex()

    assert calls, "reindex_paths must have been called"
    call = calls[0]
    assert del_path in (call["deleted"] or set()), (
        "deleted path must be in the deleted set passed to reindex_paths"
    )
    assert not (call["changed"] or set()), "changed set must be empty"


def test_do_reindex_does_not_call_full_reindex(tmp_path):
    """wiki.reindex (full) must NOT be called when using the incremental path."""
    handler = _WikiVaultHandler(delay=0.01)
    handler._pending_changed.add(tmp_path / "page.md")
    handler._busy = False

    full_reindex_called = []

    def _full_reindex(*a, **kw):
        full_reindex_called.append(True)
        return {"pages": 0, "edges": 0, "vectors": 0}

    def _inc_reindex(store, wiki_root, *, changed=None, deleted=None):
        return {"pages_updated": 1, "pages_deleted": 0, "vectors": 2, "fallback": False}

    with patch("skill_hub.activity_log.log_event", lambda *a: None):
        with patch("skill_hub.wiki.reindex", _full_reindex):
            with patch("skill_hub.wiki.reindex_paths", _inc_reindex):
                with patch("skill_hub.store.SkillStore") as MockStore:
                    instance = MagicMock()
                    instance.close = MagicMock()
                    MockStore.return_value = instance
                    handler._do_reindex()

    assert not full_reindex_called, "full wiki.reindex must NOT be called for a single-page change"


def test_do_reindex_clears_pending_sets(tmp_path):
    """Pending path sets must be drained before _do_reindex calls the indexer."""
    handler = _WikiVaultHandler(delay=0.01)
    handler._pending_changed.add(tmp_path / "page-a.md")
    handler._pending_deleted.add(tmp_path / "page-b.md")
    handler._busy = False

    with patch("skill_hub.activity_log.log_event", lambda *a: None):
        with patch("skill_hub.wiki.reindex_paths",
                   return_value={"pages_updated": 1, "pages_deleted": 1,
                                 "vectors": 2, "fallback": False}):
            with patch("skill_hub.store.SkillStore") as MockStore:
                instance = MagicMock()
                instance.close = MagicMock()
                MockStore.return_value = instance
                handler._do_reindex()

    assert not handler._pending_changed, "pending_changed must be cleared"
    assert not handler._pending_deleted, "pending_deleted must be cleared"


def test_do_reindex_logs_failure_on_exception():
    """_do_reindex must log failure and clear _busy even when reindex_paths raises."""
    handler = _WikiVaultHandler(delay=0.01)
    handler._pending_changed.add(Path("/wiki/page.md"))
    handler._busy = False

    log_lines: list[str] = []

    def _fake_log(tag, msg):
        log_lines.append(msg)

    with patch("skill_hub.activity_log.log_event", _fake_log):
        with patch("skill_hub.wiki.reindex_paths", side_effect=RuntimeError("boom")):
            with patch("skill_hub.store.SkillStore") as MockStore:
                instance = MagicMock()
                instance.close = MagicMock()
                MockStore.return_value = instance
                handler._do_reindex()

    assert not handler._busy, "_busy must be cleared even on failure"
    assert any("wiki reindex failed" in l for l in log_lines)
