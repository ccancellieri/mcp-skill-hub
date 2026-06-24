"""Tests for the wiki vault watcher (_WikiVaultHandler)."""
from __future__ import annotations

import sys
import threading
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
    def __init__(self, src_path: str) -> None:
        self.src_path = src_path


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


# ---------------------------------------------------------------------------
# _do_reindex() — patched to avoid real IO
# ---------------------------------------------------------------------------


def test_do_reindex_calls_wiki_reindex(tmp_path):
    """_do_reindex must call wiki.reindex and log the result."""
    handler = _WikiVaultHandler(delay=0.01)
    handler._last_changed = str(tmp_path / "page.md")
    handler._busy = False

    log_lines: list[str] = []
    wiki_root_used: list[Path] = []

    def _fake_log(tag, msg):
        log_lines.append(msg)

    def _fake_reindex(store, wiki_root, dry_run):
        wiki_root_used.append(wiki_root)
        return {"pages": 2, "edges": 1, "vectors": 4}

    # _do_reindex imports SkillStore inside the method, so patch at the source.
    with patch("skill_hub.activity_log.log_event", _fake_log):
        with patch("skill_hub.wiki.reindex", _fake_reindex):
            with patch("skill_hub.store.SkillStore") as MockStore:
                instance = MagicMock()
                instance.close = MagicMock()
                MockStore.return_value = instance
                handler._do_reindex()

    assert not handler._busy, "_busy must be False after reindex"
    assert wiki_root_used, "wiki.reindex must have been called"
    assert any("wiki reindex complete" in l for l in log_lines), (
        f"expected completion log; got: {log_lines}"
    )


def test_do_reindex_logs_failure_on_exception():
    """_do_reindex must log failure and clear _busy even when reindex raises."""
    handler = _WikiVaultHandler(delay=0.01)
    handler._last_changed = "/wiki/page.md"
    handler._busy = False

    log_lines: list[str] = []

    def _fake_log(tag, msg):
        log_lines.append(msg)

    with patch("skill_hub.activity_log.log_event", _fake_log):
        with patch("skill_hub.wiki.reindex", side_effect=RuntimeError("boom")):
            with patch("skill_hub.store.SkillStore") as MockStore:
                instance = MagicMock()
                instance.close = MagicMock()
                MockStore.return_value = instance
                handler._do_reindex()

    assert not handler._busy, "_busy must be cleared even on failure"
    assert any("wiki reindex failed" in l for l in log_lines)
