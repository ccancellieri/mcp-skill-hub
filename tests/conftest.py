"""Shared pytest configuration and fixtures for the skill-hub test suite.

Adds src/ to sys.path once per session so every test file can import
skill_hub without repeating the path-insertion boilerplate.

Also provides the ``assert_server_not_imported`` fixture used by test files
that must never trigger a live-DB import of skill_hub.server.  The fixture
checks against a snapshot of sys.modules taken at session-start (before any
test-function body runs), so it catches the specific hazard the guard is
designed to prevent: importing a test file's own dependencies dragging in
skill_hub.server.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Isolate the data directory BEFORE any skill_hub module is imported.
# ---------------------------------------------------------------------------
#
# ``store.DB_PATH`` and ``config.CONFIG_PATH`` are computed from ``Path.home()``
# at import time, so with the developer's real ``$HOME`` the whole suite reads
# and writes the live ``~/.claude/mcp-skill-hub`` database. Under xdist that
# shared, mutable DB (9000+ rows, concurrent writers) makes hermetic tests fail
# non-deterministically — e.g. an envelope's ``events_emitted`` picking up event
# rows another worker inserted mid-call.
#
# Point ``$HOME`` at a private temp directory per pytest process (the xdist
# controller and each worker import this conftest independently, so each gets
# its own tree and its own empty DB). Tests that pass an explicit ``db_path``
# are unaffected; those using the default store now get a clean, isolated DB.
# Opt out with ``SKILL_HUB_TEST_REAL_HOME=1`` to run against the real home.
if os.environ.get("SKILL_HUB_TEST_REAL_HOME") != "1":
    _ISOLATED_HOME = tempfile.mkdtemp(prefix="skill-hub-test-home-")
    os.environ["HOME"] = _ISOLATED_HOME
    atexit.register(shutil.rmtree, _ISOLATED_HOME, ignore_errors=True)

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Session-scoped sys.modules snapshot
# ---------------------------------------------------------------------------

# Capture the set of already-loaded module names once, during conftest
# collection, before any test-function body executes.  Individual guard
# tests compare against this baseline.
_MODULES_AT_COLLECTION: frozenset[str] = frozenset(sys.modules)


@pytest.fixture(scope="session")
def modules_at_collection() -> frozenset[str]:
    """Return the frozenset of sys.modules keys captured at collection time."""
    return _MODULES_AT_COLLECTION


# ---------------------------------------------------------------------------
# Shared server-import guard
# ---------------------------------------------------------------------------


@pytest.fixture()
def assert_server_not_imported(modules_at_collection: frozenset[str]) -> None:
    """Assert that skill_hub.server was not in sys.modules at collection time.

    Use this fixture in any test file whose imports must not pull in
    skill_hub.server (which opens the live DB via a module-level
    SkillStore() call).  The check is against the collection-time snapshot,
    not the current sys.modules, so it is immune to other test files that
    legitimately import the server later in the same suite run.
    """
    assert "skill_hub.server" not in modules_at_collection, (
        "skill_hub.server was already loaded when the test suite was collected "
        "— it must not be imported at module level, because it opens the live DB"
    )


# ---------------------------------------------------------------------------
# Reconciler-thread isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stop_reconcilers_after_test():
    """Stop every reconciler a test (or an import of skill_hub.server) started.

    skill_hub.server starts a reconciler daemon thread at import time
    (auto_reconcile is on by default). It ticks every 2s and calls real service
    start() / subprocess.Popen. Any test that imports server — directly or
    transitively — leaks that thread, which is what makes
    test_no_real_provisioning_ran flake: the thread calls the real Popen outside
    that test's patch scope in a completely unrelated later test. Draining every
    reconciler after each test keeps the background thread from bleeding across
    test boundaries (issue #143). Signalling the stop event is enough to halt the
    ticking even if the thread is momentarily blocked inside a slow start().
    """
    yield
    from skill_hub.services.registry import stop_all_reconcilers

    stop_all_reconcilers()
