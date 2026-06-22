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

import sys
from pathlib import Path

import pytest

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
