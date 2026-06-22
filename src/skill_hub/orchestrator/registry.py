"""Capability registry — declarative tool-readiness definitions.

Each ``Capability`` entry describes one developer tool: how to recognise when a
message signals intent to use it, whether a target directory is eligible, how to
probe its readiness, and how to provision (init or refresh) it.

The engine (``engine.py``) is generic — adding a new tool means adding an entry
here, not changing the engine.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .engine import Readiness, probe_codegraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------

_CODE_PROJECT_MARKERS = frozenset([
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "CMakeLists.txt",
    "Makefile",
    "src",
    ".github",
])


def is_code_project(root: Path) -> bool:
    """Return True when *root* looks like a code project.

    Checks for common code-project markers: VCS directories, build manifests,
    well-known top-level directories, etc.  Intentionally liberal — any single
    marker is sufficient.
    """
    if not root.is_dir():
        return False
    try:
        children = {p.name for p in root.iterdir()}
    except OSError:
        return False
    return bool(children & _CODE_PROJECT_MARKERS)


# ---------------------------------------------------------------------------
# Intent-signal helpers
# ---------------------------------------------------------------------------

_CODEGRAPH_VERBS = frozenset([
    "explore", "trace", "map", "where", "impact", "check", "analyze",
    "analyse", "understand", "find", "locate", "search", "navigate",
    "inspect", "review", "audit", "investigate",
])


def _signals_codegraph(message: str, root: Path) -> bool:
    """True when the message contains a code-exploration intent verb and *root*
    is a code project."""
    if not is_code_project(root):
        return False
    words = set(message.lower().split())
    return bool(words & _CODEGRAPH_VERBS)


# ---------------------------------------------------------------------------
# Provision helpers
# ---------------------------------------------------------------------------

# Resolve the codegraph executable from a hardened, fixed set of trusted
# directories rather than the inherited PATH, so a poisoned PATH entry (e.g. a
# malicious "codegraph" earlier in PATH) cannot hijack the subprocess we spawn.
# Covers the standard install locations across Linux, Intel mac, and Apple
# Silicon Homebrew. Resolution is deferred to first use so a tool installed
# after import is still found.
_CODEGRAPH_TRUSTED_DIRS = (
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/opt/homebrew/bin",
    "/opt/local/bin",
)


def _resolve_codegraph_bin() -> str | None:
    """Locate the ``codegraph`` executable in a trusted directory.

    Returns the absolute path, or ``None`` when it is not installed in any
    trusted location. Callers must treat ``None`` as "tool unavailable" and
    skip provisioning rather than falling back to an unverified path.
    """
    return shutil.which("codegraph", path=os.pathsep.join(_CODEGRAPH_TRUSTED_DIRS))


def _codegraph_refresh_argv(root: Path) -> list[str] | None:
    bin_path = _resolve_codegraph_bin()
    return [bin_path, "sync", str(root)] if bin_path else None


def _codegraph_init_argv(root: Path) -> list[str] | None:
    # ``init`` alone only creates the empty .codegraph/ scaffold — the graph has
    # zero nodes until a full index runs, and ``sync`` ("changes since last
    # index") is a no-op on a never-indexed project. The ``-i`` flag runs the
    # initial indexing as part of init, so the index is actually queryable
    # afterwards rather than just present-but-empty.
    bin_path = _resolve_codegraph_bin()
    return [bin_path, "init", "-i", str(root)] if bin_path else None


# ---------------------------------------------------------------------------
# Directive templates
# ---------------------------------------------------------------------------

_CODEGRAPH_READY_FMT = (
    "[tooling] {root} is indexed (refreshed {age}) — prefer the indexed "
    "code-graph queries (search / callers / impact) over raw text search."
)

_CODEGRAPH_MISSING_FMT = (
    "[tooling] {root} is not indexed but the task is about to explore it; "
    "offer to initialize it (via ensure_tooling) before falling back to text search."
)


# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    """Declarative description of a developer-tool capability.

    Fields:
        id:                  Stable string key (e.g. ``"code-graph"``).
        signals:             Predicate ``(message, root) -> bool`` — True when
                             the message + target suggest this capability is needed.
        scope:               Eligibility predicate ``(root) -> bool`` — whether
                             this capability can apply to the target directory at all.
        probe:               ``(root) -> Readiness`` — cheap filesystem/status check.
        provision_refresh:   ``(root) -> list[str] | None`` — argv for an in-place
                             refresh, or ``None`` when the tool is unavailable.
        provision_init:      ``(root) -> list[str] | None`` — argv for first-time
                             setup, or ``None`` when the tool is unavailable.
        directive_ready:     Format string (or ``(root, readiness) -> str`` callable)
                             for when the tool is present and fresh.
        directive_missing:   Format string (or ``(root, readiness) -> str`` callable)
                             for when the tool is absent or stale.
        probe_cache_ttl:     Seconds to cache probe results.
        sync_ttl:            Minimum seconds between auto-refresh dispatches.
    """

    id: str
    signals: Callable[[str, Path], bool]
    scope: Callable[[Path], bool]
    probe: Callable[[Path], Readiness]
    provision_refresh: Callable[[Path], list[str] | None]
    provision_init: Callable[[Path], list[str] | None]
    directive_ready: str | Callable[[Path, Readiness], str]
    directive_missing: str | Callable[[Path, Readiness], str]
    probe_cache_ttl: float = 60.0
    sync_ttl: float = 300.0

    def format_directive_ready(self, root: Path, readiness: Readiness) -> str:
        if callable(self.directive_ready):
            try:
                return self.directive_ready(root, readiness)
            except Exception:
                return ""
        age_str = (
            f"{int(readiness.stale_age)}s ago"
            if readiness.stale_age is not None
            else "recently"
        )
        return self.directive_ready.format(root=root, age=age_str)

    def format_directive_missing(self, root: Path, readiness: Readiness) -> str:
        if callable(self.directive_missing):
            try:
                return self.directive_missing(root, readiness)
            except Exception:
                return ""
        return self.directive_missing.format(root=root)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CODEGRAPH = Capability(
    id="code-graph",
    signals=_signals_codegraph,
    scope=is_code_project,
    probe=probe_codegraph,
    provision_refresh=_codegraph_refresh_argv,
    provision_init=_codegraph_init_argv,
    directive_ready=_CODEGRAPH_READY_FMT,
    directive_missing=_CODEGRAPH_MISSING_FMT,
    probe_cache_ttl=60.0,
    sync_ttl=300.0,
)

REGISTRY: list[Capability] = [CODEGRAPH]
