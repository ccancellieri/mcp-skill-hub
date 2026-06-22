"""Tooling orchestrator — holistic per-turn tool-readiness steering.

This module detects what a task needs, ensures the relevant tool is ready for
the target directory (initializing or refreshing an index when warranted), and
steers the turn toward it — transparently, without relying on the assistant to
remember.

PUBLIC API
----------

evaluate(cwd: str, message: str, *, session: dict | None = None) -> OrchestratorResult
    Per-turn evaluation.  Resolves target directories, probes registered
    capabilities (cached), and builds directives + provisioning actions.
    NEVER raises; returns an empty OrchestratorResult on any error.
    Callers must dispatch ``result.provision_actions`` asynchronously via
    ``dispatch_async()`` to preserve the tier-1 latency budget.

    Returns:
        OrchestratorResult(
            directive: str,           # text for the additional-context channel ("" if nothing)
            decisions: list[dict],    # one per (cap, target) evaluated, for logging
            provision_actions: list[list[str]],  # argv lists ready to dispatch
        )

ensure_tooling_core(path: str, *, init: bool = False, refresh: bool = True) -> dict
    Explicit, idempotent probe + optional provisioning for a single path.
    Safe to call from the ``ensure_tooling`` MCP tool or interactively.
    ``refresh=True`` dispatches a sync fire-and-forget if the index is present.
    ``init=True`` runs ``codegraph init`` blocking when the index is absent.
    Never raises.

    Returns:
        {
            "path":      str,
            "present":   bool,
            "fresh":     bool,
            "action":    "none" | "refresh_dispatched" | "init_run" | "error",
            "directive": str,
        }

dispatch_async(actions: list[list[str]]) -> None
    Fire-and-forget subprocess launch with TTL-debounce.  Never blocks, never raises.

OrchestratorResult  — see engine.py
Readiness           — see engine.py
REGISTRY            — list[Capability] from registry.py
"""

from __future__ import annotations

from .engine import (
    OrchestratorResult,
    Readiness,
    dispatch_async,
    ensure_tooling_core,
    evaluate,
    resolve_targets,
)
from .registry import REGISTRY, Capability, is_code_project

__all__ = [
    "Capability",
    "OrchestratorResult",
    "REGISTRY",
    "Readiness",
    "dispatch_async",
    "ensure_tooling_core",
    "evaluate",
    "is_code_project",
    "resolve_targets",
]
