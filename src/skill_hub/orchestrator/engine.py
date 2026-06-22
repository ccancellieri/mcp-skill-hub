"""Tooling orchestrator engine.

Every public function is non-fatal: exceptions are caught, logged at DEBUG, and
a safe default is returned.  The prompt-injection path must never raise.

Public API:
    evaluate(cwd, message, *, session=None) -> OrchestratorResult
    ensure_tooling_core(path, *, init=False, refresh=True) -> dict
    dispatch_async(actions) -> None
    probe_codegraph(root) -> Readiness
    resolve_targets(cwd, message) -> list[Path]
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Readiness:
    """Result of a cheap readiness probe for one capability + root combination.

    Attributes:
        present:    The tool index/data is present at *root*.
        fresh:      The index is recent enough to be trusted.
        stale_age:  Seconds since the index was last updated, or None if unknown.
        detail:     Human-readable explanation (for logging and directives).
    """

    present: bool
    fresh: bool
    stale_age: float | None
    detail: str


@dataclass
class OrchestratorResult:
    """Output of ``evaluate()``.

    Attributes:
        directive:          Text to inject into the additional-context channel.
                            Empty string when nothing actionable was found.
        decisions:          One dict per capability evaluated, for observability.
                            Shape: {target, capability_id, present, fresh, action}.
        provision_actions:  List of argv lists dispatched (or to be dispatched)
                            asynchronously.
    """

    directive: str = ""
    decisions: list[dict] = field(default_factory=list)
    provision_actions: list[list[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Probe cache — (capability_id, str(root)) -> (timestamp, Readiness)
# ---------------------------------------------------------------------------

_probe_cache: dict[tuple[str, str], tuple[float, Readiness]] = {}

# Debounce map for async refreshes — argv tuple -> last-dispatch timestamp.
_last_dispatch: dict[tuple[str, ...], float] = {}

# ---------------------------------------------------------------------------
# Codegraph probe
# ---------------------------------------------------------------------------

# Index directory name produced by `codegraph init`.
_CODEGRAPH_DIR = ".codegraph"

# Default sync TTL in seconds (overridden by config at call sites).
_DEFAULT_SYNC_TTL = 300.0


def probe_codegraph(root: Path) -> Readiness:
    """Probe code-graph index readiness for *root*.

    Uses filesystem mtime only — no subprocess on the hot path.

    A ``codegraph status`` call is intentionally avoided here to keep
    p99 latency within the sub-10ms tier-1 budget.  Freshness is determined
    by comparing the ``.codegraph/`` directory mtime against the configured
    sync TTL.
    """
    index_dir = root / _CODEGRAPH_DIR
    if not index_dir.exists():
        return Readiness(present=False, fresh=False, stale_age=None,
                         detail=f"{_CODEGRAPH_DIR} not found under {root}")

    try:
        index_mtime = index_dir.stat().st_mtime
    except OSError as exc:
        return Readiness(present=True, fresh=False, stale_age=None,
                         detail=f"stat failed: {exc}")

    stale_age = time.time() - index_mtime
    try:
        from .. import config as _cfg
        sync_ttl = float(_cfg.get("orchestrator_sync_ttl_secs") or _DEFAULT_SYNC_TTL)
    except Exception:
        sync_ttl = _DEFAULT_SYNC_TTL

    fresh = stale_age <= sync_ttl
    return Readiness(
        present=True,
        fresh=fresh,
        stale_age=stale_age,
        detail=f"index age {int(stale_age)}s (ttl {int(sync_ttl)}s)",
    )


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

# Match filesystem-looking tokens: start with / ~ . or contain a path separator.
_PATH_TOKEN_RE = re.compile(r"(?:^|(?<=\s))([~/.][\S]*|/\S+)")


def _resolve_project_root(p: Path) -> Path:
    """Walk up from *p* until we find a project-root marker or hit the fs root."""
    _ROOT_MARKERS = frozenset([".git", "pyproject.toml", "package.json",
                                "setup.py", "setup.cfg", "Cargo.toml", "go.mod"])
    candidate = p if p.is_dir() else p.parent
    for parent in [candidate, *candidate.parents]:
        if any((parent / m).exists() for m in _ROOT_MARKERS):
            return parent
    # No marker found — return the original directory.
    return candidate


def resolve_targets(cwd: str, message: str) -> list[Path]:
    """Extract candidate project roots from *message* and *cwd*.

    Rules:
    - Always include *cwd* as a candidate (resolved to its project root).
    - Extract tokens from *message* that look like filesystem paths.
    - Expand ``~`` and resolve to absolute paths.
    - Only include paths that exist on disk (conservative — avoids hallucinated paths).
    - Deduplicate, preserving order (cwd-derived root first).
    """
    try:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def _add(p: Path) -> None:
            root = _resolve_project_root(p)
            if root not in seen:
                seen.add(root)
                candidates.append(root)

        # Always include cwd.
        try:
            cwd_path = Path(cwd).expanduser().resolve()
            if cwd_path.exists():
                _add(cwd_path)
        except Exception as exc:
            logger.debug("resolve_targets: cwd resolution failed: %s", exc)

        # Extract path-like tokens from the message.
        for token in _PATH_TOKEN_RE.findall(message):
            try:
                p = Path(token).expanduser().resolve()
                if p.exists() and p.is_dir():
                    _add(p)
            except Exception:
                pass

        return candidates
    except Exception as exc:
        logger.debug("resolve_targets failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Async dispatch with TTL debounce
# ---------------------------------------------------------------------------

def dispatch_async(actions: list[list[str]]) -> None:
    """Fire-and-forget subprocess dispatch. Never blocks, never raises.

    Each argv list is debounced against ``orchestrator_sync_ttl_secs``: if the
    same command was dispatched within that window it is silently skipped.
    """
    if not actions:
        return

    try:
        from .. import config as _cfg
        sync_ttl = float(_cfg.get("orchestrator_sync_ttl_secs") or _DEFAULT_SYNC_TTL)
    except Exception:
        sync_ttl = _DEFAULT_SYNC_TTL

    now = time.time()
    for argv in actions:
        if not argv:
            continue
        key = tuple(argv)
        last = _last_dispatch.get(key, 0.0)
        if now - last < sync_ttl:
            logger.debug("dispatch_async: debounced %s (ran %ds ago)", argv[0], int(now - last))
            continue
        try:
            subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            _last_dispatch[key] = now
            logger.debug("dispatch_async: launched %s", argv)
        except Exception as exc:
            logger.debug("dispatch_async: Popen failed for %s: %s", argv, exc)


# ---------------------------------------------------------------------------
# Core evaluate
# ---------------------------------------------------------------------------

def evaluate(
    cwd: str,
    message: str,
    *,
    session: dict | None = None,
) -> OrchestratorResult:
    """Evaluate all registered capabilities for the current turn.

    This is the hot-path function: it must never raise, must never block
    (no subprocess.run, no network), and must complete within the tier-1 latency
    budget (~10ms typical).

    Returns an ``OrchestratorResult`` with:
    - ``directive``: text to inject into the additional-context channel (may be "").
    - ``decisions``: one entry per (capability, target) pair evaluated.
    - ``provision_actions``: argv lists collected for async dispatch by the caller.
    """
    try:
        from .. import config as _cfg
        enabled = _cfg.get("orchestrator_enabled")
        if not enabled:
            return OrchestratorResult()
    except Exception:
        pass  # if config is unreachable, proceed anyway

    try:
        return _evaluate_inner(cwd, message, session=session)
    except Exception as exc:
        logger.debug("evaluate: unexpected error, returning empty result: %s", exc)
        return OrchestratorResult()


def _evaluate_inner(
    cwd: str,
    message: str,
    *,
    session: dict | None,
) -> OrchestratorResult:
    # Import registry here (not at module level) to avoid circular imports at
    # package init time; registry imports probe_codegraph from this module.
    from .registry import REGISTRY

    try:
        from .. import config as _cfg
        auto_init_global: bool = bool(_cfg.get("orchestrator_auto_init"))
        auto_init_roots_raw = _cfg.get("orchestrator_auto_init_roots") or []
        probe_cache_ttl = float(_cfg.get("orchestrator_probe_cache_secs") or 60.0)
    except Exception:
        auto_init_global = False
        auto_init_roots_raw = []
        probe_cache_ttl = 60.0

    auto_init_roots: set[Path] = set()
    for r in (auto_init_roots_raw if isinstance(auto_init_roots_raw, list) else []):
        try:
            auto_init_roots.add(Path(r).expanduser().resolve())
        except Exception:
            pass

    targets = resolve_targets(cwd, message)

    directives: list[str] = []
    decisions: list[dict] = []
    provision_actions: list[list[str]] = []

    for cap in REGISTRY:
        for root in targets:
            # Scope check — is this capability applicable here at all?
            try:
                if not cap.scope(root):
                    continue
            except Exception as exc:
                logger.debug("cap %s scope check failed for %s: %s", cap.id, root, exc)
                continue

            # Signal check — does the message indicate this capability is wanted?
            try:
                if not cap.signals(message, root):
                    continue
            except Exception as exc:
                logger.debug("cap %s signal check failed for %s: %s", cap.id, root, exc)
                continue

            # Probe — cached. A probe that returns None (no reading) is treated
            # like a miss: skip the capability rather than emit a directive.
            cache_key = (cap.id, str(root))
            now = time.time()
            readiness: Readiness
            cached = _probe_cache.get(cache_key)
            if cached is not None and now - cached[0] < probe_cache_ttl:
                readiness = cached[1]
            else:
                try:
                    probed = cap.probe(root)
                except Exception as exc:
                    logger.debug("cap %s probe failed for %s: %s", cap.id, root, exc)
                    continue
                if probed is None:
                    continue
                readiness = probed
                _probe_cache[cache_key] = (now, readiness)

            # Plan — directive + provisioning action.
            action_taken = "none"
            directive_text = ""

            if readiness.present:
                # Index exists — always queue a TTL-debounced refresh.
                try:
                    argv = cap.provision_refresh(root)
                    provision_actions.append(argv)
                    action_taken = "refresh_queued"
                except Exception as exc:
                    logger.debug("cap %s provision_refresh failed for %s: %s", cap.id, root, exc)

                try:
                    directive_text = cap.format_directive_ready(root, readiness)
                except Exception as exc:
                    logger.debug("cap %s directive_ready failed for %s: %s", cap.id, root, exc)
            else:
                # Index absent — check autonomy policy.
                may_auto_init = auto_init_global or (root in auto_init_roots)
                if may_auto_init:
                    try:
                        argv = cap.provision_init(root)
                        provision_actions.append(argv)
                        action_taken = "init_queued"
                    except Exception as exc:
                        logger.debug("cap %s provision_init failed for %s: %s", cap.id, root, exc)
                else:
                    # Offer only — surface the directive; human decides.
                    action_taken = "offer"

                try:
                    directive_text = cap.format_directive_missing(root, readiness)
                except Exception as exc:
                    logger.debug("cap %s directive_missing failed for %s: %s", cap.id, root, exc)

            if directive_text:
                directives.append(directive_text)

            decisions.append({
                "target": str(root),
                "capability_id": cap.id,
                "present": readiness.present,
                "fresh": readiness.fresh,
                "stale_age": readiness.stale_age,
                "action": action_taken,
                "detail": readiness.detail,
            })

    return OrchestratorResult(
        directive="\n".join(directives),
        decisions=decisions,
        provision_actions=provision_actions,
    )


# ---------------------------------------------------------------------------
# ensure_tooling_core — explicit / idempotent path
# ---------------------------------------------------------------------------

def ensure_tooling_core(
    path: str,
    *,
    init: bool = False,
    refresh: bool = True,
) -> dict:
    """Probe and optionally provision the code-graph index for *path*.

    This is the explicit, idempotent entry point — safe to call from the
    ``ensure_tooling`` MCP tool or any manual invocation.

    Unlike ``evaluate()``, an init here MAY block briefly (``codegraph init``
    is a one-time, user-requested operation).  Refresh is still fire-and-forget.

    Returns a dict:
        {
            "path":      str,
            "present":   bool,
            "fresh":     bool,
            "action":    "none" | "refresh_dispatched" | "init_run" | "error",
            "directive": str,
        }
    """
    try:
        root = Path(path).expanduser().resolve()
        readiness = probe_codegraph(root)

        action = "none"
        directive = ""

        if readiness.present and refresh:
            from .registry import CODEGRAPH
            try:
                argv = CODEGRAPH.provision_refresh(root)
                dispatch_async([argv])
                action = "refresh_dispatched"
            except Exception as exc:
                logger.debug("ensure_tooling_core: refresh dispatch failed: %s", exc)
                action = "error"

            try:
                directive = CODEGRAPH.format_directive_ready(root, readiness)
            except Exception:
                pass

        elif not readiness.present and init:
            from .registry import CODEGRAPH
            try:
                argv = CODEGRAPH.provision_init(root)
                # Blocking — this is the explicit, user-confirmed path.
                result = subprocess.run(
                    argv,
                    timeout=120,
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    action = "init_run"
                    # Invalidate probe cache so next evaluate() sees the fresh index.
                    _probe_cache.pop(("code-graph", str(root)), None)
                    # Re-probe to get updated readiness.
                    readiness = probe_codegraph(root)
                else:
                    action = "error"
                    logger.debug(
                        "ensure_tooling_core: init exited %d: %s",
                        result.returncode,
                        result.stderr.decode(errors="replace")[:200],
                    )
            except Exception as exc:
                logger.debug("ensure_tooling_core: init failed: %s", exc)
                action = "error"

            try:
                directive = (
                    CODEGRAPH.format_directive_ready(root, readiness)
                    if readiness.present
                    else CODEGRAPH.format_directive_missing(root, readiness)
                )
            except Exception:
                pass

        elif not readiness.present:
            from .registry import CODEGRAPH
            try:
                directive = CODEGRAPH.format_directive_missing(root, readiness)
            except Exception:
                pass

        return {
            "path": str(root),
            "present": readiness.present,
            "fresh": readiness.fresh,
            "action": action,
            "directive": directive,
        }

    except Exception as exc:
        logger.debug("ensure_tooling_core: unexpected error for %s: %s", path, exc)
        return {
            "path": str(path),
            "present": False,
            "fresh": False,
            "action": "error",
            "directive": "",
        }
