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
        worktree_mismatch: *root* is a git worktree with no index of its own, and
                    the only reachable index belongs to a different checkout
                    (found by walking up the directory tree). Querying it would
                    return that other tree's branch, not this worktree's.
        ancestor_index: Path (as str) to the ancestor ``.codegraph/`` that would
                    be (mis)used when ``worktree_mismatch`` is True; else None.
        pending_edits: Seconds between the last source edit (codegraph's
                    ``.dirty`` marker) and the last index build. >0 means the
                    index is behind the working tree and a ``sync`` is due.
                    None when no reliable dirty marker is available.
    """

    present: bool
    fresh: bool
    stale_age: float | None
    detail: str
    worktree_mismatch: bool = False
    ancestor_index: str | None = None
    pending_edits: float | None = None


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

# SQLite database file inside the index directory.
_CODEGRAPH_DB = "codegraph.db"

# Marker file codegraph's file-watch hook rewrites (a millisecond epoch) on every
# source edit. Comparing it against the db build time tells us whether the index
# is behind the working tree — the only freshness signal that survives edits the
# orchestrator never saw.
_CODEGRAPH_DIRTY = ".dirty"

# Default sync TTL in seconds (overridden by config at call sites).
_DEFAULT_SYNC_TTL = 300.0


def _read_dirty_ts(index_dir: Path) -> float | None:
    """Return the last-edit timestamp (epoch seconds) from codegraph's ``.dirty``.

    codegraph writes a millisecond epoch into ``.codegraph/.dirty`` whenever a
    tracked source file changes. We normalise it to seconds. Falls back to the
    marker file's own mtime if the contents are unparseable, and returns ``None``
    when there is no marker at all (older codegraph, or hook not installed) so the
    caller drops back to the age-vs-TTL heuristic.
    """
    dirty = index_dir / _CODEGRAPH_DIRTY
    try:
        if not dirty.exists():
            return None
        raw = dirty.read_text().strip()
        if raw:
            val = float(raw)
            # Heuristic: values past ~2001-in-ms are millisecond epochs.
            return val / 1000.0 if val > 1e12 else val
    except Exception:
        pass
    try:
        return dirty.stat().st_mtime
    except OSError:
        return None


def _is_git_worktree(root: Path) -> bool:
    """True when *root* is a *linked* git worktree (its ``.git`` is a file).

    A primary checkout has a ``.git`` directory; a linked worktree has a ``.git``
    *file* containing a ``gitdir:`` pointer. That distinction is exactly what lets
    us warn when a worktree would silently borrow another checkout's index.
    """
    try:
        return (root / ".git").is_file()
    except OSError:
        return False


def _ancestor_codegraph(root: Path) -> Path | None:
    """Return the nearest ancestor ``.codegraph/`` (with a db), or None.

    codegraph resolves a query by walking up to the first ``.codegraph/`` it
    finds, so a worktree with none of its own silently answers from whichever
    ancestor checkout has one. We reproduce that walk to detect the trap.
    """
    for parent in root.parents:
        cg = parent / _CODEGRAPH_DIR
        try:
            if cg.is_dir() and (cg / _CODEGRAPH_DB).exists():
                return cg
        except OSError:
            continue
    return None


def _codegraph_node_count(index_dir: Path) -> int | None:
    """Return the node count in the code-graph database, cheaply.

    Returns:
        - ``0``    when the index is definitively empty/unusable — the
          ``codegraph.db`` file is missing, or its ``nodes`` table holds no
          rows. This is the "scaffold but never indexed" state a bare
          ``codegraph init`` leaves behind (``init -i`` populates it instead).
        - ``>0``   the confirmed node count.
        - ``None`` when the count cannot be determined cheaply (database locked,
          or an unexpected schema). Callers must treat ``None`` as "unknown" and
          preserve the legacy presence-based behaviour rather than downgrading —
          so a future codegraph schema change can never make the probe nag.

    Read-only and subprocess-free: opens the SQLite file in ``mode=ro`` with a
    sub-second timeout, so it stays within the tier-1 hot-path budget.
    """
    db_path = index_dir / _CODEGRAPH_DB
    if not db_path.exists():
        return 0
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.25)
        try:
            row = con.execute("SELECT count(*) FROM nodes").fetchone()
            return int(row[0]) if row else None
        finally:
            con.close()
    except Exception:
        return None


def probe_codegraph(root: Path) -> Readiness:
    """Probe code-graph index readiness for *root*.

    No subprocess on the hot path. A ``codegraph status`` call is intentionally
    avoided to keep p99 latency within the sub-10ms tier-1 budget. Readiness has
    two cheap components:

    - **presence**: the ``.codegraph/`` directory exists *and* its database holds
      at least one node. A present-but-empty index (bare ``init`` with no
      indexing) is reported as *not present*, so the orchestrator offers to
      (re-)index rather than steering to an empty graph.
    - **freshness**: the ``.codegraph/`` directory mtime against the configured
      sync TTL.
    """
    index_dir = root / _CODEGRAPH_DIR
    if not index_dir.exists():
        # A linked worktree with no index of its own is the silent-wrong-answer
        # trap: codegraph walks up and answers from the ancestor checkout's
        # index, i.e. a different branch. Surface that distinctly so the
        # orchestrator can offer a worktree-local index instead of a bare
        # "not indexed" nudge.
        if _is_git_worktree(root):
            ancestor = _ancestor_codegraph(root)
            if ancestor is not None:
                return Readiness(
                    present=False, fresh=False, stale_age=None,
                    detail=(
                        f"{root} is a git worktree with no index of its own; the "
                        f"reachable index at {ancestor} belongs to a different "
                        f"checkout/branch — queries would read that tree, not this one"
                    ),
                    worktree_mismatch=True,
                    ancestor_index=str(ancestor),
                )
        return Readiness(present=False, fresh=False, stale_age=None,
                         detail=f"{_CODEGRAPH_DIR} not found under {root}")

    # Reject a present-but-empty index. Only a *confirmed* zero count downgrades
    # presence; an unknown count (None) preserves the legacy presence behaviour.
    node_count = _codegraph_node_count(index_dir)
    if node_count == 0:
        return Readiness(
            present=False,
            fresh=False,
            stale_age=None,
            detail=f"{_CODEGRAPH_DIR} present but empty (0 nodes) — needs indexing",
        )

    # "Last indexed" = the db build time, not the directory mtime. The directory
    # mtime moves whenever codegraph touches its sidecar files (.dirty, -wal),
    # so it would read "fresh" while the actual graph is days old. The db file
    # mtime is the honest signal.
    db_path = index_dir / _CODEGRAPH_DB
    try:
        db_mtime = db_path.stat().st_mtime
    except OSError as exc:
        return Readiness(present=True, fresh=False, stale_age=None,
                         detail=f"stat failed: {exc}")

    index_age = time.time() - db_mtime

    try:
        from .. import config as _cfg
        _ttl = _cfg.get("orchestrator_sync_ttl_secs")
        sync_ttl = float(_ttl if _ttl is not None else _DEFAULT_SYNC_TTL)
    except Exception:
        sync_ttl = _DEFAULT_SYNC_TTL

    count_str = f", {node_count} nodes" if node_count else ""

    # Definitive staleness: source edited *after* the index was built. This beats
    # the age heuristic — an index touched 10s ago is still stale if a file
    # changed 5s ago.
    dirty_ts = _read_dirty_ts(index_dir)
    if dirty_ts is not None and dirty_ts > db_mtime + 1.0:
        pending = dirty_ts - db_mtime
        return Readiness(
            present=True,
            fresh=False,
            stale_age=index_age,
            detail=(
                f"index built {int(index_age)}s ago but source edited "
                f"{int(pending)}s later — needs sync{count_str}"
            ),
            pending_edits=pending,
        )

    fresh = index_age <= sync_ttl
    return Readiness(
        present=True,
        fresh=fresh,
        stale_age=index_age,
        detail=f"index age {int(index_age)}s (ttl {int(sync_ttl)}s){count_str}",
        pending_edits=0.0 if dirty_ts is not None else None,
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
        _ttl = _cfg.get("orchestrator_sync_ttl_secs")
        sync_ttl = float(_ttl if _ttl is not None else _DEFAULT_SYNC_TTL)
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
# Mode resolution
# ---------------------------------------------------------------------------

# The four behaviours the orchestrator can be in. ``orchestrator_mode`` is the
# single source of truth; the legacy boolean keys derive a mode only when it is
# unset, so pre-existing configs keep their behaviour.
VALID_MODES = ("off", "offer", "auto", "everywhere")


def resolve_mode(get=None) -> str:
    """Resolve the effective orchestrator mode.

    A valid ``orchestrator_mode`` string wins. When it is absent (``None``) the
    mode is derived from the legacy keys so existing configs are unchanged:

        orchestrator_enabled is False          -> "off"
        orchestrator_auto_init is True          -> "everywhere"
        orchestrator_auto_init_roots non-empty  -> "auto"
        otherwise                               -> "offer"

    *get* is an injectable ``config.get``-style accessor (defaults to the real
    one); the indirection keeps the function trivially testable.
    """
    if get is None:
        from .. import config as _cfg
        get = _cfg.get
    try:
        raw = get("orchestrator_mode")
        if isinstance(raw, str) and raw.lower() in VALID_MODES:
            return raw.lower()
    except Exception:
        pass
    try:
        if not get("orchestrator_enabled"):
            return "off"
        if get("orchestrator_auto_init"):
            return "everywhere"
        roots = get("orchestrator_auto_init_roots") or []
        if isinstance(roots, list) and len(roots) > 0:
            return "auto"
    except Exception:
        pass
    return "offer"


def _root_under_parents(root: Path, parents: set[Path]) -> bool:
    """True when *root* equals or is nested under any folder in *parents*.

    A path-boundary guard prevents ``/work/code`` from matching ``/work/codex``:
    nesting requires the parent to be an actual ancestor directory.
    """
    for parent in parents:
        if root == parent:
            return True
        try:
            root.relative_to(parent)
            return True
        except ValueError:
            continue
    return False


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
        if resolve_mode() == "off":
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
        mode = resolve_mode(_cfg.get)
        auto_init_roots_raw = _cfg.get("orchestrator_auto_init_roots") or []
        _pct = _cfg.get("orchestrator_probe_cache_secs")
        probe_cache_ttl = float(_pct if _pct is not None else 60.0)
    except Exception:
        mode = "offer"
        auto_init_roots_raw = []
        probe_cache_ttl = 60.0

    # "everywhere" auto-inits any eligible project; "auto" only those under a
    # configured parent folder; "offer" never auto-inits.
    auto_init_global = mode == "everywhere"
    auto_init_parents: set[Path] = set()
    if mode == "auto":
        for r in (auto_init_roots_raw if isinstance(auto_init_roots_raw, list) else []):
            try:
                auto_init_parents.add(Path(r).expanduser().resolve())
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
                # Auto-running `sync` is an action, so it follows the same
                # autonomy gate as init: only in "everywhere", or in "auto" for a
                # root under a configured parent. "offer" surfaces a stale index
                # but leaves the sync for the human to run.
                may_auto = auto_init_global or _root_under_parents(root, auto_init_parents)
                if readiness.fresh:
                    action_taken = "none"  # index trustworthy — nothing to do
                elif may_auto:
                    try:
                        argv = cap.provision_refresh(root)
                        if argv:
                            provision_actions.append(argv)
                            action_taken = "refresh_queued"
                        else:
                            action_taken = "tool_unavailable"
                    except Exception as exc:
                        logger.debug("cap %s provision_refresh failed for %s: %s", cap.id, root, exc)
                else:
                    action_taken = "refresh_offer"  # stale; human-initiated sync

                try:
                    directive_text = cap.format_directive_ready(root, readiness)
                except Exception as exc:
                    logger.debug("cap %s directive_ready failed for %s: %s", cap.id, root, exc)
            else:
                # Index absent — check autonomy policy.
                may_auto_init = auto_init_global or _root_under_parents(root, auto_init_parents)
                if may_auto_init:
                    try:
                        argv = cap.provision_init(root)
                        if argv:
                            provision_actions.append(argv)
                            action_taken = "init_queued"
                        else:
                            action_taken = "tool_unavailable"
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
                if argv:
                    dispatch_async([argv])
                    action = "refresh_dispatched"
                else:
                    action = "tool_unavailable"
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
                if not argv:
                    raise RuntimeError("codegraph executable not found in a trusted location")
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
