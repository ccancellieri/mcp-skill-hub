"""Per-session routing-verdict cache (issue #88).

The router runs as a per-prompt hook — a FRESH PROCESS each turn — so an
in-memory dict cannot persist between calls.  This module persists the
cache-stable skills/plugins block to disk, one JSON file per session, under:

    ~/.claude/mcp-skill-hub/verdict-cache/<session_id>.json

Entry schema
------------
{
  "stable_block": "<rendered skills+plugins text>",
  "domain_hints": ["hint1", ...],
  "plan_mode":    false,
  "msg_count":    7,
  "ts":           1700000000.0
}

Reuse rule (ALL of the following must hold)
-------------------------------------------
1. An entry exists on disk.
2. plan_mode is unchanged.
3. hard_switch is False  (an applied enforcement must re-evaluate immediately).
4. Domain-hint overlap: the CURRENT domain set is a subset of, or equal to,
   the CACHED domain set.  New domains entering the session mean different
   skills might be relevant — recompute.  Domains shrinking or staying the
   same are fine to reuse (the cached block was computed for a superset).
5. current_msg_count - entry.msg_count < router_verdict_cache_max_messages.
6. time.time() - entry.ts < router_verdict_cache_ttl_secs.

All disk I/O is guarded; never raises.  Cache is a pure optimisation —
correctness does not depend on it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


def cache_dir() -> Path:
    return Path.home() / ".claude" / "mcp-skill-hub" / "verdict-cache"


def cache_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return cache_dir() / f"{safe or 'unknown'}.json"


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def _load(session_id: str) -> dict[str, Any] | None:
    """Return the cached entry dict, or None on any error / missing file."""
    try:
        p = cache_path(session_id)
        if not p.is_file():
            return None
        raw = p.read_text(encoding="utf-8")
        entry = json.loads(raw)
        if not isinstance(entry, dict):
            return None
        # Minimal schema check — must have all required fields.
        required = {"stable_block", "domain_hints", "plan_mode", "msg_count", "ts"}
        if not required.issubset(entry.keys()):
            return None
        return entry
    except Exception:  # noqa: BLE001
        return None


def _save(session_id: str, entry: dict[str, Any]) -> None:
    """Persist an entry; silently swallows all errors."""
    try:
        p = cache_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(entry), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Reuse decision
# ---------------------------------------------------------------------------

def _should_reuse(
    entry: dict[str, Any],
    *,
    current_domain_hints: list[str],
    current_plan_mode: bool,
    current_msg_count: int,
    hard_switch: bool,
    max_messages: int,
    ttl_secs: float,
) -> bool:
    """Return True when the cached block can be reused unchanged.

    See module docstring for the full reuse rule.
    """
    # Rule 2: plan_mode unchanged.
    if entry.get("plan_mode") != current_plan_mode:
        return False

    # Rule 3: no applied enforcement — an applied switch means the model just
    # changed; the skills block might also need to change.
    if hard_switch:
        return False

    # Rule 4: current domain set must be a subset of (or equal to) the cached
    # set.  New domains = new skills might be needed → recompute.
    cached_domains: set[str] = set(entry.get("domain_hints") or [])
    if not set(current_domain_hints).issubset(cached_domains):
        return False

    # Rule 5: message-count staleness guard.
    try:
        age_msgs = int(current_msg_count) - int(entry.get("msg_count", 0))
    except (TypeError, ValueError):
        return False
    if age_msgs >= max_messages:
        return False

    # Rule 6: wall-clock TTL.
    try:
        age_secs = time.time() - float(entry.get("ts", 0.0))
    except (TypeError, ValueError):
        return False
    if age_secs >= ttl_secs:
        return False

    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_or_build_stable_block(
    session_id: str,
    *,
    current_domain_hints: list[str],
    current_plan_mode: bool,
    current_msg_count: int,
    hard_switch: bool,
    build_fn: Callable[[], str],
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return the stable skills/plugins block, from cache or freshly built.

    Parameters
    ----------
    session_id:
        The Claude Code session ID.  Empty string → caching is skipped.
    current_domain_hints:
        Domain hints from this turn's classification.
    current_plan_mode:
        Whether plan mode is active this turn.
    current_msg_count:
        Message counter value for this turn (from preloader.increment_message_counter).
    hard_switch:
        True when enforcement action is "hard_switch" (model just changed).
    build_fn:
        Zero-argument callable that renders the stable block fresh.  Called
        when the cache is cold, stale, or disabled.
    cfg:
        Already-loaded config dict.  When None the function loads it itself.
        Passing the caller's already-loaded dict avoids a redundant disk read.

    Returns
    -------
    str
        The stable block text (may be empty string when no skills are loaded).
    """
    # Lazy-import config to avoid circular imports at module load time.
    if cfg is None:
        try:
            from .. import config as _cfg
            cfg = _cfg.load_config()
        except Exception:  # noqa: BLE001
            return build_fn()

    enabled: bool = bool(cfg.get("router_verdict_cache_enabled", False))
    if not enabled or not session_id:
        return build_fn()

    max_messages: int = int(cfg.get("router_verdict_cache_max_messages", 20))
    ttl_secs: float = float(cfg.get("router_verdict_cache_ttl_secs", 1800))

    try:
        entry = _load(session_id)
        if entry is not None and _should_reuse(
            entry,
            current_domain_hints=current_domain_hints,
            current_plan_mode=current_plan_mode,
            current_msg_count=current_msg_count,
            hard_switch=hard_switch,
            max_messages=max_messages,
            ttl_secs=ttl_secs,
        ):
            return str(entry["stable_block"])

        # Build a fresh block and persist it.
        block = build_fn()
        _save(
            session_id,
            {
                "stable_block": block,
                "domain_hints": list(current_domain_hints),
                "plan_mode": current_plan_mode,
                "msg_count": current_msg_count,
                "ts": time.time(),
            },
        )
        return block
    except Exception:  # noqa: BLE001
        # Safety net: never let cache errors surface to the route path.
        return build_fn()
