"""Session memory compaction — ported from
`anthropics/claude-cookbooks/misc/session_memory_compaction.ipynb`.

Builds a structured 6-section summary of a Claude Code session so the
conversation can resume cleanly after ``/compact`` or a context reset. The
summary persists to ``~/.claude/mcp-skill-hub/session-memory/<session_id>.md``
and is injected as ``systemMessage`` by ``session_start_enforcer`` on the
first prompt of a resumed session.

The build runs in a daemon thread from the Stop hook so the user never waits.
Concurrent builds for the same session are suppressed via a per-session lock.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config as _cfg
from ..llm import LLMError, get_provider, load_prompt

_log = logging.getLogger(__name__)


def memory_dir() -> Path:
    return Path.home() / ".claude" / "mcp-skill-hub" / "session-memory"


def memory_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return memory_dir() / f"{safe or 'unknown'}.md"


# ---------------------------------------------------------------------------
# Build / update
# ---------------------------------------------------------------------------

def _render_prompt(transcript: str, previous_memory: str = "") -> str:
    """Format the session_memory YAML template with the given inputs."""
    prev_block = (
        "Previous session memory (refresh, don't restart):\n---\n"
        f"{previous_memory}\n---\n"
        if previous_memory.strip()
        else ""
    )
    tpl = load_prompt("session_memory").template
    # Use manual replace — .format would choke on literal braces elsewhere.
    return tpl.replace("{previous_memory_block}", prev_block).replace(
        "{transcript}", transcript
    )


def build_session_memory(
    transcript: str,
    *,
    previous_memory: str = "",
    tier: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Synchronously generate a session memory from a transcript blob.

    Raises ``LLMError`` on provider failure.
    """
    if not transcript.strip():
        raise LLMError("empty transcript")
    cfg = _cfg.load_config()
    resolved_tier = tier or str(cfg.get("session_memory_tier") or "tier_mid")
    prompt = _render_prompt(transcript, previous_memory)
    text = get_provider().complete(
        prompt,
        tier=resolved_tier,
        max_tokens=max_tokens,
        temperature=0.1,
        timeout=60.0,
        cache=True,  # ephemeral cache_control on Anthropic tiers
    )
    return text.strip()


def update_session_memory(
    previous_memory: str,
    new_messages_transcript: str,
    *,
    tier: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Incremental refresh — send only the untyped tail, keep the prior summary.

    Calls the same prompt with the previous memory in context; the model is
    instructed to refresh (not restart). Falls back to full rebuild if the
    previous memory is empty.
    """
    if not previous_memory.strip():
        return build_session_memory(
            new_messages_transcript, tier=tier, max_tokens=max_tokens
        )
    cfg = _cfg.load_config()
    resolved_tier = tier or str(cfg.get("session_memory_tier") or "tier_cheap")
    prompt = _render_prompt(new_messages_transcript, previous_memory)
    text = get_provider().complete(
        prompt,
        tier=resolved_tier,
        max_tokens=max_tokens,
        temperature=0.1,
        timeout=60.0,
        cache=True,  # ephemeral cache_control on Anthropic tiers
    )
    return text.strip()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def read_memory(session_id: str) -> str:
    p = memory_path(session_id)
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_memory(session_id: str, text: str) -> Path:
    p = memory_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Background thread orchestration
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
# session_id -> Thread (only the most recent build per session)
_ACTIVE_BUILDS: dict[str, threading.Thread] = {}


@dataclass
class BuildOutcome:
    scheduled: bool
    reason: str = ""


def schedule_build(
    session_id: str,
    transcript_provider: "callable[[], str]",
    *,
    incremental: bool = True,
) -> BuildOutcome:
    """Launch a daemon thread that builds/refreshes memory for ``session_id``.

    ``transcript_provider`` is a zero-arg callable invoked inside the thread so
    transcript reads don't block the Stop hook. If a build for the same session
    is already running, this call is a no-op.
    """
    if not _is_enabled():
        return BuildOutcome(False, "disabled")
    if not session_id:
        return BuildOutcome(False, "no session_id")

    with _STATE_LOCK:
        existing = _ACTIVE_BUILDS.get(session_id)
        if existing and existing.is_alive():
            return BuildOutcome(False, "build already running")
        t = threading.Thread(
            target=_run_build,
            args=(session_id, transcript_provider, incremental),
            name=f"session-memory-{session_id[:8]}",
            daemon=True,
        )
        _ACTIVE_BUILDS[session_id] = t
        t.start()
    return BuildOutcome(True, "scheduled")


def _is_enabled() -> bool:
    env = os.environ.get("SKILL_HUB_SESSION_MEMORY", "")
    if env == "0":
        return False
    if env == "1":
        return True
    return bool(_cfg.get("session_memory_enabled"))


def _run_build(
    session_id: str,
    transcript_provider: "callable[[], str]",
    incremental: bool,
) -> None:
    try:
        transcript = transcript_provider() or ""
    except Exception as exc:  # noqa: BLE001
        _log.debug("session_memory: transcript_provider failed: %s", exc)
        return
    if not transcript.strip():
        _log.debug("session_memory: empty transcript, skipping")
        return
    previous = read_memory(session_id) if incremental else ""
    try:
        if previous:
            new_memory = update_session_memory(previous, transcript)
        else:
            new_memory = build_session_memory(transcript)
    except LLMError as exc:
        _log.debug("session_memory: build failed (%s)", exc)
        return
    except Exception as exc:  # noqa: BLE001
        _log.debug("session_memory: unexpected failure: %s", exc)
        return
    if not new_memory:
        return
    try:
        write_memory(session_id, new_memory)
    except OSError as exc:
        _log.debug("session_memory: write failed: %s", exc)


def wait_for_active_builds(timeout: float = 5.0) -> None:
    """Join all active build threads (used in tests)."""
    with _STATE_LOCK:
        threads = list(_ACTIVE_BUILDS.values())
    for t in threads:
        t.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Transcript utilities
# ---------------------------------------------------------------------------

def read_transcript_tail(
    transcript_path: str | Path,
    *,
    max_bytes: int | None = None,
) -> str:
    """Read the tail of a Claude Code transcript JSONL file as plain text.

    ``max_bytes=None`` reads the full file. For long sessions the caller
    should supply ``session_memory_max_transcript_bytes`` (default 200_000).
    """
    p = Path(transcript_path)
    if not p.is_file():
        return ""
    cap = max_bytes
    if cap is None:
        cap = int(_cfg.get("session_memory_max_transcript_bytes") or 200_000)
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if cap and size > cap:
                f.seek(size - cap)
                # Skip partial line at the truncation point.
                f.readline()
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return raw


__all__ = [
    "BuildOutcome",
    "build_session_memory",
    "memory_dir",
    "memory_path",
    "read_memory",
    "read_transcript_tail",
    "schedule_build",
    "update_session_memory",
    "wait_for_active_builds",
    "write_memory",
]
