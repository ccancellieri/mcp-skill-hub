"""Swarm-lite: launch N Claude subprocesses, each on a distinct worktree+claim.

Provides the "swarm" capability without any runtime dependency on
``claude-flow`` / ``ruflo``. Each claim is mapped to one ``subprocess.Popen``
invocation of the ``claude`` CLI, with ``cwd`` set to the claim's worktree
path and an initial prompt that includes the claim's task summary.

Lifecycle
---------
1. ``swarm_launch(claims, ...)`` -> list[SwarmHandle], one per claim.
   * Each handle records the OS PID and per-claim log file
     (``<swarm_dir>/<group_id>/<claim_id>.log``).
   * An optional ``status_callback(claim_id, "running")`` is invoked once the
     subprocess is up; callers (e.g. the claims board) can use this to
     transition claim status from ``claimed`` -> ``running``.

2. ``swarm_reap(handles, ...)`` -> list[SwarmHandle] with updated state.
   * Polls each child; for finished children captures the exit code and
     transitions status to ``done`` (rc == 0) or ``failed`` (rc != 0).
   * Invokes ``status_callback(claim_id, "done"|"failed", returncode=rc)``.

Design constraints (from issue m4-#20)
--------------------------------------
* **No-ruflo-dep**: this module never imports ``claude_flow`` / ``ruflo`` and
  pyproject.toml carries no such dependency.
* **Pure subprocess.Popen**: no threads, no asyncio. The reaper is a plain
  poll loop the caller drives at the cadence they want.
* **Claim semantics live elsewhere**: the claims board (issue m1-#9) is the
  source of truth for claim_id / worktree_path / task_summary. This module
  just consumes a list of ``Claim`` records and reports status back via
  ``status_callback``, so it is forward-compatible with whatever claims
  storage layer ships.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

# Log root for swarm subprocesses. Matches the wider project convention
# (~/.claude/mcp-skill-hub/...); the issue body uses "~/.skill_hub/swarm/"
# colloquially but every other state path in this repo lives under
# ~/.claude/mcp-skill-hub/, so we keep it consistent here.
SWARM_LOG_DIR: Path = Path.home() / ".claude" / "mcp-skill-hub" / "swarm"


SwarmStatus = Literal["pending", "running", "done", "failed", "lost"]


class SwarmError(RuntimeError):
    """Raised for swarm lifecycle problems (missing binary, bad claim, etc.)."""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """One unit of work the swarm subprocess will pick up.

    * ``claim_id`` is the stable identifier from the claims board (m1-#9).
      When the claims board lands, callers will pass the row's primary key
      here; until then any unique string works (issue numbers, uuids, ...).
    * ``worktree_path`` is the absolute path to the isolated worktree
      created by ``skill_hub.worktree.ensure_worktree`` (m1-#6).
    * ``task_summary`` is a short human-readable description of the work,
      injected into the initial prompt so the spawned Claude has context.
    * ``prompt`` overrides the default prompt template if set.
    """
    claim_id: str
    worktree_path: str
    task_summary: str = ""
    prompt: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SwarmHandle:
    """Live state of one swarm subprocess. Mutated by ``swarm_reap``."""
    claim_id: str
    worktree_path: str
    pid: int
    log_path: str
    status: SwarmStatus = "running"
    returncode: Optional[int] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


StatusCallback = Callable[..., None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def swarm_launch(
    claims: Sequence[Claim | dict],
    *,
    group_id: Optional[str] = None,
    claude_binary: Optional[str] = None,
    log_dir: Optional[Path | str] = None,
    status_callback: Optional[StatusCallback] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> list[SwarmHandle]:
    """Spawn one Claude subprocess per claim. Returns one handle per claim.

    Parameters
    ----------
    claims:
        Iterable of ``Claim`` records (or equivalent dicts with the same
        keys). Each claim must carry a non-empty ``claim_id`` and an
        existing ``worktree_path``.
    group_id:
        Optional grouping label used as a sub-directory under ``log_dir`` so
        callers can tear down logs for a batch by removing one directory.
        Auto-generated if omitted.
    claude_binary:
        Path to the ``claude`` CLI. Defaults to ``shutil.which("claude")``.
        Raises ``SwarmError`` if the binary is not found.
    log_dir:
        Override the swarm log root. Defaults to :data:`SWARM_LOG_DIR`.
    status_callback:
        Optional ``callback(claim_id, status, **extras)`` invoked with
        ``status="running"`` once each subprocess is launched. The claims
        board (m1-#9) can pass its own transition function here.
    extra_args:
        Extra CLI args appended after ``--print`` (and before the prompt).
        Mostly useful for testing with a dummy binary.
    """
    resolved_claims = [_coerce_claim(c) for c in claims]
    if not resolved_claims:
        raise SwarmError("swarm_launch: claims is empty")

    binary = claude_binary or _resolve_claude_binary()
    root = Path(log_dir) if log_dir is not None else SWARM_LOG_DIR
    gid = group_id or f"swarm-{uuid.uuid4().hex[:8]}"
    group_dir = root / gid
    group_dir.mkdir(parents=True, exist_ok=True)

    handles: list[SwarmHandle] = []
    seen_claim_ids: set[str] = set()
    for claim in resolved_claims:
        if claim.claim_id in seen_claim_ids:
            raise SwarmError(f"duplicate claim_id in batch: {claim.claim_id!r}")
        seen_claim_ids.add(claim.claim_id)

        wt = Path(claim.worktree_path)
        if not wt.is_dir():
            raise SwarmError(
                f"worktree_path does not exist for claim {claim.claim_id!r}: {wt}"
            )
        log_path = group_dir / f"{_safe_id(claim.claim_id)}.log"
        prompt = claim.prompt or _default_prompt(claim)
        args = [binary, "--print", *list(extra_args or []), prompt]

        log_fp = open(log_path, "a", buffering=1)
        log_fp.write(
            f"\n=== swarm launch claim={claim.claim_id} "
            f"at {datetime.now(timezone.utc).isoformat()} ===\n"
        )
        try:
            proc = subprocess.Popen(  # noqa: S603 -- claude binary, typed args
                args,
                cwd=str(wt),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:  # pragma: no cover -- depends on host PATH
            log_fp.close()
            raise SwarmError(
                f"failed to spawn claude for claim {claim.claim_id!r}: {exc}"
            ) from exc

        handle = SwarmHandle(
            claim_id=claim.claim_id,
            worktree_path=str(wt),
            pid=proc.pid,
            log_path=str(log_path),
            status="running",
        )
        handles.append(handle)
        if status_callback is not None:
            try:
                status_callback(claim.claim_id, "running", pid=proc.pid)
            except Exception:  # noqa: BLE001 -- callbacks must not break launch
                pass

    return handles


def swarm_reap(
    handles: Sequence[SwarmHandle],
    *,
    timeout: Optional[float] = None,
    poll_interval: float = 0.1,
    status_callback: Optional[StatusCallback] = None,
) -> list[SwarmHandle]:
    """Poll each handle; for finished children record exit code + status.

    Mutates the handles in-place AND returns them so callers can chain.

    Parameters
    ----------
    handles:
        Handles returned by :func:`swarm_launch`.
    timeout:
        Max seconds to wait for *all* children. ``None`` means a single
        non-blocking sweep. ``0`` is equivalent to a single sweep too.
    poll_interval:
        Seconds between sweeps when waiting.
    status_callback:
        Optional ``callback(claim_id, status, returncode=int)`` invoked once
        per child that transitions out of ``running`` during this call.
    """
    deadline: Optional[float] = None
    if timeout is not None and timeout > 0:
        deadline = time.monotonic() + timeout

    while True:
        any_running = False
        for handle in handles:
            if handle.status != "running":
                continue
            rc = _poll_pid(handle.pid)
            if rc is None:
                any_running = True
                continue
            handle.returncode = rc
            handle.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            handle.status = "done" if rc == 0 else "failed"
            if status_callback is not None:
                try:
                    status_callback(handle.claim_id, handle.status, returncode=rc)
                except Exception:  # noqa: BLE001
                    pass

        if not any_running:
            return list(handles)
        if deadline is None:
            return list(handles)
        if time.monotonic() >= deadline:
            return list(handles)
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_claim(c: Claim | dict) -> Claim:
    if isinstance(c, Claim):
        claim = c
    elif isinstance(c, dict):
        if "claim_id" not in c or "worktree_path" not in c:
            raise SwarmError(
                "claim dict must contain 'claim_id' and 'worktree_path' "
                f"(got keys: {sorted(c)})"
            )
        claim = Claim(
            claim_id=str(c["claim_id"]),
            worktree_path=str(c["worktree_path"]),
            task_summary=str(c.get("task_summary", "")),
            prompt=c.get("prompt"),
        )
    else:
        raise SwarmError(
            f"unsupported claim type: {type(c).__name__} (expected Claim or dict)"
        )
    if not claim.claim_id:
        raise SwarmError("claim_id must be non-empty")
    if not claim.worktree_path:
        raise SwarmError(f"worktree_path must be non-empty for {claim.claim_id!r}")
    return claim


def _resolve_claude_binary() -> str:
    found = shutil.which("claude")
    if not found:
        raise SwarmError(
            "`claude` CLI not found on PATH. Install Claude Code or pass "
            "claude_binary=<path>."
        )
    return found


def _default_prompt(claim: Claim) -> str:
    """Build the initial prompt injected into the spawned Claude session."""
    summary = (claim.task_summary or "(no task summary provided)").strip()
    return (
        f"You are working on claim {claim.claim_id} in an isolated worktree "
        f"at {claim.worktree_path}.\n\n"
        f"Task summary:\n{summary}\n\n"
        "Stay inside this worktree. When done, summarize the diff and stop."
    )


def _safe_id(claim_id: str) -> str:
    """Filesystem-safe form of a claim_id for log filenames."""
    keep = []
    for ch in claim_id:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep) or "claim"


def _poll_pid(pid: int) -> Optional[int]:
    """Non-blocking ``waitpid(pid, WNOHANG)``. Returns exit code or None.

    Mirrors ``Popen.poll`` but works on bare PIDs (so callers can persist
    just the integer pid between processes if they want to).
    """
    if pid <= 0:
        return None
    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        # Not our child (e.g. handle restored from disk). Fall back to
        # signal-0 liveness check.
        return None if _process_alive(pid) else 0
    except OSError:
        return None
    if wpid == 0:
        return None
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        # Encode signal as 128+sig so it round-trips through "non-zero exit".
        return 128 + os.WTERMSIG(status)
    return 1


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
