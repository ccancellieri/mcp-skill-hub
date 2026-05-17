"""The autopilot foreground loop.

``AutopilotRunner.run()`` is the body of the ``autopilot_run`` command. It
drains the claims board one entry at a time:

    while not stopped:
        c = claim_next(db, runner_id)
        if c is None:
            sleep(poll_interval)
            continue
        try:
            launcher(c)
        except Exception as exc:
            mark_failed(db, c.id, str(exc))
        else:
            mark_done(db, c.id)

The launcher is pluggable so issue #20 (swarm-lite) can drop in the real
``subprocess.Popen`` call without changing this module. The default launcher
just logs and returns — useful for the synthetic-queue acceptance test in
``tests/autopilot/test_loop.py``.

Stopping
--------
* SIGINT / SIGTERM trigger an in-process flag.
* ``autopilot_stop`` flips ``autopilot_state.stop_requested`` in SQLite; the
  next poll cycle sees the flag and returns cleanly.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import claims as _claims
from .claims import Claim

_log = logging.getLogger(__name__)

# A launcher takes a Claim and runs it to completion. It may raise — the
# runner catches and marks the claim failed.
Launcher = Callable[[Claim], None]


def default_launcher(claim: Claim) -> None:
    """Placeholder until issue #20 wires the real swarm_launch subprocess.

    Treats ``payload["cmd"]`` (a list of argv) as a shell command if present,
    otherwise just logs the claim. Either way the call is synchronous — the
    autopilot loop blocks until the subprocess exits, then marks the claim
    done. This mirrors the ruflo autopilot's "one claim at a time" semantics
    and keeps the loop trivially testable.
    """
    cmd = claim.payload.get("cmd") if isinstance(claim.payload, dict) else None
    if isinstance(cmd, list) and cmd:
        _log.info("autopilot: launching claim #%d cmd=%r", claim.id, cmd)
        # check=True so non-zero exits become CalledProcessError -> mark_failed.
        subprocess.run(cmd, check=True)
    else:
        _log.info("autopilot: claim #%d has no cmd payload — noop", claim.id)


@dataclass
class RunResult:
    runner_id: str
    drained: int = 0
    failed: int = 0
    claim_ids: list[int] = field(default_factory=list)
    stopped_by: str = ""  # "signal" | "stop_requested" | "max_claims" | "empty"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class AutopilotRunner:
    """Foreground loop. One instance per process.

    Parameters
    ----------
    db_path:        SQLite file backing the claims board.
    runner_id:      Identity recorded on each claim (for audit / stop).
    poll_interval:  Seconds to sleep when the queue is empty.
    max_claims:     Drain at most N claims then return ("0" = unbounded).
    drain_and_exit: If True, exit as soon as the queue is empty once.
    launcher:       Callable that runs one claim. See ``default_launcher``.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        runner_id: str = "",
        poll_interval: float = 5.0,
        max_claims: int = 0,
        drain_and_exit: bool = False,
        launcher: Launcher | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.runner_id = runner_id or f"autopilot-{uuid.uuid4().hex[:8]}"
        self.poll_interval = max(0.01, float(poll_interval))
        self.max_claims = int(max_claims)
        self.drain_and_exit = bool(drain_and_exit)
        self.launcher = launcher or default_launcher
        self._stop_flag = threading.Event()
        self._sleeper = threading.Event()  # interruptible sleep

    # ----- public control surface -----

    def request_stop(self, reason: str = "stop_requested") -> None:
        """Wake the loop and exit at the next safe point."""
        self._stop_reason = reason
        self._stop_flag.set()
        self._sleeper.set()

    def _install_signal_handlers(self) -> bool:
        """Return True iff handlers were installed (only works on main thread)."""
        if threading.current_thread() is not threading.main_thread():
            return False
        def _handler(signum, _frame):
            name = signal.Signals(signum).name if signum else "signal"
            _log.info("autopilot: received %s — exiting after current claim", name)
            self.request_stop(reason="signal")
        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
            return True
        except (ValueError, OSError):
            # Embedded contexts (pytest, threads) may forbid signal install.
            return False

    # ----- main loop -----

    def run(self) -> RunResult:
        _claims.upsert_runner(self.db_path, self.runner_id)
        self._install_signal_handlers()

        result = RunResult(runner_id=self.runner_id)
        self._stop_reason = ""

        while not self._stop_flag.is_set():
            # Cooperative stop signal from autopilot_stop (other process).
            if _claims.is_stop_requested(self.db_path, self.runner_id):
                self._stop_reason = self._stop_reason or "stop_requested"
                break

            claim = _claims.claim_next(self.db_path, self.runner_id)
            if claim is None:
                if self.drain_and_exit:
                    self._stop_reason = self._stop_reason or "empty"
                    break
                # Sleep, but break early if anyone calls request_stop().
                self._sleeper.wait(timeout=self.poll_interval)
                self._sleeper.clear()
                continue

            _claims.heartbeat(self.db_path, self.runner_id, claim.id)
            try:
                self.launcher(claim)
            except BaseException as exc:  # noqa: BLE001 — capture *anything* the launcher raises
                _claims.mark_failed(self.db_path, claim.id, repr(exc))
                result.failed += 1
                # KeyboardInterrupt bubbled out of the launcher should still stop us cleanly.
                if isinstance(exc, KeyboardInterrupt):
                    self._stop_reason = "signal"
                    break
            else:
                _claims.mark_done(self.db_path, claim.id)

            result.drained += 1
            result.claim_ids.append(claim.id)

            if self.max_claims and result.drained >= self.max_claims:
                self._stop_reason = "max_claims"
                break

        result.stopped_by = self._stop_reason or "stop_flag"
        return result


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def run_autopilot(
    db_path: str | Path,
    *,
    runner_id: str = "",
    poll_interval: float = 5.0,
    max_claims: int = 0,
    drain_and_exit: bool = False,
    launcher: Launcher | None = None,
) -> RunResult:
    """One-shot helper — construct an ``AutopilotRunner`` and call ``run()``."""
    runner = AutopilotRunner(
        db_path,
        runner_id=runner_id,
        poll_interval=poll_interval,
        max_claims=max_claims,
        drain_and_exit=drain_and_exit,
        launcher=launcher,
    )
    return runner.run()


def request_stop(db_path: str | Path, runner_id: str = "") -> int:
    """Signal the autopilot loop to exit at its next safe checkpoint.

    Returns the number of runners marked for stop. ``runner_id=""`` stops all.
    """
    return _claims.request_runner_stop(db_path, runner_id)
