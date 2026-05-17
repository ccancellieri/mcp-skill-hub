"""Autopilot-lite — overnight loop that drains the claims board.

Pure SQLite + subprocess. No ruflo runtime dependency. The loop:

1. Polls the claims board for entries where ``stealable_at <= now()`` and
   ``claimed_by IS NULL``.
2. Picks the top one by ``priority`` (lower = more urgent), then ``created_at``.
3. Atomically claims it (sets ``claimed_by`` + ``started_at``) and invokes the
   launcher — by default a placeholder that simulates ``swarm_launch`` (issue
   #20 will wire the real subprocess).
4. On launcher exit, marks the claim ``done`` (or ``failed``) and sleeps the
   configured interval before polling again.

Stop signals:
    * SIGINT / SIGTERM         — handled inside ``AutopilotRunner.run()``
    * ``request_stop(db_path)`` — sets ``autopilot_state.stop_requested``;
      the next poll cycle returns cleanly.

Layout:
    claims.py — claims board schema + atomic claim/release helpers
    loop.py   — ``AutopilotRunner`` (the foreground command body)
"""
from __future__ import annotations

from .claims import (
    Claim,
    claim_next,
    ensure_schema,
    insert_claim,
    list_claims,
    mark_done,
    mark_failed,
)
from .loop import AutopilotRunner, request_stop, run_autopilot

__all__ = [
    "AutopilotRunner",
    "Claim",
    "claim_next",
    "ensure_schema",
    "insert_claim",
    "list_claims",
    "mark_done",
    "mark_failed",
    "request_stop",
    "run_autopilot",
]
