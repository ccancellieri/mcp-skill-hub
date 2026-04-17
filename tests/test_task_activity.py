"""Tests for compute_activity_state() module-level helper in store.py."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _now_minus(seconds: float) -> str:
    """Return an ISO timestamp (UTC, no tzinfo — matches SQLite datetime('now'))."""
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)
    # SQLite stores naive UTC; strip tzinfo to mimic store output.
    return dt.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def test_active_within_threshold():
    from skill_hub.store import compute_activity_state
    ts = _now_minus(30)  # 30 seconds ago → within 60s active threshold
    assert compute_activity_state(ts, "open") == "active"


def test_idle_between_thresholds():
    from skill_hub.store import compute_activity_state
    ts = _now_minus(30 * 60)  # 30 minutes ago → within 3600s idle threshold
    assert compute_activity_state(ts, "open") == "idle"


def test_open_beyond_idle_threshold():
    from skill_hub.store import compute_activity_state
    ts = _now_minus(2 * 3600)  # 2 hours ago → beyond 3600s → "open"
    assert compute_activity_state(ts, "open") == "open"


def test_closed_status_overrides_activity():
    from skill_hub.store import compute_activity_state
    # Even with a very recent timestamp, closed status → "closed"
    ts = _now_minus(5)
    assert compute_activity_state(ts, "closed") == "closed"


def test_none_last_activity_returns_open():
    from skill_hub.store import compute_activity_state
    assert compute_activity_state(None, "open") == "open"


def test_invalid_timestamp_returns_open():
    from skill_hub.store import compute_activity_state
    assert compute_activity_state("not-a-date", "open") == "open"


def test_closed_with_none_activity():
    from skill_hub.store import compute_activity_state
    assert compute_activity_state(None, "closed") == "closed"
