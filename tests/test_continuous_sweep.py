"""Tests for skill_hub.continuous_sweep.

Covers:
  (a) flag OFF  → sweep never invokes the optimize/promote path
  (b) flag ON + IDLE + interval elapsed → promote called once
  (c) flag ON + pressure NOT idle → skipped
  (d) flag ON + interval NOT elapsed (recent run) → skipped

All tests are hermetic: config is isolated, pressure snapshot is mocked,
and the promote callable is replaced with a spy — no real optimization or
LLM calls occur.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(pressure_name: str):
    """Build a fake SystemSnapshot with the given pressure level."""
    from skill_hub.resource_monitor import Pressure, SystemSnapshot
    p = Pressure[pressure_name]
    return SystemSnapshot(
        pressure=p,
        cpu_load_1m=0.0,
        memory_used_pct=0.0,
        memory_available_mb=16000,
        total_memory_mb=16000,
        timestamp=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# (a) flag OFF → promote never called
# ---------------------------------------------------------------------------


def test_sweep_disabled_never_calls_promote(tmp_path: Path, monkeypatch) -> None:
    """When continuous_sweep_enabled is False the sweep is a no-op."""
    from skill_hub import continuous_sweep as cs

    state_file = tmp_path / "sweep.json"
    promote_spy = MagicMock()

    with patch("skill_hub.continuous_sweep._cfg") as mock_cfg:
        mock_cfg.get.return_value = False  # disabled for every key

        cs._run_sweep(
            promote_fn=promote_spy,
            state_file=state_file,
            _interval_minutes=60,
            _reschedule=False,
        )

    promote_spy.assert_not_called()
    assert not state_file.exists(), "state file should not be written when disabled"


# ---------------------------------------------------------------------------
# (b) flag ON + IDLE + interval elapsed → promote called once
# ---------------------------------------------------------------------------


def test_sweep_runs_when_idle_and_interval_elapsed(tmp_path: Path) -> None:
    """promote_fn is called when flag is on, machine is IDLE, and interval has passed."""
    from skill_hub import continuous_sweep as cs

    state_file = tmp_path / "sweep.json"
    # Write a last_run timestamp that is older than the interval.
    old_ts = time.time() - 7200  # 2 hours ago
    cs._write_last_run(old_ts, state_file)

    promote_spy = MagicMock(return_value='{"dry_run": false, "count": 0, "actions": []}')

    idle_snap = _make_snapshot("IDLE")

    with patch("skill_hub.continuous_sweep._cfg") as mock_cfg, \
         patch("skill_hub.continuous_sweep._snapshot", return_value=idle_snap):

        def cfg_get(key):
            return {
                "continuous_sweep_enabled": True,
                "continuous_sweep_interval_minutes": 60,
            }.get(key, None)

        mock_cfg.get.side_effect = cfg_get

        cs._run_sweep(
            promote_fn=promote_spy,
            state_file=state_file,
            _interval_minutes=60,
            _reschedule=False,
        )

    promote_spy.assert_called_once_with(dry_run=False)

    # Verify the state file was updated.
    written = json.loads(state_file.read_text())
    assert written["last_run"] > old_ts


# ---------------------------------------------------------------------------
# (c) flag ON + pressure NOT IDLE → skipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pressure", ["LOW", "MODERATE", "HIGH"])
def test_sweep_skipped_under_pressure(pressure: str, tmp_path: Path) -> None:
    """promote_fn is NOT called when machine pressure is above IDLE."""
    from skill_hub import continuous_sweep as cs

    state_file = tmp_path / "sweep.json"
    # No previous run → interval definitely elapsed.
    old_ts = time.time() - 7200
    cs._write_last_run(old_ts, state_file)

    promote_spy = MagicMock()
    loaded_snap = _make_snapshot(pressure)

    with patch("skill_hub.continuous_sweep._cfg") as mock_cfg, \
         patch("skill_hub.continuous_sweep._snapshot", return_value=loaded_snap):

        def cfg_get(key):
            return {
                "continuous_sweep_enabled": True,
                "continuous_sweep_interval_minutes": 60,
            }.get(key, None)

        mock_cfg.get.side_effect = cfg_get

        cs._run_sweep(
            promote_fn=promote_spy,
            state_file=state_file,
            _interval_minutes=60,
            _reschedule=False,
        )

    promote_spy.assert_not_called()

    # State file must NOT have been updated (no completed sweep).
    written = json.loads(state_file.read_text())
    assert written["last_run"] == pytest.approx(old_ts, abs=1.0)


# ---------------------------------------------------------------------------
# (d) flag ON + interval NOT elapsed → skipped
# ---------------------------------------------------------------------------


def test_sweep_skipped_when_interval_not_elapsed(tmp_path: Path) -> None:
    """promote_fn is NOT called when the interval has not yet passed."""
    from skill_hub import continuous_sweep as cs

    state_file = tmp_path / "sweep.json"
    # Write a very recent last_run.
    recent_ts = time.time() - 60  # only 1 minute ago
    cs._write_last_run(recent_ts, state_file)

    promote_spy = MagicMock()
    # Even if idle, the interval guard should fire first.
    idle_snap = _make_snapshot("IDLE")

    with patch("skill_hub.continuous_sweep._cfg") as mock_cfg, \
         patch("skill_hub.continuous_sweep._snapshot", return_value=idle_snap):

        def cfg_get(key):
            return {
                "continuous_sweep_enabled": True,
                "continuous_sweep_interval_minutes": 60,  # 60-minute interval
            }.get(key, None)

        mock_cfg.get.side_effect = cfg_get

        cs._run_sweep(
            promote_fn=promote_spy,
            state_file=state_file,
            _interval_minutes=60,
            _reschedule=False,
        )

    promote_spy.assert_not_called()

    # State file should not have been updated.
    written = json.loads(state_file.read_text())
    assert written["last_run"] == pytest.approx(recent_ts, abs=1.0)


# ---------------------------------------------------------------------------
# start() idempotency
# ---------------------------------------------------------------------------


def test_start_is_noop_when_disabled(monkeypatch) -> None:
    """start() with flag OFF must not create a timer thread."""
    from skill_hub import continuous_sweep as cs

    with patch("skill_hub.continuous_sweep._cfg") as mock_cfg:
        mock_cfg.get.return_value = False

        # Reset any previously running thread.
        with cs._sweep_lock:
            cs._sweep_thread = None

        cs.start()

        with cs._sweep_lock:
            assert cs._sweep_thread is None, (
                "start() must not create a timer when continuous_sweep_enabled=False"
            )
