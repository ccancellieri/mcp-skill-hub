"""Smoke test for the reconciler daemon thread."""
from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.services.base import Service, Status  # noqa: E402
from skill_hub.services.monitor import PressureTracker  # noqa: E402
from skill_hub.services.registry import ServiceRegistry, start_reconciler  # noqa: E402


class _FakeService(Service):
    def __init__(self, name: str, start_state: Status = "stopped"):
        self.name = name
        self.label = name
        self.description = ""
        self._state: Status = start_state
        self.starts = 0
        self.stops = 0

    def status(self) -> Status:
        return self._state

    def is_available(self):
        return True, ""

    def start(self):
        self.starts += 1
        self._state = "running"
        return True, "ok"

    def stop(self):
        self.stops += 1
        self._state = "stopped"
        return True, "ok"


def test_reconciler_picks_up_config_changes(tmp_path):
    svc = _FakeService("alpha")
    reg = ServiceRegistry([svc])

    cfg_holder = {"services": {"alpha": {"enabled": False}}}

    def load_cfg():
        return dict(cfg_holder)

    # Write a real file so mtime gating has something to observe.
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}")

    class _StubPressure:
        def sample(self):
            import skill_hub.services.monitor as m
            return m.ResourceSample(0, 0, 0.0, 1, False, time.time())

        def sustained_seconds(self):
            return 0.0

        def last_sample(self):
            return None

    handle = start_reconciler(
        reg,
        _StubPressure(),
        config_path=cfg_path,
        load_config=load_cfg,
        interval_sec=0.05,
    )
    try:
        # Initially disabled → reconciler should not start it.
        time.sleep(0.2)
        assert svc.starts == 0

        # Flip config; reconciler should start it within a few ticks.
        cfg_holder["services"]["alpha"]["enabled"] = True
        cfg_path.write_text("{}")  # bump mtime
        t0 = time.time()
        while time.time() - t0 < 2:
            if svc.starts >= 1:
                break
            time.sleep(0.05)
        assert svc.starts >= 1
    finally:
        handle.stop()
