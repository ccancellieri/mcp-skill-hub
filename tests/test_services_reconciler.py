"""Smoke test for the reconciler daemon thread."""
from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.services.base import Service, Status  # noqa: E402
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


class _SlowStartService(_FakeService):
    """A service whose start() blocks — mimics Ollama/SearXNG subprocess timeouts."""

    def __init__(self, name: str, start_delay: float):
        super().__init__(name)
        self._delay = start_delay

    def start(self):
        time.sleep(self._delay)
        return super().start()


class _StubPressure:
    def sample(self):
        import skill_hub.services.monitor as m
        return m.ResourceSample(0, 0, 0.0, 1, False, time.time())

    def sustained_seconds(self):
        return 0.0

    def last_sample(self):
        return None


def test_reconciler_does_not_block_caller_on_slow_startup_align(tmp_path):
    """start_reconciler() must return promptly even if startup_align is slow.

    Regression: startup_align() ran synchronously on the CALLER's thread before
    the daemon thread was spawned. Because the MCP server runs it during
    `import skill_hub.server` (before `mcp.run(transport="stdio")`), a slow
    service.start() — e.g. Ollama down → blocking `brew services`/`docker run`
    timeouts — stalled the import past the client's ~30s `initialize` handshake
    window, so the server was abandoned/orphaned before it ever served MCP.

    Alignment must happen in the background thread, not block the caller.
    """
    slow = _SlowStartService("slow", start_delay=2.0)
    reg = ServiceRegistry([slow])
    cfg = {"services": {"slow": {"enabled": True, "auto_start": True}}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}")

    t0 = time.time()
    handle = start_reconciler(
        reg,
        _StubPressure(),
        config_path=cfg_path,
        load_config=lambda: dict(cfg),
        interval_sec=0.05,
    )
    elapsed = time.time() - t0
    try:
        assert elapsed < 0.5, (
            f"start_reconciler blocked the caller for {elapsed:.2f}s on a slow "
            "startup_align — this stalls the MCP stdio handshake"
        )
        # Alignment still runs, just off the caller's thread.
        t1 = time.time()
        while time.time() - t1 < 5:
            if slow.starts >= 1:
                break
            time.sleep(0.05)
        assert slow.starts >= 1, "startup_align never ran in the background thread"
    finally:
        handle.stop()
        assert not handle.is_alive(), (
            "reconciler thread did not stop — it would keep ticking in the "
            "background and could bleed into a later test (issue #143)"
        )


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
        assert not handle.is_alive(), (
            "reconciler thread did not stop — it would keep ticking in the "
            "background and could bleed into a later test (issue #143)"
        )


def test_stop_all_reconcilers_drains_a_leaked_thread(tmp_path):
    """A reconciler whose handle is dropped without stop() is still stoppable.

    This is the mechanism the autouse conftest fixture relies on to stop the
    reconciler skill_hub.server starts at import time (whose handle a test never
    holds) before it can tick real subprocess.Popen calls into an unrelated
    later test — the root cause of #143.
    """
    import threading

    from skill_hub.services.registry import stop_all_reconcilers

    reg = ServiceRegistry([_FakeService("alpha")])
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}")

    # Start a reconciler and drop the handle without stopping it — a leak.
    start_reconciler(
        reg,
        _StubPressure(),
        config_path=cfg_path,
        load_config=lambda: {"services": {}},
        interval_sec=0.05,
    )

    stragglers = stop_all_reconcilers(timeout=5.0)
    assert stragglers == [], f"leaked reconciler did not stop: {stragglers}"
    assert not [
        t for t in threading.enumerate()
        if t.name == "skill-hub-reconciler" and t.is_alive()
    ], "a skill-hub-reconciler thread survived stop_all_reconcilers()"
