"""Tests for searxng container lifecycle."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.services import searxng as sx_mod  # noqa: E402
from skill_hub.services import _proc  # noqa: E402


def _router(script):
    calls: list[list[str]] = []

    def _run(argv, timeout=15.0):
        calls.append(list(argv))
        return script(list(argv))

    return calls, _run


def test_unavailable_when_docker_missing(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: None)
    svc = sx_mod.SearxngContainer()
    ok, msg = svc.is_available()
    assert ok is False
    assert "docker" in msg.lower()
    assert svc.status() == "unavailable"


def test_unavailable_when_container_missing(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/local/bin/docker")

    def script(argv):
        if argv == ["docker", "info"]:
            return 0, "Docker running"
        if argv[:2] == ["docker", "ps"]:
            return 0, ""  # no matching container
        return 1, ""

    _, run = _router(script)
    monkeypatch.setattr(_proc, "run", run)

    svc = sx_mod.SearxngContainer()
    ok, msg = svc.is_available()
    assert ok is False
    assert "container not created" in msg
    assert svc.status() == "unavailable"


def test_running_container(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/local/bin/docker")

    def script(argv):
        if argv == ["docker", "info"]:
            return 0, "ok"
        if argv[:2] == ["docker", "ps"]:
            return 0, "skill-hub-searxng\n"
        if argv[:2] == ["docker", "inspect"]:
            return 0, "true\n"
        return 1, ""

    _, run = _router(script)
    monkeypatch.setattr(_proc, "run", run)

    svc = sx_mod.SearxngContainer()
    assert svc.status() == "running"


def test_stop_invokes_docker_stop(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/local/bin/docker")

    def script(argv):
        if argv[:3] == ["docker", "ps", "-a"]:
            return 0, "skill-hub-searxng\n"
        if argv[:2] == ["docker", "stop"]:
            return 0, ""
        return 0, ""

    calls, run = _router(script)
    monkeypatch.setattr(_proc, "run", run)

    ok, _ = sx_mod.SearxngContainer().stop()
    assert ok
    assert any(c[:2] == ["docker", "stop"] for c in calls)
