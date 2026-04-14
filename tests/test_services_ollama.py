"""Tests for ollama service lifecycle."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.services import ollama as ollama_mod  # noqa: E402
from skill_hub.services import _proc  # noqa: E402


def _mk_run_stub(script):
    """Build a callable simulating _proc.run, recording argv and returning (code, out)."""
    calls: list[list[str]] = []

    def _run(argv, timeout=15.0):
        calls.append(list(argv))
        return script(list(argv))

    return calls, _run


def test_daemon_status_running(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/bin/" + c if c in ("ollama", "pgrep") else None)

    def script(argv):
        if argv[0] == "pgrep":
            return 0, "1234\n"
        return 0, ""

    calls, run = _mk_run_stub(script)
    monkeypatch.setattr(_proc, "run", run)
    monkeypatch.setattr(ollama_mod, "_brew_services_manages_ollama", lambda: False)

    assert ollama_mod.OllamaDaemon().status() == "running"
    assert ["pgrep", "-f", "ollama serve"] in calls


def test_daemon_status_stopped(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/bin/ollama" if c == "ollama" else None)
    monkeypatch.setattr(_proc, "run", lambda argv, timeout=15.0: (1, ""))  # pgrep miss
    monkeypatch.setattr(ollama_mod, "_brew_services_manages_ollama", lambda: False)
    assert ollama_mod.OllamaDaemon().status() == "stopped"


def test_daemon_unavailable_when_missing(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: None)
    svc = ollama_mod.OllamaDaemon()
    ok, msg = svc.is_available()
    assert ok is False
    assert "brew install ollama" in msg
    assert svc.status() == "unavailable"


def test_daemon_stop_uses_pkill_without_brew(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/bin/ollama")
    monkeypatch.setattr(ollama_mod, "_brew_services_manages_ollama", lambda: False)
    calls, run = _mk_run_stub(lambda argv: (0, ""))
    monkeypatch.setattr(_proc, "run", run)
    ok, _ = ollama_mod.OllamaDaemon().stop()
    assert ok
    assert ["pkill", "-TERM", "-f", "ollama serve"] in calls


def test_model_status_running(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/bin/ollama")

    def script(argv):
        if argv[:1] == ["pgrep"]:
            return 0, "1\n"  # daemon running
        if argv == ["ollama", "ps"]:
            return 0, "NAME                  ID      SIZE\nqwen2.5:3b            abc     1.8 GB\n"
        return 0, ""

    _, run = _mk_run_stub(script)
    monkeypatch.setattr(_proc, "run", run)

    svc = ollama_mod.OllamaModel("ollama_router", "L", "d", "qwen2.5:3b")
    assert svc.status() == "running"


def test_model_stop_argv(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/bin/ollama")
    calls, run = _mk_run_stub(lambda argv: (0, ""))
    monkeypatch.setattr(_proc, "run", run)

    svc = ollama_mod.OllamaModel("ollama_router", "L", "d", "qwen2.5:3b")
    ok, _ = svc.stop()
    assert ok
    assert ["ollama", "stop", "qwen2.5:3b"] in calls


def test_model_start_refuses_if_not_pulled(monkeypatch):
    monkeypatch.setattr(_proc, "which", lambda c: "/usr/bin/ollama")
    # ollama list returns empty
    monkeypatch.setattr(_proc, "run", lambda argv, timeout=15.0: (0, "NAME\n"))
    svc = ollama_mod.OllamaModel("ollama_router", "L", "d", "qwen2.5:3b")
    ok, msg = svc.start()
    assert ok is False
    assert "not installed" in msg


def test_model_resource_footprint():
    svc = ollama_mod.OllamaModel("ollama_embed", "L", "d", "nomic-embed-text", approx_ram_mb=500)
    fp = svc.resource_footprint()
    assert fp["ram_mb_approx"] == 500
    assert 0 < fp["cpu_share"] < 1
