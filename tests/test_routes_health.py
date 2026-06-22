"""Tests for the /health system-health webapp routes."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from skill_hub import system_health as sh  # noqa: E402


@pytest.fixture
def fake_snapshot(monkeypatch):
    """A deterministic snapshot with one stale claude proc + a docker stack."""
    mem = sh.MemInfo(
        ram_used_mb=15000, ram_total_mb=16000, ram_pct=93.7,
        swap_used_mb=19000, swap_total_mb=20000, swap_pct=95.0,
        cpu_load_1m=21.0, cpu_count=12, load_pct=175.0,
    )
    claude = [
        sh.ClaudeProc(111, 1, "2.1.183", "session", 300000, "3d", 5.0, 140,
                      "abc", True, "old version 2.1.183 (current 2.1.185)", "claude --resume abc"),
        sh.ClaudeProc(222, 999, "2.1.185", "daemon", 100, "1m", 0.1, 40,
                      None, False, "daemon root (never killed)", "claude daemon run"),
    ]
    docker = [
        sh.DockerContainer("geoid_db", "Up", "geoid", 240.0, 1200, essential=True),
        sh.DockerContainer("geoid_catalog", "Up", "geoid", 7.0, 760, essential=True),
        sh.DockerContainer("geoid_elasticsearch", "Up", "geoid", 18.0, 940, essential=True),
        sh.DockerContainer("geoid_keycloak", "Up", "geoid", 1.0, 530, essential=False),
        sh.DockerContainer("geoid_kibana", "Up", "geoid", 18.0, 160, essential=False),
    ]
    monkeypatch.setattr(sh, "memory_info", lambda: mem)
    monkeypatch.setattr(sh, "scan_claude_processes", lambda: claude)
    monkeypatch.setattr(sh, "scan_docker", lambda: docker)
    monkeypatch.setattr(sh, "top_processes", lambda n=8: [])
    return mem, claude, docker


@pytest.fixture
def client(tmp_path, monkeypatch, fake_snapshot):
    from skill_hub import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
    # Disable the background watcher thread during tests.
    monkeypatch.setattr(sh, "start_health_watcher", lambda *a, **k: None)
    from skill_hub.webapp.main import create_app
    app = create_app(store=None)
    return TestClient(app)


def test_health_page_renders(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "System Health" in r.text
    assert 'href="/health"' in r.text  # nav entry


def test_panel_shows_metrics_and_issues(client):
    r = client.get("/health/panel")
    assert r.status_code == 200
    assert "Swap" in r.text
    # Stale claude proc -> kill action button present.
    assert "kill-daemons" in r.text
    # Non-essential containers -> stop-nonessential action present.
    assert "stop-nonessential" in r.text


def test_json_snapshot_shape(client):
    j = client.get("/health/json").json()
    assert set(j.keys()) >= {"mem", "claude", "docker", "issues", "stale_daemon_count"}
    assert j["stale_daemon_count"] == 1
    titles = [i["title"] for i in j["issues"]]
    assert any("Swap" in t for t in titles)
    assert any("non-essential" in i["detail"] for i in j["issues"])


def test_kill_daemons_action_invokes_killer(client, monkeypatch):
    called = {}

    def _fake_kill_stale():
        called["stale"] = True
        return {"killed": [111], "failed": [], "skipped": []}

    monkeypatch.setattr(sh, "kill_stale_claude", _fake_kill_stale)
    r = client.post("/health/action/kill-daemons")
    assert r.status_code == 200
    assert called.get("stale")
    assert "killed [111]" in r.text


def test_kill_specific_pid(client, monkeypatch):
    got = {}

    def _fake_kill(pids):
        got["pids"] = pids
        return {"killed": pids, "failed": [], "skipped": []}

    monkeypatch.setattr(sh, "kill_claude", _fake_kill)
    r = client.post("/health/action/kill-daemons?pids=111")
    assert r.status_code == 200
    assert got["pids"] == [111]


def test_stop_nonessential_keeps_minimum(client, monkeypatch):
    monkeypatch.setattr(sh, "stop_nonessential_docker",
                        lambda: {"stopped": ["geoid_keycloak", "geoid_kibana"],
                                 "kept": ["geoid_db", "geoid_catalog", "geoid_elasticsearch"]})
    r = client.post("/health/action/stop-nonessential")
    assert r.status_code == 200
    assert "Stopped 2" in r.text
    assert "kept" in r.text


def test_purge_memory_action(client, monkeypatch):
    monkeypatch.setattr(sh, "purge_memory", lambda: {"ok": True, "note": "Flushed inactive memory."})
    r = client.post("/health/action/purge-memory")
    assert r.status_code == 200
    assert "Flushed inactive memory" in r.text


def test_essential_classification():
    assert sh._is_essential("geoid_db")
    assert sh._is_essential("geoid_catalog")
    assert sh._is_essential("geoid_elasticsearch")
    assert not sh._is_essential("geoid_keycloak")
    assert not sh._is_essential("geoid_kibana")
