"""Tests for plugin registry helpers + plugin panel routes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub import plugin_registry as pr  # noqa: E402
from skill_hub.services import registry as reg_mod  # noqa: E402
from skill_hub.services.base import Service, Status  # noqa: E402


class _FakeService(Service):
    name = "noop"
    label = "Noop"
    description = ""

    def status(self) -> Status:
        return "stopped"

    def is_available(self):
        return True, ""

    def start(self):
        return True, ""

    def stop(self):
        return True, ""


def _seed_plugin_dir(tmp_path: Path, name: str, *, description: str = "") -> Path:
    base = tmp_path / "plugins" / name
    base.mkdir(parents=True)
    (base / "plugin.json").write_text(json.dumps({"name": name, "description": description}))
    return base


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(pr, "SETTINGS_PATH", settings_path)

    # Two plugin directories under one source.
    _seed_plugin_dir(tmp_path, "alpha-plugin", description="the alpha")
    _seed_plugin_dir(tmp_path, "beta-plugin", description="the beta")

    cfg_path.write_text(json.dumps({
        "bundled_plugins_enabled": False,
        "extra_plugin_dirs": [{
            "path": str(tmp_path / "plugins"),
            "source": "local",
            "enabled": True,
        }],
        "profiles": {
            "minimal": {"description": "", "plugins": ["alpha-plugin"]},
            "full":    {"description": "", "plugins": "__all__"},
        },
    }))
    settings_path.write_text(json.dumps({
        "enabledPlugins": {"alpha-plugin@local": True, "beta-plugin@local": False},
    }))

    reg_mod.set_registry(reg_mod.ServiceRegistry([_FakeService()]))

    from skill_hub.webapp.main import create_app

    return TestClient(create_app(store=None)), settings_path


def test_iter_all_plugins_yields_both(env):
    _, _ = env
    plugins = list(pr.iter_all_plugins())
    names = {p["name"] for p in plugins}
    assert names == {"alpha-plugin", "beta-plugin"}


def test_iter_all_plugins_reflects_enabled_state(env):
    _, _ = env
    by_name = {p["name"]: p for p in pr.iter_all_plugins()}
    assert by_name["alpha-plugin"]["enabled"] is True
    assert by_name["beta-plugin"]["enabled"] is False


def test_toggle_persists(env):
    _, settings_path = env
    msg = pr.toggle("alpha-plugin@local", False)
    assert "disabled" in msg
    data = json.loads(settings_path.read_text())
    assert data["enabledPlugins"]["alpha-plugin@local"] is False


def test_apply_profile_minimal(env):
    _, settings_path = env
    pr.apply_profile("minimal")
    data = json.loads(settings_path.read_text())
    # minimal has only alpha-plugin
    assert data["enabledPlugins"]["alpha-plugin@local"] is True
    assert data["enabledPlugins"]["beta-plugin@local"] is False


def test_apply_profile_full_enables_all(env):
    _, settings_path = env
    pr.apply_profile("full")
    data = json.loads(settings_path.read_text())
    assert data["enabledPlugins"]["alpha-plugin@local"] is True
    assert data["enabledPlugins"]["beta-plugin@local"] is True


def test_plugins_route_renders(env):
    client, _ = env
    r = client.get("/control/plugins")
    assert r.status_code == 200
    assert "alpha-plugin" in r.text
    assert "beta-plugin" in r.text


def test_plugins_toggle_route(env):
    client, settings_path = env
    r = client.post("/control/plugins/beta-plugin@local/toggle")
    assert r.status_code == 200
    data = json.loads(settings_path.read_text())
    assert data["enabledPlugins"]["beta-plugin@local"] is True
