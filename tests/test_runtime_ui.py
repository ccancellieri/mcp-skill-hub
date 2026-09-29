"""Provider-neutral model and runtime-session UI coverage."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub.services import registry as reg_mod  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
    cfg_mod.save_config({
        "llm_providers": {
            "tier_cheap": "legacy/current-model",
            "tier_smart": "astra::codex-astra",
        },
    })
    reg_mod.set_registry(reg_mod.ServiceRegistry([]))

    class _Pressure:
        def sample(self):
            from skill_hub.services.monitor import ResourceSample
            return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)

        def sustained_seconds(self): return 0.0
        def last_sample(self): return self.sample()

    reg_mod.set_pressure(_Pressure())

    from skill_hub.webapp.main import create_app
    return TestClient(create_app(store=None))


def test_models_card_is_provider_neutral_and_lists_concurrent_sessions(client, monkeypatch):
    from skill_hub.webapp.routes import control

    monkeypatch.setattr(control, "_model_options", lambda: [
        {
            "id": "codex-astra", "provider": "astra", "kind": "codex",
            "configured": True, "availability": "configured",
            "value": "astra::codex-astra", "price": None,
        },
        {
            "id": "grok-4", "provider": "pi-gateway", "kind": "openai_compatible",
            "configured": False, "availability": "catalog",
            "value": "pi-gateway::grok-4", "price": None,
        },
    ])
    monkeypatch.setattr(control, "_runtime_sessions", lambda: [
        {
            "client_id": "codex", "client_version": "1.2", "session_id": "codex-session",
            "turn_id": "turn-1", "model_id": "codex-astra", "model_provider": "openai",
            "model_display_name": "Codex Astra", "effort_value": "high",
            "effort_scheme": "codex", "source": "mixed",
            "observed_at": "2026-09-24T09:00:00Z",
            "provenance": {"model_id": "native_event", "effort_value": "caller_reported"},
            "observed_at_by_field": {
                "model_id": "2026-09-24T07:30:00Z",
                "effort_value": "2026-09-24T07:45:00Z",
            },
        },
        {
            "client_id": "pi", "client_version": "0.50", "session_id": "pi-session",
            "turn_id": "", "model_id": "grok-4", "model_provider": "xai",
            "model_display_name": "Pi Grok", "effort_value": "medium",
            "effort_scheme": "pi", "source": "adapter_reported",
            "observed_at": "2026-09-24T07:59:00Z",
            "provenance": {"model_id": "adapter_reported", "effort_value": "adapter_reported"},
            "observed_at_by_field": {
                "model_id": "2026-09-24T07:59:00Z",
                "effort_value": "2026-09-24T07:58:00Z",
            },
        },
    ])

    response = client.get("/control/llm/card")

    assert response.status_code == 200
    assert "Skill Hub model services" in response.text
    assert "Client agent sessions (L3)" in response.text
    assert "Codex Astra" in response.text and "high" in response.text
    assert "model: native event" in response.text
    assert "effort: caller-reported" in response.text
    assert 'title="2026-09-24T07:30:00Z">2026-09-24 07:30:00</time>' in response.text
    assert 'title="2026-09-24T07:45:00Z">2026-09-24 07:45:00</time>' in response.text
    assert "mixed evidence" in response.text
    assert "Pi Grok" in response.text
    assert "model: adapter-reported" in response.text
    assert "effort: adapter-reported" in response.text
    assert 'title="2026-09-24T07:59:00Z">2026-09-24 07:59:00</time>' in response.text
    assert 'title="2026-09-24T07:58:00Z">2026-09-24 07:58:00</time>' in response.text
    assert response.text.count("adapter-reported") >= 3
    assert "Not reported" in response.text
    assert 'value="astra::codex-astra"' in response.text
    assert "legacy/current-model" in response.text
    # Existing saved values remain visible, but catalog choices come from the
    # provider registry rather than a baked-in Claude option list.
    assert 'value="pi-gateway::grok-4"' in response.text
    assert "tier_planner" in response.text
    assert "Active" not in response.text


def test_saved_planner_service_can_be_updated_without_accepting_unknown_keys(client, monkeypatch):
    from skill_hub.webapp.routes import control

    monkeypatch.setattr(control, "_model_options", lambda: [])
    monkeypatch.setattr(control, "_runtime_sessions", lambda: [])

    planner = client.post("/control/llm/tier", data={
        "tier": "tier_planner", "model_id": "work::frontier",
    })
    unknown = client.post("/control/llm/tier", data={
        "tier": "invented_tier", "model_id": "work::frontier",
    })

    assert planner.status_code == 200
    assert cfg_mod.get("llm_providers")["tier_planner"] == "work::frontier"
    assert unknown.status_code == 400


def test_models_card_labels_unknown_price_and_missing_runtime_fields(client, monkeypatch):
    from skill_hub.webapp.routes import control

    monkeypatch.setattr(control, "_model_options", lambda: [{
        "id": "grok-4", "provider": "pi-gateway", "kind": "openai_compatible",
        "configured": True, "availability": "unknown",
        "value": "pi-gateway::grok-4", "price": None,
    }])
    monkeypatch.setattr(control, "_runtime_sessions", lambda: [{
        "client_id": "old-client", "client_version": "", "session_id": "old-session",
        "turn_id": "", "model_id": "", "model_provider": "",
        "model_display_name": "", "effort_value": "", "effort_scheme": "",
        "source": "mcp_client_info", "observed_at": "", "provenance": {},
        "observed_at_by_field": {},
    }])

    response = client.get("/control/llm/card")

    assert response.status_code == 200
    assert "Price not reported" in response.text
    assert response.text.count("Not reported") >= 3
    assert "MCP client info" in response.text


def test_models_card_survives_clients_without_runtime_observations(client, monkeypatch):
    from skill_hub.webapp.routes import control

    monkeypatch.setattr(control, "_model_options", lambda: [])
    monkeypatch.setattr(control, "_runtime_sessions", lambda: [])

    response = client.get("/control/llm/card")

    assert response.status_code == 200
    assert "No runtime observations reported." in response.text


def test_report_does_not_borrow_a_price_for_unknown_models(monkeypatch):
    from skill_hub import model_registry
    from skill_hub.webapp.routes.report import _estimate_usd

    monkeypatch.setattr(
        model_registry,
        "blended_usd_per_m",
        lambda model: 12.0 if model == "known-model" else None,
    )

    assert _estimate_usd({"known-model": 1_000_000}) == 12.0
    assert _estimate_usd({"unknown-model": 1_000_000}) is None
