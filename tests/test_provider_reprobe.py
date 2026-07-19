"""Provider auto-reprobe: re-enable transiently-disabled providers on recovery."""
from __future__ import annotations

import email.message
import urllib.error

import pytest

from skill_hub.llm import health


def _provider(name, *, enabled, auto_reenable):
    rec = {
        "name": name,
        "kind": "openai_compatible",
        "api_base": "",
        "api_key": {"source": "opencode", "ref": name},
        "enabled": enabled,
        "order": 30,
        "models": [{"id": "vendor/model-x", "complexity": "light"}],
    }
    if auto_reenable:
        rec["auto_reenable"] = True
    return rec


@pytest.fixture
def patched(monkeypatch):
    """Stub config get/set + credential resolution; capture any persisted write."""
    state: dict = {"registry": [], "saved": None}

    def fake_get(key):
        return state["registry"] if key == "llm_provider_registry" else None

    def fake_set(key, value):
        if key == "llm_provider_registry":
            state["saved"] = value

    monkeypatch.setattr(health._cfg, "get", fake_get)
    monkeypatch.setattr(health._cfg, "set", fake_set)
    monkeypatch.setattr(health._creds, "resolve_credentials",
                        lambda prov: ("https://gw.example/v1", "sk-test"))
    return state


def test_reenables_a_reachable_marked_provider(patched, monkeypatch):
    patched["registry"] = [_provider("gw", enabled=False, auto_reenable=True)]
    monkeypatch.setattr(health, "probe_models", lambda *a, **k: (True, 200))

    changed = health.reprobe_disabled_providers()

    assert changed == [{"name": "gw", "status": 200}]
    assert patched["registry"][0]["enabled"] is True   # flipped in place
    assert patched["saved"] is not None                # persisted


def test_leaves_a_still_blocked_provider_disabled(patched, monkeypatch):
    patched["registry"] = [_provider("gw", enabled=False, auto_reenable=True)]
    monkeypatch.setattr(health, "probe_models", lambda *a, **k: (False, 403))

    changed = health.reprobe_disabled_providers()

    assert changed == []
    assert patched["registry"][0]["enabled"] is False
    assert patched["saved"] is None   # no change → no write


def test_ignores_disabled_without_the_flag(patched, monkeypatch):
    patched["registry"] = [_provider("manual", enabled=False, auto_reenable=False)]
    probed = {"called": False}

    def _probe(*a, **k):
        probed["called"] = True
        return (True, 200)

    monkeypatch.setattr(health, "probe_models", _probe)

    changed = health.reprobe_disabled_providers()

    assert changed == []
    assert probed["called"] is False           # a deliberately-off provider is never probed
    assert patched["registry"][0]["enabled"] is False


def test_ignores_already_enabled_provider(patched, monkeypatch):
    patched["registry"] = [_provider("gw", enabled=True, auto_reenable=True)]
    monkeypatch.setattr(health, "probe_models", lambda *a, **k: (True, 200))

    assert health.reprobe_disabled_providers() == []
    assert patched["saved"] is None


def test_skips_when_base_url_unresolvable(patched, monkeypatch):
    patched["registry"] = [_provider("gw", enabled=False, auto_reenable=True)]
    monkeypatch.setattr(health._creds, "resolve_credentials", lambda prov: (None, None))
    monkeypatch.setattr(health, "probe_models",
                        lambda *a, **k: pytest.fail("must not probe without a base URL"))

    assert health.reprobe_disabled_providers() == []
    assert patched["registry"][0]["enabled"] is False


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_probe_models_ok(monkeypatch):
    monkeypatch.setattr(health.urllib.request, "urlopen", lambda req, timeout=5.0: _Resp(200))
    assert health.probe_models("https://gw.example/v1", "sk") == (True, 200)


def test_probe_models_http_error_stays_down(monkeypatch):
    def _raise(req, timeout=5.0):
        raise urllib.error.HTTPError("u", 403, "Forbidden", email.message.Message(), None)

    monkeypatch.setattr(health.urllib.request, "urlopen", _raise)
    assert health.probe_models("https://gw.example/v1", "sk") == (False, 403)


def test_probe_models_network_error_is_false_zero(monkeypatch):
    def _raise(req, timeout=5.0):
        raise OSError("connection refused")

    monkeypatch.setattr(health.urllib.request, "urlopen", _raise)
    assert health.probe_models("https://gw.example/v1", "sk") == (False, 0)
