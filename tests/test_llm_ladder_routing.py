"""Integration tests: ladder routing in LitellmProvider.

Verifies two behaviours:
1. A 429/quota error on the first ladder selection causes cooldown and rotation
   to the next provider, which succeeds.
2. When the personal-Claude budget cap is zero and the only registered provider
   is personal-level, the ladder exhausts and raises LLMError.
"""
from __future__ import annotations

import json
import importlib

import pytest


_REG = {
    "llm_metering_enabled": False,
    "llm_provider_registry": [
        {
            "name": "gw",
            "level": "L3",
            "kind": "openai_compatible",
            "api_base": "https://gw/v1",
            "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True,
            "order": 30,
            "models": [{"id": "model-a", "complexity": "light"}],
        },
        {
            "name": "claude",
            "level": "personal",
            "kind": "anthropic",
            "api_base": "",
            "api_key": {"source": "inline", "ref": "sk2"},
            "enabled": True,
            "order": 90,
            "models": [{"id": "anthropic/claude-haiku-4-5", "complexity": "light"}],
        },
    ],
}


def _write_cfg(monkeypatch, tmp_path, data: dict):
    """Write *data* to a temp config file and point cfg.CONFIG_PATH at it."""
    import skill_hub.config as cfg
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)


def _reload_llm_modules():
    """Reload all LLM sub-modules so they pick up the new CONFIG_PATH."""
    import skill_hub.llm.registry as registry
    import skill_hub.llm.credentials as credentials
    import skill_hub.llm.escalation as escalation
    import skill_hub.llm.provider as provider
    import skill_hub.llm.litellm_adapter as litellm_adapter
    for m in (registry, credentials, escalation, provider, litellm_adapter):
        importlib.reload(m)
    return escalation, litellm_adapter, provider


def test_quota_429_rotates_and_cools_first_model(monkeypatch, tmp_path):
    _write_cfg(monkeypatch, tmp_path, _REG)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()

    calls: list[str] = []

    class FakeLitellm:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "model-a":
                raise Exception("HTTP 429 rate limit exceeded")
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    p = litellm_adapter.LitellmProvider()
    p._litellm = FakeLitellm()
    out = p.chat([provider.Message(role="user", content="hi")], complexity=0.1)
    assert out == "ok"
    assert calls == ["model-a", "anthropic/claude-haiku-4-5"]
    assert escalation.is_cooled("model-a")


def test_budget_cap_excludes_personal(monkeypatch, tmp_path):
    # Registry with only the personal provider, cap at 0 → ladder exhausted.
    cfg_data = {
        "llm_metering_enabled": False,
        "llm_personal_daily_usd_cap": 0.0,
        "llm_provider_registry": [_REG["llm_provider_registry"][1]],
    }
    _write_cfg(monkeypatch, tmp_path, cfg_data)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()

    p = litellm_adapter.LitellmProvider()
    p._litellm = type(
        "X",
        (),
        {
            "suppress_debug_info": True,
            "drop_params": True,
            "completion": lambda self, **k: 1 / 0,
        },
    )()

    with pytest.raises(provider.LLMError):
        p.chat([provider.Message(role="user", content="hi")], complexity=0.1)


def test_spend_decodes_json_string_payload_and_caps(monkeypatch, tmp_path):
    """Regression: store.get_events returns ``payload`` as a JSON *string*.

    The spend accounting must json.loads it (not call .get() on a str), or the
    budget cap is silently disabled. With a priced personal-model event over
    the cap and only the personal provider registered, the ladder exhausts.
    """
    cfg_data = {
        "llm_metering_enabled": False,
        "llm_personal_daily_usd_cap": 1.0,
        "llm_provider_registry": [_REG["llm_provider_registry"][1]],
    }
    _write_cfg(monkeypatch, tmp_path, cfg_data)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()

    import time as _t

    class _FakeStore:
        def get_events(self, *, session_id="", since=0.0, kind="", limit=200):
            # payload is a raw JSON STRING, exactly as the real store returns it.
            return [{
                "ts": _t.time(),
                "payload": json.dumps({
                    "model": "anthropic/claude-haiku-4-5",
                    "total_tokens": 1_000_000,
                }),
            }]

    import skill_hub.store as store_mod
    monkeypatch.setattr(store_mod, "get_store", lambda: _FakeStore())

    p = litellm_adapter.LitellmProvider()
    # Spend must be computed (proves json.loads ran) and exceed the $1 cap.
    assert p._personal_spend_today_usd() > 1.0
    assert p._personal_over_cap() is True

    p._litellm = type("X", (), {
        "suppress_debug_info": True, "drop_params": True,
        "completion": lambda self, **k: 1 / 0,
    })()
    with pytest.raises(provider.LLMError):
        p.chat([provider.Message(role="user", content="hi")], complexity=0.1)
