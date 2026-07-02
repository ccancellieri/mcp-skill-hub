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
            "kind": "openai_compatible",
            "api_base": "https://gw/v1",
            "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True,
            "order": 30,
            "models": [{"id": "model-a", "complexity": "light"}],
        },
        {
            "name": "claude",
            "personal": True,
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
            # gw is openai_compatible → litellm model is prefixed ``openai/``.
            if kwargs["model"] == "openai/model-a":
                raise Exception("HTTP 429 rate limit exceeded")
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    p = litellm_adapter.LitellmProvider()
    p._litellm = FakeLitellm()
    out = p.chat([provider.Message(role="user", content="hi")], complexity=0.1)
    assert out == "ok"
    assert calls == ["openai/model-a", "anthropic/claude-haiku-4-5"]
    assert escalation.is_cooled("model-a")   # cooldown keys on the registry id


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


def test_op_routing_engages_ladder_and_routes_by_domain(monkeypatch, tmp_path):
    """A known ``op`` (no explicit complexity/model) engages the ladder and the
    policy domain selects the matching specialist."""
    reg = {
        "llm_metering_enabled": False,
        "llm_provider_registry": [{
            "name": "gw", "kind": "openai_compatible",
            "api_base": "https://gw/v1", "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True, "order": 30,
            "models": [{"id": "fast-m", "complexity": "light", "tags": ["fast"]},
                       {"id": "py-m", "complexity": "light", "tags": ["python"]}],
        }],
    }
    _write_cfg(monkeypatch, tmp_path, reg)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()

    calls: list[str] = []

    class FakeLitellm:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append(kwargs["model"])
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    p = litellm_adapter.LitellmProvider()
    p._litellm = FakeLitellm()
    # 'rerank' maps to (0.2, 'fast') in _OP_ROUTING.
    out = p.complete("x", op="rerank")
    assert out == "ok"
    assert calls == ["openai/fast-m"]   # 'fast' specialist (openai_compatible → openai/ prefix)


def test_unknown_op_with_no_signal_skips_ladder(monkeypatch, tmp_path):
    """An op absent from _OP_ROUTING and no complexity/domain must NOT engage
    the ladder — preserves the prior tier-resolution path (no regression)."""
    _write_cfg(monkeypatch, tmp_path, _REG)
    escalation, litellm_adapter, provider = _reload_llm_modules()

    p = litellm_adapter.LitellmProvider()
    engaged = {"v": False}
    monkeypatch.setattr(p, "_chat_via_ladder",
                        lambda *a, **k: (engaged.__setitem__("v", True) or "ladder"))
    monkeypatch.setattr(p, "_resolve_model", lambda tier, model: "resolved-x")
    monkeypatch.setattr(p, "_api_base", lambda m: None)
    monkeypatch.setattr(p, "_chat_once", lambda *a, **k: "direct")

    out = p.complete("x", op="totally_unknown_op")
    assert out == "direct"
    assert engaged["v"] is False


# --- local-daemon-down → skip the dead level, use the ladder ----------------

_REG_LOCAL_PLUS_GW = {
    "llm_metering_enabled": False,
    "llm_provider_registry": [
        {"name": "local", "kind": "ollama",
         "api_base": "", "api_key": {}, "enabled": True, "order": 10,
         "models": [{"id": "qwen-local", "complexity": "light", "tags": ["digest"]}]},
        {"name": "gw", "kind": "openai_compatible",
         "api_base": "https://gw/v1", "api_key": {"source": "inline", "ref": "sk"},
         "enabled": True, "order": 30,
         "models": [{"id": "gw-fast", "complexity": "light", "tags": ["fast"]}]},
    ],
}


def _fake_litellm(calls: list[str]):
    class FakeLitellm:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append(kwargs["model"])
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    return FakeLitellm()


def test_pinned_local_skips_dead_daemon_and_routes_to_gateway(monkeypatch, tmp_path):
    """A hook pins ``ollama/...`` but the daemon is down: the call must skip the
    dead local model entirely and route through the ladder to the gateway."""
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm(calls)

    out = p.complete("x", model="ollama/qwen-local", op="conversation_digest")
    assert out == "ok"
    assert calls == ["openai/gw-fast"]     # never issued a doomed local call
    assert escalation.is_cooled("qwen-local")  # local cooled so the ladder skips it


def test_model_none_skips_dead_local_via_ladder(monkeypatch, tmp_path):
    """model=None + a known op: the ladder picks the level, skipping a dead L0."""
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm(calls)

    out = p.complete("x", op="conversation_digest")
    assert out == "ok"
    assert calls == ["openai/gw-fast"]


def test_local_used_first_when_daemon_up(monkeypatch, tmp_path):
    """Daemon up: the pinned local model is used (free), gateway untouched."""
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: True)

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm(calls)

    out = p.complete("x", model="ollama/qwen-local", op="conversation_digest")
    assert out == "ok"
    assert calls == ["ollama/qwen-local"]   # local-first preserved when up


def test_local_runtime_failure_falls_through_to_ladder(monkeypatch, tmp_path):
    """Daemon passes the probe but the call fails mid-flight: fall through to the
    ladder, excluding the failed model."""
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: True)

    calls: list[str] = []

    class FlakyLocal:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "ollama/qwen-local":
                raise Exception("connection refused")
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    p = litellm_adapter.LitellmProvider()
    p._litellm = FlakyLocal()
    out = p.complete("x", model="ollama/qwen-local", op="conversation_digest")
    assert out == "ok"
    assert calls == ["ollama/qwen-local", "openai/gw-fast"]   # tried local, then ladder


def test_unsignalled_local_down_rescued_by_gateway(monkeypatch, tmp_path):
    """NEW fallback: an op with NO routing signal whose resolved model is a dead
    local daemon is rescued via the ladder instead of issuing a doomed call.

    This is the high-volume path (classifier/triage with op='' on tier_cheap)
    that previously failed straight to local. It must now reach the gateway.
    """
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm(calls)
    monkeypatch.setattr(p, "_resolve_model", lambda tier, model: "ollama/qwen-local")

    out = p.complete("x", op="totally_unknown_op")   # not in _OP_ROUTING, no signal
    assert out == "ok"
    assert calls == ["openai/gw-fast"]               # rescued, no doomed local call
    assert escalation.is_cooled("qwen-local")


def test_unsignalled_local_runtime_failure_rescued(monkeypatch, tmp_path):
    """NEW fallback: daemon passes the probe but a local call for an unsignalled
    op fails mid-flight — it now falls through to the ladder (previously raised)."""
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: True)

    calls: list[str] = []

    class FlakyLocal:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "ollama/qwen-local":
                raise Exception("connection refused")
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    p = litellm_adapter.LitellmProvider()
    p._litellm = FlakyLocal()
    monkeypatch.setattr(p, "_resolve_model", lambda tier, model: "ollama/qwen-local")

    out = p.complete("x", op="totally_unknown_op")
    assert out == "ok"
    assert calls == ["ollama/qwen-local", "openai/gw-fast"]


def test_cool_ollama_makes_select_skip_local(monkeypatch, tmp_path):
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)
    escalation, _litellm_adapter, _provider = _reload_llm_modules()
    escalation.reset_cooldowns()

    sel = escalation.select(0.3, domain="digest")
    assert sel is not None and sel.provider == "local"   # local is the digest specialist
    escalation.cool_ollama()
    sel2 = escalation.select(0.3, domain="digest")
    assert sel2 is not None and sel2.provider == "gw"    # local cooled → next reachable level


def test_smart_memory_write_routes_to_ladder_when_no_local(monkeypatch):
    """No local model reachable: smart_memory_write must route through the ladder
    (model=None + its op) instead of returning ``no_local_model``."""
    import skill_hub.embeddings as emb

    monkeypatch.setattr(emb, "ollama_available", lambda *a, **k: False)
    captured: dict = {}

    def fake_generate(prompt, *, model, timeout, temperature=0.2, num_predict=512, op=""):
        captured["model"] = model
        captured["op"] = op
        return ('{"filename":"x.md","name":"X","description":"d","type":"project",'
                '"content":"a sufficiently long memory entry describing the work done",'
                '"key_entities":["work"],"detail_score":0.9}')

    monkeypatch.setattr(emb, "_generate", fake_generate)
    out = emb.smart_memory_write("session content here", "")

    assert captured["model"] is None                 # routed via the ladder, not a pinned local model
    assert captured["op"] == "smart_memory_write"
    assert out["escalate"] is False                  # ladder produced a usable entry → no Claude escalation


_GOOD_MEMORY_JSON = (
    '{"filename":"x.md","name":"X","description":"d","type":"project",'
    '"content":"a sufficiently long memory entry describing the work done",'
    '"key_entities":["work"],"detail_score":0.9}'
)


def test_smart_memory_write_retries_ladder_on_weak_local(monkeypatch):
    """A low-quality LOCAL result retries once through the ladder (model=None)
    instead of immediately asking Claude to write the memory (#133)."""
    import skill_hub.embeddings as emb
    import skill_hub.llm.escalation as escalation

    monkeypatch.setattr(emb, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_hot_path", lambda: False)
    monkeypatch.setattr(escalation, "has_remote_provider", lambda: True)
    cooled: list[int] = []
    monkeypatch.setattr(escalation, "cool_ollama",
                        lambda *, seconds=30: cooled.append(seconds))

    models_called: list = []

    def fake_generate(prompt, *, model, timeout, temperature=0.2, num_predict=512, op=""):
        models_called.append(model)
        if model is not None:      # local attempt → garbage
            return '{"filename":"x.md","content":"tiny","key_entities":[],"detail_score":0.1}'
        return _GOOD_MEMORY_JSON   # ladder attempt → good entry

    monkeypatch.setattr(emb, "_generate", fake_generate)
    out = emb.smart_memory_write("session content here " * 20, "")

    assert models_called[0] is not None and models_called[1] is None
    assert cooled, "local rung must be cooled so the retry walk skips it"
    assert out["escalate"] is False
    assert "directive" not in out


def test_smart_memory_write_no_ladder_retry_on_hot_path(monkeypatch):
    """On the hook hot path a weak local result must NOT trigger a remote
    round-trip — the escalate directive is returned immediately."""
    import skill_hub.embeddings as emb

    monkeypatch.setattr(emb, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_hot_path", lambda: True)

    models_called: list = []

    def fake_generate(prompt, *, model, timeout, temperature=0.2, num_predict=512, op=""):
        models_called.append(model)
        return '{"filename":"x.md","content":"tiny","key_entities":[],"detail_score":0.1}'

    monkeypatch.setattr(emb, "_generate", fake_generate)
    out = emb.smart_memory_write("session content here " * 20, "")

    assert len(models_called) == 1          # no second (ladder) attempt
    assert out["escalate"] is True
    assert "directive" in out


def test_smart_memory_write_keeps_best_when_retry_also_weak(monkeypatch):
    """If the ladder retry is no better, the original outcome (and the Claude
    escalation directive) is preserved."""
    import skill_hub.embeddings as emb
    import skill_hub.llm.escalation as escalation

    monkeypatch.setattr(emb, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_hot_path", lambda: False)
    monkeypatch.setattr(escalation, "has_remote_provider", lambda: True)
    monkeypatch.setattr(escalation, "cool_ollama", lambda *, seconds=30: None)

    def fake_generate(prompt, *, model, timeout, temperature=0.2, num_predict=512, op=""):
        return '{"filename":"x.md","content":"tiny","key_entities":[],"detail_score":0.1}'

    monkeypatch.setattr(emb, "_generate", fake_generate)
    out = emb.smart_memory_write("session content here " * 20, "")

    assert out["escalate"] is True
    assert "directive" in out


# --- gateway dispatch correctness ------------------------------------------

def test_litellm_model_prefixes_openai_compatible_only():
    from skill_hub.llm.litellm_adapter import _litellm_model
    # OpenAI-compatible gateway must route via the ``openai/`` provider.
    assert _litellm_model("openai_compatible", "anthropic/claude-haiku-4-5") == "openai/anthropic/claude-haiku-4-5"
    assert _litellm_model("openai_compatible", "zai-org/glm-4.7-maas") == "openai/zai-org/glm-4.7-maas"
    assert _litellm_model("openai_compatible", "openai/already") == "openai/already"   # idempotent
    # Native providers keep their own routing prefix.
    assert _litellm_model("anthropic", "anthropic/claude-haiku-4-5") == "anthropic/claude-haiku-4-5"
    assert _litellm_model("ollama", "ollama/qwen") == "ollama/qwen"


def test_ladder_success_emits_activity_line(monkeypatch, tmp_path):
    """A successful ladder call surfaces a precise activity line naming the work
    model that served it — so gateway usage is visible, not a generic 'ladder'."""
    _write_cfg(monkeypatch, tmp_path, {**_REG, "llm_metering_enabled": True})
    escalation, litellm_adapter, provider = _reload_llm_modules()
    escalation.reset_cooldowns()

    import skill_hub.store as store_mod
    monkeypatch.setattr(store_mod, "get_store",
                        lambda: type("S", (), {"append_event": lambda self, **k: None})())

    lines: list[str] = []
    import skill_hub.activity_log as al
    monkeypatch.setattr(al, "append_line", lambda s: lines.append(s))

    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm([])
    out = p.chat([provider.Message(role="user", content="hi")], complexity=0.1)
    assert out == "ok"
    assert any("model-a" in ln and "ladder" in ln for ln in lines)


def test_chat_once_falls_back_to_reasoning_content(monkeypatch, tmp_path):
    """A reasoning model returns ``content: null`` with text in
    ``reasoning_content``; the parse must return that rather than fail/empty."""
    _write_cfg(monkeypatch, tmp_path, _REG)
    _escalation, litellm_adapter, _provider = _reload_llm_modules()

    class FakeLitellm:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            return {"choices": [{"message": {"content": None,
                                             "reasoning_content": "the answer is 42"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    p = litellm_adapter.LitellmProvider()
    p._litellm = FakeLitellm()
    out = p.complete("x", model="ollama/qwen", op="")   # explicit model, no ladder
    assert out == "the answer is 42"
