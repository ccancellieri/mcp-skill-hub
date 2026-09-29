"""Auxiliary consumers share routing policy without changing user model IDs."""
import importlib


def test_hook_context_never_resolves_a_provider(monkeypatch):
    module = importlib.import_module("skill_hub.llm.request")
    monkeypatch.setenv("SKILL_HUB_LOCAL_ONLY", "1")
    def forbidden():
        raise AssertionError("Hook must not resolve a model provider")
    assert module.request("mid", "prompt", get_provider_fn=forbidden) == ""


def test_session_memory_uses_common_dispatch_and_preserves_tier(monkeypatch):
    from skill_hub.router import session_memory
    calls = []
    def dispatch(tier, prompt, **kwargs):
        calls.append((tier, prompt, kwargs))
        return "stored memory"
    monkeypatch.setattr(session_memory, "request", dispatch)
    assert session_memory.build_session_memory("user: work", tier="smart") == "stored memory"
    assert session_memory.update_session_memory("old", "new", tier="mid") == "stored memory"
    assert [c[0] for c in calls] == ["smart", "mid"]
    assert all(c[2]["op"] == "session_memory" for c in calls)


def test_classifier_uses_configured_tier_without_claude_pin(monkeypatch):
    from skill_hub.router import haiku_client
    calls = []
    monkeypatch.setattr(haiku_client, "is_enabled", lambda cfg: True)
    def dispatch(tier, prompt, **kwargs):
        calls.append((tier, kwargs))
        return '{"classification":{"complexity":0.4}}'
    monkeypatch.setattr(haiku_client, "request", dispatch)
    assert haiku_client.classify("task", {}) is not None
    assert calls[0][1].get("model") is None
    assert calls[0][1]["op"] == "router_classify"
