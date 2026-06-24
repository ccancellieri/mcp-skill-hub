"""Tests for specialization (domain) routing: per-model tags, domain-aware
select(), the pick_model_for() selector, and the adapter op-routing policy."""
import json

import skill_hub.config as cfg
from skill_hub.llm import escalation, registry

_REG = {
    "llm_provider_registry": [
        # L1 cheap generalist: light, tagged for fast/digest only.
        {"name": "local", "level": "L1", "kind": "ollama",
         "api_base": "", "api_key": {}, "enabled": True, "order": 10,
         "models": [{"id": "qwen", "complexity": "light",
                     "tags": ["fast", "digest"]}]},
        # L3 gateway specialists.
        {"name": "gw", "level": "L3", "kind": "openai_compatible",
         "api_base": "https://gw/v1", "api_key": {"source": "inline", "ref": "sk"},
         "enabled": True, "order": 30,
         "models": [{"id": "glm-light", "complexity": "light",
                     "tags": ["programming", "python"]},
                    {"id": "gemini", "complexity": "heavy",
                     "tags": ["web", "ui-ux", "design"]},
                    {"id": "deepseek", "complexity": "heavy",
                     "tags": ["python", "reasoning"]}]},
        # Personal fallback specialist.
        {"name": "claude", "level": "personal", "kind": "anthropic",
         "api_base": "", "api_key": {"source": "inline", "ref": "sk2"},
         "enabled": True, "order": 90,
         "models": [{"id": "sonnet", "complexity": "heavy",
                     "tags": ["python", "git", "programming"]}]},
    ],
}


def _write(monkeypatch, tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)


# --- registry parsing ------------------------------------------------------

def test_registry_parses_and_normalizes_tags(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, {
        "llm_provider_registry": [
            {"name": "gw", "level": "L3", "kind": "openai_compatible",
             "api_base": "x", "api_key": {"source": "inline", "ref": "sk"},
             "enabled": True, "order": 1,
             "models": [{"id": "m", "tags": ["Python", " Git ", "", 123]}]},
        ]})
    provs = registry.load_registry()
    assert provs[0].models[0].tags == ["python", "git"]   # lowercased, trimmed, junk dropped


def test_registry_tags_default_empty(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _REG)
    local = next(p for p in registry.load_registry() if p.name == "local")
    gw = next(p for p in registry.load_registry() if p.name == "gw")
    assert local.models[0].tags == ["fast", "digest"]
    assert "python" in gw.models[0].tags


# --- domain-aware select() -------------------------------------------------

def test_domain_filter_picks_specialist_over_cheaper_tier(monkeypatch, tmp_path):
    """A python task skips the cheap L1 (no python tag) for the gw specialist."""
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.2, domain="python")
    assert sel is not None
    assert sel.provider == "gw" and sel.model == "glm-light"   # light python at cheapest matching tier


def test_domain_and_complexity_combine(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.9, domain="python")   # heavy python
    assert sel is not None
    assert sel.provider == "gw" and sel.model == "deepseek"


def test_domain_web_ui_routes_to_gemini(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.5, domain="ui-ux")
    assert sel is not None
    assert sel.model == "gemini"


def test_domain_is_case_insensitive(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.2, domain="PYTHON")
    assert sel is not None and sel.model == "glm-light"


def test_unknown_domain_falls_back_to_complexity_ladder(monkeypatch, tmp_path):
    """No specialist for the domain -> degrade to the normal availability ladder."""
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.2, domain="cobol")
    assert sel is not None
    assert sel.provider == "local" and sel.model == "qwen"   # cheapest light, ignoring domain


def test_domain_specialist_cooldown_then_fallback(monkeypatch, tmp_path):
    """Cooling all python specialists falls back to the complexity ladder."""
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    escalation.mark_cooldown("glm-light")
    escalation.mark_cooldown("deepseek")
    escalation.mark_cooldown("sonnet")
    sel = escalation.select(0.2, domain="python")
    assert sel is not None
    assert sel.provider == "local"   # no python specialist left -> generalist ladder


def test_digest_domain_prefers_cheap_local(monkeypatch, tmp_path):
    """A digest task stays on the free local generalist (tagged digest)."""
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.3, domain="digest")
    assert sel is not None
    assert sel.provider == "local"


# --- pick_model_for selector ----------------------------------------------

def test_pick_model_for_selector(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.pick_model_for("design")
    assert sel is not None and sel.model == "gemini"


def test_pick_model_for_git(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    # 'git' only on the personal sonnet specialist.
    sel = escalation.pick_model_for("git")
    assert sel is not None and sel.model == "sonnet"
