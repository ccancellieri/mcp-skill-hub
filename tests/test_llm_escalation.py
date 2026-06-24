"""Tests for the availability/complexity/quota-aware escalation engine."""
import json

import skill_hub.config as cfg
from skill_hub.llm import escalation

_REG = {
    "llm_cooldown_seconds": 100,
    "llm_provider_registry": [
        {
            "name": "gw",
            "level": "L3",
            "kind": "openai_compatible",
            "api_base": "https://gw/v1",
            "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True,
            "order": 30,
            "models": [
                {"id": "light-a", "complexity": "light"},
                {"id": "heavy-a", "complexity": "heavy"},
            ],
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


def _write_registry(monkeypatch, tmp_path, data: dict):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)


def test_complexity_picks_heavy(monkeypatch, tmp_path):
    _write_registry(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.9)
    assert sel is not None
    assert sel.model == "heavy-a" and sel.provider == "gw"


def test_complexity_picks_light(monkeypatch, tmp_path):
    _write_registry(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    sel = escalation.select(0.1)
    assert sel is not None
    assert sel.model == "light-a"


def test_cooldown_rotates_to_next_provider(monkeypatch, tmp_path):
    _write_registry(monkeypatch, tmp_path, _REG)
    escalation.reset_cooldowns()
    escalation.mark_cooldown("light-a")
    escalation.mark_cooldown("heavy-a")
    sel = escalation.select(0.1)
    assert sel is not None
    assert sel.provider == "claude" and sel.level == "personal"


def test_missing_key_skips_provider(monkeypatch, tmp_path):
    cfg_data = {
        "llm_provider_registry": [
            {
                "name": "gw",
                "level": "L3",
                "kind": "openai_compatible",
                "api_base": "https://gw/v1",
                "api_key": {"source": "env", "ref": "ABSENT_KEY"},
                "enabled": True,
                "order": 30,
                "models": [{"id": "light-a", "complexity": "light"}],
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
        ]
    }
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    _write_registry(monkeypatch, tmp_path, cfg_data)
    escalation.reset_cooldowns()
    sel = escalation.select(0.1)
    assert sel is not None
    assert sel.provider == "claude"


def test_quota_error_detection():
    assert escalation.looks_like_quota_error(Exception("HTTP 429 Too Many Requests"))
    assert escalation.looks_like_quota_error(Exception("insufficient_quota"))
    assert not escalation.looks_like_quota_error(Exception("connection refused"))


def test_cooldown_expires_and_is_evicted():
    import time
    escalation.reset_cooldowns()
    escalation.mark_cooldown("m", seconds=1)
    assert escalation.is_cooled("m") is True
    # A `now` past the expiry returns False and evicts the entry.
    assert escalation.is_cooled("m", now=time.time() + 2) is False
    assert "m" not in escalation._COOLDOWN
