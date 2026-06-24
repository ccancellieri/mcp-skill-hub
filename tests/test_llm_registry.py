"""Tests for the data-driven auxiliary LLM provider registry (issue #117)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_hub import config as cfg


def test_load_registry_sorts_by_order_and_skips_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    cfg.CONFIG_PATH.write_text(json.dumps({
        "llm_provider_registry": [
            {"name": "b", "level": "L3", "kind": "openai_compatible", "api_base": "",
             "api_key": {"source": "env", "ref": "B_API_KEY"}, "enabled": True, "order": 30,
             "models": [{"id": "m1", "complexity": "light"}]},
            {"name": "a", "level": "L1", "kind": "ollama", "api_base": "", "api_key": {},
             "enabled": True, "order": 10, "models": []},
            {"name": "off", "level": "L3", "kind": "openai_compatible", "api_base": "",
             "api_key": {}, "enabled": False, "order": 5, "models": []},
        ]
    }))
    from skill_hub.llm import registry
    provs = registry.load_registry()
    assert [p.name for p in provs] == ["a", "b"]
    assert provs[1].models[0].complexity == "light"


def test_load_registry_skips_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    cfg.CONFIG_PATH.write_text(json.dumps(
        {"llm_provider_registry": [{"no_name": True}, "garbage", 42]}
    ))
    from skill_hub.llm import registry
    assert registry.load_registry() == []
