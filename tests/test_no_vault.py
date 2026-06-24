"""Guard test: credential vault is removed; load_config has no vault keys."""
from __future__ import annotations

import importlib
import json

import pytest


def test_vault_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("skill_hub.vault")


def test_load_config_has_no_vault_keys(monkeypatch, tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("{}")
    import skill_hub.config as c
    # reload first (re-runs module-level CONFIG_PATH assignment), then patch
    importlib.reload(c)
    monkeypatch.setattr(c, "CONFIG_PATH", p)
    cfg = c.load_config()
    assert "vault_backend" not in cfg
    assert "voyage_api_key_ref" not in cfg
    assert "anthropic_api_key_ref" not in cfg
