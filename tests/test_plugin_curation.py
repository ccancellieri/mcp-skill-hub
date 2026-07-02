"""L1/L2 autonomous plugin curation — disable decisions on the cheap ladder tier."""
from __future__ import annotations

import importlib
import json

import pytest

from skill_hub import plugin_curation as pc
from skill_hub.store import SkillStore


@pytest.fixture()
def store(tmp_path):
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


def _patch_request(monkeypatch, payload):
    _req = importlib.import_module("skill_hub.llm.request")
    monkeypatch.setattr(_req, "request", lambda *a, **k: payload)


def test_parse_decision():
    assert pc._parse_decision('x {"action":"disable","reason":"stale"} y') == {
        "action": "disable", "reason": "stale"
    }
    assert pc._parse_decision("no json") is None
    assert pc._parse_decision("[1,2]") is None   # array, not object


def test_decide_returns_disable(store, monkeypatch):
    monkeypatch.setattr(pc, "_cfg", pc._cfg)  # keep real config for stale_days default
    from skill_hub import plugin_registry
    monkeypatch.setattr(plugin_registry, "_enabled_map",
                        lambda: {"foo@src": True, "bar@src": True})
    _patch_request(monkeypatch, json.dumps({"action": "disable", "reason": "no activity"}))

    out = pc.decide(store, {"plugin_id": "bar@src", "last_used_at": None,
                            "sessions_last_window": 0})
    assert out == {"plugin_id": "bar@src", "action": "disable", "reason": "no activity"}


def test_decide_rejects_bad_action(store, monkeypatch):
    from skill_hub import plugin_registry
    monkeypatch.setattr(plugin_registry, "_enabled_map", lambda: {"foo@src": True})
    _patch_request(monkeypatch, json.dumps({"action": "nonsense", "reason": "?"}))
    assert pc.decide(store, {"plugin_id": "foo@src"}) is None


def test_decide_none_without_plugin_id(store):
    assert pc.decide(store, {"last_used_at": None}) is None


def test_run_curation_gated_off(store, monkeypatch):
    monkeypatch.setattr(pc._cfg, "get",
                        lambda k, d=None: False if k == "plugin_curation_enabled" else d)
    assert pc.run_curation(store) == []


def test_run_curation_dry_run_does_not_toggle(store, monkeypatch):
    def cfg(k, d=None):
        return {"plugin_curation_enabled": True, "plugin_curation_auto": False,
                "plugin_curation_max_per_session": 5}.get(k, d)
    monkeypatch.setattr(pc._cfg, "get", cfg)
    monkeypatch.setattr(pc, "candidates", lambda s, **k: [{"plugin_id": "stale@src"}])
    monkeypatch.setattr(pc, "decide",
                        lambda s, c, **k: {"plugin_id": c["plugin_id"],
                                           "action": "disable", "reason": "dormant"})
    from skill_hub import plugin_registry
    calls = []
    monkeypatch.setattr(plugin_registry, "toggle",
                        lambda pid, enabled: calls.append((pid, enabled)))

    out = pc.run_curation(store)   # apply defaults to plugin_curation_auto (False)
    assert out == [{"plugin_id": "stale@src", "action": "disable",
                    "reason": "dormant", "applied": False}]
    assert calls == []             # nothing written in dry-run


def test_run_curation_apply_toggles_off(store, monkeypatch):
    def cfg(k, d=None):
        return {"plugin_curation_enabled": True, "plugin_curation_auto": True,
                "plugin_curation_max_per_session": 5}.get(k, d)
    monkeypatch.setattr(pc._cfg, "get", cfg)
    monkeypatch.setattr(pc, "candidates",
                        lambda s, **k: [{"plugin_id": "stale@src"}, {"plugin_id": "keep@src"}])

    def fake_decide(s, c, **k):
        action = "disable" if c["plugin_id"] == "stale@src" else "keep"
        return {"plugin_id": c["plugin_id"], "action": action, "reason": "r"}
    monkeypatch.setattr(pc, "decide", fake_decide)

    from skill_hub import plugin_registry
    calls = []
    monkeypatch.setattr(plugin_registry, "toggle",
                        lambda pid, enabled: calls.append((pid, enabled)))

    out = pc.run_curation(store)
    assert calls == [("stale@src", False)]          # only the disable was applied
    applied = {r["plugin_id"]: r["applied"] for r in out}
    assert applied == {"stale@src": True, "keep@src": False}


def test_run_curation_respects_limit(store, monkeypatch):
    def cfg(k, d=None):
        return {"plugin_curation_enabled": True, "plugin_curation_auto": False}.get(k, d)
    monkeypatch.setattr(pc._cfg, "get", cfg)
    monkeypatch.setattr(pc, "candidates",
                        lambda s, **k: [{"plugin_id": f"p{i}@src"} for i in range(10)])
    seen = []
    monkeypatch.setattr(pc, "decide",
                        lambda s, c, **k: seen.append(c["plugin_id"]) or
                        {"plugin_id": c["plugin_id"], "action": "keep", "reason": "r"})
    pc.run_curation(store, limit=3)
    assert len(seen) == 3
