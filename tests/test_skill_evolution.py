"""L2 skill evolution: proposals from feedback signals (#137)."""
from __future__ import annotations

import json

import pytest

from skill_hub import skill_evolution as se
from skill_hub.store import SkillStore


@pytest.fixture()
def store(tmp_path):
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


def test_parse_proposal():
    assert se._parse_proposal('noise {"content":"x","reason":"y"} tail') == {
        "content": "x", "reason": "y"
    }
    assert se._parse_proposal("no json") is None
    assert se._parse_proposal("[1,2]") is None   # array, not object


def test_candidates_filters_and_sorts(store, monkeypatch):
    stats = [
        {"id": "good", "injections": 10, "helpful": 8, "unhelpful": 0, "feedback_score": 1.3},
        {"id": "rarely-seen", "injections": 1, "helpful": 0, "unhelpful": 1, "feedback_score": 0.5},
        {"id": "unhelpful", "injections": 6, "helpful": 1, "unhelpful": 5, "feedback_score": 0.7},
        {"id": "low-score", "injections": 4, "helpful": 2, "unhelpful": 1, "feedback_score": 0.85},
    ]
    monkeypatch.setattr(store, "get_skill_usage_stats", lambda: stats)
    got = se.candidates(store, limit=5, min_injections=3)
    ids = [c["id"] for c in got]
    # "good" excluded (score>=floor, no unhelpful); "rarely-seen" excluded (few injections)
    assert ids == ["unhelpful", "low-score"]   # worst score first


def test_propose_evolution_records_proposal(store, monkeypatch):
    monkeypatch.setattr(store, "get_skill", lambda sid: {"id": sid, "content": "old skill body"})
    monkeypatch.setattr(store, "get_skill_usage_stats", lambda: [
        {"id": "s:one", "injections": 5, "helpful": 0, "unhelpful": 4, "feedback_score": 0.6},
    ])
    import importlib
    _req = importlib.import_module("skill_hub.llm.request")
    monkeypatch.setattr(_req, "request",
                        lambda *a, **k: json.dumps({"content": "sharper skill body",
                                                    "reason": "clearer steps"}))

    out = se.propose_evolution(store, "s:one")
    assert out["skill_id"] == "s:one"
    assert out["version"] == 1
    assert "clearer steps" in out["change_reason"]

    versions = store.get_skill_versions("s:one")
    assert len(versions) == 1
    payload = json.loads(versions[0]["skill_json"])
    assert payload["proposed_content"] == "sharper skill body"
    assert payload["source"] == "ladder"


def test_propose_evolution_rejects_unchanged(store, monkeypatch):
    monkeypatch.setattr(store, "get_skill", lambda sid: {"id": sid, "content": "same body"})
    monkeypatch.setattr(store, "get_skill_usage_stats", lambda: [])
    import importlib
    _req = importlib.import_module("skill_hub.llm.request")
    monkeypatch.setattr(_req, "request",
                        lambda *a, **k: json.dumps({"content": "same body", "reason": "noop"}))
    assert se.propose_evolution(store, "s:one") is None
    assert store.get_skill_versions("s:one") == []


def test_propose_evolution_unknown_skill(store, monkeypatch):
    monkeypatch.setattr(store, "get_skill", lambda sid: None)
    monkeypatch.setattr(store, "get_skill_content", lambda sid: None)
    assert se.propose_evolution(store, "missing") is None


def test_run_evolution_gated_off(store, monkeypatch):
    monkeypatch.setattr(se._cfg, "get",
                        lambda k, d=None: False if k == "skill_evolution_auto" else d)
    assert se.run_evolution(store) == []


def test_run_evolution_caps_and_collects(store, monkeypatch):
    def cfg(k, d=None):
        return {"skill_evolution_auto": True, "skill_evolution_max_per_session": 2}.get(k, d)
    monkeypatch.setattr(se._cfg, "get", cfg)
    monkeypatch.setattr(se, "candidates",
                        lambda s, limit=5: [{"id": f"s:{i}"} for i in range(limit)])
    monkeypatch.setattr(se, "propose_evolution",
                        lambda s, sid, reason="": {"skill_id": sid, "version": 1, "change_reason": "x"})
    made = se.run_evolution(store)
    assert len(made) == 2   # capped at max_per_session
