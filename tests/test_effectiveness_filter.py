"""Effectiveness-based skill load gating + Skill-tool usage attribution.

Covers:
- ``SkillStore.get_ineffective_skill_ids`` — never-helpful and
  injected-but-unused criteria, cleared by any positive signal.
- ``SkillStore.resolve_skill_id`` — exact id, unique bare name, ambiguous.
- ``cli._drop_ineffective_skills`` — candidate filtering + config off-switch.
- Keyword context injection skips gated skills end-to-end.
- ``post_tool_observer._maybe_emit_skill_used`` — the built-in Skill tool
  emits a ``skill.used`` event for the resolved skill id.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
HOOKS = Path(__file__).resolve().parent.parent / "hooks"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


def _seed_skill(store, skill_id="local:probe", name="probe",
                description="probe skill", content="Body."):
    from skill_hub.store import Skill
    store.upsert_skill(Skill(
        id=skill_id, name=name, description=description, content=content,
        file_path="", plugin="", target="claude",
    ))


# ---------------------------------------------------------------------------
# get_ineffective_skill_ids
# ---------------------------------------------------------------------------

def test_never_helpful_feedback_gates_skill(store):
    _seed_skill(store, "local:bad")
    store.record_feedback("local:bad", "q", [], helpful=False)
    assert "local:bad" not in store.get_ineffective_skill_ids()  # below floor
    store.record_feedback("local:bad", "q", [], helpful=False)
    assert "local:bad" in store.get_ineffective_skill_ids()


def test_helpful_feedback_clears_never_helpful(store):
    _seed_skill(store, "local:mixed")
    store.record_feedback("local:mixed", "q", [], helpful=False)
    store.record_feedback("local:mixed", "q", [], helpful=False)
    store.record_feedback("local:mixed", "q", [], helpful=True)
    assert "local:mixed" not in store.get_ineffective_skill_ids()


def test_injected_but_unused_gates_skill(store):
    _seed_skill(store, "local:noisy")
    for _ in range(7):
        store.log_skill_injection("local:noisy", "q", "sess-1")
    assert "local:noisy" not in store.get_ineffective_skill_ids()  # below floor
    store.log_skill_injection("local:noisy", "q", "sess-1")
    assert "local:noisy" in store.get_ineffective_skill_ids()


def test_skill_used_event_clears_injected_but_unused(store):
    _seed_skill(store, "local:redeemed")
    for _ in range(8):
        store.log_skill_injection("local:redeemed", "q", "sess-1")
    store.record_skill_used("local:redeemed", "sess-1")
    assert "local:redeemed" not in store.get_ineffective_skill_ids()


# ---------------------------------------------------------------------------
# resolve_skill_id
# ---------------------------------------------------------------------------

def test_resolve_skill_id_exact_and_bare_name(store):
    _seed_skill(store, "superpowers:brainstorming", name="brainstorming")
    assert store.resolve_skill_id("superpowers:brainstorming") == \
        "superpowers:brainstorming"
    assert store.resolve_skill_id("brainstorming") == "superpowers:brainstorming"
    assert store.resolve_skill_id("no-such-skill") is None
    assert store.resolve_skill_id("") is None


def test_resolve_skill_id_ambiguous_name_returns_none(store):
    _seed_skill(store, "discord:configure", name="configure",
                content="discord body")
    _seed_skill(store, "telegram:configure", name="configure",
                content="telegram body")
    assert store.resolve_skill_id("configure") is None


# ---------------------------------------------------------------------------
# cli._drop_ineffective_skills
# ---------------------------------------------------------------------------

def test_drop_ineffective_skills_filters_candidates(store):
    import skill_hub.cli as cli
    _seed_skill(store, "local:good")
    _seed_skill(store, "local:bad")
    store.record_feedback("local:bad", "q", [], helpful=False)
    store.record_feedback("local:bad", "q", [], helpful=False)

    candidates = [{"id": "local:good"}, {"id": "local:bad"}]
    kept = cli._drop_ineffective_skills(store, candidates, {})
    assert [s["id"] for s in kept] == ["local:good"]


def test_drop_ineffective_skills_config_off_switch(store):
    import skill_hub.cli as cli
    _seed_skill(store, "local:bad")
    store.record_feedback("local:bad", "q", [], helpful=False)
    store.record_feedback("local:bad", "q", [], helpful=False)

    candidates = [{"id": "local:bad"}]
    kept = cli._drop_ineffective_skills(
        store, candidates, {"hook_context_effectiveness_filter": False},
    )
    assert kept == candidates


# ---------------------------------------------------------------------------
# End-to-end: keyword context injection skips gated skills
# ---------------------------------------------------------------------------

def test_keyword_context_injection_skips_ineffective(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore, Skill
    monkeypatch.setenv("HOME", str(tmp_path))
    st = SkillStore(db_path=tmp_path / "kw_gate.db")
    st.upsert_skill(Skill(
        id="local:git-pr", name="git-pr",
        description="create a git commit and open a pull request",
        content="Steps: stage, commit, push, open PR.",
        file_path="", plugin="", target="claude",
    ))
    st.record_feedback("local:git-pr", "q", [], helpful=False)
    st.record_feedback("local:git-pr", "q", [], helpful=False)

    import skill_hub.cli as cli
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **k: st)

    out = cli._build_keyword_context_injection(
        "how do I commit and open a pull request")
    assert out is None or "local:git-pr" not in out


# ---------------------------------------------------------------------------
# post_tool_observer — Skill tool emits skill.used
# ---------------------------------------------------------------------------

def _load_observer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "post_tool_observer", HOOKS / "post_tool_observer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skill_tool_invocation_records_skill_used(store, monkeypatch):
    import skill_hub.store as store_mod
    _seed_skill(store, "superpowers:brainstorming", name="brainstorming")
    db_path = store.db_path
    real_cls = store_mod.SkillStore
    monkeypatch.setattr(store_mod, "SkillStore",
                        lambda *a, **k: real_cls(db_path=db_path))

    observer = _load_observer()
    observer._maybe_emit_skill_used(
        "Skill", None, "sess-42", {"skill": "superpowers:brainstorming"})

    rows = store._conn.execute(
        "SELECT payload, tool_name FROM events WHERE kind = 'skill.used'"
    ).fetchall()
    assert len(rows) == 1
    import json
    payload = json.loads(rows[0]["payload"])
    assert payload["skill_id"] == "superpowers:brainstorming"
    assert rows[0]["tool_name"] == "Skill"


def test_skill_tool_unknown_ref_is_noop(store, monkeypatch):
    import skill_hub.store as store_mod
    db_path = store.db_path
    real_cls = store_mod.SkillStore
    monkeypatch.setattr(store_mod, "SkillStore",
                        lambda *a, **k: real_cls(db_path=db_path))

    observer = _load_observer()
    observer._maybe_emit_skill_used("Skill", None, "sess-42", {"skill": "ghost"})

    rows = store._conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind = 'skill.used'"
    ).fetchone()
    assert rows["n"] == 0
