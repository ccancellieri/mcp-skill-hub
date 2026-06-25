"""Tests for S5 F-PROMPT — pluggable prompt rewriters."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


def test_registry_has_builtins():
    from skill_hub.router import rewriters

    names = rewriters.available()
    assert "add_skill_context" in names
    assert "add_recent_tasks" in names
    assert "normalize_language" in names


def test_add_recent_tasks_prefixes_open_task(store):
    from skill_hub.router import rewriters

    store.save_task(
        title="Wire up bandit",
        summary="Bandit MCP tools need docs",
        vector=[0.0] * 8,
    )
    result = rewriters.improve_prompt(
        "what next?", store, rewriters=["add_recent_tasks"]
    )
    assert "add_recent_tasks" in result.applied
    assert "Wire up bandit" in result.prompt
    assert result.original == "what next?"


def test_unknown_rewriter_is_noted_not_raised(store):
    from skill_hub.router import rewriters

    result = rewriters.improve_prompt(
        "hello", store, rewriters=["does_not_exist"]
    )
    assert result.applied == []
    assert any("does_not_exist" in n for n in result.notes)
    assert result.prompt == "hello"


def test_rewriter_errors_are_contained(store, monkeypatch):
    from skill_hub.router import rewriters

    def boom(prompt, store, cfg):
        raise RuntimeError("kaboom")

    rewriters.register("boom", boom)
    try:
        result = rewriters.improve_prompt(
            "hi", store, rewriters=["boom"]
        )
        assert result.applied == []
        assert any("error" in n for n in result.notes)
    finally:
        rewriters._REGISTRY.pop("boom", None)


def test_default_chain_runs_when_rewriters_none(store, monkeypatch):
    from skill_hub.router import rewriters

    calls: list[str] = []

    def fake_skill(prompt, store, cfg):
        calls.append("skill")
        return rewriters.RewriterResult(prefix="[skills]", note="ok", applied=True)

    def fake_tasks(prompt, store, cfg):
        calls.append("tasks")
        return rewriters.RewriterResult(note="no tasks")

    monkeypatch.setitem(rewriters._REGISTRY, "add_skill_context", fake_skill)
    monkeypatch.setitem(rewriters._REGISTRY, "add_recent_tasks", fake_tasks)

    result = rewriters.improve_prompt("hello", store, cfg={})
    assert calls == ["skill", "tasks"]
    assert result.prompt.startswith("[skills]")
    assert "hello" in result.prompt


def test_normalize_language_routes_via_ladder(store, monkeypatch):
    """G4: the optimizer must hit the gateway ladder, not pin a local model.

    It calls the provider with ``op="improve_prompt"`` (ladder-eligible) and no
    explicit ``model``, so chat() routes through L1 -> work gateway.
    """
    from skill_hub.router import rewriters
    from skill_hub import llm as _llm

    captured: dict = {}

    class _FakeProvider:
        def chat(self, messages, **kwargs):
            captured.update(kwargs)
            captured["messages"] = messages
            return "tighten the OLAP pagination question"

    monkeypatch.setattr(_llm, "get_provider", lambda: _FakeProvider())
    result = rewriters.improve_prompt(
        "please help me figure out how olap dimension pagination should work",
        store, rewriters=["normalize_language"],
    )
    assert captured.get("op") == "improve_prompt"
    assert "model" not in captured  # no pinned model -> ladder picks it
    assert result.applied == ["normalize_language"]
    assert result.prompt == "tighten the OLAP pagination question"


def test_normalize_language_skips_short_prompt(store):
    from skill_hub.router import rewriters

    result = rewriters.improve_prompt("hi", store, rewriters=["normalize_language"])
    assert result.applied == []
    assert any("too short" in n for n in result.notes)


def test_improve_prompt_op_is_ladder_eligible():
    """The op must carry a routing signal so chat() escalates to the gateway."""
    from skill_hub.llm import litellm_adapter as la

    assert "improve_prompt" in la._OP_ROUTING
    complexity, domain = la._OP_ROUTING["improve_prompt"]
    assert 0.0 < complexity <= 0.5 and domain  # cheap + signalled


def test_body_rewrite_replaces_prompt(store, monkeypatch):
    from skill_hub.router import rewriters

    def replacer(prompt, store, cfg):
        return rewriters.RewriterResult(body="REWRITTEN", note="ok", applied=True)

    monkeypatch.setitem(rewriters._REGISTRY, "replacer", replacer)
    result = rewriters.improve_prompt(
        "original text", store, rewriters=["replacer"]
    )
    assert result.prompt == "REWRITTEN"
    assert result.applied == ["replacer"]
