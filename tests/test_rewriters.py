"""Compatibility tests for prompt enrichment backed by scoped context."""
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
    assert "normalize_language" not in names


def test_builtin_context_uses_scoped_task_and_preserves_original_body(store):
    from skill_hub.router import rewriters

    store.save_task(
        title="Wire up bandit",
        summary="Bandit MCP tools need docs",
        vector=[0.0] * 8,
        session_id="session-a",
        cwd="/projects/alpha",
    )
    prompt = "Keep this exact line.\nReview the Bandit MCP docs."
    result = rewriters.improve_prompt(
        prompt, store, rewriters=["add_recent_tasks"],
        cwd="/projects/alpha", session_id="session-a",
    )
    assert result.applied == ["context"]
    assert "Wire up bandit" in result.prompt
    assert result.original == prompt
    assert result.prompt.startswith(prompt + "\n\n")


def test_builtin_context_abstains_from_unrelated_scoped_task(store):
    from skill_hub.router import rewriters

    store.save_task(
        title="Wire up bandit", summary="Bandit MCP tools need docs",
        vector=[0.0] * 8, session_id="session-a", cwd="/projects/alpha",
    )
    prompt = "Keep this exact line.\nAnd this one too."
    result = rewriters.improve_prompt(
        prompt, store, rewriters=["add_recent_tasks"],
        cwd="/projects/alpha", session_id="session-a",
    )
    assert result.applied == []
    assert result.prompt == result.original == prompt


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


def test_default_chain_uses_shared_context_builder(store, monkeypatch):
    from skill_hub.router import rewriters

    received = {}

    def fake_build_context(prompt, **kwargs):
        received.update(prompt=prompt, **kwargs)
        return {"context": "[memory] project note", "warnings": []}

    monkeypatch.setattr("skill_hub.context_service.build_context", fake_build_context)

    result = rewriters.improve_prompt(
        "hello", store, cfg={"context_enabled": True}, cwd="/project",
        session_id="session-1", task_id=4,
    )
    assert received == {
        "prompt": "hello", "store": store, "cfg": {"context_enabled": True},
        "cwd": "/project", "session_id": "session-1", "task_id": 4,
    }
    assert result.prompt == "hello\n\n[memory] project note"
    assert result.applied == ["context"]


def test_normalize_language_is_retired_and_preserves_original_prompt(store):
    from skill_hub.router import rewriters
    prompt = "please help me figure out pagination"
    result = rewriters.improve_prompt(
        prompt,
        store, rewriters=["normalize_language"],
    )
    assert result.applied == []
    assert result.prompt == prompt
    assert any("retired" in note for note in result.notes)


def test_body_replacement_is_ignored_and_original_body_is_preserved(store, monkeypatch):
    from skill_hub.router import rewriters

    def replacer(prompt, store, cfg):
        return rewriters.RewriterResult(body="REWRITTEN", note="ok", applied=True)

    monkeypatch.setitem(rewriters._REGISTRY, "replacer", replacer)
    result = rewriters.improve_prompt(
        "original text", store, rewriters=["replacer"]
    )
    assert result.prompt == "original text"
    assert result.applied == []
    assert any("body replacement ignored" in note for note in result.notes)


def test_no_cwd_does_not_leak_foreign_task(store):
    from skill_hub.router import rewriters

    store.save_task(
        title="Foreign task", summary="This must remain scoped to beta.", vector=[0.0] * 8,
        session_id="session-b", cwd="/projects/beta",
    )

    result = rewriters.improve_prompt(
        "continue", store, rewriters=["add_recent_tasks"], session_id="session-a",
    )

    assert result.prompt == "continue"
    assert "Foreign task" not in result.prompt


def test_context_builder_errors_are_contained(store, monkeypatch):
    from skill_hub.router import rewriters

    def boom(*_args, **_kwargs):
        raise RuntimeError("context unavailable")

    monkeypatch.setattr("skill_hub.context_service.build_context", boom)

    result = rewriters.improve_prompt("original text", store)

    assert result.prompt == "original text"
    assert any("context" in note and "error" in note for note in result.notes)
