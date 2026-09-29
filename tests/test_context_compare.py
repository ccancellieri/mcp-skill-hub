"""Behavior tests for the reproducible context-route comparison harness."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "context_compare.py"
_SPEC = importlib.util.spec_from_file_location("context_compare", _SCRIPT)
assert _SPEC and _SPEC.loader
context_compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(context_compare)


def _corpus() -> dict:
    return {
        "skills": [{
            "id": "skill-alpha",
            "name": "Exact Skill Alpha",
            "description": "alpha routing evidence",
            "content": "EVIDENCE_SKILL_ALPHA",
            "plugin": "fixture",
            "file_path": "/synthetic/skills/alpha.md",
        }],
        "tasks": [{
            "key": "TASK_ALPHA",
            "title": "Alpha task",
            "summary": "EVIDENCE_TASK_ALPHA",
            "context": "alpha context",
            "cwd": "/synthetic/project",
            "session_id": "session-alpha",
        }],
        "memories": [{
            "key": "MEMORY_ALPHA",
            "project": "/synthetic/project",
            "text": "EVIDENCE_MEMORY_ALPHA",
        }],
        "wiki": [{
            "key": "WIKI_ALPHA",
            "project": "/synthetic/project",
            "text": "EVIDENCE_WIKI_ALPHA",
        }],
        "cases": [{
            "id": "alpha",
            "group": "answerable",
            "prompt": "find alpha",
            "cwd": "/synthetic/project",
            "session_id": "session-alpha",
            "task_key": "TASK_ALPHA",
            "expected_sources": ["skill:skill-alpha", "task:TASK_ALPHA"],
            "forbidden_sources": ["memory:MEMORY_ALPHA"],
        }],
    }


def test_validate_corpus_rejects_unknown_canonical_source():
    corpus = _corpus()
    corpus["cases"][0]["expected_sources"] = ["task:NOT_IN_FIXTURE"]

    with pytest.raises(ValueError, match="unknown source"):
        context_compare.validate_corpus(corpus)


def test_extract_sources_ignores_original_prompt_occurrences_only():
    corpus = _corpus()
    catalog = context_compare.source_catalog(corpus)
    prompt = "Exact Skill Alpha EVIDENCE_TASK_ALPHA"

    assert context_compare.extract_sources(prompt, prompt, catalog) == []

    injection = prompt + "\nEVIDENCE_MEMORY_ALPHA\nExact Skill Alpha"
    assert context_compare.extract_sources(injection, prompt, catalog) == [
        "memory:MEMORY_ALPHA",
        "skill:skill-alpha",
    ]


def test_tokenizer_requires_an_explicit_populated_offline_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="TIKTOKEN_CACHE_DIR"):
        context_compare.load_tokenizer()

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="not cached"):
        context_compare.load_tokenizer()

    cache_key = "fb374d419588a4632f3f557e76b4b70aebbca790"
    (tmp_path / cache_key).write_bytes(b"corrupt tokenizer cache")
    with pytest.raises(RuntimeError, match="checksum"):
        context_compare.load_tokenizer()


def test_measurement_counts_original_and_additional_context_but_not_ui_warning():
    import os

    if not os.environ.get("TIKTOKEN_CACHE_DIR"):
        pytest.skip("requires a pre-provisioned offline tokenizer cache")
    tokenizer = context_compare.load_tokenizer()
    prompt = "Preserve this original prompt."
    output = {
        "systemMessage": "A UI-only warning with EVIDENCE_TASK_ALPHA.",
        "hookSpecificOutput": {"additionalContext": "EVIDENCE_TASK_ALPHA"},
    }

    measured = context_compare.measure_output(prompt, output, tokenizer)

    assert measured["original_tokens"] == len(tokenizer.encode(prompt))
    assert measured["additional_context_tokens"] == len(tokenizer.encode("EVIDENCE_TASK_ALPHA"))
    assert measured["primary_total_tokens"] == (
        measured["original_tokens"] + measured["additional_context_tokens"]
    )
    assert measured["ui_warning_tokens"] == len(tokenizer.encode(output["systemMessage"]))


def test_quality_keeps_no_answer_and_foreign_leakage_separate():
    answerable = context_compare.score_sources(
        expected=["task:TASK_ALPHA"],
        forbidden=["memory:MEMORY_ALPHA"],
        retrieved=["task:TASK_ALPHA", "memory:MEMORY_ALPHA"],
    )
    no_answer = context_compare.score_sources(
        expected=[],
        forbidden=["task:TASK_ALPHA"],
        retrieved=["task:TASK_ALPHA"],
    )

    assert answerable["precision"] == 0.5
    assert answerable["recall"] == 1.0
    assert answerable["foreign_source_leakage"] is True
    assert no_answer["no_answer_correct"] is False
    assert no_answer["precision"] is None
    assert no_answer["recall"] is None


def test_claim_gate_requires_lower_tokens_and_higher_quality_for_every_old_profile():
    comparisons = {
        "old_default": {"lower_tokens": True, "higher_quality": True},
        "old_rewriter_on": {"lower_tokens": True, "higher_quality": False},
    }

    assert context_compare.claim_gate(comparisons) is False

    comparisons["old_rewriter_on"]["higher_quality"] = True
    assert context_compare.claim_gate(comparisons) is True
