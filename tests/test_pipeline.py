"""Unit tests for the 4-tier pre-conversation pipeline (pipeline.py)."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.pipeline import (
    Pipeline,
    PipelineResult,
    TierResult,
    _run_with_timeout,
    _yake_classify,
)


# ---------------------------------------------------------------------------
# _yake_classify: deterministic, no deps
# ---------------------------------------------------------------------------

def test_yake_classify_short_message_complexity_low():
    result = _yake_classify("short msg")
    assert result["complexity"] == "low"


def test_yake_classify_long_message_complexity_high():
    long_msg = "word " * 100  # 500 chars -> high
    result = _yake_classify(long_msg)
    assert result["complexity"] == "high"


def test_yake_classify_medium_message():
    msg = "x " * 80  # 160 chars -> medium
    result = _yake_classify(msg)
    assert result["complexity"] == "medium"


def test_yake_classify_returns_required_keys():
    result = _yake_classify("Hello world test message here")
    for key in ("intent_tags", "domain_keywords", "complexity", "scope"):
        assert key in result


def test_yake_classify_intent_tags_max_4():
    result = _yake_classify("alpha beta gamma delta epsilon zeta")
    assert len(result["intent_tags"]) <= 4


# ---------------------------------------------------------------------------
# _run_with_timeout: timeout produces None
# ---------------------------------------------------------------------------

def test_run_with_timeout_fast_fn():
    def fast():
        return {"ok": True}

    result, elapsed_ms, timed_out = _run_with_timeout(fast, 1000)
    assert result == {"ok": True}
    assert timed_out is False
    assert elapsed_ms >= 0


def test_run_with_timeout_slow_fn_falls_back():
    def slow():
        time.sleep(5)
        return {"ok": True}

    result, elapsed_ms, timed_out = _run_with_timeout(slow, 50)
    assert result is None
    assert timed_out is True


def test_run_with_timeout_exception_treated_as_fallback():
    def boom():
        raise RuntimeError("boom")

    result, elapsed_ms, timed_out = _run_with_timeout(boom, 1000)
    assert result is None
    assert timed_out is False  # exception, not timeout


# ---------------------------------------------------------------------------
# PipelineResult.fallbacks property
# ---------------------------------------------------------------------------

def test_pipeline_result_fallbacks_empty_when_no_fallback():
    pr = PipelineResult()
    pr.tier1 = TierResult(ran=True, fallback_used=False)
    pr.tier2 = TierResult(ran=True, fallback_used=False)
    pr.tier3 = TierResult(ran=True, fallback_used=False)
    pr.tier4 = TierResult(ran=False, fallback_used=False)
    assert pr.fallbacks == []


def test_pipeline_result_fallbacks_lists_only_fallen_back_tiers():
    pr = PipelineResult()
    pr.tier1 = TierResult(ran=True, fallback_used=True)
    pr.tier2 = TierResult(ran=True, fallback_used=False)
    pr.tier3 = TierResult(ran=True, fallback_used=True)
    pr.tier4 = TierResult(ran=False, fallback_used=False)
    assert pr.fallbacks == ["tier1", "tier3"]


def test_pipeline_result_fallbacks_all_tiers():
    pr = PipelineResult()
    pr.tier1 = TierResult(ran=True, fallback_used=True)
    pr.tier2 = TierResult(ran=True, fallback_used=True)
    pr.tier3 = TierResult(ran=True, fallback_used=True)
    pr.tier4 = TierResult(ran=True, fallback_used=True)
    assert set(pr.fallbacks) == {"tier1", "tier2", "tier3", "tier4"}


# ---------------------------------------------------------------------------
# Helpers: minimal in-memory SkillStore substitute
# ---------------------------------------------------------------------------

def _make_mock_store(existing_task_row=None, similar_tasks=None):
    """Build a mock store with the methods used by Pipeline.run()."""
    store = MagicMock()
    store.get_open_task_for_session.return_value = existing_task_row
    store.search_tasks.return_value = similar_tasks or []
    store.save_task.return_value = 42
    store.update_task.return_value = True
    store.record_pipeline_run.return_value = 1
    return store


# ---------------------------------------------------------------------------
# Pipeline.run() — new task creation path
# ---------------------------------------------------------------------------

def test_pipeline_run_creates_task_when_no_open_task(monkeypatch):
    """When no existing session task, Pipeline.run() must call save_task."""
    # Patch config to disable LLM calls
    monkeypatch.setattr(
        "skill_hub.config.get",
        lambda key: {
            "pre_conversation_pipeline_enabled": True,
            "pipeline_tier1_timeout_ms": 500,
            "pipeline_tier2_timeout_ms": 400,
            "pipeline_tier3_timeout_ms": 1200,
            "pipeline_tier4_timeout_ms": 1500,
            "pipeline_tier4_min_complexity": "medium",
            "task_similarity_threshold": 0.75,
            "task_auto_create_min_chars": 0,
            "pipeline_synthesis_max_sentences": 5,
            "classify_backend": "yake_keywords",
            "synthesis_backend": "none",
            "rewrite_backend": "none",
        }.get(key),
    )

    # Patch embeddings to return empty (triggers FTS fallback in L2)
    monkeypatch.setattr(
        "skill_hub.embeddings.embed_available", lambda: False
    )

    store = _make_mock_store(existing_task_row=None, similar_tasks=[])
    store.search_text.return_value = []

    pipe = Pipeline()
    result = pipe.run(message="Fix the login bug in auth.py", session_id="s1", store=store)

    store.save_task.assert_called_once()
    assert result.task_id == 42
    store.record_pipeline_run.assert_called_once_with(
        session_id="s1",
        task_id=42,
        tier_ms=pytest.approx(
            {
                "tier1": result.tier1.duration_ms,
                "tier2": result.tier2.duration_ms,
                "tier3": result.tier3.duration_ms,
                "tier4": result.tier4.duration_ms,
            }
        ),
        fallbacks=result.fallbacks,
        top_similarity=None,
        token_cost_usd=0.0,
    )


def test_pipeline_run_updates_existing_task(monkeypatch):
    """When get_open_task_for_session returns a row, update_task is called."""
    monkeypatch.setattr(
        "skill_hub.config.get",
        lambda key: {
            "pre_conversation_pipeline_enabled": True,
            "pipeline_tier1_timeout_ms": 500,
            "pipeline_tier2_timeout_ms": 400,
            "pipeline_tier3_timeout_ms": 1200,
            "pipeline_tier4_timeout_ms": 1500,
            "pipeline_tier4_min_complexity": "medium",
            "task_similarity_threshold": 0.75,
            "task_auto_create_min_chars": 0,
            "pipeline_synthesis_max_sentences": 5,
            "classify_backend": "yake_keywords",
            "synthesis_backend": "none",
            "rewrite_backend": "none",
        }.get(key),
    )
    monkeypatch.setattr(
        "skill_hub.embeddings.embed_available", lambda: False
    )

    # Simulate existing open task for this session
    existing_row = MagicMock()
    existing_row.__getitem__ = lambda self, key: {"id": 99}[key]
    # Make dict() work on it
    existing_dict = {"id": 99}
    existing_row.keys = lambda: list(existing_dict.keys())

    # Use a simple sqlite3.Row-like dict instead
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT, summary TEXT)"
    )
    conn.execute("INSERT INTO tasks VALUES (99, 'old task', 'old summary')")
    conn.commit()
    row = conn.execute("SELECT * FROM tasks WHERE id=99").fetchone()

    store = _make_mock_store(existing_task_row=row)
    store.search_text.return_value = []

    pipe = Pipeline()
    result = pipe.run(
        message="Continue fixing the login bug", session_id="s1", store=store
    )

    store.update_task.assert_called_once_with(99, summary=pytest.approx("Continue fixing the login bug", rel=None, abs=None))
    store.save_task.assert_not_called()
    assert result.task_id == 99


# ---------------------------------------------------------------------------
# Tier timeout degrades gracefully (fallback_used=True)
# ---------------------------------------------------------------------------

def test_tier1_timeout_falls_back_gracefully(monkeypatch):
    """If tier1 classify times out, result.tier1.fallback_used == True."""
    monkeypatch.setattr(
        "skill_hub.config.get",
        lambda key: {
            "pre_conversation_pipeline_enabled": True,
            "pipeline_tier1_timeout_ms": 10,  # very short — will time out
            "pipeline_tier2_timeout_ms": 400,
            "pipeline_tier3_timeout_ms": 1200,
            "pipeline_tier4_timeout_ms": 1500,
            "pipeline_tier4_min_complexity": "medium",
            "task_similarity_threshold": 0.75,
            "task_auto_create_min_chars": 0,
            "pipeline_synthesis_max_sentences": 5,
            "classify_backend": "haiku_json",  # triggers LLM path (which we'll slow down)
        }.get(key),
    )
    monkeypatch.setattr(
        "skill_hub.embeddings.embed_available", lambda: False
    )

    # Patch the LLM provider to sleep (simulate slow response)
    slow_provider = MagicMock()
    slow_provider.complete.side_effect = lambda *a, **kw: (time.sleep(5) or "{}")

    monkeypatch.setattr(
        "skill_hub.llm.get_provider", lambda: slow_provider
    )

    store = _make_mock_store()
    store.search_text.return_value = []

    pipe = Pipeline()
    result = pipe.run(
        message="Fix the login bug", session_id="timeout-test", store=store
    )

    assert result.tier1.fallback_used is True
    # fallback classify still sets intent data
    assert "complexity" in result.tier1.data


# ---------------------------------------------------------------------------
# record_pipeline_run is called with correct session_id and task_id
# ---------------------------------------------------------------------------

def test_record_pipeline_run_called_with_correct_args(monkeypatch):
    monkeypatch.setattr(
        "skill_hub.config.get",
        lambda key: {
            "pre_conversation_pipeline_enabled": True,
            "pipeline_tier1_timeout_ms": 500,
            "pipeline_tier2_timeout_ms": 400,
            "pipeline_tier3_timeout_ms": 1200,
            "pipeline_tier4_timeout_ms": 1500,
            "pipeline_tier4_min_complexity": "high",  # L4 won't run
            "task_similarity_threshold": 0.75,
            "task_auto_create_min_chars": 0,
            "pipeline_synthesis_max_sentences": 5,
            "classify_backend": "yake_keywords",
        }.get(key),
    )
    monkeypatch.setattr(
        "skill_hub.embeddings.embed_available", lambda: False
    )

    store = _make_mock_store()
    store.search_text.return_value = []

    pipe = Pipeline()
    result = pipe.run(
        message="Test message for telemetry", session_id="telem-session", store=store
    )

    call_kwargs = store.record_pipeline_run.call_args
    assert call_kwargs is not None
    kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert kw["session_id"] == "telem-session"
    assert kw["task_id"] == result.task_id
    assert kw["token_cost_usd"] == 0.0
    assert isinstance(kw["fallbacks"], list)
