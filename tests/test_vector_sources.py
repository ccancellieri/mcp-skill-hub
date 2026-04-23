"""Tests for vector_sources module — protocol, dataclasses, registry."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from skill_hub.vector_sources import (
    IndexStats,
    MergeItem,
    MergeDraft,
    CommitResult,
    MergeMode,
)


def test_index_stats_defaults():
    s = IndexStats(
        name="tasks", source_type="first_class_table", doc_count=10,
        embedded_count=8, avg_age_days=2.5, last_indexed="2026-04-23",
        embedding_model="local-mini", level_breakdown={},
        scatter_url="/vector?source=tasks", supports_merge=True,
        merge_mode=MergeMode.MECHANICAL,
    )
    assert s.name == "tasks"
    assert s.supports_merge is True
    assert s.merge_mode == MergeMode.MECHANICAL


def test_merge_draft_minimal():
    d = MergeDraft(
        proposed_label="merged", proposed_body="body",
        proposed_raw={}, tier_used="local", tokens_used=None,
    )
    assert d.proposed_label == "merged"


def test_commit_result_shape():
    r = CommitResult(new_id="42", closed_ids=["1", "2"], audit_id=7)
    assert r.closed_ids == ["1", "2"]
