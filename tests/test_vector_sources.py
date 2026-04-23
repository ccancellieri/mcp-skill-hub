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


from skill_hub.vector_sources import SourceRegistry


class _FakeSource:
    name = "fake"
    merge_mode = MergeMode.REJECTED

    def index_stats(self):
        return IndexStats(
            name="fake", source_type="first_class_table", doc_count=0,
            embedded_count=0, avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=fake", supports_merge=False,
            merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): raise NotImplementedError
    def commit_merge(self, items, draft): raise NotImplementedError


def test_registry_register_and_get():
    reg = SourceRegistry()
    s = _FakeSource()
    reg.register(s)
    assert reg.get("fake") is s
    assert [x.name for x in reg.all()] == ["fake"]


def test_registry_get_missing_raises():
    reg = SourceRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")
