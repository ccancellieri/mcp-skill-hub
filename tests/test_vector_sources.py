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


import sqlite3

from skill_hub.store import SkillStore


@pytest.fixture()
def seeded_store(tmp_path):
    db = tmp_path / "hub.db"
    store = SkillStore(db_path=db)
    store._conn.execute(
        "INSERT INTO tasks (title, summary, context, tags, vector, status) "
        "VALUES (?, ?, ?, ?, ?, 'open')",
        ("Task A", "summary A", "ctx A", "foo", "[0.1,0.2]"),
    )
    store._conn.execute(
        "INSERT INTO tasks (title, summary, context, tags, vector, status) "
        "VALUES (?, ?, ?, ?, ?, 'open')",
        ("Task B", "summary B", "ctx B", "bar", "[0.1,0.3]"),
    )
    store._conn.commit()
    yield store
    store._conn.close()


def test_task_source_index_stats(seeded_store):
    from skill_hub.vector_sources import TaskSource
    src = TaskSource(seeded_store)
    stats = src.index_stats()
    assert stats.name == "tasks"
    assert stats.doc_count == 2
    assert stats.embedded_count == 2
    assert stats.supports_merge is True
    assert stats.merge_mode == MergeMode.MECHANICAL


def test_task_source_mechanical_draft(seeded_store):
    from skill_hub.vector_sources import TaskSource
    src = TaskSource(seeded_store)
    ids = [str(r["id"]) for r in seeded_store._conn.execute(
        "SELECT id FROM tasks ORDER BY id").fetchall()]
    items = src.fetch_for_merge(ids)
    assert len(items) == 2

    draft = src.draft_merge(items, tier="mechanical", instruction="")
    assert "Task A" in draft.proposed_body
    assert "Task B" in draft.proposed_body
    assert draft.tier_used == "mechanical"


def test_task_source_commit_closes_originals(seeded_store):
    from skill_hub.vector_sources import TaskSource
    src = TaskSource(seeded_store)
    ids = [str(r["id"]) for r in seeded_store._conn.execute(
        "SELECT id FROM tasks ORDER BY id").fetchall()]
    items = src.fetch_for_merge(ids)
    draft = src.draft_merge(items, tier="mechanical", instruction="")
    result = src.commit_merge(items, draft)
    assert result.new_id is not None
    assert sorted(result.closed_ids) == sorted(ids)
    row = seeded_store._conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (int(ids[0]),)).fetchone()
    assert row["status"] == "closed"
