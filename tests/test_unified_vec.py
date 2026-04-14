"""Tests for S6 F-MEM — unified sqlite-vec search for tasks + teachings."""
from __future__ import annotations

import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


VEC_DIM = 768


def _rand_vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(VEC_DIM)]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


def _vec_engine_available(store) -> bool:
    return store._vec_engine == "sqlite-vec"


def test_tasks_mirror_written_on_save(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    task_id = store.save_task("bandit docs", "document route_to_model", _rand_vec(1))
    rows = store._conn.execute(
        "SELECT task_id FROM tasks_vec_bin WHERE task_id = ?", (task_id,)
    ).fetchall()
    assert len(rows) == 1
    rows_f32 = store._conn.execute(
        "SELECT task_id FROM tasks_vec_f32 WHERE task_id = ?", (task_id,)
    ).fetchall()
    assert len(rows_f32) == 1


def test_search_tasks_returns_nearest(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    target_vec = _rand_vec(42)
    # Add 3 tasks — one close to target, two far.
    store.save_task("close task", "this one is the hit", target_vec)
    store.save_task("far task A", "irrelevant", _rand_vec(99))
    store.save_task("far task B", "irrelevant too", _rand_vec(123))

    hits = store.search_tasks(target_vec, top_k=1, min_sim=0.1)
    assert len(hits) == 1
    assert hits[0]["title"] == "close task"


def test_delete_task_cleans_vec_tables(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    tid = store.save_task("t", "s", _rand_vec(1))
    assert store.delete_task(tid) is True
    rows = store._conn.execute(
        "SELECT task_id FROM tasks_vec_bin WHERE task_id = ?", (tid,)
    ).fetchall()
    assert rows == []


def test_teachings_mirror_written_on_add(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    tid = store.add_teaching(
        "when I give a URL suggest chrome-devtools",
        _rand_vec(2),
        "suggest", "plugin", "chrome-devtools-mcp",
        weight=1.0,
    )
    rows = store._conn.execute(
        "SELECT teaching_id FROM teachings_vec_bin WHERE teaching_id = ?", (tid,)
    ).fetchall()
    assert len(rows) == 1


def test_search_teachings_returns_nearest(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    target_vec = _rand_vec(7)
    store.add_teaching("rule A", target_vec, "suggest", "plugin", "pluginA", weight=1.0)
    store.add_teaching("rule B", _rand_vec(88), "suggest", "plugin", "pluginB", weight=1.0)
    hits = store.search_teachings(target_vec, min_sim=0.1)
    assert hits
    assert hits[0]["rule"] == "rule A"


def test_remove_teaching_cleans_vec_tables(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    tid = store.add_teaching("x", _rand_vec(1), "suggest", "plugin", "p", weight=1.0)
    assert store.remove_teaching(tid) is True
    rows = store._conn.execute(
        "SELECT teaching_id FROM teachings_vec_bin WHERE teaching_id = ?", (tid,)
    ).fetchall()
    assert rows == []


def test_update_task_vector_updates_mirror(store):
    if not _vec_engine_available(store):
        pytest.skip("sqlite-vec not available")
    tid = store.save_task("t", "s", _rand_vec(10))
    new_vec = _rand_vec(77)
    assert store.update_task(tid, vector=new_vec) is True
    # mirror row should still exist after update
    rows = store._conn.execute(
        "SELECT task_id FROM tasks_vec_bin WHERE task_id = ?", (tid,)
    ).fetchall()
    assert len(rows) == 1
