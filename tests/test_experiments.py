"""Tests for experimentation framework: pipeline presets + A/B experiments."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest

from skill_hub.store import SkillStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """Isolated SkillStore with fresh DB."""
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------


def test_list_presets_returns_builtins(store):
    """5 built-in presets must be seeded on DB creation."""
    presets = store.list_presets()
    assert len(presets) == 5
    names = {p["name"] for p in presets}
    assert names == {"fast-local", "cheap-cloud", "quality-cloud", "offline-only", "balanced"}


def test_builtin_presets_have_is_builtin_flag(store):
    """All seeded built-ins must carry is_builtin=1."""
    presets = store.list_presets()
    for p in presets:
        assert p["is_builtin"] == 1, f"expected is_builtin=1 for {p['name']}"


def test_save_and_get_preset(store):
    """Custom preset round-trip: save then get_preset by id."""
    config = {"classify_backend": "yake", "synthesis_backend": "haiku"}
    pid = store.save_preset("my-custom", "custom test preset", config)
    assert isinstance(pid, int) and pid > 0

    got = store.get_preset(pid)
    assert got is not None
    assert got["name"] == "my-custom"
    assert got["description"] == "custom test preset"
    assert got["config"]["classify_backend"] == "yake"
    assert got["is_builtin"] == 0


def test_preset_config_is_dict(store):
    """config_json column is deserialized to a plain dict, never a string."""
    presets = store.list_presets()
    for p in presets:
        assert isinstance(p["config"], dict), f"config for {p['name']} is not dict"


def test_delete_builtin_preset_fails(store):
    """delete_preset must return False (and not delete) for built-in presets."""
    builtin = next(p for p in store.list_presets() if p["is_builtin"])
    deleted = store.delete_preset(builtin["id"])
    assert deleted is False
    # still present
    assert store.get_preset(builtin["id"]) is not None


def test_delete_custom_preset_succeeds(store):
    """delete_preset returns True and removes a custom preset."""
    pid = store.save_preset("to-delete", "", {})
    assert store.get_preset(pid) is not None
    deleted = store.delete_preset(pid)
    assert deleted is True
    assert store.get_preset(pid) is None


def test_list_presets_ordering(store):
    """Built-ins come before custom presets (is_builtin DESC, name ASC)."""
    store.save_preset("zzz-custom", "", {})
    presets = store.list_presets()
    builtin_idx = [i for i, p in enumerate(presets) if p["is_builtin"]]
    custom_idx  = [i for i, p in enumerate(presets) if not p["is_builtin"]]
    assert max(builtin_idx) < min(custom_idx)


# ---------------------------------------------------------------------------
# Experiment tests
# ---------------------------------------------------------------------------


def test_list_experiments_empty(store):
    """No experiments on a fresh DB."""
    exps = store.list_experiments()
    assert exps == []


def test_create_experiment(store):
    """create_experiment returns a valid integer id."""
    presets = store.list_presets()
    a_id = presets[0]["id"]
    b_id = presets[1]["id"]
    eid = store.create_experiment("test-exp", a_id, b_id, target_runs=5, notes="hello")
    assert isinstance(eid, int) and eid > 0


def test_list_experiments_with_data(store):
    """list_experiments returns created experiment with joined preset names."""
    presets = store.list_presets()
    a = next(p for p in presets if p["name"] == "fast-local")
    b = next(p for p in presets if p["name"] == "balanced")
    store.create_experiment("local-vs-balanced", a["id"], b["id"])

    exps = store.list_experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp["name"] == "local-vs-balanced"
    assert exp["preset_a_name"] == "fast-local"
    assert exp["preset_b_name"] == "balanced"
    assert exp["status"] == "active"
    assert exp["target_runs"] == 10


def test_get_experiment_stats_empty(store):
    """Stats for an experiment with no runs returns zero counts."""
    presets = store.list_presets()
    eid = store.create_experiment("empty-exp", presets[0]["id"], presets[1]["id"])
    stats = store.get_experiment_stats(eid)
    assert stats["A"]["runs"] == 0
    assert stats["B"]["runs"] == 0
    assert stats["A"]["avg_cost"] is None
    assert stats["B"]["avg_cost"] is None


def test_get_experiment_stats_with_runs(store):
    """Stats aggregate correctly when experiment_runs rows are inserted directly."""
    presets = store.list_presets()
    eid = store.create_experiment("stats-exp", presets[0]["id"], presets[1]["id"])

    # Insert mock runs directly into the table
    conn = store._conn
    conn.executemany(
        "INSERT INTO experiment_runs (experiment_id, preset_tag, token_cost_usd, user_rating) "
        "VALUES (?, ?, ?, ?)",
        [
            (eid, "A", 0.004, 4),
            (eid, "A", 0.006, 5),
            (eid, "B", 0.010, 3),
        ],
    )
    conn.commit()

    stats = store.get_experiment_stats(eid)
    assert stats["A"]["runs"] == 2
    assert stats["B"]["runs"] == 1
    assert abs(stats["A"]["avg_cost"] - 0.005) < 1e-6
    assert abs(stats["A"]["avg_rating"] - 4.5) < 1e-6
    assert abs(stats["B"]["avg_rating"] - 3.0) < 1e-6
