"""Tests for dynamic vec0 dimension resolution (issue #35).

Verifies that:
- SkillStore sizes vec0 tables to the ACTUAL embedding dim on first write.
- SkillStore rebuilds vec0 tables when the declared dim mismatches existing data.

Uses tmp_path only — never touches the live DB at DB_PATH.
Imports only skill_hub.store (not skill_hub.server which instantiates a live store).
"""
import json

import pytest

from skill_hub.store import SkillStore


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "d.db"


def test_lazy_sizes_to_first_vector_dim(db_path):
    """vec0 tables are created at the dim of the first embedding written."""
    s = SkillStore(db_path)
    if s._vec_engine != "sqlite-vec":
        pytest.skip("sqlite-vec not available in this environment")

    # Before any write, vec0 tables may not exist yet (or may be pre-created
    # via expected_embedding_dim lookup).  Either way, after the first write
    # the dim must be 384.
    s.upsert_embedding("s1", "all-MiniLM-L6-v2", [0.01] * 384)

    assert s._vec_dim == 384, f"expected _vec_dim=384, got {s._vec_dim}"
    declared = s._vec0_declared_dim("skills_vec_f32")
    assert declared == 384, f"expected declared dim=384, got {declared}"

    # Verify a row actually landed in the vec table
    row = s._conn.execute(
        "SELECT skill_id FROM skills_vec_f32 WHERE skill_id = 's1'"
    ).fetchone()
    assert row is not None, "no row found in skills_vec_f32 after upsert_embedding"


def test_rebuild_on_dim_change(tmp_path):
    """On reopen, if existing vec0 tables differ in dim from stored embeddings, rebuild."""
    db_p = tmp_path / "rebuild.db"

    # --- Phase 1: create a store, manually force a stale 768-dim vec table ---
    s1 = SkillStore(db_p)
    if s1._vec_engine != "sqlite-vec":
        pytest.skip("sqlite-vec not available in this environment")

    # Drop whatever dim may exist and replace with a stale 768-dim table.
    s1._conn.execute("DROP TABLE IF EXISTS skills_vec_f32")
    s1._conn.execute(
        "CREATE VIRTUAL TABLE skills_vec_f32 USING vec0("
        "skill_id TEXT PRIMARY KEY, embedding float[768])"
    )
    # Insert a 384-dim row into the embeddings table directly to simulate
    # existing data at the correct (new) dimension.
    s1._conn.execute(
        "INSERT OR IGNORE INTO skills(id, name, description, content, plugin, target) "
        "VALUES('x','x','','','',  'claude')"
    )
    s1._conn.execute(
        "INSERT INTO embeddings(skill_id, model, vector, norm) VALUES('x','m',?,0)",
        (json.dumps([0.01] * 384),),
    )
    # Persist meta vec_dim at the stale value to simulate a pre-existing DB.
    s1._conn.execute(
        "INSERT INTO meta(key, value) VALUES('vec_dim','768') "
        "ON CONFLICT(key) DO UPDATE SET value='768'"
    )
    s1._conn.commit()

    # --- Phase 2: reopen; the store should detect dim mismatch and rebuild ---
    s2 = SkillStore(db_p)
    if s2._vec_engine != "sqlite-vec":
        pytest.skip("sqlite-vec not available after reopen")

    assert s2._vec_dim == 384, (
        f"after reopen _vec_dim should be 384 (from existing embeddings), got {s2._vec_dim}"
    )
    declared = s2._vec0_declared_dim("skills_vec_f32")
    assert declared == 384, (
        f"skills_vec_f32 should be rebuilt at float[384], got float[{declared}]"
    )
