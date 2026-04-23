"""Tests for session_binding — resume-or-create match ladder + marker writer."""
import json
from pathlib import Path

import pytest

from skill_hub.store import SkillStore
from skill_hub import session_binding


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    marker = tmp_path / "active_task.json"
    monkeypatch.setattr(session_binding, "ACTIVE_TASK_MARKER", marker)
    store = SkillStore(db_path=tmp_path / "sh.db")
    yield store, marker
    store.close()


def _seed(store, title, cwd, branch, session_id="old-sid", vector=None):
    return store.save_task(
        title=title, summary="seed", vector=vector or [],
        cwd=cwd, branch=branch, session_id=session_id,
    )


def test_create_new_when_no_open_tasks(tmp_store):
    store, marker = tmp_store
    action, task_id, title, reason = session_binding.bind_session_to_task(
        session_id="new-sid", message="Fix the CORS bug in proxy",
        cwd="/tmp/projA", branch="main", store=store,
    )
    assert action == "created"
    assert task_id > 0
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert data["task_id"] == task_id
    assert data["session_id"] == "new-sid"


def test_resume_on_cwd_branch_match(tmp_store):
    store, marker = tmp_store
    tid = _seed(store, "old work", "/tmp/projA", "main")
    action, task_id, _, reason = session_binding.bind_session_to_task(
        session_id="new-sid", message="continue",
        cwd="/tmp/projA", branch="main", store=store,
    )
    assert action == "resumed"
    assert task_id == tid
    assert reason == "cwd+branch"
    row = store._conn.execute("SELECT session_id FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["session_id"] == "new-sid"


def test_no_resume_when_branch_differs(tmp_store):
    store, _ = tmp_store
    _seed(store, "on main", "/tmp/projA", "main")
    action, _, _, _ = session_binding.bind_session_to_task(
        session_id="new-sid", message="x",
        cwd="/tmp/projA", branch="feature/x", store=store,
    )
    assert action == "created"


def test_no_resume_outside_window(tmp_store, monkeypatch):
    store, _ = tmp_store
    tid = _seed(store, "stale", "/tmp/projA", "main")
    store._conn.execute(
        "UPDATE tasks SET updated_at = datetime('now', '-30 days') WHERE id=?", (tid,)
    )
    store._conn.commit()
    monkeypatch.setattr(session_binding, "_get_config",
                        lambda: {"enabled": True, "strategy": "cwd_branch",
                                 "window_days": 7, "semantic_threshold": 0.75})
    action, _, _, _ = session_binding.bind_session_to_task(
        session_id="sid", message="x", cwd="/tmp/projA", branch="main", store=store,
    )
    assert action == "created"


def test_semantic_match_resumes_when_embeddings_available(tmp_store, monkeypatch):
    store, _ = tmp_store
    vec = [0.1, 0.9, 0.0]
    tid = _seed(store, "proxy bug", "/tmp/other", "main", vector=vec)
    monkeypatch.setattr(session_binding, "_embed_message", lambda _m: vec)
    monkeypatch.setattr(session_binding, "_get_config",
                        lambda: {"enabled": True, "strategy": "semantic",
                                 "window_days": 7, "semantic_threshold": 0.5})
    action, task_id, _, reason = session_binding.bind_session_to_task(
        session_id="sid", message="investigate the proxy cors issue",
        cwd="/tmp/different", branch="x", store=store,
    )
    assert action == "resumed"
    assert task_id == tid
    assert reason.startswith("semantic")


def test_semantic_skipped_when_embeddings_fail(tmp_store, monkeypatch):
    store, _ = tmp_store
    _seed(store, "t", "/tmp/other", "main", vector=[0.1, 0.9, 0.0])
    monkeypatch.setattr(session_binding, "_embed_message", lambda _m: [])
    monkeypatch.setattr(session_binding, "_get_config",
                        lambda: {"enabled": True, "strategy": "hybrid",
                                 "window_days": 7, "semantic_threshold": 0.5})
    action, _, _, _ = session_binding.bind_session_to_task(
        session_id="sid", message="anything long enough to try semantic",
        cwd="/tmp/new", branch="b", store=store,
    )
    assert action == "created"


def test_disabled_skips_all(tmp_store, monkeypatch):
    store, marker = tmp_store
    _seed(store, "t", "/tmp/projA", "main")
    monkeypatch.setattr(session_binding, "_get_config",
                        lambda: {"enabled": False, "strategy": "hybrid",
                                 "window_days": 7, "semantic_threshold": 0.75})
    action, task_id, _, _ = session_binding.bind_session_to_task(
        session_id="sid", message="x", cwd="/tmp/projA", branch="main", store=store,
    )
    assert action == "skipped"
    assert task_id == 0
    assert not marker.exists()


def test_marker_refreshed_on_resume(tmp_store):
    store, marker = tmp_store
    tid = _seed(store, "t", "/tmp/projA", "main")
    session_binding.bind_session_to_task(
        session_id="brand-new", message="x",
        cwd="/tmp/projA", branch="main", store=store,
    )
    data = json.loads(marker.read_text())
    assert data["task_id"] == tid
    assert data["session_id"] == "brand-new"
