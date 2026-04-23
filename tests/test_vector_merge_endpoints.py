"""Endpoint tests for /vector/merge/draft and /vector/merge/commit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from skill_hub.store import SkillStore
from skill_hub.vector_sources import (
    SourceRegistry,
    TaskSource,
    VerdictSource,
    NamespaceSource,
)
from skill_hub import vector_sources as vs_mod
from skill_hub.webapp.routes import vector as vector_routes


TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent / "src/skill_hub/webapp/templates"
)


@pytest.fixture()
def client(tmp_path):
    db = tmp_path / "hub.db"
    store = SkillStore(db_path=db)
    store._conn.execute(
        "INSERT INTO tasks (title, summary, context, tags, vector, status) "
        "VALUES ('A', 'a', '', '', '[0.1]', 'open')"
    )
    store._conn.execute(
        "INSERT INTO tasks (title, summary, context, tags, vector, status) "
        "VALUES ('B', 'b', '', '', '[0.2]', 'open')"
    )
    store._conn.commit()

    app = FastAPI()
    app.state.store = store
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    reg = SourceRegistry()
    reg.register(TaskSource(store))
    reg.register(VerdictSource(None))
    app.state.source_registry = reg
    app.include_router(vector_routes.router)
    return TestClient(app), store


def test_draft_verdicts_rejected(client):
    c, _ = client
    r = c.post("/vector/merge/draft", json={"source": "verdicts", "ids": ["1", "2"]})
    assert r.status_code == 400
    assert "merge_rejected" in r.json()["error"]


def test_draft_too_few_ids(client):
    c, _ = client
    r = c.post("/vector/merge/draft", json={"source": "tasks", "ids": ["1"]})
    assert r.status_code == 400


def test_draft_tasks_returns_mechanical_body(client):
    c, store = client
    ids = [
        str(r["id"])
        for r in store._conn.execute("SELECT id FROM tasks ORDER BY id").fetchall()
    ]
    r = c.post(
        "/vector/merge/draft",
        json={"source": "tasks", "ids": ids, "tier": "mechanical"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "proposed_body" in body
    assert body["tier_used"] == "mechanical"


def test_commit_tasks_closes_originals(client):
    c, store = client
    ids = [
        str(r["id"])
        for r in store._conn.execute("SELECT id FROM tasks ORDER BY id").fetchall()
    ]
    draft = c.post(
        "/vector/merge/draft",
        json={"source": "tasks", "ids": ids, "tier": "mechanical"},
    ).json()
    commit = c.post(
        "/vector/merge/commit",
        json={"source": "tasks", "ids": ids, "draft": draft},
    )
    assert commit.status_code == 200, commit.text
    body = commit.json()
    assert body["new_id"]
    assert sorted(body["closed_ids"]) == sorted(ids)

    row = store._conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (int(ids[0]),)
    ).fetchone()
    assert row["status"] == "closed"


def test_commit_rejects_verdicts(client):
    c, _ = client
    r = c.post(
        "/vector/merge/commit",
        json={
            "source": "verdicts",
            "ids": ["a", "b"],
            "draft": {
                "proposed_label": "x",
                "proposed_body": "y",
                "proposed_raw": {},
                "tier_used": "mechanical",
            },
        },
    )
    assert r.status_code == 400


def test_legacy_merge_shim_tasks(client):
    c, store = client
    ids = [
        str(r["id"])
        for r in store._conn.execute("SELECT id FROM tasks ORDER BY id").fetchall()
    ]
    r = c.post("/vector/merge", json={"source": "tasks", "ids": ids})
    assert r.status_code == 200, r.text
    assert "merged_into_id" in r.json()


@pytest.fixture()
def memory_client(tmp_path, monkeypatch):
    db = tmp_path / "hub.db"
    store = SkillStore(db_path=db)
    for i in range(3):
        store._conn.execute(
            "INSERT INTO vectors (doc_id, namespace, level, metadata, vector, "
            "access_count, indexed_at, last_accessed) "
            "VALUES (?, 'skills', 'L2', ?, '[0.1]', ?, datetime('now'), datetime('now'))",
            (
                f"doc-{i}",
                json.dumps({"title": f"T{i}", "content": f"body{i}"}),
                i + 1,
            ),
        )
    store._conn.commit()

    class _Stub:
        def complete(self, prompt, **kw):
            return "CONSOLIDATED"

    monkeypatch.setattr(vs_mod, "_get_provider", lambda: _Stub())

    app = FastAPI()
    app.state.store = store
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    reg = SourceRegistry()
    reg.register(NamespaceSource(store, namespace="skills"))
    app.state.source_registry = reg
    app.include_router(vector_routes.router)
    return TestClient(app), store


def test_end_to_end_memory_merge(memory_client):
    c, store = memory_client
    ids = ["doc-0", "doc-1", "doc-2"]

    draft = c.post(
        "/vector/merge/draft",
        json={"source": "skills", "ids": ids, "tier": "local"},
    ).json()
    assert draft["proposed_body"] == "CONSOLIDATED"
    assert draft["proposed_raw"]["access_count"] == 6

    commit = c.post(
        "/vector/merge/commit",
        json={"source": "skills", "ids": ids, "draft": draft},
    ).json()
    assert commit["new_id"]
    assert sorted(commit["closed_ids"]) == ids

    remaining = store._conn.execute(
        "SELECT doc_id, access_count FROM vectors WHERE namespace='skills'"
    ).fetchall()
    assert len(remaining) == 1
    assert remaining[0]["doc_id"] == commit["new_id"]
    assert remaining[0]["access_count"] == 6

    audit = store._conn.execute(
        "SELECT reason FROM memory_audit WHERE id = ?", (commit["audit_id"],)
    ).fetchone()
    reason = json.loads(audit["reason"])
    assert reason["closed_ids"] == ids
    assert "rollback" in reason
    assert len(reason["rollback"]["restore_docs"]) == 3
