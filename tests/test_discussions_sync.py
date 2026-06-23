"""Tests for discussions_sync module (issue #41).

Covers:
(a) Happy path: 2 discussions (one with 2 comments, one with 0) → checked=2,
    discussions=2, comments=2, indexed=4, 4 rows in the discussions namespace.
(b) dry_run=True: same counts returned, but 0 rows written to the DB.
(c) Graceful failure when _gh_graphql returns None → error key, indexed==0,
    no exception.
(d) Graceful failure when _resolve_repo returns None → error key, no exception.

All tests are fast and offline-safe:
- No subprocess is called (helpers are monkeypatched).
- No ML model is loaded (embeddings.embed is monkeypatched).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """A fresh DB-backed SkillStore that does not touch the live DB."""
    from skill_hub.store import SkillStore

    db_path = tmp_path / "test_discussions_sync.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


# Fixed two-discussion payload returned by the mocked _gh_graphql.
_TWO_DISCUSSIONS_PAYLOAD = {
    "data": {
        "repository": {
            "discussions": {
                "nodes": [
                    {
                        "number": 1,
                        "title": "How do I do X?",
                        "url": "https://github.com/o/n/discussions/1",
                        "body": "I tried X but it didn't work.",
                        "updatedAt": "2024-01-10T12:00:00Z",
                        "category": {"name": "Q&A", "isAnswerable": True},
                        "author": {"login": "alice"},
                        "answerChosenAt": "2024-01-11T08:00:00Z",
                        "comments": {
                            "nodes": [
                                {
                                    "id": "DC_abc1",
                                    "body": "Try Y instead.",
                                    "url": "https://github.com/o/n/discussions/1#comment-1",
                                    "updatedAt": "2024-01-10T13:00:00Z",
                                    "author": {"login": "bob"},
                                },
                                {
                                    "id": "DC_abc2",
                                    "body": "Also check the docs.",
                                    "url": "https://github.com/o/n/discussions/1#comment-2",
                                    "updatedAt": "2024-01-10T14:00:00Z",
                                    "author": {"login": "carol"},
                                },
                            ]
                        },
                    },
                    {
                        "number": 2,
                        "title": "Feature request: support Z",
                        "url": "https://github.com/o/n/discussions/2",
                        "body": "It would be great to have Z.",
                        "updatedAt": "2024-01-09T10:00:00Z",
                        "category": {"name": "Ideas", "isAnswerable": False},
                        "author": {"login": "dave"},
                        "answerChosenAt": None,
                        "comments": {"nodes": []},
                    },
                ]
            }
        }
    }
}


# ---------------------------------------------------------------------------
# (a) Happy path
# ---------------------------------------------------------------------------

def test_sync_happy_path(isolated_store, monkeypatch):
    """2 discussions (disc1 with 2 comments, disc2 with 0) → indexed=4, 4 DB rows."""
    monkeypatch.setattr("skill_hub.embeddings.embed", lambda text, **kw: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )
    monkeypatch.setattr(
        "skill_hub.discussions_sync._gh_graphql",
        lambda query, variables, **kw: _TWO_DISCUSSIONS_PAYLOAD,
    )

    from skill_hub import discussions_sync

    report = discussions_sync.sync_discussions(isolated_store, repo="o/n")

    assert report.get("error") is None, f"Unexpected error: {report.get('error')}"
    assert report["checked"] == 2
    assert report["discussions"] == 2
    assert report["comments"] == 2
    assert report["indexed"] == 4
    assert report["skipped"] == 0
    assert report["dry_run"] is False

    # Verify the 4 rows are actually in the DB
    rows = isolated_store._conn.execute(
        "SELECT doc_id, source FROM vectors WHERE namespace = 'discussions'"
    ).fetchall()
    assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}: {[r['doc_id'] for r in rows]}"

    doc_ids = {r["doc_id"] for r in rows}
    assert "discussion:1" in doc_ids
    assert "discussion:2" in doc_ids
    assert "discussion:1:comment:DC_abc1" in doc_ids
    assert "discussion:1:comment:DC_abc2" in doc_ids

    for r in rows:
        assert r["source"] == "discussion"


# ---------------------------------------------------------------------------
# (b) dry_run=True: same counts, no DB writes
# ---------------------------------------------------------------------------

def test_sync_dry_run(isolated_store, monkeypatch):
    """dry_run=True returns same counts but writes nothing to the DB."""
    monkeypatch.setattr("skill_hub.embeddings.embed", lambda text, **kw: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )
    monkeypatch.setattr(
        "skill_hub.discussions_sync._gh_graphql",
        lambda query, variables, **kw: _TWO_DISCUSSIONS_PAYLOAD,
    )

    from skill_hub import discussions_sync

    report = discussions_sync.sync_discussions(isolated_store, repo="o/n", dry_run=True)

    assert report["checked"] == 2
    assert report["discussions"] == 2
    assert report["comments"] == 2
    assert report["indexed"] == 4
    assert report["dry_run"] is True

    # Nothing written
    count = isolated_store._conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace = 'discussions'"
    ).fetchone()[0]
    assert count == 0, f"Expected 0 rows in dry_run, got {count}"


# ---------------------------------------------------------------------------
# (c) Graceful failure when _gh_graphql returns None
# ---------------------------------------------------------------------------

def test_sync_graphql_failure(isolated_store, monkeypatch):
    """_gh_graphql returning None → error key, indexed==0, no exception raised."""
    monkeypatch.setattr("skill_hub.embeddings.embed", lambda text, **kw: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )
    monkeypatch.setattr(
        "skill_hub.discussions_sync._gh_graphql",
        lambda query, variables, **kw: None,
    )

    from skill_hub import discussions_sync

    report = discussions_sync.sync_discussions(isolated_store, repo="o/n")

    assert "error" in report
    assert report["indexed"] == 0


# ---------------------------------------------------------------------------
# (d) Graceful failure when _resolve_repo returns None
# ---------------------------------------------------------------------------

def test_sync_resolve_repo_failure(isolated_store, monkeypatch):
    """_resolve_repo returning None → error key, no exception raised."""
    monkeypatch.setattr("skill_hub.embeddings.embed", lambda text, **kw: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: None,
    )

    from skill_hub import discussions_sync

    report = discussions_sync.sync_discussions(isolated_store, repo="bad-repo")

    assert "error" in report
    assert report["indexed"] == 0
