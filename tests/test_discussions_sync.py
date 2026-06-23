"""Tests for discussions_sync module (issue #41).

Covers (Wave 2 — discussions land as mechanical wiki source pages):
(a) Happy path: 2 discussions (one with 2 comments, one with 0) → checked=2,
    discussions=2, comments=2, indexed=2 (one source page each, comments folded).
(b) dry_run=True: same counts returned, but no pages and no vectors written.
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

def _isolate_wiki_root(monkeypatch, tmp_path):
    """Point config's wiki_root at a temp vault (the default is baked at import)."""
    import skill_hub.config as _cfg_mod
    wiki_root = tmp_path / ".claude" / "mcp-skill-hub" / "wiki"
    _orig = _cfg_mod.get
    monkeypatch.setattr(
        _cfg_mod, "get",
        lambda k, *a, **kw: str(wiki_root) if k == "wiki_root" else _orig(k, *a, **kw),
    )
    return wiki_root


def test_sync_happy_path(isolated_store, tmp_path, monkeypatch):
    """Wave 2: 2 discussions → 2 wiki source pages (comments folded in)."""
    wiki_root = _isolate_wiki_root(monkeypatch, tmp_path)
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
    # One source page per discussion (comments folded into the body).
    assert report["indexed"] == 2
    assert report["skipped"] == 0
    assert report["dry_run"] is False

    # Source pages written to the wiki, comments folded into the body.
    src = wiki_root / "pages" / "source"
    assert (src / "source-discussion-1.md").exists()
    assert (src / "source-discussion-2.md").exists()
    d1 = (src / "source-discussion-1.md").read_text()
    assert "Try Y instead." in d1
    assert "Also check the docs." in d1

    # The raw 'discussions' namespace is retired — content is in 'wiki'.
    assert isolated_store._conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace='discussions'"
    ).fetchone()[0] == 0
    assert isolated_store._conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace='wiki'"
    ).fetchone()[0] > 0


# ---------------------------------------------------------------------------
# (b) dry_run=True: same counts, no writes
# ---------------------------------------------------------------------------

def test_sync_dry_run(isolated_store, tmp_path, monkeypatch):
    """dry_run=True returns page counts but writes no pages and no vectors."""
    wiki_root = _isolate_wiki_root(monkeypatch, tmp_path)
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
    assert report["indexed"] == 2
    assert report["dry_run"] is True

    assert not (wiki_root / "pages" / "source").exists()
    assert isolated_store._conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace='wiki'"
    ).fetchone()[0] == 0


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
