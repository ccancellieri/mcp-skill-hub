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


# ---------------------------------------------------------------------------
# Write path: create_discussion (issue #87)
# ---------------------------------------------------------------------------

# Shared GraphQL mock payloads for the write path.
_CATEGORY_PAYLOAD = {
    "data": {
        "repository": {
            "id": "R_kgDO_test123",
            "discussionCategories": {
                "nodes": [
                    {"id": "DIC_kwDO_general", "name": "General"},
                    {"id": "DIC_kwDO_ideas",   "name": "Ideas"},
                ]
            },
        }
    }
}

_CREATE_DISCUSSION_PAYLOAD = {
    "data": {
        "createDiscussion": {
            "discussion": {
                "number": 42,
                "url": "https://github.com/o/n/discussions/42",
                "title": "Design note: foo",
            }
        }
    }
}


def _make_graphql_router(category_payload, create_payload):
    """Return a _gh_graphql mock that dispatches on mutation keyword."""
    def _mock(query, variables, **kw):
        if "createDiscussion" in query:
            return create_payload
        # category query and any other read
        return category_payload
    return _mock


def _enable_write(monkeypatch):
    """Patch config.get to return discussions_write_enabled=True."""
    import skill_hub.config as _cfg_mod
    _orig = _cfg_mod.get
    def _patched(k, *a, **kw):
        if k == "discussions_write_enabled":
            return True
        if k == "discussions_category":
            return "General"
        return _orig(k, *a, **kw)
    monkeypatch.setattr(_cfg_mod, "get", _patched)


# (e) Flag off → no-op.

def test_create_discussion_disabled_by_default(monkeypatch):
    """When discussions_write_enabled is False (default), returns status=disabled."""
    import skill_hub.config as _cfg_mod
    _orig = _cfg_mod.get
    monkeypatch.setattr(
        _cfg_mod, "get",
        lambda k, *a, **kw: False if k == "discussions_write_enabled" else _orig(k, *a, **kw),
    )

    from skill_hub import discussions_sync

    result = discussions_sync.create_discussion(repo="o/n", title="T", body="B")

    assert result == {"status": "disabled"}


# (f) Flag on, mock GraphQL → category resolved + mutation called + event emitted.

def test_create_discussion_happy_path(monkeypatch):
    """When enabled and GraphQL succeeds, returns ok + emits discussion.created event."""
    _enable_write(monkeypatch)
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )
    monkeypatch.setattr(
        "skill_hub.discussions_sync._gh_graphql",
        _make_graphql_router(_CATEGORY_PAYLOAD, _CREATE_DISCUSSION_PAYLOAD),
    )

    emitted: list[tuple] = []

    from skill_hub import discussions_sync

    result = discussions_sync.create_discussion(
        repo="o/n",
        title="Design note: foo",
        body="Long-form design text.",
        emit=lambda kind, tool, payload: emitted.append((kind, tool, payload)),
    )

    assert result["status"] == "ok"
    assert result["number"] == 42
    assert result["url"] == "https://github.com/o/n/discussions/42"
    assert result["title"] == "Design note: foo"

    assert len(emitted) == 1
    kind, tool, payload = emitted[0]
    assert kind == "discussion.created"
    assert tool is None
    assert payload["number"] == 42
    assert payload["repo"] == "o/n"
    assert payload["category"] == "General"


# (g) Category missing → error, no mutation called.

def test_create_discussion_missing_category(monkeypatch):
    """Category not found in repo → error dict, no createDiscussion call."""
    _enable_write(monkeypatch)
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )

    mutation_calls: list[str] = []

    def _mock_graphql(query, variables, **kw):
        if "createDiscussion" in query:
            mutation_calls.append("create")
            return _CREATE_DISCUSSION_PAYLOAD
        # category payload with no "General" category
        return {
            "data": {
                "repository": {
                    "id": "R_kgDO_test",
                    "discussionCategories": {
                        "nodes": [{"id": "DIC_kwDO_qa", "name": "Q&A"}]
                    },
                }
            }
        }

    monkeypatch.setattr("skill_hub.discussions_sync._gh_graphql", _mock_graphql)

    from skill_hub import discussions_sync

    result = discussions_sync.create_discussion(repo="o/n", title="T", body="B")

    assert result["status"] == "error"
    assert "General" in result["error"]
    assert mutation_calls == [], "createDiscussion must NOT be called when category is missing"


# (h) _resolve_repo fails → error dict.

def test_create_discussion_resolve_repo_fails(monkeypatch):
    """_resolve_repo returning None → error dict."""
    _enable_write(monkeypatch)
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: None,
    )

    from skill_hub import discussions_sync

    result = discussions_sync.create_discussion(repo="bad", title="T", body="B")

    assert result["status"] == "error"
    assert "resolve" in result["error"]


# (i) GraphQL errors block in response → error dict.

def test_create_discussion_graphql_errors(monkeypatch):
    """GraphQL errors list in response → error dict."""
    _enable_write(monkeypatch)
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )

    def _mock_graphql(query, variables, **kw):
        if "createDiscussion" in query:
            return {"errors": [{"message": "Insufficient permissions"}]}
        return _CATEGORY_PAYLOAD

    monkeypatch.setattr("skill_hub.discussions_sync._gh_graphql", _mock_graphql)

    from skill_hub import discussions_sync

    result = discussions_sync.create_discussion(repo="o/n", title="T", body="B")

    assert result["status"] == "error"
    assert "Insufficient permissions" in result["error"]


# (j) emit callback failure is non-fatal.

def test_create_discussion_emit_failure_nonfatal(monkeypatch):
    """A raising emit callback must not prevent a successful return."""
    _enable_write(monkeypatch)
    monkeypatch.setattr(
        "skill_hub.discussions_sync._resolve_repo",
        lambda repo: ("o", "n"),
    )
    monkeypatch.setattr(
        "skill_hub.discussions_sync._gh_graphql",
        _make_graphql_router(_CATEGORY_PAYLOAD, _CREATE_DISCUSSION_PAYLOAD),
    )

    def _bad_emit(kind, tool, payload):
        raise RuntimeError("emit exploded")

    from skill_hub import discussions_sync

    result = discussions_sync.create_discussion(
        repo="o/n", title="T", body="B", emit=_bad_emit
    )

    assert result["status"] == "ok"
