"""Tests for FTS5 keyword fallback when embedding backend is unavailable (#8).

Covers:
- ``SkillStore.search_skills_text()`` and ``suggest_plugins_text()`` store APIs.
- ``server.search_skills`` / ``suggest_plugins`` / ``search_context`` routing:
  when ``embed_available()`` returns False, they fall back to FTS5 BM25 search
  and surface ``mode=keyword-fts5`` in the result envelope instead of erroring.
- When ``embed_available()`` returns True, the result envelope reports
  ``mode=vector`` and the original semantic path is taken (no regression).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


@pytest.fixture()
def populated_store(store):
    """Seed a couple of skills and plugins via direct SQL (FTS triggers populate)."""
    # Skills — INSERT through public API so triggers fire.
    from skill_hub.store import Skill
    store.upsert_skill(
        Skill(
            id="local:cooking-pasta",
            name="cooking-pasta",
            description="Boil water and cook pasta al dente",
            content="Detailed pasta cooking recipe.",
            file_path="",
            plugin="",
            target="claude",
        )
    )
    store.upsert_skill(
        Skill(
            id="local:fts-fallback-debug",
            name="fts-fallback-debug",
            description="Debug a SQLite FTS5 BM25 query fallback",
            content="How to use SQLite FTS5 in degraded mode.",
            file_path="",
            plugin="",
            target="claude",
        )
    )
    # Plugins — via public API so triggers fire.
    store.upsert_plugin(
        "fts-debugger@official", "fts-debugger",
        "Debug SQLite FTS5 indexes and BM25 ranking",
    )
    store.upsert_plugin(
        "cookbook@official", "cookbook",
        "Italian recipes and pasta cooking guides",
    )
    return store


# ---------------------------------------------------------------------------
# Store-level: search_skills_text + suggest_plugins_text
# ---------------------------------------------------------------------------

def test_skills_fts_table_and_triggers_exist(store):
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='skills_fts'"
    ).fetchone()
    assert row is not None
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='plugins_fts'"
    ).fetchone()
    assert row is not None
    triggers = {
        r["name"] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    }
    for t in (
        "skills_fts_insert", "skills_fts_delete", "skills_fts_update",
        "plugins_fts_insert", "plugins_fts_delete", "plugins_fts_update",
    ):
        assert t in triggers, f"missing trigger {t}"


def test_search_skills_text_returns_keyword_match(populated_store):
    hits = populated_store.search_skills_text("pasta", top_k=5)
    assert hits, "expected at least one keyword hit for 'pasta'"
    assert any("pasta" in (h["name"] or "").lower() for h in hits)
    # Result shape compatible with vector search() rows
    for key in ("id", "name", "description", "content", "score"):
        assert key in hits[0]


def test_search_skills_text_empty_query(populated_store):
    assert populated_store.search_skills_text("") == []
    assert populated_store.search_skills_text("   ") == []


def test_search_skills_text_no_match(populated_store):
    assert populated_store.search_skills_text("xyzzy_nope_unlikely") == []


def test_search_skills_text_top_k_respected(populated_store):
    hits = populated_store.search_skills_text("debug fts", top_k=1)
    assert len(hits) <= 1


def test_suggest_plugins_text_returns_keyword_match(populated_store):
    hits = populated_store.suggest_plugins_text("BM25 FTS5")
    assert hits, "expected at least one plugin hit"
    short_names = [h["short_name"] for h in hits]
    assert "fts-debugger" in short_names
    # Same envelope shape as suggest_plugins() so the renderer doesn't break.
    for key in ("plugin_id", "short_name", "description",
                "embed_score", "teaching_score", "session_score",
                "total_score"):
        assert key in hits[0]


def test_suggest_plugins_text_empty_query(populated_store):
    assert populated_store.suggest_plugins_text("") == []


def test_skills_fts_trigger_update_syncs(populated_store):
    """Updating a skill keeps FTS5 row in sync."""
    populated_store._conn.execute(
        "UPDATE skills SET description = ? WHERE id = ?",
        ("Now mentions parmesan cheese topping", "local:cooking-pasta"),
    )
    populated_store._conn.commit()
    hits = populated_store.search_skills_text("parmesan")
    assert any(h["id"] == "local:cooking-pasta" for h in hits)


def test_skills_fts_trigger_delete_syncs(populated_store):
    populated_store._conn.execute(
        "DELETE FROM skills WHERE id = ?", ("local:cooking-pasta",)
    )
    populated_store._conn.commit()
    hits = populated_store.search_skills_text("pasta")
    assert not any(h["id"] == "local:cooking-pasta" for h in hits)


def test_rebuild_fts_index_repopulates_skills_and_plugins(populated_store):
    """Rebuild deletes + repopulates, must be idempotent and complete."""
    populated_store._rebuild_fts_index(populated_store._conn)
    populated_store._rebuild_fts_index(populated_store._conn)
    hits = populated_store.search_skills_text("pasta")
    assert hits
    phits = populated_store.suggest_plugins_text("FTS5")
    assert phits


# ---------------------------------------------------------------------------
# Server-level: routing based on embed_available()
# ---------------------------------------------------------------------------

@pytest.fixture()
def server_with_store(populated_store, monkeypatch, tmp_path):
    """Wire the global server module to our isolated populated store.

    Returns the imported ``server`` module so individual tests can monkeypatch
    ``embed_available`` per case.
    """
    from skill_hub import server
    monkeypatch.setattr(server, "_store", populated_store)
    monkeypatch.setattr(server, "SETTINGS_PATH", tmp_path / "settings.json")
    # Reset transient state so tests are independent.
    server._last_search_state["query"] = ""
    server._last_search_state["vector"] = []
    server._last_search_state["skills"] = []
    server._session["topic"] = ""
    server._session["topic_vector"] = []
    return server


def test_search_skills_falls_back_to_fts5_when_no_embed(server_with_store, monkeypatch):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: False)
    out = server.search_skills("pasta cooking", top_k=3)
    assert "mode=keyword-fts5" in out
    assert "local:cooking-pasta" in out
    assert "degraded-search" in out.lower()


def test_search_skills_vector_path_when_embed_available(
    server_with_store, monkeypatch
):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: True)
    # Stub embed() so we don't need a real backend.
    monkeypatch.setattr(server, "embed", lambda q: [0.0] * 16)
    # The vector store will return [] for this fake vector — that's fine,
    # we just need to verify the path was taken.
    out = server.search_skills("pasta cooking", top_k=3)
    # When embeddings ARE available we never see the keyword-fts5 marker.
    assert "mode=keyword-fts5" not in out


def test_suggest_plugins_falls_back_to_fts5_when_no_embed(
    server_with_store, monkeypatch
):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: False)
    out = server.suggest_plugins("BM25 FTS5")
    assert "mode=keyword-fts5" in out
    assert "fts-debugger" in out


def test_suggest_plugins_keyword_no_match_message(server_with_store, monkeypatch):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: False)
    out = server.suggest_plugins("xyzzy_no_such_plugin")
    assert "mode=keyword-fts5" in out
    assert "no plugin suggestions" in out.lower()


def test_search_context_falls_back_to_fts5_when_no_embed(
    server_with_store, monkeypatch
):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: False)
    # Insert an open task so the tasks branch has something to find.
    server._store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("Investigate FTS5 BM25 fallback", "Look at how to degrade gracefully"),
    )
    server._store._conn.commit()

    out = server.search_context("FTS5 fallback", top_k=3)
    assert "mode=keyword-fts5" in out
    assert "Investigate FTS5 BM25 fallback" in out


def test_search_context_vector_marker_when_embed_available(
    server_with_store, monkeypatch
):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: True)
    monkeypatch.setattr(server, "embed", lambda q: [0.0] * 16)
    # Insert a task; with no real vector match it just produces the marker.
    server._store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("vector path probe", "ensure mode marker present"),
    )
    server._store._conn.commit()
    out = server.search_context("anything", top_k=3)
    # Either we get the marker (with hits) OR the "no relevant context" fallback.
    # Both are acceptable; the assertion below just guarantees we are not on the
    # degraded code path.
    assert "mode=keyword-fts5" not in out


def test_search_skills_no_match_keyword_returns_helpful_message(
    server_with_store, monkeypatch
):
    server = server_with_store
    monkeypatch.setattr(server, "embed_available", lambda: False)
    out = server.search_skills("zzz_not_a_real_keyword_in_corpus", top_k=3)
    assert "mode=keyword-fts5" in out
    assert "no matching skills" in out.lower()


def test_keyword_context_injection_enriches_without_embeddings(tmp_path, monkeypatch):
    """The prompt-enrichment hook degrades to FTS5 keyword context when no
    embedding backend is reachable, instead of passing through with nothing."""
    from skill_hub.store import SkillStore, Skill
    monkeypatch.setenv("HOME", str(tmp_path))
    st = SkillStore(db_path=tmp_path / "kw.db")
    st.upsert_skill(Skill(
        id="local:git-pr", name="git-pr",
        description="create a git commit and open a pull request",
        content="Steps: stage, commit, push, open PR.",
        file_path="", plugin="", target="claude",
    ))
    import skill_hub.cli as cli
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **k: st)

    out = cli._build_keyword_context_injection("how do I commit and open a pull request")
    assert out is not None
    assert "local:git-pr" in out
    assert "keyword fallback" in out


def test_keyword_context_injection_returns_none_on_no_match(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    monkeypatch.setenv("HOME", str(tmp_path))
    st = SkillStore(db_path=tmp_path / "kw2.db")
    import skill_hub.cli as cli
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **k: st)
    assert cli._build_keyword_context_injection("zzz_no_such_corpus_token") is None
