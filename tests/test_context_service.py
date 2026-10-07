"""Deterministic, scoped retrieval for context assembly."""
from __future__ import annotations

import json

import pytest

from skill_hub.store import Skill, SkillStore


@pytest.mark.parametrize("prompt, source, expected", [
    ("cat", "allocation", 0),
    ("cache", "cached unrelated", 0),
    ("migration", "migrazione", 0),
    ("cache", "CACHE: migration", 1),
    ("cache_key", "cache_key cache_keys", 1),
    ("cache_key", "other_cache_key cache_keys", 0),
])
def test_relevance_matches_normalized_tokens_not_substrings(prompt, source, expected):
    from skill_hub.context_service import _relevance

    assert _relevance(prompt, source) == expected


@pytest.fixture()
def store(tmp_path):
    result = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield result
    result.close()


def _insert_skill(store: SkillStore) -> None:
    store.upsert_skill(Skill(
        id="backend:postgres",
        name="postgres",
        description="Use PostgreSQL migration and query patterns.",
        content="# PostgreSQL\nFull instructions stay on demand.",
        file_path="/trusted/skills/postgres/SKILL.md",
        plugin="backend",
    ))


def _insert_memory(store: SkillStore, *, project: str, doc_id: str,
                   text: str) -> None:
    store._conn.execute(
        "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("memory:project", doc_id, "[]", 0, "{}", "L3", f"memory:{doc_id}", project),
    )
    store._conn.execute(
        "INSERT INTO context_digests (key, content_hash, digest, content, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (f"memory:{doc_id}", "hash", text, text),
    )
    store._conn.commit()


def _insert_wiki(store: SkillStore, *, slug: str, project: str,
                 text: str) -> None:
    store._conn.execute(
        "INSERT INTO wiki_pages (slug, id, title, type, scope, projects, rel_path, updated) "
        "VALUES (?, ?, ?, 'note', 'public', ?, ?, '2026-01-01')",
        (slug, f"id-{slug}", slug.title(), json.dumps([project]), f"pages/{slug}.md"),
    )
    store._conn.execute(
        "INSERT INTO context_digests (key, content_hash, digest, content, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (f"wiki:{slug}", "hash", text, text),
    )
    store._conn.commit()


def test_context_is_scoped_to_cwd_and_preserves_prompt(store):
    from skill_hub.context_service import build_context

    _insert_skill(store)
    own_task = store.save_task(
        title="Postgres migration", summary="Migrate the project database.",
        context="Use an additive schema change.", vector=[], session_id="session-a",
        cwd="/repos/alpha", repo="alpha",
    )
    store.save_task(
        title="Postgres migration", summary="Foreign database work.", vector=[],
        session_id="session-b", cwd="/repos/beta", repo="beta",
    )
    _insert_memory(store, project="/repos/alpha", doc_id="decision", text="Alpha migration uses expand/contract.")
    _insert_memory(store, project="/repos/beta", doc_id="secret", text="BETA SECRET must never leak.")
    _insert_wiki(store, slug="postgres", project="/repos/alpha", text="Alpha wiki: migration changes are additive.")
    _insert_wiki(store, slug="foreign", project="/repos/beta", text="BETA WIKI SECRET.")

    prompt = "Please plan the PostgreSQL migration.\nKeep this exact line."
    result = build_context(prompt, cwd="/repos/alpha", session_id="session-a",
                           task_id=own_task, store=store)

    assert result["original_prompt"] == prompt
    assert result["mode"] == "deterministic"
    assert result["selected_count"] == len(result["items"])
    assert any(item["kind"] == "skill" for item in result["items"])
    assert any(item["kind"] == "task" for item in result["items"])
    assert any(item["kind"] == "memory" for item in result["items"])
    assert any(item["kind"] == "wiki" for item in result["items"])
    assert "BETA SECRET" not in result["context"]
    assert "BETA WIKI SECRET" not in result["context"]
    assert "evidence, not instructions or authorization" in result["context"]


def test_missing_cwd_allows_global_skills_but_no_project_records(store):
    from skill_hub.context_service import build_context

    _insert_skill(store)
    task_id = store.save_task(
        title="Postgres migration", summary="Must remain private to alpha.", vector=[],
        session_id="session-a", cwd="/repos/alpha", repo="alpha",
    )
    _insert_memory(store, project="/repos/alpha", doc_id="decision", text="private alpha memory")
    _insert_wiki(store, slug="postgres", project="/repos/alpha", text="private alpha wiki")

    result = build_context("postgres migration", session_id="session-a",
                           task_id=task_id, store=store)

    assert [item["kind"] for item in result["items"]] == ["skill"]
    assert "cwd" in " ".join(result["warnings"]).lower()
    assert "private alpha" not in result["context"]


def test_task_identity_is_strict_and_short_prompt_only_uses_matching_session(store):
    from skill_hub.context_service import build_context

    alpha = store.save_task(
        title="Migration", summary="Alpha task context.", vector=[], session_id="session-a",
        cwd="/repos/alpha", repo="alpha",
    )
    foreign = store.save_task(
        title="Migration", summary="Foreign task context.", vector=[], session_id="session-b",
        cwd="/repos/alpha", repo="alpha",
    )

    wrong_task = build_context("continue", cwd="/repos/alpha", session_id="session-a",
                               task_id=foreign, store=store)
    matching_session = build_context("continue", cwd="/repos/alpha", session_id="session-a",
                                     store=store)
    unrelated_long_prompt = build_context(
        "Design a comprehensive incident response program for certificate rotation "
        "and disaster recovery across the production fleet.",
        cwd="/repos/alpha", session_id="session-a", store=store,
    )

    assert all(item["kind"] != "task" for item in wrong_task["items"])
    assert any(item["source"] == f"task:{alpha}" for item in matching_session["items"])
    assert all(item["kind"] != "task" for item in unrelated_long_prompt["items"])


def test_cwd_scoped_topic_can_retrieve_relevant_task_without_identity(store):
    from skill_hub.context_service import build_context

    relevant = store.save_task(
        title="FastAPI concurrency", summary="Optimize connection pool concurrency.", vector=[],
        cwd="/repos/alpha", repo="alpha",
    )
    store.save_task(
        title="Release notes", summary="Prepare the next release.", vector=[],
        cwd="/repos/alpha", repo="alpha",
    )
    store.save_task(
        title="FastAPI concurrency", summary="Foreign project work.", vector=[],
        cwd="/repos/beta", repo="beta",
    )

    relevant_result = build_context("FastAPI concurrency", cwd="/repos/alpha", store=store)
    generic_result = build_context("proceed", cwd="/repos/alpha", store=store)

    assert any(item["source"] == f"task:{relevant}" for item in relevant_result["items"])
    assert all(item["kind"] != "task" for item in generic_result["items"])


def test_cwd_scoped_task_search_does_not_rank_the_global_fts_corpus(store):
    from skill_hub.context_service import build_context

    task_id = store.save_task(
        title="FastAPI concurrency", summary="Optimize the connection pool.", vector=[],
        cwd="/repos/alpha", repo="alpha",
    )
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)
    try:
        result = build_context("FastAPI concurrency", cwd="/repos/alpha", store=store)
    finally:
        store._conn.set_trace_callback(None)

    assert any(item["source"] == f"task:{task_id}" for item in result["items"])
    assert not any("tasks_fts" in statement.lower() for statement in statements)


def test_disabled_and_limits_are_hard_bounded_without_llm(store, monkeypatch):
    from skill_hub.context_service import build_context

    _insert_skill(store)
    monkeypatch.setattr(store, "search_vectors", lambda *_args, **_kwargs: pytest.fail("no embeddings"))

    disabled = build_context("postgres", store=store, cfg={"hook_context_injection": False})
    limited = build_context("postgres", store=store, cfg={"context_max_chars": 1, "context_max_items": 0})

    assert disabled["items"] == []
    assert disabled["warnings"]
    assert limited["items"] == []
    assert len(limited["context"]) <= 1
    assert limited["omitted_count"] >= 1


def test_no_store_uses_a_read_only_database_connection(tmp_path, monkeypatch):
    from skill_hub.context_service import build_context

    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".claude" / "mcp-skill-hub" / "skill_hub.db"
    db_path.parent.mkdir(parents=True)
    seeded = SkillStore(db_path=db_path)
    _insert_skill(seeded)
    seeded.close()

    result = build_context("postgres migration")

    assert any(item["kind"] == "skill" for item in result["items"])
    assert result["warnings"] == ["No cwd scope was provided; project context was not queried."]


def test_paused_task_is_only_evidence_for_an_explicit_task_id(store):
    from skill_hub.context_service import build_context

    task_id = store.save_task(
        title="Migration", summary="Wait for approval.", vector=[], session_id="session-a",
        cwd="/repos/alpha", repo="alpha",
    )
    store._conn.execute(
        "UPDATE tasks SET options = ? WHERE id = ?",
        (json.dumps({"work_state": "paused"}), task_id),
    )
    store._conn.commit()

    matched = build_context("continue", cwd="/repos/alpha", session_id="session-a", store=store)
    explicit = build_context("continue", cwd="/repos/alpha", session_id="session-a",
                             task_id=task_id, store=store)

    assert all(item["kind"] != "task" for item in matched["items"])
    task = next(item for item in explicit["items"] if item["kind"] == "task")
    assert "Work state (literal): paused" in task["text"]


def test_memory_provenance_recovers_absolute_metadata_path_and_configured_labels(store):
    from skill_hub.context_service import build_context

    encoded = "-repos-alpha"
    store._conn.execute(
        "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
        "VALUES ('memory:user-project', 'absolute', '[]', 0, ?, 'L3', NULL, NULL)",
        (json.dumps({"path": "/repos/alpha/.memory/decision.md"}),),
    )
    store._conn.execute(
        "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
        "VALUES ('memory:user-project', 'encoded', '[]', 0, '{}', 'L3', ?, NULL)",
        (encoded,),
    )
    for doc_id, text in (("absolute", "Alpha rollback decision."),
                         ("encoded", "Alpha encoded migration decision.")):
        store._conn.execute(
            "INSERT INTO context_digests (key, content_hash, digest, content) VALUES (?, 'h', ?, ?)",
            (f"memory:{doc_id}", text, text),
        )
    store._conn.commit()

    automatic = build_context("alpha decision", cwd="/repos/alpha", store=store)
    mapped = build_context(
        "alpha migration", cwd="/repos/alpha", store=store,
        cfg={"context_project_roots": ["/repos/alpha"]},
    )

    assert "Alpha rollback decision." in automatic["context"]
    assert "Alpha encoded migration decision." not in automatic["context"]
    assert "Alpha encoded migration decision." in mapped["context"]


def test_bare_legacy_project_names_require_an_explicit_alias(store):
    from skill_hub.context_service import build_context

    _insert_memory(store, project="", doc_id="legacy", text="Alpha legacy decision.")
    store._conn.execute("UPDATE vectors SET source = 'alpha' WHERE doc_id = 'legacy'")
    _insert_wiki(store, slug="legacy", project="alpha", text="Alpha legacy wiki.")
    store._conn.commit()

    omitted = build_context("alpha legacy", cwd="/repos/alpha", store=store)
    aliased = build_context(
        "alpha legacy", cwd="/repos/alpha", store=store,
        cfg={"context_project_aliases": {"/repos/alpha": ["alpha"]}},
    )

    assert "Alpha legacy decision." not in omitted["context"]
    assert "Alpha legacy wiki." not in omitted["context"]
    assert "Alpha legacy decision." in aliased["context"]
    assert "Alpha legacy wiki." in aliased["context"]


def test_ambiguous_aliases_are_not_used_for_project_recovery(store):
    from skill_hub.context_service import build_context

    _insert_memory(store, project="", doc_id="shared", text="Shared legacy decision.")
    store._conn.execute("UPDATE vectors SET source = 'legacy-project' WHERE doc_id = 'shared'")
    store._conn.commit()

    result = build_context(
        "shared legacy", cwd="/repos/alpha", store=store,
        cfg={"context_project_aliases": {
            "/repos/alpha": ["legacy-project"],
            "/repos/beta": ["legacy-project"],
        }},
    )

    assert "Shared legacy decision." not in result["context"]


def test_relative_cwd_is_not_resolved_against_the_daemon_working_directory(store):
    from skill_hub.context_service import build_context

    _insert_memory(store, project="/repos/alpha", doc_id="decision", text="Alpha decision.")

    result = build_context("alpha decision", cwd="repos/alpha", store=store)

    assert "Alpha decision." not in result["context"]
    assert any("cwd" in warning.lower() for warning in result["warnings"])


def test_memory_source_paths_treat_sql_like_metacharacters_as_literals(store):
    from skill_hub.context_service import build_context

    for doc_id, source, text in (
        ("underscore", "/repos/axb/memory.md", "Underscore sibling secret."),
        ("percent", "/repos/abc/memory.md", "Percent sibling secret."),
    ):
        store._conn.execute(
            "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
            "VALUES ('memory:project', ?, '[]', 0, '{}', 'L3', ?, NULL)",
            (doc_id, source),
        )
        store._conn.execute(
            "INSERT INTO context_digests (key, content_hash, digest, content) VALUES (?, 'h', ?, ?)",
            (f"memory:{doc_id}", text, text),
        )
    store._conn.commit()

    underscore = build_context("sibling secret", cwd="/repos/a_b", store=store)
    percent = build_context("sibling secret", cwd="/repos/a%", store=store)

    assert "Underscore sibling secret." not in underscore["context"]
    assert "Percent sibling secret." not in percent["context"]


def test_memory_chunks_with_the_same_evidence_are_deduplicated(store):
    from skill_hub.context_service import build_context

    for index in range(2):
        store._conn.execute(
            "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
            "VALUES ('memory:project', ?, '[]', 0, ?, 'L3', NULL, ?)",
            (f"decision#chunk-{index:03d}", json.dumps({"path": "/repos/alpha/memory.md"}), "/repos/alpha"),
        )
    store._conn.execute(
        "INSERT INTO context_digests (key, content_hash, digest, content) "
        "VALUES ('memory:decision', 'h', 'Alpha decision evidence.', 'Alpha decision evidence.')"
    )
    store._conn.commit()

    result = build_context("alpha decision", cwd="/repos/alpha", store=store)

    assert len([item for item in result["items"] if item["kind"] == "memory"]) == 1


def test_skill_ranking_prefers_exact_fastapi_metadata_over_content_distractors(store):
    from skill_hub.context_service import build_context

    store.upsert_skill(Skill(
        id="backend:fastapi", name="fastapi",
        description="Optimize FastAPI concurrency while preserving behavior.",
        content="# FastAPI\nPrecise concurrency guidance.",
        file_path="/trusted/skills/fastapi/SKILL.md", plugin="backend",
    ))
    for name in ("before-you-build", "python-testing", "airflow-migration", "e2e"):
        store.upsert_skill(Skill(
            id=f"distractor:{name}", name=name,
            description=">-",
            content="Keep existing behavior and discuss concurrency. " * 30,
            file_path=f"/trusted/skills/{name}/SKILL.md", plugin="distractor",
        ))
    store.upsert_skill(Skill(
        id="distractor:maintenance", name="maintenance",
        description="Optimize behavior while keeping existing systems stable.",
        content="# Maintenance", file_path="/trusted/skills/maintenance/SKILL.md",
        plugin="distractor",
    ))

    result = build_context("  Optimize FastAPI concurrency\nKeep existing behavior.\n", store=store)

    skills = [item for item in result["items"] if item["kind"] == "skill"]
    assert skills[0]["source"] == "skill:backend:fastapi"
    assert all(item["text"] != ">-" for item in skills)
    assert all(item["source"] != "skill:distractor:maintenance" for item in skills)


@pytest.mark.parametrize("prompt, expected", [
    ("Improve MCP", {"skill:tools:mcp-builder"}),
    ("Migliora MCP", {"skill:tools:mcp-builder"}),
    ("MCP tool to inspect accessibility", {
        "skill:tools:mcp-builder", "skill:chrome-devtools-mcp:a11y-debugging",
    }),
    ("Strumento MCP per verificare accessibilità", {
        "skill:tools:mcp-builder", "skill:chrome-devtools-mcp:a11y-debugging",
    }),
    ("Use chrome-devtools-mcp:a11y-debugging", {
        "skill:chrome-devtools-mcp:a11y-debugging",
    }),
    ("Use chrome-devtools-mcp:a11y-debugging with MCP builder", {
        "skill:chrome-devtools-mcp:a11y-debugging", "skill:tools:mcp-builder",
    }),
    ("Use tools:mcp-builder", {"skill:tools:mcp-builder"}),
    ("Review PostgreSQL migration", set()),
])
def test_skill_namespace_does_not_make_unrelated_skills_relevant(store, prompt, expected):
    from skill_hub.context_service import build_context

    store.upsert_skill(Skill(
        id="tools:mcp-builder", name="MCP builder",
        description="Build and improve MCP tools and servers.", content="# MCP builder",
        file_path="/trusted/skills/mcp-builder/SKILL.md", plugin="tools",
    ))
    store.upsert_skill(Skill(
        id="chrome-devtools-mcp:a11y-debugging", name="a11y debugging",
        description="Inspect accessibility and accessibilità with Chrome DevTools MCP.",
        content="# Accessibility", file_path="/trusted/skills/a11y/SKILL.md",
        plugin="chrome-devtools-mcp",
    ))
    store.upsert_skill(Skill(
        id="tools:filesystem", name="filesystem",
        description="Improve file system diagnostics.", content="# Filesystem",
        file_path="/trusted/skills/filesystem/SKILL.md", plugin="tools",
    ))
    store.upsert_skill(Skill(
        id="tools:mcp-build", name="helper",
        description="MCP diagnostics for a different helper.", content="# Helper",
        file_path="/trusted/skills/helper/SKILL.md", plugin="tools",
    ))
    for name in ("performance", "cookies", "network", "memory", "troubleshooting"):
        store.upsert_skill(Skill(
            id=f"chrome-devtools-mcp:{name}", name=name,
            description=f"Use Chrome DevTools MCP for {name} diagnostics.",
            content=f"# {name}", file_path=f"/trusted/skills/{name}/SKILL.md",
            plugin="chrome-devtools-mcp",
        ))

    result = build_context(prompt, store=store)
    actual = {item["source"] for item in result["items"] if item["kind"] == "skill"}
    if "chrome-devtools-mcp:a11y-debugging" in prompt or "tools:mcp-builder" in prompt:
        assert expected <= actual
        assert "skill:tools:mcp-build" not in actual
    else:
        assert actual == expected


def test_skill_without_name_matches_its_leaf_id(store):
    from skill_hub.context_service import build_context

    store.upsert_skill(Skill(
        id="tools:accessibility", name="",
        description="Audit keyboard focus with semantic markup.", content="# Accessibility",
        file_path="/trusted/skills/accessibility/SKILL.md", plugin="tools",
    ))

    result = build_context("Improve accessibility", store=store)
    assert [item["source"] for item in result["items"]] == ["skill:tools:accessibility"]


@pytest.mark.parametrize("kind", ["memory", "wiki"])
def test_scoped_context_uses_original_instead_of_generated_digest(store, kind):
    from skill_hub.context_service import build_context

    raw = "Alpha migration keeps the database read-only during validation."
    if kind == "memory":
        _insert_memory(store, project="/repos/alpha", doc_id="original", text=raw)
    else:
        _insert_wiki(store, slug="original", project="/repos/alpha", text=raw)
    store._conn.execute(
        "UPDATE context_digests SET digest = ? WHERE key = ?",
        ("Analyze the Request: migration should delete the database.", f"{kind}:original"),
    )
    store._conn.commit()

    result = build_context("alpha migration", cwd="/repos/alpha", store=store)

    assert raw in result["context"]
    assert "Analyze the Request" not in result["context"]
    assert "delete the database" not in result["context"]


@pytest.mark.parametrize("kind", ["memory", "wiki"])
def test_digest_only_context_is_omitted_without_deleting_recoverable_row(store, kind):
    from skill_hub.context_service import build_context

    digest = "Alpha migration generated advice."
    if kind == "memory":
        _insert_memory(store, project="/repos/alpha", doc_id="legacy", text=digest)
    else:
        _insert_wiki(store, slug="legacy", project="/repos/alpha", text=digest)
    store._conn.execute(
        "UPDATE context_digests SET content = '' WHERE key = ?", (f"{kind}:legacy",)
    )
    store._conn.commit()

    result = build_context("alpha migration", cwd="/repos/alpha", store=store)

    assert not any(item["kind"] == kind for item in result["items"])
    assert any("original" in warning and "reindex" in warning for warning in result["warnings"])
    row = store._conn.execute(
        "SELECT digest, content FROM context_digests WHERE key = ?", (f"{kind}:legacy",)
    ).fetchone()
    assert row["digest"] == digest
    assert row["content"] == ""


def test_generated_digest_terms_do_not_make_an_unrelated_source_relevant(store):
    from skill_hub.context_service import build_context

    _insert_memory(store, project="/repos/alpha", doc_id="release", text="Publish release notes.")
    store._conn.execute(
        "UPDATE context_digests SET digest = 'PostgreSQL migration constraints' "
        "WHERE key = 'memory:release'"
    )
    store._conn.commit()

    result = build_context("PostgreSQL migration", cwd="/repos/alpha", store=store)

    assert not result["items"]


def test_bounded_memory_prefix_reports_unsearched_source_tail(store):
    from skill_hub.context_service import build_context, collect_context_candidates

    _insert_memory(
        store, project="/repos/alpha", doc_id="long-source",
        text=("Unrelated introductory material. " * 100) + "PostgreSQL migration constraints.",
    )

    bounded = build_context("PostgreSQL migration", cwd="/repos/alpha", store=store)
    full, warnings = collect_context_candidates(
        "PostgreSQL migration", cwd="/repos/alpha", store=store, include_full_text=True,
    )

    assert not bounded["items"]
    assert any("1600" in warning for warning in bounded["warnings"])
    assert any(item["kind"] == "memory" for item in full)
    assert not any("1600" in warning for warning in warnings)
