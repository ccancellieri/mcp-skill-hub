"""Deterministic reviewable context composition."""
from __future__ import annotations

import json

import pytest

from skill_hub.store import Skill, SkillStore


@pytest.fixture()
def store(tmp_path):
    result = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield result
    result.close()


def _seed(store: SkillStore) -> tuple[int, int]:
    store.upsert_skill(Skill(
        id="backend:postgres", name="postgres",
        description="Use PostgreSQL migration and rollback patterns.",
        content="# PostgreSQL\nFull instructions.",
        file_path="/trusted/postgres/SKILL.md", plugin="backend",
    ))
    alpha = store.save_task(
        title="Postgres migration", summary="Alpha uses expand and contract.",
        context="Preserve the rollback path.", vector=[], session_id="session-a",
        cwd="/repos/alpha", repo="alpha",
    )
    beta = store.save_task(
        title="Postgres migration", summary="BETA SECRET.", vector=[],
        session_id="session-b", cwd="/repos/beta", repo="beta",
    )
    for project, doc_id, text in (
        ("/repos/alpha", "alpha", "Alpha migration requires an additive schema."),
        ("/repos/beta", "beta", "BETA MEMORY SECRET."),
    ):
        store._conn.execute(
            "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
            "VALUES ('memory:project', ?, '[]', 0, '{}', 'L3', ?, ?)",
            (doc_id, f"memory:{doc_id}", project),
        )
        store._conn.execute(
            "INSERT INTO context_digests (key, content_hash, digest, content, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (f"memory:{doc_id}", f"hash-{doc_id}", text, text),
        )
    store._conn.commit()
    return alpha, beta


def test_prepare_is_bounded_scoped_and_persisted(store):
    from skill_hub.context_composer import get_composition_candidate, prepare_composition

    _seed(store)
    prompt = "Plan the PostgreSQL migration.\nKeep this exact constraint."
    result = prepare_composition(
        prompt, project_roots=["/repos/alpha", "/repos/beta"],
        token_budget=400, mode="training", store=store,
    )

    assert result["original_prompt"] == prompt
    assert result["needs_review"] is True
    assert len(result["candidates"]) <= 20
    assert len({item["candidate_id"] for item in result["candidates"]}) == len(result["candidates"])
    assert all(set((
        "candidate_id", "kind", "title", "source", "text", "project_root",
        "source_hash", "updated_at", "reason", "estimated_tokens", "score", "features",
    )) <= item.keys() for item in result["candidates"])
    assert not any("BETA SECRET" in item["text"] for item in result["candidates"] if item["project_root"] == "/repos/alpha")
    candidate = result["candidates"][0]
    expanded = get_composition_candidate(result["draft_id"], candidate["candidate_id"], store=store)
    assert expanded["candidate_id"] == candidate["candidate_id"]
    assert expanded["source_hash"] == candidate["source_hash"]


def test_task_id_requires_exactly_one_project_and_verified_scope(store):
    from skill_hub.context_composer import prepare_composition

    alpha, beta = _seed(store)
    with pytest.raises(ValueError, match="single project"):
        prepare_composition("continue", project_roots=["/repos/alpha", "/repos/beta"], task_id=alpha, store=store)

    result = prepare_composition(
        "continue", project_roots=["/repos/alpha"], task_id=beta,
        session_id="session-b", store=store,
    )
    assert all(item["kind"] != "task" for item in result["candidates"])
    assert result["needs_review"] is True


@pytest.mark.parametrize("mode", ["mixed", "automatic"])
def test_untrained_nontraining_modes_require_review_without_claiming_confidence(store, mode):
    from skill_hub.context_composer import prepare_composition

    _seed(store)
    result = prepare_composition(
        "postgres migration", project_roots=["/repos/alpha"], mode=mode, store=store,
    )

    assert result["needs_review"] is True
    assert any("promoted selector" in warning.lower() for warning in result["warnings"])
    assert all("confidence" not in item["features"] for item in result["candidates"])


def test_compose_revalidates_stale_sources_and_excerpt_substrings(store):
    from skill_hub.context_composer import compose_context, prepare_composition

    _seed(store)
    draft = prepare_composition(
        "postgres migration additive", project_roots=["/repos/alpha"],
        token_budget=300, store=store,
    )
    memory = next(item for item in draft["candidates"] if item["kind"] == "memory")
    with pytest.raises(ValueError, match="substring"):
        compose_context(
            draft["draft_id"], selected_ids=[memory["candidate_id"]],
            excerpts={memory["candidate_id"]: "fabricated evidence"}, store=store,
        )

    store._conn.execute(
        "UPDATE context_digests SET content = 'Changed evidence.' WHERE key = 'memory:alpha'"
    )
    store._conn.commit()
    with pytest.raises(ValueError, match="stale"):
        compose_context(draft["draft_id"], selected_ids=[memory["candidate_id"]], store=store)


def test_exact_excerpt_is_lossy_without_claiming_budget_truncation(store):
    from skill_hub.context_composer import compose_context, prepare_composition

    _seed(store)
    draft = prepare_composition(
        "alpha additive migration", project_roots=["/repos/alpha"],
        token_budget=1000, store=store,
    )
    memory = next(item for item in draft["candidates"] if item["kind"] == "memory")
    excerpt = "requires an additive schema"
    result = compose_context(
        draft["draft_id"], selected_ids=[memory["candidate_id"]],
        excerpts={memory["candidate_id"]: excerpt}, store=store,
    )

    assert result["items"][0]["text"] == excerpt
    assert result["items"][0]["lossy"] is True
    assert result["selection"]["lossy"] is True
    assert any("exact source excerpt" in warning for warning in result["warnings"])
    assert not any("truncated to fit the token budget" in warning for warning in result["warnings"])


def test_expansion_uses_full_indexed_source_and_hashes_edits_beyond_preview(store):
    from skill_hub.context_composer import (
        compose_context,
        get_composition_candidate,
        prepare_composition,
    )

    _seed(store)
    full = "Alpha additive migration. " + ("x" * 1800) + " exact ending"
    store._conn.execute(
        "UPDATE context_digests SET digest = ?, content = ? WHERE key = 'memory:alpha'",
        (full, full),
    )
    store._conn.commit()
    draft = prepare_composition(
        "alpha additive migration", project_roots=["/repos/alpha"], store=store,
    )
    memory = next(item for item in draft["candidates"] if item["kind"] == "memory")
    assert len(memory["text"]) <= 1200
    expanded = get_composition_candidate(draft["draft_id"], memory["candidate_id"], store=store)
    assert expanded["text"] == full

    changed = full[:-1] + "!"
    store._conn.execute(
        "UPDATE context_digests SET digest = ?, content = ? WHERE key = 'memory:alpha'",
        (changed, changed),
    )
    store._conn.commit()
    with pytest.raises(ValueError, match="stale"):
        compose_context(draft["draft_id"], selected_ids=[memory["candidate_id"]], store=store)


def test_digest_preview_expands_raw_source_and_raw_only_change_is_stale(store):
    from skill_hub.context_composer import (
        compose_context,
        get_composition_candidate,
        prepare_composition,
    )

    _seed(store)
    digest = "Alpha additive migration digest."
    raw_v1 = digest + (" original detail" * 100)
    store._conn.execute(
        "UPDATE context_digests SET digest = ?, content = ? WHERE key = 'memory:alpha'",
        (digest, raw_v1),
    )
    store._conn.commit()
    draft = prepare_composition(
        "alpha additive migration", project_roots=["/repos/alpha"], store=store,
    )
    memory = next(item for item in draft["candidates"] if item["kind"] == "memory")
    assert memory["text"] == raw_v1[:1200]
    assert get_composition_candidate(
        draft["draft_id"], memory["candidate_id"], store=store,
    )["text"] == raw_v1

    raw_v2 = raw_v1 + " changed tail"
    store._conn.execute(
        "UPDATE context_digests SET content = ? WHERE key = 'memory:alpha'",
        (raw_v2,),
    )
    store._conn.commit()
    with pytest.raises(ValueError, match="stale"):
        compose_context(
            draft["draft_id"], selected_ids=[memory["candidate_id"]], store=store,
        )


def test_legacy_digest_only_source_is_omitted_until_original_is_restored(store):
    from skill_hub.context_composer import prepare_composition

    _seed(store)
    store._conn.execute(
        "UPDATE context_digests SET digest = ?, content = '' WHERE key = 'memory:alpha'",
        ("Alpha additive migration digest.",),
    )
    store._conn.commit()

    draft = prepare_composition(
        "alpha additive migration", project_roots=["/repos/alpha"], store=store,
    )
    assert any("reindex" in warning.lower() for warning in draft["warnings"])
    assert not any(item["kind"] == "memory" for item in draft["candidates"])


def test_compose_deduplicates_and_accounts_for_rendering_budget(store):
    from skill_hub.context_composer import compose_context, prepare_composition

    _seed(store)
    store._conn.execute(
        "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
        "VALUES ('memory:project', 'duplicate', '[]', 0, '{}', 'L3', 'memory:duplicate', '/repos/alpha')"
    )
    store._conn.execute(
        "INSERT INTO context_digests (key, content_hash, digest, content) "
        "VALUES ('memory:duplicate', 'dup', 'Alpha migration requires an additive schema.', "
        "'Alpha migration requires an additive schema.')"
    )
    store._conn.commit()
    draft = prepare_composition(
        "alpha migration additive", project_roots=["/repos/alpha"],
        token_budget=80, store=store,
    )
    selected = [item["candidate_id"] for item in draft["candidates"]]
    result = compose_context(draft["draft_id"], selected_ids=selected, store=store)

    assert result["estimated_tokens"] <= 80
    assert result["context"].count("Alpha migration requires an additive schema.") <= 1
    assert result["original_prompt"] == draft["original_prompt"]
    assert any("lossy" in warning.lower() for warning in result["warnings"])


def test_fabricated_ids_are_rejected_and_composition_is_json_safe(store):
    from skill_hub.context_composer import compose_context, prepare_composition

    _seed(store)
    draft = prepare_composition("postgres migration", project_roots=["/repos/alpha"], store=store)
    with pytest.raises(ValueError, match="candidate"):
        compose_context(draft["draft_id"], selected_ids=["invented"], store=store)
    candidate_id = draft["candidates"][0]["candidate_id"]
    result = compose_context(draft["draft_id"], selected_ids=[candidate_id], store=store)
    json.dumps(result)


def test_only_confirmed_training_or_mixed_compositions_create_learning_labels(store):
    from skill_hub.context_composer import compose_context, prepare_composition

    _seed(store)
    draft = prepare_composition("postgres migration", project_roots=["/repos/alpha"], mode="training", store=store)
    selected, rejected = draft["candidates"][:2]
    compose_context(
        draft["draft_id"], selected_ids=[selected["candidate_id"]],
        rejected_ids=[rejected["candidate_id"]], confirmed=False, store=store,
    )
    assert store._conn.execute(
        "SELECT COUNT(*) FROM context_learning_compositions"
    ).fetchone()[0] == 0

    compose_context(
        draft["draft_id"], selected_ids=[selected["candidate_id"]],
        rejected_ids=[rejected["candidate_id"]], confirmed=True, store=store,
    )
    assert store._conn.execute(
        "SELECT COUNT(*) FROM context_learning_compositions"
    ).fetchone()[0] == 1
    assert store._conn.execute(
        "SELECT COUNT(*) FROM context_learning_labels"
    ).fetchone()[0] == 2


def test_optimize_prompt_only_normalizes_safe_whitespace():
    from skill_hub.context_composer import optimize_prompt

    prompt = 'Keep   this constraint.\n\n\n"Do   not change this quote."\n```py\nx  =  1\n```\n'
    result = optimize_prompt(prompt)

    assert result["original_prompt"] == prompt
    assert '"Do   not change this quote."' in result["optimized_prompt"]
    assert "x  =  1" in result["optimized_prompt"]
    assert "Keep   this constraint." in result["optimized_prompt"]
    assert "\n\n\n" not in result["optimized_prompt"]
    assert result["after_estimated_tokens"] <= result["before_estimated_tokens"]
    assert result["transformations"] == ["collapsed_excess_blank_lines"]


def test_optimize_prompt_preserves_fenced_code_and_json_values():
    from skill_hub.context_composer import optimize_prompt

    prompt = (
        "Compact surrounding prose.\n\n\n\n"
        "```text\nkeep\n\n\n\nthese blank lines\n```\n"
        '{"duplicate": 1, "duplicate": 999999999999999999999999999999}\n'
    )
    result = optimize_prompt(prompt)

    assert "keep\n\n\n\nthese blank lines" in result["optimized_prompt"]
    assert '"duplicate": 1, "duplicate": 999999999999999999999999999999' in result["optimized_prompt"]
    assert "Compact surrounding prose.\n\n\n" not in result["optimized_prompt"]


def test_optimize_prompt_keeps_blank_lines_inside_valid_long_fence():
    from skill_hub.context_composer import optimize_prompt

    prompt = (
        "Outside prose.\n\n\n"
        "````text\nfirst line\n```python\n\n\nsecond line\n```\n\n\nthird line\n````\n"
        "After prose.\n\n\n"
    )
    optimized = optimize_prompt(prompt)["optimized_prompt"]

    assert "Outside prose.\n\n\n" not in optimized
    assert "```python\n\n\nsecond line\n```\n\n\nthird line" in optimized
    assert "After prose.\n\n\n" not in optimized
