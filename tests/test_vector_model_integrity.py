from __future__ import annotations

import logging

import pytest

from skill_hub.embeddings import EmbeddingVector, embed, embed_with_metadata
from skill_hub.store import SkillStore


@pytest.fixture()
def store(tmp_path):
    value = SkillStore(db_path=tmp_path / "vectors.db")
    yield value
    value.close()


def test_embed_keeps_list_api_and_reports_fallback_model(monkeypatch):
    import skill_hub.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_hot_path", lambda: False)
    monkeypatch.setattr(embeddings._cfg, "get", lambda key: {
        "embedding_backend_priority": ["ollama", "sentence_transformers"],
        "embed_model": "requested-model",
        "sentence_transformers_model": "actual-fallback-model",
    }.get(key))
    monkeypatch.setattr(
        embeddings, "_embed_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")),
    )
    monkeypatch.setattr(embeddings, "_embed_sentence_transformers", lambda text: [1.0, 0.0])

    result = embed_with_metadata("query")
    public = embed("query")
    assert result.vector == (1.0, 0.0)
    assert result.model == "actual-fallback-model"
    assert result.backend == "sentence_transformers"
    assert isinstance(public, list)
    assert public == [1.0, 0.0]
    assert public.model == "actual-fallback-model"


def test_upserts_persist_actual_model_from_vector_provenance(store, monkeypatch):
    import skill_hub.embeddings as embeddings

    monkeypatch.setattr(
        embeddings,
        "embed",
        lambda *args, **kwargs: EmbeddingVector(
            [1.0, 0.0], model="fallback-model", backend="sentence_transformers"
        ),
    )
    store.upsert_vector("docs", "one", "text", model="requested-model")
    row = store._conn.execute("SELECT model FROM vectors WHERE doc_id='one'").fetchone()
    assert row[0] == "fallback-model"

    vector = EmbeddingVector([0.0, 1.0], model="fallback-model", backend="sentence_transformers")
    store._conn.execute(
        "INSERT INTO skills(id,name,content,target) VALUES('s','s','s','claude')"
    )
    store.upsert_embedding("s", "requested-model", vector)
    row = store._conn.execute("SELECT model FROM embeddings WHERE skill_id='s'").fetchone()
    assert row[0] == "fallback-model"


def test_search_omits_mixed_models_dimensions_and_legacy_rows(store, monkeypatch, caplog):
    import skill_hub.embeddings as embeddings

    monkeypatch.setattr(
        embeddings,
        "embed",
        lambda *args, **kwargs: EmbeddingVector([1.0, 0.0], model="model-a", backend="ollama"),
    )
    rows = [
        ("ok", "model-a", [1.0, 0.0]),
        ("wrong-model", "model-b", [1.0, 0.0]),
        ("wrong-dimension", "model-a", [1.0, 0.0, 0.0]),
        ("legacy", None, [1.0, 0.0]),
    ]
    for doc_id, model, vector in rows:
        store._conn.execute(
            "INSERT INTO vectors(namespace,doc_id,model,vector,norm) VALUES(?,?,?,?,1.0)",
            ("docs", doc_id, model, __import__("json").dumps(vector)),
        )
    store._conn.commit()

    with caplog.at_level(logging.WARNING):
        results = store.search_vectors("query", namespaces=["docs"])
    assert [row["doc_id"] for row in results] == ["ok"]
    assert "omitted 3 incompatible or ambiguous vectors" in caplog.text


def _skill(store, skill_id: str) -> None:
    store._conn.execute(
        "INSERT INTO skills(id,name,content,target) VALUES(?,?,?,'claude')",
        (skill_id, skill_id, skill_id),
    )
    store._conn.commit()


def test_primary_legacy_search_requires_matching_model_and_dimension(store):
    _skill(store, "a")
    _skill(store, "b")
    _skill(store, "wrong-dim")
    store.upsert_embedding("a", "model-a", [1.0, 0.0])
    store.upsert_embedding("b", "model-b", [1.0, 0.0])
    store.upsert_embedding("wrong-dim", "model-a", [1.0, 0.0, 0.0])
    query = EmbeddingVector([1.0, 0.0], model="model-a", backend="ollama")

    results = store.search(query, similarity_threshold=0.0)
    assert [result["id"] for result in results] == ["a"]


def test_primary_sqlite_vec_search_requires_matching_model(store):
    if store._vec_engine != "sqlite-vec":
        pytest.skip("sqlite-vec not available")
    vector = [1.0] + [0.0] * 7
    _skill(store, "a")
    _skill(store, "b")
    store.upsert_embedding("a", "model-a", vector)
    store.upsert_embedding("b", "model-b", vector)
    query = EmbeddingVector(vector, model="model-a", backend="ollama")

    results = store.search(query, similarity_threshold=0.0)
    assert [result["id"] for result in results] == ["a"]
