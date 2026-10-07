"""Optional namespace projection preserves model and transform provenance."""
import builtins
import json

import pytest

np = pytest.importorskip("numpy", reason="FastRP requires the optional fastrp extra")

from skill_hub.embeddings import EmbeddingResult, EmbeddingVector
from skill_hub.fastrp import ProjectionSpec
from skill_hub.store import SkillStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    import skill_hub.embeddings as embeddings
    rng = np.random.default_rng(31)
    vectors = {name: rng.normal(size=16).tolist() for name in ("one", "two", "three")}
    monkeypatch.setattr(embeddings, "embed", lambda text, **kwargs: EmbeddingVector(
        vectors[text], model="actual-model", backend="test"))
    value = SkillStore(db_path=tmp_path / "projection.db")
    yield value
    value.close()


def search(store, query="one"):
    return store.search_vectors(query, top_k=20, similarity_threshold=-1,
                                apply_level_weight=False, apply_recency_decay=False)


def test_full_default_is_unchanged_and_opt_in_persists_backup(store):
    store.upsert_vector("full", "one", "one")
    store.upsert_vector("compressed", "one", "one", fast_rp=True,
                        fast_rp_components=8, fast_rp_seed=0)
    rows = store._conn.execute("SELECT * FROM vectors ORDER BY namespace").fetchall()
    compressed, full = rows
    assert full["projection"] is None and full["original_vector"] is None
    assert len(json.loads(full["vector"])) == 16
    assert len(json.loads(compressed["vector"])) == 8
    assert json.loads(compressed["original_vector"]) == json.loads(full["vector"])
    assert json.loads(compressed["projection"])["seed"] == 0
    assert compressed["model"] == "actual-model"
    assert all(r["raw_score"] == pytest.approx(1) for r in search(store))


def test_namespace_configuration_and_seed_changes_use_saved_row_transform(store):
    store.configure_vector_projection("docs", fast_rp=True, n_components=8, seed=1)
    store.upsert_vector("docs", "old", "one")
    old_spec = store._conn.execute("SELECT projection FROM vectors").fetchone()[0]
    store.configure_vector_projection("docs", fast_rp=True, n_components=4, seed=2)
    store.upsert_vector("docs", "new", "one")
    rows = search(store)
    assert len(rows) == 2
    assert all(r["raw_score"] == pytest.approx(1) for r in rows)
    assert store._conn.execute("SELECT projection FROM vectors WHERE doc_id='old'").fetchone()[0] == old_spec
    path = store._conn.execute("PRAGMA database_list").fetchone()[2]
    reopened = SkillStore(db_path=__import__('pathlib').Path(path))
    try:
        assert all(r["raw_score"] == pytest.approx(1) for r in search(reopened))
    finally:
        reopened.close()


def test_disable_uses_original_and_full_upsert_clears_projection(store):
    store.configure_vector_projection("docs", fast_rp=True, n_components=4)
    store.upsert_vector("docs", "one", "one")
    store.upsert_vector("docs", "two", "two")
    store.configure_vector_projection("docs", fast_rp=False)
    expected = {}
    for text in ("one", "two"):
        store.upsert_vector("baseline", text, text)
    for row in search(store):
        expected.setdefault(row["doc_id"], []).append(row["raw_score"])
    assert all(scores[0] == pytest.approx(scores[1]) for scores in expected.values())
    store.upsert_vector("docs", "one", "one")
    row = store._conn.execute("SELECT projection, original_vector FROM vectors WHERE namespace='docs' AND doc_id='one'").fetchone()
    assert tuple(row) == (None, None)


def test_missing_numpy_and_unknown_version_fall_back_safely(store, monkeypatch):
    store.upsert_vector("docs", "one", "one", fast_rp=True, fast_rp_components=8)
    original_import = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name.endswith("fastrp"):
            raise ImportError("optional dependency unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert search(store)[0]["raw_score"] == pytest.approx(1)
    monkeypatch.setattr(builtins, "__import__", original_import)
    spec = ProjectionSpec(16, 8).metadata()
    spec["version"] = "future-version"
    store._conn.execute("UPDATE vectors SET projection=?", (json.dumps(spec),))
    assert search(store)[0]["raw_score"] == pytest.approx(1)


def test_projected_rows_do_not_match_other_model_or_dimension(store, monkeypatch):
    import skill_hub.embeddings as embeddings
    store.upsert_vector("docs", "one", "one", fast_rp=True, fast_rp_components=8)
    monkeypatch.setattr(embeddings, "embed", lambda *args, **kwargs: EmbeddingVector(
        [1.] * 16, model="other-model", backend="test"))
    assert search(store) == []
    monkeypatch.setattr(embeddings, "embed", lambda *args, **kwargs: EmbeddingVector(
        [1.] * 8, model="actual-model", backend="test"))
    assert search(store) == []


def test_standalone_projection_keeps_list_api_and_distinct_identity(monkeypatch):
    import skill_hub.embeddings as embeddings
    monkeypatch.setattr(embeddings, "embed_with_metadata", lambda *args, **kwargs:
                        EmbeddingResult(tuple(range(16)), "actual", "test"))
    result = embeddings.embed("text", fast_rp=True, fast_rp_components=4, fast_rp_seed=0)
    assert isinstance(result, list) and len(result) == 4
    assert result.model != "actual"
    assert result.provenance["original_model"] == "actual"
    assert result.provenance["projection"]["seed"] == 0


def test_plugin_vector_index_registration_is_optional(store, monkeypatch):
    import skill_hub.plugin_registry as registry
    monkeypatch.setattr(registry, "iter_enabled_plugins", lambda: [{"name": "example", "manifest": {
        "vector_indexes": [{"name": "example:docs", "projection": {"type": "fastrp", "n_components": 8, "seed": 9}}]
    }}])
    assert registry.register_plugin_vector_indexes(store) == 1
    store.upsert_vector("example:docs", "one", "one")
    assert json.loads(store._conn.execute("SELECT projection FROM vectors").fetchone()[0])["seed"] == 9


def test_plugin_memory_projection_registration(store, monkeypatch, tmp_path):
    import skill_hub.memory_index as memory
    (tmp_path / "one.txt").write_text("one")
    monkeypatch.setattr(memory, "iter_enabled_plugins", lambda: [{"name": "example", "path": tmp_path,
        "manifest": {"memory": {"indexes": [{"name": "example:memory", "reads": ["*.txt"],
                       "projection": {"type": "fastrp", "n_components": 4}}]}}}])
    assert memory.index_plugin_memory(store) == {"example:memory": 1}
    row = store._conn.execute("SELECT vector, projection FROM vectors").fetchone()
    assert len(json.loads(row["vector"])) == 4 and row["projection"]


def test_per_write_opt_in_works_in_default_namespace(store):
    from skill_hub.fastrp import ProjectionSpec
    store.upsert_vector("memory:user-project", "one", "one", fast_rp=True, fast_rp_components=4)
    store.upsert_vector("memory:user-project", "two", "two", fast_rp=True, fast_rp_components=4)
    row = store._conn.execute("SELECT * FROM vectors WHERE doc_id='two'").fetchone()
    q = json.loads(store._conn.execute("SELECT original_vector FROM vectors WHERE doc_id='one'").fetchone()[0])
    p = ProjectionSpec(**json.loads(row["projection"])).transform(q)
    expected = np.dot(p, json.loads(row["vector"])) / (np.linalg.norm(p) * row["norm"])
    result = next(r for r in search(store) if r["doc_id"] == "two")
    assert result["raw_score"] == pytest.approx(expected, abs=1e-6)


def test_explicit_disable_does_not_import_projection(store, monkeypatch):
    store.configure_vector_projection("docs", fast_rp=True, n_components=8)
    store.upsert_vector("docs", "one", "one")
    store.configure_vector_projection("docs", fast_rp=False)
    original_import = builtins.__import__
    def forbidden(name, *args, **kwargs):
        if name.endswith("fastrp"):
            raise AssertionError("disabled retrieval imported optional projection")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", forbidden)
    assert search(store)[0]["raw_score"] == pytest.approx(1)


def test_merge_audit_snapshot_keeps_projection_and_model(store):
    from skill_hub.vector_sources import NamespaceSource
    store.upsert_vector("docs", "one", "one", fast_rp=True, fast_rp_components=8)
    source = NamespaceSource(store, "docs")
    items = source.fetch_for_merge(["one"])
    draft = source.draft_merge(items, "local", "")
    result = source.commit_merge(items, draft)
    reason = json.loads(store._conn.execute("SELECT reason FROM memory_audit WHERE id=?", (result.audit_id,)).fetchone()[0])
    snapshot = reason["rollback"]["restore_docs"][0]
    assert snapshot["model"] == "actual-model"
    assert snapshot["projection"] and snapshot["original_vector"] and snapshot["norm"]


def test_explicit_disable_is_independent_of_json_whitespace(store, monkeypatch):
    store.configure_vector_projection("docs", fast_rp=True, n_components=8)
    store.upsert_vector("docs", "one", "one")
    store._conn.execute(
        "UPDATE vector_index_config SET projection=? WHERE name='docs'",
        ('{"type":"full"}',),
    )
    original_import = builtins.__import__
    def forbidden(name, *args, **kwargs):
        if name.endswith("fastrp"):
            raise AssertionError("disabled retrieval imported optional projection")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", forbidden)
    assert search(store)[0]["raw_score"] == pytest.approx(1)


def test_wiki_disable_restores_full_ranking_and_preserves_markdown(store, tmp_path):
    from skill_hub.wiki import WikiPage, page_path, query, render_page
    root = tmp_path / "wiki"
    originals = {}
    for text in ("one", "two", "three"):
        page = WikiPage(id=text, slug=text, title=text, type="concept",
                        projects=["_global"], scope="public", body=text)
        path = page_path(root, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_page(page))
        originals[path] = path.read_bytes()
        metadata = {"slug": text, "rel_path": str(path.relative_to(root))}
        store.upsert_vector("wiki", text, text, metadata=metadata)
    baseline = query(store, root, "one", top_k=3)
    store.configure_vector_projection("wiki", fast_rp=True, n_components=4)
    for text in ("one", "two", "three"):
        path = next(path for path in originals if path.stem == text)
        store.upsert_vector("wiki", text, text,
                            metadata={"slug": text, "rel_path": str(path.relative_to(root))})
    store.configure_vector_projection("wiki", fast_rp=False)
    restored = query(store, root, "one", top_k=3)
    assert [r["slug"] for r in restored["results"]] == [r["slug"] for r in baseline["results"]]
    assert [r["score"] for r in restored["results"]] == [r["score"] for r in baseline["results"]]
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_memory_reindex_removes_obsolete_derived_chunks_only(store, tmp_path, monkeypatch):
    import skill_hub.embeddings as embeddings
    from skill_hub.memory_index import _embed_file
    monkeypatch.setattr(embeddings, "embed", lambda text, **kwargs: EmbeddingVector(
        [1.] * 16, model="actual-model", backend="test"))
    path = tmp_path / "note%_.md"
    path.write_text("a" * 30)
    store._conn.execute("INSERT INTO vector_index_config(name, chunk_size, chunk_overlap) VALUES ('docs', 10, 0)")
    assert _embed_file(store, path, "docs", "example")
    assert store._conn.execute("SELECT COUNT(*) FROM vectors WHERE namespace='docs'").fetchone()[0] == 3
    store.upsert_vector("other", str(path) + "#chunk-002", "other namespace")
    unrelated = tmp_path / "note-other.md"
    store.upsert_vector("docs", str(unrelated), "other file", metadata={"path": str(unrelated)})
    store.upsert_vector("docs", "malformed-metadata", "unrelated")
    store._conn.execute("UPDATE vectors SET metadata='invalid json' WHERE doc_id='malformed-metadata'")
    path.write_text("short")
    assert _embed_file(store, path, "docs", "example")
    assert {row[0] for row in store._conn.execute("SELECT doc_id FROM vectors WHERE namespace='docs'")} == {str(path), str(unrelated), "malformed-metadata"}
    assert store._conn.execute("SELECT COUNT(*) FROM vectors WHERE namespace='other'").fetchone()[0] == 1
    assert path.read_text() == "short"


def test_memory_failed_replacement_keeps_previous_chunks(store, tmp_path, monkeypatch):
    import skill_hub.embeddings as embeddings
    from skill_hub.memory_index import _embed_file
    monkeypatch.setattr(embeddings, "embed", lambda text, **kwargs: EmbeddingVector(
        [1.] * 16, model="actual-model", backend="test"))
    path = tmp_path / "note.md"
    path.write_text("a" * 30)
    store._conn.execute("INSERT INTO vector_index_config(name, chunk_size, chunk_overlap) VALUES ('docs', 10, 0)")
    assert _embed_file(store, path, "docs", "example")
    original = store.upsert_vector
    def fail_second(**kwargs):
        if kwargs["doc_id"].endswith("#chunk-001"):
            raise RuntimeError("embedding unavailable")
        return original(**kwargs)
    monkeypatch.setattr(store, "upsert_vector", fail_second)
    path.write_text("b" * 20)
    assert not _embed_file(store, path, "docs", "example")
    assert store._conn.execute("SELECT COUNT(*) FROM vectors WHERE namespace='docs'").fetchone()[0] == 3
    assert path.read_text() == "b" * 20
