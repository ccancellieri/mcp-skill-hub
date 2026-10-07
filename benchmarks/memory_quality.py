#!/usr/bin/env python3
"""Offline, source-grounded comparison of memory and wiki retrieval arms.

This benchmark measures retrieved evidence only. It does not generate answers,
call hosted services, or claim that source recall is answer quality. The
relationship arm is a one-hop traversal over existing wiki ``[[links]]``; it
is not Microsoft GraphRAG.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = REPO_ROOT / "benchmarks/fixtures/memory_quality_questions.json"
DEFAULT_WIKI = REPO_ROOT / "benchmarks/fixtures/memory_quality_wiki.json"
DEFAULT_OUTPUT = Path("/private/tmp/mcp-skill-hub-memory-quality")
MAX_TOKENS = 1500
TOP_K = 10
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
MODEL_ID = "all-MiniLM-L6-v2"
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:Users|private/var)/[^\s\]\[()<>\"']+")


def _sanitize(text: str) -> str:
    """Remove machine-specific absolute paths from saved benchmark material."""
    return _PATH_RE.sub("<LOCAL_PATH>", str(text))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_text(text: str, size: int = CHUNK_CHARS,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on headings/paragraph boundaries, then overlap long sections."""
    text = _sanitize(text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    sections = re.split(r"(?=^#{1,6}\s)|(?<=\n\n)", text, flags=re.MULTILINE)
    chunks: list[str] = []
    pending = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) > size:
            if pending:
                chunks.append(pending)
                pending = ""
            start = 0
            while start < len(section):
                end = min(start + size, len(section))
                chunks.append(section[start:end])
                if end == len(section):
                    break
                start = max(start + 1, end - overlap)
            continue
        candidate = f"{pending}\n\n{section}" if pending else section
        if len(candidate) <= size:
            pending = candidate
        else:
            chunks.append(pending)
            # Carry an overlap from the prior paragraph to retain local context.
            carry = pending[-overlap:] if overlap else ""
            pending = f"{carry}\n\n{section}" if carry else section
    if pending:
        chunks.append(pending)
    return chunks


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tokenizer():
    spec = importlib.util.spec_from_file_location(
        "memory_quality_context_compare", REPO_ROOT / "benchmarks/context_compare.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared offline tokenizer helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tokenizer()


def _load_model(model_id: str):
    """Load an already-cached embedding model without permitting downloads."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, local_files_only=True)


def _model_fingerprint(model: Any, model_id: str) -> dict:
    config = getattr(getattr(model, "_modules", {}).get("0"), "auto_model", None)
    config = getattr(config, "config", None)
    return {
        "requested_id": model_id,
        "resolved_id": getattr(config, "_name_or_path", None) or model_id,
        "revision": getattr(config, "_commit_hash", None),
        "dimension": int(model.get_sentence_embedding_dimension()),
    }


def _encode_many(model: Any, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False,
                           show_progress_bar=False, batch_size=32)
    return [[float(value) for value in row] for row in vectors]


def _new_store(path: Path):
    from skill_hub.store import SkillStore

    path.parent.mkdir(parents=True, exist_ok=True)
    return SkillStore(db_path=path)


def _insert_vector(store: Any, *, namespace: str, doc_id: str, model_id: str,
                   vector: list[float], metadata: dict,
                   projection: dict | None = None) -> None:
    """Insert precomputed evidence; projected candidates intentionally omit backup."""
    norm = math.sqrt(sum(value * value for value in vector))
    store._conn.execute(
        "INSERT INTO vectors (namespace,doc_id,model,vector,norm,metadata,level,source,project,tags,projection,original_vector) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)",
        (namespace, doc_id, model_id, json.dumps(vector), norm,
         json.dumps(metadata, ensure_ascii=False), "L3", "memory-quality", None,
         None, json.dumps(projection) if projection else None),
    )


def _chunk_records(questions: dict, wiki: dict, repo_root: Path) -> tuple[list[dict], list[dict], dict]:
    records: list[dict] = []
    canonical_bytes = 0
    source_files = list(dict.fromkeys(questions.get("source_corpus", []) +
                                     wiki.get("source_files", [])))
    for relative in source_files:
        path = (repo_root / relative).resolve()
        if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
            raise ValueError(f"invalid or missing source_files entry: {relative}")
        content = path.read_text(encoding="utf-8")
        canonical_bytes += len(content.encode("utf-8"))
        for index, chunk in enumerate(_chunk_text(content)):
            records.append({"id": f"{relative}#chunk-{index:04d}",
                            "text": chunk, "source_file": relative,
                            "source_refs": [relative], "title": path.name})

    wiki_pages: list[dict] = []
    for page in wiki.get("pages", []):
        body = _sanitize(page["body"])
        refs = [_sanitize(ref) for ref in page.get("source_refs", [])]
        quotes = page.get("source_quotes", [])
        wiki_pages.append({**page, "body": body, "source_refs": refs,
                           "source_quotes": quotes})
    wiki_bytes = sum(len((page["body"] + page["title"]).encode("utf-8"))
                     for page in wiki_pages)
    return records, wiki_pages, {"source_files": source_files,
                                 "canonical_source_bytes": canonical_bytes,
                                 "wiki_page_body_bytes": wiki_bytes}


def _build_raw_store(path: Path, records: list[dict], vectors: list[list[float]],
                     model_id: str, projection_dim: int | None = None) -> Any:
    from skill_hub.fastrp import ProjectionSpec

    store = _new_store(path)
    spec = ProjectionSpec(len(vectors[0]), projection_dim, 42) if projection_dim else None
    namespace = "memory:user-project" if projection_dim is None else f"mq:fastrp-{projection_dim}"
    for record, vector in zip(records, vectors):
        stored = spec.transform(vector).tolist() if spec else vector
        _insert_vector(store, namespace=namespace, doc_id=record["id"],
                       model_id=model_id, vector=stored,
                       metadata={"path": record["source_file"] or "",
                                 "title": record["title"],
                                 "source_refs": record["source_refs"],
                                 "chunk_id": record["id"]},
                       projection=spec.metadata() if spec else None)
    store._conn.commit()
    store._benchmark_record_map = {record["id"]: record for record in records}
    return store


def _build_wiki_store(path: Path, wiki_root: Path, pages: list[dict],
                      section_vectors: dict[str, list[float]], model_id: str) -> Any:
    from skill_hub.wiki import WikiPage, _index_pages, page_path, render_page

    store = _new_store(path)
    try:
        from unittest.mock import patch
        from skill_hub import embeddings
        from skill_hub.embeddings import EmbeddingVector

        def embed(text: str, **_: Any):
            try:
                vec = section_vectors[text]
            except KeyError as exc:
                raise RuntimeError("wiki section absent from the frozen encode batch") from exc
            return EmbeddingVector(vec, model=model_id, backend="offline-local-fixture")

        patcher = patch.object(embeddings, "embed", side_effect=embed)
        patcher.start()
        store._benchmark_patcher = patcher
    except Exception:
        store.close()
        raise

    wiki_root.mkdir(parents=True, exist_ok=True)
    wiki_pages = []
    for page in pages:
        item = WikiPage(
            id=f"mq-{page['slug']}", slug=page["slug"], title=page["title"],
            type="concept", projects=["_global"], scope="public", body=page["body"],
            source_refs=page.get("source_refs", []), created="2026-01-01", updated="2026-01-01",
        )
        out = page_path(wiki_root, item)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_page(item), encoding="utf-8")
        wiki_pages.append(item)
    (wiki_root / "index.md").write_text(
        "# Index\n\n" + "\n".join(
            f"- [[{page.slug}]] — {page.title}" for page in wiki_pages
        ) + "\n", encoding="utf-8")
    _index_pages(store, wiki_root, wiki_pages)
    store._conn.commit()
    patcher.stop()
    return store


def _make_lexical_wiki_store(source_path: Path, target_path: Path, wiki_root: Path) -> Any:
    """Clone derived wiki metadata, then remove dense vectors for lexical-only."""
    source = sqlite3.connect(source_path)
    try:
        source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        source.close()
    shutil.copy2(source_path, target_path)
    lexical = sqlite3.connect(target_path)
    try:
        lexical.execute("DELETE FROM vectors")
        lexical.commit()
        lexical.execute("VACUUM")
        lexical.commit()
        count = lexical.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        if count:
            raise RuntimeError("lexical-only wiki clone unexpectedly retains vectors")
    finally:
        lexical.close()
    store = _new_store(target_path)
    store._benchmark_wiki_root = wiki_root
    return store


def _load_hits(store: Any, namespace: str, question: str) -> list[dict]:
    hits = store.search_vectors(
        question, namespaces=[namespace], top_k=TOP_K, similarity_threshold=0.0,
        apply_level_weight=False, apply_recency_decay=False,
    )
    return [_hit_payload(hit, store) for hit in hits]


def _hit_payload(hit: dict, store: Any) -> dict:
    metadata = hit.get("metadata") or {}
    record = getattr(store, "_benchmark_record_map", {}).get(hit["doc_id"], {})
    return {"id": hit["doc_id"], "score": hit.get("raw_score", hit["score"]),
            "namespace": hit["namespace"], "text": record.get("text", ""),
            "source_file": record.get("source_file") or metadata.get("path") or None,
            "source_refs": record.get("source_refs", metadata.get("source_refs", [])),
            "wiki_slug": record.get("wiki_slug"),
            "title": record.get("title", metadata.get("title", ""))}


def _wiki_query(store: Any, root: Path, question: str, *, lexical_only: bool = False) -> list[dict]:
    from skill_hub import wiki
    if lexical_only:
        from unittest.mock import patch
        with patch.object(store, "search_vectors", return_value=[]):
            result = wiki.query(store, root, question, top_k=TOP_K)
    else:
        result = wiki.query(store, root, question, top_k=TOP_K)
    hits = []
    for item in result["results"]:
        hits.append({"id": item["slug"], "score": item["score"],
                     "namespace": "wiki", "text": item["body"],
                     "source_file": None, "source_refs": item["source_refs"],
                     "wiki_slug": item["slug"], "title": item["title"]})
    return hits


GRAPH_SEED_K = 3


def _wiki_one_hop(store: Any, seeds: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Add linked pages one hop after ranked seeds; never synthesize relations."""
    selected = list(seeds[:GRAPH_SEED_K])
    seen = {hit["wiki_slug"] for hit in selected}
    for hit in seeds[:GRAPH_SEED_K]:
        if not hit.get("wiki_slug"):
            continue
        rows = store._conn.execute(
            "SELECT dst_slug FROM wiki_edges WHERE src_slug=? AND resolved=1 ORDER BY dst_slug",
            (hit["wiki_slug"],),
        ).fetchall()
        for row in rows:
            slug = row[0]
            if slug in seen:
                continue
            page = store._conn.execute(
                "SELECT rel_path FROM wiki_pages WHERE slug=?", (slug,)
            ).fetchone()
            if not page:
                continue
            from skill_hub.wiki import _load_page
            item = _load_page(store._benchmark_wiki_root / page[0])
            if item is None:
                continue
            selected.append({"id": slug, "score": 0.0, "namespace": "wiki-one-hop",
                             "text": item.body, "source_file": None,
                             "source_refs": item.source_refs,
                             "wiki_slug": slug, "title": item.title})
            seen.add(slug)
            if len(selected) >= top_k:
                return selected
    return selected[:top_k]


def _budget_hits(hits: list[dict], tokenizer: Any, budget: int = MAX_TOKENS) -> tuple[list[dict], int]:
    selected: list[dict] = []
    used = 0
    for hit in hits[:TOP_K]:
        payload = f"[{hit['id']}] {hit.get('title', '')}\n{hit['text']}\nSources: {', '.join(hit.get('source_refs') or [])}"
        cost = len(tokenizer.encode(payload))
        if used + cost <= budget:
            selected.append({**hit, "context_text": payload, "context_tokens": cost})
            used += cost
            continue
        remaining = budget - used
        if remaining > 0:
            ids = tokenizer.encode(payload)[:remaining]
            trimmed = tokenizer.decode(ids)
            selected.append({**hit, "context_text": trimmed, "context_tokens": len(ids),
                             "truncated_for_budget": True})
            used += len(ids)
        break
    return selected, used


def _fact_scores(question: dict, selected: list[dict]) -> dict:
    required = question.get("required_facts", [])
    evidence = "\n".join(hit.get("context_text", hit["text"]) for hit in selected)
    rendered_hits = [(hit, hit.get("context_text", hit["text"])) for hit in selected]
    fact_rows = []
    for fact in required:
        source = _sanitize(fact["source_file"])
        quote = _sanitize(fact["source_quote"])
        source_hits = [text for hit, text in rendered_hits
                       if source in ([hit.get("source_file")] + hit.get("source_refs", []))
                       and source in text]
        source_present = bool(source_hits)
        quote_present = bool(quote and quote in evidence)
        paired = bool(quote and any(quote in text for text in source_hits))
        fact_rows.append({"fact": fact["fact"], "source_file": source,
                          "source_present": source_present,
                          "source_quote_present_in_context": quote_present,
                          "covered": paired,
                          "provenance_only_unknown": bool(source_present and not paired)})
    covered = sum(row["covered"] for row in fact_rows)
    answerable = bool(question.get("answerable", bool(required)))
    abstention_evidence = question.get("abstention_evidence", [])
    abstention_rows = []
    for fact in abstention_evidence:
        source = _sanitize(fact["source_file"])
        quote = _sanitize(fact["source_quote"])
        source_hits = [text for hit, text in rendered_hits
                       if source in ([hit.get("source_file")] + hit.get("source_refs", []))
                       and source in text]
        source_present = bool(source_hits)
        quote_present = bool(quote and any(quote in text for text in source_hits))
        abstention_rows.append({"fact": fact["fact"], "source_file": source,
                                "source_present": source_present,
                                "source_quote_present_in_context": quote_present})
    return {"required_fact_count": len(required), "covered_fact_count": covered,
            "source_coverage_recall": covered / len(required) if required else None,
            "facts": fact_rows,
            "abstention_response_assessment": "not_evaluated_no_answer_generation",
            "abstention_evidence": abstention_rows,
            "abstention_evidence_source_coverage": (
                sum(row["source_present"] and row["source_quote_present_in_context"]
                    for row in abstention_rows) / len(abstention_rows)
                if abstention_rows else None),
            "quality_unknown_fact_count": sum(row["provenance_only_unknown"] for row in fact_rows),
            "metric_scope": "source_quote_recall_only; not answer/content quality"}


def _db_size(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in
               (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
               if candidate.exists())


def _source_quotes_map(wiki: dict) -> dict[str, list[str]]:
    # Preserved in the input fixture and exported output for blinded review.
    return {page["slug"]: [_sanitize(str(q)) for q in page.get("source_quotes", [])]
            for page in wiki.get("pages", [])}


def run_benchmark(*, questions_path: Path, wiki_path: Path, repo_root: Path,
                  output_dir: Path, model_id: str = MODEL_ID,
                  tokenizer: Any | None = None, model: Any | None = None) -> dict:
    # Validate offline prerequisites and output isolation before creating any
    # directory, database, or partial benchmark artifact.
    token_counter = tokenizer if tokenizer is not None else _load_tokenizer()
    work = output_dir / "work"
    _require_empty_workdir(work)
    questions = _read_json(questions_path)
    wiki_fixture = _read_json(wiki_path)
    records, wiki_pages, sizing = _chunk_records(questions, wiki_fixture, repo_root)
    if not records:
        raise ValueError("no raw or wiki evidence records were loaded")
    query_rows = questions.get("questions", [])
    from skill_hub.wiki import WikiPage, _split_page_sections
    wiki_section_texts: list[str] = []
    for page in wiki_pages:
        item = WikiPage(id=f"mq-{page['slug']}", slug=page["slug"], title=page["title"],
                        type="concept", projects=["_global"], scope="public", body=page["body"],
                        source_refs=page.get("source_refs", []))
        wiki_section_texts.extend(text for _, text in _split_page_sections(item))
    texts = ([record["text"] for record in records] + wiki_section_texts
             + [q["question"] for q in query_rows])
    if model is None:
        model = _load_model(model_id)
    vectors = _encode_many(model, texts)
    record_vectors = vectors[:len(records)]
    section_vectors_list = vectors[len(records):len(records) + len(wiki_section_texts)]
    section_vectors = dict(zip(wiki_section_texts, section_vectors_list))
    query_vectors = vectors[len(records) + len(wiki_section_texts):]
    query_vector_map = dict(zip((q["question"] for q in query_rows), query_vectors))
    # Store.search_vectors needs an embedding function, but it consumes these
    # precomputed exact model vectors so evaluation is network-free/repeatable.
    from skill_hub import embeddings
    from skill_hub.embeddings import EmbeddingVector
    from unittest.mock import patch

    def embed(query: str, **_: Any):
        return EmbeddingVector(query_vector_map[query], model=model_id,
                              backend="offline-local-fixture")

    output_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)
    arms: dict[str, Any] = {}
    db_paths: dict[str, Path] = {}
    model_key = model_id
    arm_specs = [("memory_rag_full", None), ("memory_rag_fastrp128", 128),
                 ("memory_rag_fastrp256", 256)]
    for arm, dim in arm_specs:
        path = work / f"{arm}.sqlite3"
        db_paths[arm] = path
        store = _build_raw_store(path, records, record_vectors, model_key, dim)
        namespace = "memory:user-project" if dim is None else f"mq:fastrp-{dim}"
        arms[arm] = (store, namespace)

    wiki_arm = "wiki_hybrid"
    wiki_db = work / f"{wiki_arm}.sqlite3"
    wiki_root = work / "wiki"
    wiki_store = _build_wiki_store(wiki_db, wiki_root, wiki_pages, section_vectors, model_key)
    wiki_store._benchmark_wiki_root = wiki_root
    db_paths[wiki_arm] = wiki_db
    lexical_db = work / "wiki_lexical.sqlite3"
    lexical_store = _make_lexical_wiki_store(wiki_db, lexical_db, wiki_root)
    db_paths["wiki_lexical"] = lexical_db

    output_queries = []
    review_packets = []
    grading_key = []
    key_map: dict[str, str] = {}
    try:
        with patch.object(embeddings, "embed", side_effect=embed):
            for question in query_rows:
                q_text = _sanitize(question["question"])
                raw_arm_hits: dict[str, list[dict]] = {}
                latencies: dict[str, float] = {}
                for arm, (store, namespace) in list(arms.items())[:3]:
                    start = time.perf_counter()
                    raw_arm_hits[arm] = _load_hits(store, namespace, question["question"])
                    latencies[arm] = (time.perf_counter() - start) * 1000
                for arm, lexical in (("wiki_hybrid", False), ("wiki_lexical", True)):
                    start = time.perf_counter()
                    query_store = lexical_store if lexical else wiki_store
                    raw_arm_hits[arm] = _wiki_query(query_store, wiki_root,
                                                    question["question"],
                                                    lexical_only=lexical)
                    latencies[arm] = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                raw_arm_hits["wiki_relationship_one_hop"] = _wiki_one_hop(
                    wiki_store, raw_arm_hits["wiki_hybrid"])
                latencies["wiki_relationship_one_hop"] = (
                    latencies["wiki_hybrid"] + (time.perf_counter() - start) * 1000)

                arm_results = {}
                for arm, hits in raw_arm_hits.items():
                    selected, token_count = _budget_hits(hits, token_counter)
                    score = _fact_scores(question, selected)
                    arm_results[arm] = {
                        "latency_ms": round(latencies[arm], 3),
                        "budget": {"maximum_tokens": MAX_TOKENS, "used_tokens": token_count,
                                   "top_k": TOP_K},
                        "raw_hit_count": len(hits),
                        "selected_hit_count": len(selected),
                        "graph_seed_count": GRAPH_SEED_K if arm == "wiki_relationship_one_hop" else None,
                        "metrics": score,
                        "hits": [{**hit, "selected": any(item["id"] == hit["id"] for item in selected)}
                                 for hit in hits],
                        "context": selected,
                    }
                # Stable source labels are fixture IDs; randomized aliases
                # support review packets without disclosing the retrieval arm.
                order = sorted(arm_results)
                seed = int(hashlib.sha256(question["id"].encode()).hexdigest()[:8], 16)
                import random
                random.Random(seed).shuffle(order)
                arm_ids = {name: f"arm_{i + 1}" for i, name in enumerate(order)}
                key_map[question["id"]] = {v: k for k, v in arm_ids.items()}
                output_queries.append({"id": question["id"], "split": question.get("split"),
                                      "language": question.get("language"),
                                      "type": question.get("type"), "question": q_text,
                                      "answerable": question.get("answerable"),
                                      "review_arms": {arm_ids[name]: arm_results[name] for name in order}})
                review_packets.append({"id": question["id"], "split": question.get("split"),
                                       "language": question.get("language"),
                                       "type": question.get("type"), "question": q_text,
                                       "arms": {arm_ids[name]: [item["context_text"] for item in arm_results[name]["context"]]
                                                for name in order}})
                grading_key.append({"id": question["id"],
                                    "answerable": question.get("answerable"),
                                    "required_facts": question.get("required_facts", []),
                                    "forbidden_claims": question.get("forbidden_claims", []),
                                    "reference_answer": question.get("reference_answer"),
                                    "abstention_evidence": question.get("abstention_evidence", [])})
    finally:
        for store, _ in arms.values():
            store.close()
        wiki_store.close()
        lexical_store.close()

    fixture_files = [questions_path, wiki_path]
    fixture_files.extend((repo_root / rel).resolve() for rel in sizing["source_files"])
    fixture_files.append(Path(__file__).resolve())
    run = {
        "protocol": {"top_k": TOP_K, "max_context_tokens": MAX_TOKENS,
                     "tokenizer": "o200k_base", "similarity_threshold": 0.0,
                     "projection_seed": 42, "answer_model_calls": 0,
                     "relationship_latency_includes": "wiki_hybrid seed query plus one-hop expansion",
                     "lexical_storage_basis": "independent wiki DB clone with vectors deleted and VACUUMed",
                     "graph_arm": "one-hop existing [[wikilinks]], not Microsoft GraphRAG",
                     "source_metric_warning": "quote/source coverage is retrieval evidence only, not answer quality"},
        "model": _model_fingerprint(model, model_id),
        "fixtures": {"path_hashes": {str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else path.name: _sha256(path)
                                      for path in fixture_files},
                     **sizing},
        "vector_backups": {arm: 0 for arm in
                           ("memory_rag_full", "memory_rag_fastrp128",
                            "memory_rag_fastrp256", "wiki_hybrid")},
        "storage": {"canonical_source_bytes": sizing["canonical_source_bytes"],
                    "arms": {arm: {"sqlite_bytes": _db_size(path),
                                   "vector_backup_bytes": _backup_bytes(path)}
                             for arm, path in db_paths.items()},
                    "wiki_vault_bytes": _tree_size(wiki_root)},
        "questions": output_queries,
        "code_revision": _git_revision(repo_root),
    }
    report_path = output_dir / "memory_quality_results.json"
    report_path.write_text(json.dumps(_sanitize_obj(run), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (output_dir / "memory_quality_review_packets.json").write_text(
        json.dumps(_sanitize_obj(review_packets), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (output_dir / "memory_quality_grading_key.json").write_text(
        json.dumps(_sanitize_obj({"questions": grading_key,
                                  "wiki_source_quotes": _source_quotes_map(wiki_fixture)}),
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (output_dir / "memory_quality_arm_key.json").write_text(
        json.dumps(_sanitize_obj(key_map), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return run


def _backup_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT COALESCE(SUM(length(original_vector)),0) FROM vectors").fetchone()
        return int(row[0] or 0)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _require_empty_workdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"benchmark work directory is not empty: {path}; choose a fresh --output-dir"
        )


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _git_revision(repo_root: Path) -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                              check=True, capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sanitize_obj(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize(value)
    if isinstance(value, list):
        return [_sanitize_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_obj(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--wiki", type=Path, default=DEFAULT_WIKI)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)
    run_benchmark(questions_path=args.questions, wiki_path=args.wiki,
                  repo_root=args.repo_root.resolve(), output_dir=args.output_dir,
                  model_id=args.model)
    print(args.output_dir / "memory_quality_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
