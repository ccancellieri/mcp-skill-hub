#!/usr/bin/env python3
"""Reproducible namespace search benchmark; no model or network calls.

Optional NPZ input contains documents (N,D) and queries (Q,D). Output is JSON.
Synthetic Gaussian data is a stress fixture, not a semantic-relevance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill_hub import embeddings
from skill_hub.embeddings import EmbeddingVector
from skill_hub.fastrp import ProjectionSpec, fast_rp_batch
from skill_hub.store import SkillStore


def benchmark(documents, queries, *, n_components=128, seed=42, top_k=10, repeats=3):
    documents = np.asarray(documents, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if documents.ndim != 2 or queries.ndim != 2 or documents.shape[1] != queries.shape[1]:
        raise ValueError("documents and queries must be 2-D arrays with the same dimension")
    if not len(documents) or not len(queries) or not 0 < top_k <= len(documents) or repeats <= 0:
        raise ValueError("nonempty corpus/queries, valid top_k and positive repeats required")
    spec = ProjectionSpec(documents.shape[1], n_components, seed)
    started = time.perf_counter()
    projected = fast_rp_batch(documents, n_components, seed=seed)
    projection_seconds = time.perf_counter() - started
    original_embed = embeddings.embed
    embeddings.embed = lambda text, **kwargs: EmbeddingVector(
        queries[int(text)].tolist(), model="benchmark-fixed-embeddings", backend="fixture")
    try:
        with tempfile.TemporaryDirectory(prefix="fastrp-benchmark-") as directory:
            stores = [SkillStore(db_path=Path(directory) / f"{kind}.db")
                      for kind in ("full", "projected")]
            try:
                for i, store in enumerate(stores):
                    rows = []
                    for doc_id, full in enumerate(documents):
                        vector = full if i == 0 else projected[doc_id]
                        rows.append(("benchmark", str(doc_id), "benchmark-fixed-embeddings",
                                     json.dumps(vector.tolist()), float(np.linalg.norm(vector)),
                                     None if i == 0 else json.dumps(spec.metadata()),
                                     None if i == 0 else json.dumps(full.tolist())))
                    store._conn.executemany(
                        "INSERT INTO vectors(namespace,doc_id,model,vector,norm,projection,original_vector) "
                        "VALUES (?,?,?,?,?,?,?)", rows)
                    store._conn.commit()
                timings, rankings, sizes, payloads = [], [], [], []
                for i, store in enumerate(stores):
                    def search(index):
                        return store.search_vectors(str(index), namespaces=["benchmark"],
                            top_k=top_k, similarity_threshold=-1,
                            apply_level_weight=False, apply_recency_decay=False)
                    search(0)  # warm imports, matrix cache, and SQLite pages
                    samples, results = [], []
                    for index in range(len(queries)):
                        for _ in range(repeats):
                            started = time.perf_counter()
                            result = search(index)
                            samples.append((time.perf_counter() - started) * 1000)
                        results.append({r["doc_id"] for r in result})
                    timings.append({"median_ms": statistics.median(samples),
                                    "p95_ms": float(np.percentile(samples, 95))})
                    rankings.append(results)
                    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    page_size = store._conn.execute("PRAGMA page_size").fetchone()[0]
                    sizes.append(page_size * store._conn.execute("PRAGMA page_count").fetchone()[0])
                    payloads.append(store._conn.execute(
                        "SELECT SUM(length(vector)) FROM vectors").fetchone()[0])
                recall = statistics.mean(len(a & b) / top_k for a, b in zip(*rankings))
                reduction = 100 * (1 - payloads[1] / payloads[0])
                latency = 100 * (1 - timings[1]["median_ms"] / timings[0]["median_ms"])
                return {
                    "dataset_sha256": hashlib.sha256(documents.tobytes() + queries.tobytes()).hexdigest(),
                    "documents": len(documents), "queries": len(queries), "input_dim": documents.shape[1],
                    "components": n_components, "seed": seed, "top_k": top_k, "repeats": repeats,
                    "python": platform.python_version(), "numpy": np.__version__, "platform": platform.system(),
                    "projection_seconds": projection_seconds,
                    "full": {**timings[0], "retrieval_vector_json_bytes": payloads[0], "database_bytes": sizes[0]},
                    "projected_with_backup": {**timings[1], "retrieval_vector_json_bytes": payloads[1],
                                               "database_bytes": sizes[1]},
                    "retrieval_payload_reduction_percent": reduction,
                    "total_database_reduction_percent": 100 * (1 - sizes[1] / sizes[0]),
                    "median_latency_improvement_percent": latency,
                    "recall_at_k_vs_full": recall, "recall_drop_percentage_points": 100 * (1 - recall),
                    "criteria": {"retrieval_payload_at_least_50_percent_smaller": reduction >= 50,
                                 "total_database_at_least_50_percent_smaller": sizes[1] <= sizes[0] / 2,
                                 "median_latency_at_least_30_percent_faster": latency >= 30,
                                 "recall_drop_under_2_percentage_points": recall > .98},
                }
            finally:
                for store in stores:
                    store.close()
    finally:
        embeddings.embed = original_embed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, help="NPZ with documents and queries arrays")
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=False) as data:
            documents, queries = data["documents"], data["queries"]
        dataset = "supplied-embeddings"
    else:
        rng = np.random.default_rng(args.seed)
        documents = rng.normal(size=(args.documents, args.dimensions))
        queries = rng.normal(size=(args.queries, args.dimensions))
        dataset = "synthetic-gaussian-stress"
    result = benchmark(documents, queries, n_components=args.components, seed=args.seed,
                       top_k=args.top_k, repeats=args.repeats)
    result["dataset"] = dataset
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
