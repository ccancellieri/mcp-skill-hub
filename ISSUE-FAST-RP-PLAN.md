# Optional random projection: implementation and qualification

## Scope

The proposal's FastRP name is retained for compatibility. The implementation
is seeded Gaussian projection of existing embeddings, not graph FastRP or an
approximate nearest-neighbor index. Namespaced vectors continue to use exact
cosine search. Core skill, task, and teaching indexes keep their existing
engines. Wiki pages and raw memory sources remain authoritative.

A 768-dimensional float32 vector contains 3 KiB of numeric payload. SQLite
JSON serialization, indexes, and retained originals add storage overhead;
reducing dimensions does not imply the same reduction in total database size.

## Delivered phases

1. Projection: `src/skill_hub/fastrp.py` provides single-vector and bounded
   input-batch APIs, seeded shared transforms, float32 output, validation, and
   versioned transform identity. NumPy is an optional `fastrp` extra.
2. Integration: `FastRPIndexer` in `vector_sources.py` adapts namespace writes.
   `store.py` retains full originals and each row's transform; queries use the
   saved transform, even after configuration changes. Explicit disablement
   restores full-vector retrieval. Plugin `vector_indexes[]` and
   `memory.indexes[]` support opt-in projection declarations.
3. Qualification: focused unit/integration tests and `benchmarks/fastrp.py`
   compare actual SQLite retrieval, storage, latency, and neighbor recall.
   [Configuration](docs/plugin-extension-points.md) and
   [measured results](benchmarks/FASTRP_RESULTS.md) document the limits.

## Acceptance and current decision

Evaluate at least 50% less **total index storage**, at least 30% lower median
search latency, and less than two percentage points of recall@k loss against
full-vector retrieval. The original latency phrase, “≤30% improvement,” was
ambiguous; measured percentages are reported explicitly.

The fixed synthetic stress benchmark measured 67.0% smaller retrieval payload,
63.2% lower median search latency, **48.1% larger total SQLite storage**, and
recall@10 of **0.13**, an 87-percentage-point loss. It fails the storage and
recall gates. This is a synthetic neighbor-agreement result, not semantic
relevance or a representative production benchmark.

Keep projection experimental and disabled by default. Do not replace the
existing RAG or wiki memory management, delete original vectors, or remove
markdown sources based on these results. Full vectors provide recovery when
projection is disabled, unavailable, or incompatible.

## Follow-up qualification

Use fixed representative document/query embeddings with independent relevance
labels before considering replacement. Evaluate RAG and wiki separately,
including lexical fallback, source fidelity, freshness, scope rejection,
recovery, complete storage, and end-to-end query cost. Publish regressions and
every omitted result. A reduction in vector payload alone does not justify
promotion or cleanup of the authoritative memory paths.

IVF-PQ, graph embeddings, automatic corpus migration, and a 100,000-document
scale qualification have not been implemented or demonstrated. They require
separate evidence rather than being prerequisites hidden in this integration.
