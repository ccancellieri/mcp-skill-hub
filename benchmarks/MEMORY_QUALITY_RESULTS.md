# Memory retrieval content qualification

The earlier random-vector recall@10 measured neighbor agreement, not whether
retrieved passages could support a correct answer. This study evaluates the
rendered evidence by meaning. Hybrid wiki is the strongest content candidate
on this bounded fixture. It is not qualified for a wholesale live migration.

## Frozen comparison

The corpus contains ten tracked project documents, 24 questions (nine Italian),
20 answerable questions with 48 required facts, and four unanswerable questions.
Odd question IDs are calibration; even IDs are holdout. Questions and gold
facts were authored independently of 32 source-grounded curated wiki pages,
before retrieval. Neither corpus nor ranking was tuned after inspecting results.

All arms use the same local cached `all-MiniLM-L6-v2` model, top-10 ceiling,
1,500 `o200k_base` token budget, and rendered citations. The projection arms
use seed 42 and dimensions 128/256. Full and projected RAG use actual
`SkillStore.search_vectors`; wiki uses the production hybrid query. Lexical
wiki has its own vector-free database. The relationship arm adds one hop of
existing wiki links after three seeds; it is **not Microsoft GraphRAG**.

The retrieved content was reviewed under randomized per-question arm labels.
Model-based semantic adjudication distinguishes fully supported, partial, and
missing facts, with independent spot checks and exact evidence-quote
validation. The provisional phrase screen was rejected before reporting these
results. Partial support receives no credit in the fully supported count.
Source references alone cannot establish a fact. Exact source-quote recall is
retained as a diagnostic and is not used to judge wiki paraphrases.

## Results

| Approach | Fully supported facts | Calibration / holdout | Explicit unknown evidence | Mean noise, 0–3 | Median retrieval ms | Total candidate bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full raw RAG | 38/48 (79.2%) | 18/24 / 20/24 | 4/4 | 2.00 | 7.81 | 1,161,272 |
| Projected raw RAG, 128 | 33/48 (68.8%) | 15/24 / 18/24 | 3/4 | 2.25 | 2.91 | 837,688 |
| Projected raw RAG, 256 | 34/48 (70.8%) | 17/24 / 17/24 | 4/4 | 2.08 | 5.42 | 976,952 |
| Hybrid wiki | **46/48 (95.8%)** | **22/24 / 24/24** | 3/4 | 2.00 | 38.84 | 913,491 |
| Lexical wiki | 31/48 (64.6%) | 14/24 / 17/24 | 2/4 | 2.25 | 4.20 | **622,675** |
| Wiki with one-hop links | 45/48 (93.8%) | 22/24 / 23/24 | **4/4** | 1.92 | 42.96 | 913,491 |

Noise: 0 means nearly all items directly relevant; 1 means a clear answer amid
some unrelated items; 2 means unrelated items predominate but decisive evidence
is accessible; 3 means mostly off-topic or decisive evidence absent. No retrieved
packet asserted a listed forbidden claim. This does **not** establish absence of
hallucinations in generated answers: no answers were generated. Explicit unknown
evidence measures availability of a caveat, not whether an answerer abstains.

Hybrid wiki improves supported facts by 16.7 percentage points and uses 21.3%
less total candidate storage than full raw RAG. Its retrieval is about 5.0 times
slower in this small fixture. One-hop links improve unknown-evidence coverage
but lose one fully supported fact and add latency; there is no clear reason to
promote the extra graph traversal. The smallest arm, lexical wiki, loses facts.
Projection also loses facts despite lower storage and latency.

Examples demonstrate content quality rather than length:

- MQ06: wiki includes both rejecting changed sources and rejecting posted bodies
  as replacements for verified evidence. Raw full retrieval mentions stale checks
  but misses the posted-body rule.
- MQ16: wiki states named namespace **and declared level** for per-index memory.
  Full raw retrieval includes the legacy routing rule but misses that fact.
- MQ20: condensed wiki preserves the negative 48.1% storage increase. Several raw
  contexts say storage grows without retrieving the exact result.
- MQ19: every arm misses the historical-table fact. A larger context is not
  automatically sufficient.

## Promotion decision and limits

Choose **hybrid wiki as the candidate architecture to qualify**, with original
sources retained for provenance and rebuilding. Do not add graph traversal or
projection to that candidate on this evidence. Do not keep two serving memory
indexes as the steady-state design.

Production now has a single `memory_retrieval_backend` selector: `raw` or `wiki`.
The conservative default is `raw`, because the existing wiki has uncovered raw
sources and insufficient freshness metadata. Choosing `wiki` affects serving
and automatic maintenance together; it does not synthesize missing pages or
delete old indexes. The unselected data remains available for an explicit,
verified migration rather than being destroyed to obtain a smaller number.
See [backend operation](../docs/memory-backends.md).

The wiki fixture was curated independently from public documentation; it is
not the production ingestion pipeline or live user/plugin memory. Its quality
gain may reflect good distillation. There is no Microsoft GraphRAG arm in this
repository, no answer-generation study, and no complete-task cost measurement.
The foreground hook uses deterministic scoped retrieval, not these vector arms.
Explicit context search also uses a different wiki vector/rendering path from
the hybrid `wiki.query` benchmark. Backend selection does not merge those
interfaces into an identical ranking algorithm.
This study does not qualify the hook's relevance, scope leakage, or latency.

Storage includes canonical sources once, the candidate SQLite database, and
wiki Markdown where applicable. Databases include common schema overhead, which
is substantial at this corpus size. Projected candidates deliberately omit
full-vector backups and rely on hypothetical source reindex recovery; production
projection retains backups. These numbers do not overturn the earlier measured
storage increase with backups. Raw chunk bodies are preloaded for this isolated
experiment; wiki query reads files, so timing is not a complete production-path
comparison. Embedding generation, wiki synthesis, and maintenance are excluded.
Native sqlite-vec was unavailable in the runtime; timings use the legacy path.

Before live wiki promotion: verify source coverage and freshness, qualify
automatic ingestion on representative scoped memories, inspect rendered
answers and abstention, then retire the raw derived index with a recoverable
maintenance procedure. A reference-coverage audit alone cannot certify quality.

## Reproduce

Provision the local embedding model and a checksum-verified offline tokenizer
cache before running. Use a fresh output directory:

```bash
TIKTOKEN_CACHE_DIR=/path/to/tokenizer-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python benchmarks/memory_quality.py --output-dir /path/to/fresh-results
python benchmarks/audit_memory_backend.py
```

The harness emits rendered contexts, randomized review packets, grading keys,
and arm keys separately. Keep the arm key hidden during content review. It
rejects reuse of populated work directories and checks tokenizer availability
before creating indexes. [The result JSON](MEMORY_QUALITY_RESULTS.json) records
fixture/model/harness fingerprints, per-arm measurements, and evidence reviews.
Repeated runs produced byte-identical review packets; timing is from the final run.
Latency uses one sample per question in fixed arm order without warmup; these
exploratory timings cannot qualify a latency improvement. Cached model revision
metadata is recorded, but model weights were not independently hashed.
