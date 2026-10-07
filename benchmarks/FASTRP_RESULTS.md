# FastRP namespace search qualification

Run from the repository root with the `fastrp` extra installed:

```bash
python3 benchmarks/fastrp.py
python3 benchmarks/fastrp.py --embeddings corpus.npz --components 128
```

The optional NPZ file supplies `documents` (N,D) and `queries` (Q,D) float
arrays; loading disables pickled objects. No embedding models or external
services are called. Default input is deterministic independent Gaussian
vectors: 2,000 documents, 20 queries, dimension 384, target 128, seed 42,
recall@10, three repetitions per query. Its hash and runtime versions are
recorded in [the JSON result](FASTRP_RESULTS.json).

This measures actual `SkillStore.search_vectors` over separate SQLite
stores, including query projection, SQL reads, JSON decoding, cosine scoring,
and sorting. Embedding generation is replaced by fixed fixture vectors;
level weighting and recency decay are disabled to isolate vector retrieval.
Each store gets one warm-up query; medians/p95 cover the repeated queries.
Native sqlite-vec was unavailable in this Python build; namespace retrieval
already uses the exact cosine scan regardless of the core skill engine.
Source insertion is batched and excluded from query timing. The output
separates retrieval JSON bytes from the complete SQLite database, including
original-vector backups, schema, indexes, and unused default tables.

Measured on macOS, Python 3.12.3, NumPy 2.5.0:

| Measurement | Full | Projected with backup |
| --- | ---: | ---: |
| Retrieval vector JSON bytes | 15,844,145 | 5,226,150 |
| Complete SQLite bytes | 17,031,168 | 25,223,168 |
| Median search milliseconds | 175.29 | 64.48 |
| p95 search milliseconds | 185.73 | 70.59 |
| Recall@10 against full neighbors | 1.00 | 0.13 |

Retrieval payload shrank **67.0%** and median latency fell **63.2%**. Total
SQLite storage grew **48.1%** because full originals are retained. Neighbor
recall dropped **87 percentage points** on this high-dimensional stress
fixture. Batch projection took about 3.5 ms; query timings include projection.

The proposal's “≤30% latency improvement” is ambiguous; the benchmark reports
the measured percentage and uses **at least 30% faster** as a useful latency
target. It qualifies ≥50% smaller retrieval payload, but **does not qualify
≥50% smaller total storage or <2 percentage points recall loss**. Synthetic
neighbor agreement is not semantic relevance or a production-corpus result.
Timings vary with hardware and system load. Repeat with representative,
fixed embedding/query fixtures before opting in for a project; do not infer
success on 100,000 documents or sublinear ANN performance from this run.

Phases 1–3 delivered the projection, opt-in integration, tests, and benchmark.
The feature remains experimental with default full vectors. IVF-PQ and
automatic corpus migration are intentionally outside this measured exact-scan
implementation; introducing another index would require separate evidence.
