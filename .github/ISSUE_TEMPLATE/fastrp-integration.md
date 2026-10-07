---
name: Random projection qualification
about: Evaluate opt-in embedding projection against full retrieval
title: "Qualify optional embedding projection on a representative corpus"
labels: ["enhancement"]
assignees: ""
---

## Scope and dataset

Identify the namespace, embedding model, fixed document/query fixture, corpus
hash, and independent relevance labels. Use public or sanitized fixtures.
Evaluate RAG and wiki separately when proposing memory-management replacement.

## Existing implementation

Seeded Gaussian projection and opt-in namespace/plugin-memory integration are
implemented. Queries use persisted per-row transforms; full originals remain
available for fallback. This is exact cosine retrieval, not graph FastRP or ANN.

The initial synthetic qualification failed total-storage and recall targets.
See [the benchmark report](../../benchmarks/FASTRP_RESULTS.md) and
[implementation scope](../../ISSUE-FAST-RP-PLAN.md).

## Required evidence

- [ ] Compare full and projected retrieval with identical queries and settings.
- [ ] Report retrieval payload and complete database size, including backups.
- [ ] Report median/p95 latency, projection overhead, and all failures.
- [ ] Report recall@k against full neighbors and independently labeled relevance.
- [ ] Preserve project scope, source fidelity, freshness, and recovery.
- [ ] Verify at least 50% less total storage, at least 30% lower median latency,
      and less than two percentage points recall loss.
- [ ] Record the promotion or rejection decision and remaining limitations.

Keep full retrieval as the default unless these gates pass. Do not remove
original vectors or authoritative memory/wiki sources to manufacture a storage
gain. No new indexing engine or automatic migration is implied by this issue.
