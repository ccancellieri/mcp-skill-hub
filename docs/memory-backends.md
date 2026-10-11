# One serving memory backend

Set `memory_retrieval_backend` to `raw` or `wiki`. The default is `raw`; there is
no mixed mode. Invalid values omit both memory sources rather than guessing.
Skills and tasks retain their existing retrieval rules and verified scope.

The setting controls deterministic context collection, explicit context search,
thin-prompt wiki enrichment, and automatic indexing/refresh of raw memory and
wiki vectors. Explicit wiki tools remain available for preparation and review
when raw memory is selected. Original source files and existing derived indexes
are not deleted by changing the setting.

The [content comparison](../benchmarks/MEMORY_QUALITY_RESULTS.md) favors hybrid
wiki on independently curated project-document evidence. It does not qualify
live ingestion fidelity or establish that all existing memories are covered.
Keep raw selected until a wiki migration has passed those checks. Projection
and graph traversal remain experimental and are not part of the selected wiki
candidate.

The later [answer pilot](../benchmarks/MEMORY_ANSWER_PILOT.md) evaluates generated
answers and abstention on the existing public-document holdout. Wiki retains
more required facts, but that single diagnostic does not qualify live ingestion
or the configured answer provider.

Run the read-only aggregate prerequisite audit:

```bash
python benchmarks/audit_memory_backend.py
```

It inspects both public and private page roots and reports raw-source reference
coverage, source observation freshness, and canonical ingestion receipts without
printing source paths or content. Generated listings and backups do not count as
source coverage. `reference_coverage_complete`
is a prerequisite only. A current first-source hash cannot verify every cited
source or the meaning of a distilled page. `semantic_quality_qualified` stays
false because this tool does not judge content.

`source_hash` records the observed source. `ingest_hash` records a completed live
distillation into canonical Markdown; migration and connector writes do not set
it. An unchanged observation must still be distilled once. Legacy completed
queue entries without a receipt return to review. A receipt establishes neither
fact fidelity nor vector readiness: embeddings can fail independently, and a
successful reindex remains a separate serving prerequisite.

Private disk placement owns the project scope. Retrieval, candidate discovery,
indexes and writes must respect that scope even when frontmatter or a symbolic
link suggests another project. Private ingestion logs and assumptions remain in
the private project directory.

Before selecting wiki for an existing corpus, repair uncovered/stale pages,
evaluate content and evidence-grounded answers on frozen representative queries,
and check abstention and project scope. Retain original sources as provenance.
After selection and successful qualification, remove the unused derived index
in a backed-up maintenance window and verify that automatic refresh does not
recreate it. Switching configuration alone does not shrink an existing database.
