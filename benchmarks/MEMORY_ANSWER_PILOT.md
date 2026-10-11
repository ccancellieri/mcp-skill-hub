# Evidence-grounded answer pilot

On 2026-10-11, separate native answer-generation and semantic-grading agents
evaluated full raw RAG and hybrid wiki using the twelve existing holdout
questions. Each answerer input contained only the question and its retrieved
evidence, with opaque per-question arm labels. The grader independently checked
required facts, evidence support, citations, forbidden claims and abstention.
No configured wiki provider, live memories or production answer tool was used.

| Measure | Full raw RAG | Hybrid wiki |
| --- | ---: | ---: |
| Required facts fully answered and supported | 18/24 | 22/24 |
| Answerable responses satisfying all required facts | 6/10 | 8/10 |
| Justified abstention on unknown questions | 2/2 | 2/2 |
| Unsupported claims / forbidden claims / citation errors | 0 / 0 / 0 | 0 / 0 / 0 |

Partial facts receive no full credit. Raw had one partial and five missing
facts; wiki had two missing facts. This measures the actual answers' content,
not response length or neighbor agreement. All prompts, rendered evidence,
answers and individual semantic assessments are retained in
[the pilot artifact](MEMORY_ANSWER_PILOT.json).

Wiki preserved the changed-source/posted-body rules (MQ06), the named namespace
and declared level (MQ16), and the exact 48.1% storage increase (MQ20). Raw
appropriately declined where its retrieved evidence was incomplete. Both
answerers omitted the explicit boundary that synthetic neighbor agreement does
not establish semantic relevance, even though their packets contained it.
Both also omitted the historical log-line example in MQ14. These omissions show
why sufficient retrieval alone cannot certify answer quality.

The inputs' full review-packet hash matches the previously reported frozen
content comparison. Rankings and corpora were not retuned. Generation used
gpt-6.1-sol with medium reasoning; a separate context graded with the same model
family at high reasoning. This is a single diagnostic run with correlated model
judgments, no variance estimate, no new sealed task suite, and no whole-task
usage or latency measurement. It supports hybrid wiki as the preparation
candidate, but does not qualify the configured provider or live ingestion.

The [read-only live audit](MEMORY_INTEGRITY_AUDIT.json) includes private pages:
1,077 of 1,226 raw paths are referenced, leaving 149 uncovered; 1,139 source
pages lack observation hashes and no source page has a current canonical ingest
receipt. The previous 165 gap count omitted private pages. Global wiki promotion
and deletion of the raw derived index remain unqualified. Original sources
remain authoritative; use the [migration prerequisites](../docs/memory-backends.md).
