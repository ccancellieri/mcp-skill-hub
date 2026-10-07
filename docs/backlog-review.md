# Open issue review — 2026-10-07

Reviewed all ten open issues against the integrated source, tests, and tracked
qualification reports. Implementation and operational/research acceptance are
separate. None of these issues can be closed solely because optional vector
projection has been added.

| Issue | Current implementation | Remaining acceptance |
| --- | --- | --- |
| [#112](https://github.com/ccancellieri/mcp-skill-hub/issues/112) — Source integrity | Scoped source selection, retained originals, freshness fingerprints, and model/dimension compatibility checks are integrated. | Audit ingestion and individual source fragments for historical contamination; qualify sanitized bilingual adversarial cases. Do not delete originals to repair polluted summaries. |
| [#128](https://github.com/ccancellieri/mcp-skill-hub/issues/128) — Auxiliary routing | Search summarization, wiki file answers, fan-out synthesis, session memory creation/update, and classifier use the common request API. Mocked regressions cover explicit models/providers. | Live configured-provider and endpoint smoke checks; issue wording that describes all implementation as unintegrated is stale. No new endpoint or paid call is implied by this review. |
| [#129](https://github.com/ccancellieri/mcp-skill-hub/issues/129) — Client qualification | Shared adapters, bounded deterministic context, optional hooks, and installer cleanup are integrated. | Hook-free/manual workflows and registration/trust/invocation evidence on available real Claude, Pi, and OpenClaw clients. Adapter tests do not establish native-client qualification. |
| [#131](https://github.com/ccancellieri/mcp-skill-hub/issues/131) — Offline schema cleanup | The migration script and schema/idempotence tests exist. | Separate live maintenance window: stop, back up, migrate, restart, verify. No live database migration was performed in this review. |
| [#152](https://github.com/ccancellieri/mcp-skill-hub/issues/152) — Whole-task value | Tracked offline studies and bounded native qualification machinery retain negative results. | Preregistered real-task study meeting at least 15% median paired total-token reduction without correctness loss. Payload size and query latency cannot substitute for complete-task usage. |
| [#156](https://github.com/ccancellieri/mcp-skill-hub/issues/156) — MCP overhead | Full and static minimal profiles are integrated. The frozen historical schema report measured 87 to 8 tools and 92.41% fewer serialized schema tokens. | Current native-client capability/selection checks and whole-task evaluation, including discovery overhead. Historical schema estimates are not billing or current-client measurements. |
| [#157](https://github.com/ccancellieri/mcp-skill-hub/issues/157) — Compression fidelity | Lexical JSON compaction preserves duplicate members, numeric spellings, strings, and escapes; UTF-8 byte accounting is tested. | Update stale integration status and separately qualify any bounded recoverable command-output filter. That optional filter remains deferred. |
| [#158](https://github.com/ccancellieri/mcp-skill-hub/issues/158) — Relevance and abstention | Skill matching uses title/description token evidence; the reviewed memory/task/wiki substring false positive is repaired with token matching. | Frozen bilingual calibration/holdout and abstention qualification. The boundary fix does not claim a measured aggregate relevance gain. |
| [#163](https://github.com/ccancellieri/mcp-skill-hub/issues/163) — Compact representations | Existing deterministic compression and benchmark interfaces are reusable. | Freeze and review the five-arm contract; implement and test representations, expansion, observation import, and bounded native calibration. No new native inference result is claimed. |
| [#164](https://github.com/ccancellieri/mcp-skill-hub/issues/164) — Free pre-client entrypoint | Preview-only OpenAI-compatible catalog discovery is integrated; candidates stay unqualified and disabled. | Dedicated CLI, eligible endpoint/private-context qualification, frozen 12-case study, and direct-provider versus gateway comparison. A catalog listing does not prove zero-cost inference. |

## Memory management decision

The optional Gaussian projection benchmark fails total-storage and recall
acceptance: keeping originals grows the database, and the synthetic stress
fixture loses substantial neighbor agreement. Keep full-vector retrieval as
the default for RAG and wiki. Canonical markdown and raw source text remain
authoritative. See [projection qualification](../benchmarks/FASTRP_RESULTS.md).

Wiki uses the shared namespace store and lexical backfill; lexical matches can
mask vector ranking loss. Regression coverage confirms that disabling
projection restores full-vector ranking without changing markdown. This is
correctness evidence, not proof that projection improves semantic answers.

Derived memory chunks may be removed when their source is successfully
reindexed with a different chunk layout. This cleanup must be scoped to that
file and namespace and happen only after replacement writes succeed. Failed
reindexing must retain the prior evidence; unrelated files and source text
must remain intact.

## Follow-up order

1. Qualify source integrity and deterministic relevance (#112, #158).
2. Verify native clients and configured provider endpoints (#129, #128).
3. Evaluate complete-task value (#152), including minimal profiles (#156).
4. Run separately bounded representation/pre-client experiments (#163, #164)
   only with an explicit frozen protocol.
5. Schedule offline database maintenance (#131) independently of retrieval.

Keep research stop rules and negative results visible. Do not revive automatic
model layers, remove recovery data, or widen project scope to make a benchmark
look favorable.
