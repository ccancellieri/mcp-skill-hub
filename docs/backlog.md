# Current conclusions and remaining work

Updated 2026-10-11. This is the current work list; the earlier
[backlog review](backlog-review.md) remains a historical evidence snapshot.

## Conclusions

- JSON fidelity repair is delivered; #157 is closed. Optional recoverable
  tool-output filtering has not shipped and is outside the completed repair.
  Any renewed experiment belongs behind #152's complete-task gate.
- Static minimal MCP profile is delivered; #156 is closed. Native deferred
  discovery is not adopted. Actual-client checks belong to #129 and complete-task
  comparison/discovery overhead to #152, rather than a third parallel tracker.
- Use one memory backend. Hybrid wiki is the strongest content candidate in the
  [frozen document comparison](../benchmarks/MEMORY_QUALITY_RESULTS.md): 46/48
  required facts fully supported versus 38/48 for raw RAG and 33/48 for projected
  RAG at 128 dimensions. Live coverage, ingestion fidelity and answer quality
  remain unqualified, so raw is the sole default. Follow
  [the migration prerequisites](memory-backends.md); do not delete originals.
- Projection and one-hop graph traversal have no evidence sufficient for
  promotion. Keep the existing experimental capability outside the selected
  default; do not add another memory engine or claim whole-task savings.
- The corrected live read-only audit includes private pages: 1,077 of 1,226
  raw source paths are referenced, leaving 149 uncovered. There are 1,139
  source pages without observation hashes and no source pages with a current
  canonical ingestion receipt. These are migration gaps, not evidence of
  answer quality. Receipt, queue and private-scope repairs make the preparation
  path usable; they do not qualify global promotion or index deletion.
- A separate [answer pilot](../benchmarks/MEMORY_ANSWER_PILOT.md) checks actual
  generated content on twelve existing holdout questions: wiki answers fully
  satisfy 22/24 required facts versus raw 18/24. Both abstain on the two unknown
  questions. This single native-assisted public-document run does not qualify
  live ingestion, the configured wiki provider or whole-task value.

## Active gates

| Work | Owner | Done when |
| --- | --- | --- |
| Trustworthy memory | [#112](https://github.com/ccancellieri/mcp-skill-hub/issues/112) | Source fragments, wiki coverage/freshness and ingestion are audited; sanitized bilingual adversarial integrity cases qualify. |
| Relevant foreground context | [#158](https://github.com/ccancellieri/mcp-skill-hub/issues/158) | A separately frozen bilingual skill/candidate holdout reduces irrelevant injection without unacceptable recall loss and supports abstention. The wiki document comparison is a different path. |
| Provider integration | [#128](https://github.com/ccancellieri/mcp-skill-hub/issues/128) | Configured provider/endpoint smoke checks qualify the integrated six auxiliary call paths. |
| Client integration | [#129](https://github.com/ccancellieri/mcp-skill-hub/issues/129) | Available real clients qualify hook-free/manual workflows, minimal-profile capabilities/tool choice, trust/invocation, isolation and bounded failure behavior. |
| Product value | [#152](https://github.com/ccancellieri/mcp-skill-hub/issues/152) | Preregistered real tasks achieve at least 15% median paired native-token reduction without correctness degradation or critical errors, counting all calls and recovery. |
| Database maintenance | [#131](https://github.com/ccancellieri/mcp-skill-hub/issues/131) | The live service is stopped, backed up, migrated offline, restarted and verified. Code delivery does not complete maintenance. |

Do memory integrity/relevance first, runtime qualification next, and task value
after those qualify. Keep offline maintenance a separate operation.

The [foreground skill qualification](../benchmarks/FOREGROUND_RELEVANCE_RESULTS.md)
grades actual selected source IDs on 48 frozen public English/Italian cases.
Holdout irrelevant injections fall from 9 to 1 in English and 7 to 1 in Italian;
recall is 100% and 91.7%. Both languages abstain correctly on only five of six
negative cases, below the fixed 90% requirement: #158 remains open. Independent
review fixes protect operational requests and explicit invocations, including
bounded lookup beyond description distractors. The post-review rerun is
diagnostic after holdout inspection, not a new independent qualification.

## Archived proposals

[#163](https://github.com/ccancellieri/mcp-skill-hub/issues/163) (compressed
representations) and [#164](https://github.com/ccancellieri/mcp-skill-hub/issues/164)
(free pre-client/gateway) are closed as **not planned**. Their proposals and
negative or unknown evidence remain available in the closed issues. This is a
scope decision, not successful research or product promotion. Existing provider
qualification belongs to #128 and complete-task value to #152.

The [database maintenance rehearsal](../benchmarks/DATABASE_MAINTENANCE_REHEARSAL.json)
applied the existing migration to a consistent copy of the live database:
retired schema removed, all 98 ordinary nonretired table counts preserved and
integrity check `ok`. Ten virtual tables were not queried by that runtime.
The six migration tests pass. The live operation remains open in #131: active
stdio MCP clients must be disconnected and subsequently restarted and verified.

Current closure verification: 26 focused compression/fidelity/MCP-profile tests
passed. The integrated code at `9317b78` passed 2,061 offline tests, with 18 skips.
The October 11 integrity repair passed a 2,085-test offline run with 18 skips;
after the final write/backup refinements, all 214 affected wiki, audit, route,
watcher, enrichment and benchmark tests passed. Independent focused review
found no remaining actionable defects in the examined paths.
Protocol schema size, content fact support and completed-task value are separate
measurements; preserve their negative and unknown results.

The foreground repair passed a 2,138-test offline run with 18 skips; all 130
affected tests passed after its final legacy-name lookup refinement. Final
independent review passed 39 focused tests without an actionable defect in the
examined paths. These checks do not replace the failed relevance gate.
