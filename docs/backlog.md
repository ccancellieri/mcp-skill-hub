# Current conclusions and remaining work

Updated 2026-10-07. This is the current work list; the earlier
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

## Deferred research

[#163](https://github.com/ccancellieri/mcp-skill-hub/issues/163) (compressed
representations) and [#164](https://github.com/ccancellieri/mcp-skill-hub/issues/164)
(free pre-client/gateway) remain separate deferred studies. Neither has a
successful native evaluation or product promotion. They are not prerequisites
for the six active gates above; do not expand them automatically.

Current closure verification: 26 focused compression/fidelity/MCP-profile tests
passed. The integrated code at `9317b78` passed 2,061 offline tests, with 18 skips.
Protocol schema size, content fact support and completed-task value are separate
measurements; preserve their negative and unknown results.
