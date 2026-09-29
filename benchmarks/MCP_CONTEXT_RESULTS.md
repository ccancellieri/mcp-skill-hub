# Compact context responses over real MCP stdio

This continues the [offline input/modular study](INPUT_CONTEXT_RESULTS.md).
The index and compact batch are now actual opt-in MCP response modes, exercised
through a minimal server subprocess. Existing response defaults remain compatible.

## Method

The versioned `mcp_context_payloads.py` script starts a fresh source-checkout
server for each fixture with an isolated home, configuration and SQLite database.
The corpus has six memory modules and foreign-project copies. English and
Italian prompts are run against short and long variants. Fixture time and opaque
IDs are fixed for reproducible payload counts; retrieval, expansion, scope checks,
MCP schemas and response serialization use the real implementation.

For each fixed schedule of one, two or six source reads, compare:

- Existing path: default preview response followed by individual full reads.
- Compact path: `detail="index"` followed by one `detail="compact"` batch.

The requested original source texts must match exactly across conditions. The
script checks original-prompt and warning preservation, absence of source bodies
in the index, and exclusion of foreign-project fixtures. It does not ask an LLM
to choose the sources or complete a task.

## Observations

The following English figures use `o200k_base`, counting each serialized tool
request plus structured response once. Tool schemas, transcript replay, model
reasoning and final answers are excluded from this table.

| Fixture / reads | Existing request + response tokens | Compact request + response tokens | Reduction |
| --- | ---: | ---: | ---: |
| Short modules / 1 | 1,665 | 1,076 | 35.38% |
| Short modules / 2 | 1,916 | 1,213 | 36.69% |
| Short modules / 6 | 2,916 | 1,767 | 39.40% |
| Long modules / 1 | 3,434 | 1,394 | 59.41% |
| Long modules / 2 | 4,009 | 1,856 | 53.70% |
| Long modules / 6 | 6,336 | 3,737 | 41.02% |

The Italian conditions show the same direction, with reductions between 35.12%
and 59.20%. The long English initial response alone shrinks from 2,820 to 825
structured tokens; its contents differ intentionally because the index omits
source previews. Exact source text is recovered only for the requested items.

The six-source flow uses two MCP calls instead of seven. This is a count of tool
calls, not a measured latency or model-turn reduction. A client may already
parallelize independent calls.

Eight tools remain available. The script separately records the serialized
catalog and instructions (1,263 tokens with its compact, null-omitting JSON
representation). It also reports text-content and reconstructed result-envelope counts and
SHA-256 values. The envelope is built from the parsed tool result; it is not a
raw JSON-RPC transport capture. **Structured, text-content and result-envelope counts are alternative
representations: do not add them.** Client handling of structured results and
caching determines the actual input sent to a model.

## Boundaries

This is an improvement over the existing MCP response path, not proof that MCP
beats no enrichment. The earlier lean full-context baseline remains important:
six short rules fit in 190 tokens, and six long modules in 2,182. In particular,
reading every long module through the compact path still costs more than that
lean full packet. Do not add an index when the evidence is already short.

The composer still loads source bodies internally; no I/O or RAM reduction is
claimed. Batches are limited to six unique IDs and 100,000 source characters.
Unknown, stale or out-of-scope references reject the batch before text is
returned; oversize batches must be split explicitly. There is no silent
truncation. Mandatory task constraints belong in the unchanged user request or
trusted client instructions, not behind an optional source read.

No automatic hook changes, remote inference, model selector or prompt rewrite
was enabled. Native complete-task usage, missed constraints, extra reads, retries
and correctness remain the promotion gate tracked by issues #152 and #156.

## Reproduce

Use the project dependencies, Python 3.11 or later, `tiktoken==0.12.0`, and a
verified offline tokenizer cache as in the earlier report:

```sh
export TIKTOKEN_CACHE_DIR=/path/to/verified-tokenizer-cache
python benchmarks/mcp_context_payloads.py --output /tmp/mcp-context-payloads.json
python -m pytest tests/test_context_composer.py tests/test_context_mcp.py -q
```

The separate `installed_smoke.py` check builds on the existing installed-wheel
stdio test and now verifies index plus compact batch against the installed
package. The source-checkout payload measurements above must not be confused
with that installation check or with LLM task evaluation.

Verification on the measured implementation: 38 focused composer/MCP tests
passed; the full Python suite passed 1,979 tests with five skips and eight
existing HTTPX deprecation warnings. The installed-wheel stdio smoke test also
passed, including index retrieval and exact source expansion through a compact
batch. No model calls were requested by these checks.

## Design sources

- [OpenAI latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization):
  filter context and reduce unnecessary requests; measure the trade-offs.
- [Anthropic effective tools](https://www.anthropic.com/engineering/writing-tools-for-agents):
  useful response content, explicit concise/detailed options and actionable limits.
- [Anthropic code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp):
  progressive disclosure and filtering before information enters model context.

These sources motivate the implementation. Their published benchmark results
are not attributed to this project.
