# Local auxiliary and wiki smoke

On 2026-10-12, LM Studio served existing GGUF weights on loopback only, without
starting Ollama or downloading additional model weights. Both models completed
real HTTP calls through the common dispatch and adapter. **Neither qualifies
for memory or wiki synthesis.** Keep #112 and #128 open.

| Measurement | DeepSeek-R1 1.5B | Qwen 2.5 Coder 3B |
| --- | --- | --- |
| Reported loaded memory | 1.04 GiB | 1.80 GiB |
| Recorded load time | 12.72 s | 2.35 s |
| Integrated calls | 7 | 7 |
| Mechanical/fact checks passing | 2/7 | 6/7 |
| Truncated completions | 3 | 0 |
| Recorded completion durations | 2.074–5.728 s | 0.645–4.631 s |

The seven actual helper paths were search summarization, wiki file answers,
fanout prompt synthesis, session memory creation, session memory update, the
legacy classifier, and public wiki ingestion in dry-run mode. All use one
synthetic source: Orchid has port 4321 and retry limit 3. The memory-update
input says tests now pass and the README is next. Raw completions, usage,
automatic checks and manual findings are retained in
[the full report](LOCAL_PROVIDER_SMOKE_RESULTS.json).

DeepSeek preserves the two facts in its summary and cited wiki answer. Its
fanout and classification truncate, memory sections violate the requested
format, and memory update retains completed tests as active work. Wiki ingestion
invents architecture, performance, a URL and shutdown behavior; truncated JSON
is rejected. A preceding 128-token readiness request also truncated without
producing its required exact answer.

Qwen is better at completing the bounded requests, but manual content review
finds failures missed by the mechanical checks:

- Memory update keeps “Add tests” under Pending Tasks after tests pass.
- Fanout invents specific file paths and client connection behavior, and
  changes a retry limit into a maximum number of attempts.
- Wiki raw index metadata invents a static-site CMS description. Current
  ingestion ignores `index_entries`, so this is an unsupported raw claim,
  not a persisted index change.
- The proposed page update cites `[[orchid-public-demo]]`, while ingestion
  assigns the canonical source slug `source-synthetic-public-orchid`. This
  would leave a dangling reference if applied; the dry-run wrote no pages.

Both models' cited wiki answers preserve the supplied facts. Qwen memory
creation also preserves facts and required headings. Those successes do not
qualify broad answer quality, memory state transitions or source integrity.

## Method and limits

This is a sequential exploratory comparison, not a sealed evaluation. Qwen was
tested after inspecting DeepSeek and correcting two measurement defects: stale
tests must be rejected, and vault write detection must compare paths and file
content hashes rather than file counts. Original DeepSeek results are retained
unchanged; these graders and the remaining manual findings must not be treated
as an independent holdout.

Each run caps output at 512 tokens and requests at 30 seconds, with lower native
caps retained for summarization/classification. These imposed caps are below
native fanout/wiki budgets: truncation describes this bounded test. There were
15 model calls including readiness, with 8,165 reported integrated tokens plus
137 readiness tokens. Latency excludes startup and is not a paired task-value
benchmark; the two models have different tokenizers.

Provider getters and explicit model selection are injected into actual helper
calls using temporary configuration. The transport intercept permits only the
chosen model at `http://127.0.0.1:1234/v1`, with retries disabled. It does not
globally monkeypatch sockets. Retrieval vector hits are fixtures, while page
loading, scope gates, parsing and dispatch are real. No live registry, memory,
wiki or client configuration was changed; embedding retrieval and canonical
write durability remain untested. This is not live-service model-selection
qualification and does not close #128.

The [reproduction harness](local_provider_smoke.py) defaults to a non-executing
plan. With an already loaded LM Studio model alias, run it with the repository
Python, `--repo . --model <loaded-alias> --output <report.json> --execute`.
Model setup is separate; the harness does not download or start a runtime.
Twenty routing/backend-unavailable regression tests passed. Both harnesses
compiled, and the revised harness executed all seven Qwen helper paths.
