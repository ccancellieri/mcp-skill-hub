# Context Value Evaluation

This benchmark provides reproducible evidence for context retrieval and
compression without claiming that retrieved context improves a coding agent.
Its fixture corpus is synthetic and public-shaped. It does not read live Skill
Hub data, private repositories, authenticated services, or private endpoints.

## Offline suites

`context_value.py` creates an isolated SQLite database and evaluates 120 unique
retrieval cases: 40 calibration cases and 80 frozen holdout cases. The cases
span two separate synthetic project scopes and include English and Italian
prompts, relevant evidence, same-topic foreign-scope decoys, and independently
declared expected markers. A second suite contains 24 memory questions split
evenly among updated, conflicting, and unanswerable facts, with both languages
represented in every category.

The declared conditions are:

- A: no supplemental Skill Hub context.
- B: the current deterministic `build_context` path.
- C: the production deterministic context composer using
  `prepare_composition` and `compose_context`.

Learned ranking, RAG, and OpenJev are named as separate ablations and remain
`separate_not_run` in the manifest. They are not silently folded into condition
C. Precision, recall, abstention, constraint adherence, and foreign-scope
leakage are calculated from fixture markers. The raw row output remains
available for failure inspection.

Offline `context_token_estimate` values use UTF-8 bytes divided by four. They
are explicitly estimates for payload comparison; they are not provider token
counts, billing counts, or evidence of end-to-end savings.

Run the isolated suites with:

```bash
.venv/bin/python benchmarks/context_value.py \
  --output-dir benchmarks/results/context-value-latest
```

The output directory contains `corpus.json`, `manifest.json`, and
`offline-results.json`. The manifest freezes the corpus revision and SHA-256
hashes of the corpus and harness.

## Paired development evaluation

`context_value_tasks.py` defines 24 synthetic-development tasks: twelve
Skill-Hub-shaped Python tasks and twelve independently written,
Tellurion-shaped Rust tasks. They are not real issue results from either
project. No GeoID code, tests, fixtures, manager architecture, or private
project material is copied. The harness materializes a fresh standalone source
snapshot for every run. Its behavioral test must fail before the agent starts,
the test is hash-protected, and the same test must pass afterward. Every task
runs twice under A, B, and C. Condition order is counterbalanced, producing 144
required slots.

`context_value_e2e.py` has a concrete native adapter for the Codex binary in the
ChatGPT application. It uses `codex exec --json`, ephemeral sessions, ignored
user configuration, the workspace-write sandbox, native approval denial,
`gpt-6-astra`, and high reasoning effort. It keeps exec-policy rules enabled
and uses no approval or sandbox bypass. Prompts are passed on stdin using an
argv list, never a shell command. Each workspace is temporary, and every run
is capped at 600 seconds. Isolation
comes from `--ignore-user-config` plus a fresh fixture with no integration
configuration. Global native instructions and tool definitions still contribute
to client input. The harness does not claim an unused environment variable
disables hooks, and it does not recursively start a coding client from a prompt
hook.

The harness saves native JSONL traces and extracts usage from the final native
usage event. Cached input is a subset of input tokens and reasoning is a subset
of output tokens, so neither is added twice. When the native event omits either
breakdown it remains `null`, rather than becoming a measured zero. After the
client returns, the harness directly runs the fixture test; agent self-reported
success is ignored. One monotonic 600-second budget covers fixture setup,
baseline test, Codex, post-test, and snapshot verification together.

Without a configured client, this command writes all 144 slots as pending and
counts zero completions:

```bash
.venv/bin/python benchmarks/context_value_e2e.py \
  --output benchmarks/results/context-value-latest/e2e-results.json
```

An authorized qualification run is explicit and bounded. The recommended first
stage selects two Python and two Rust tasks, one repeat each, across A/B/C for
12 balanced runs. It reports the remaining 132 slots as pending and cannot pass
the promotion gate:

```bash
python3 benchmarks/context_value_e2e.py --run --qualification \
  --candidate-version SELECTOR_VERSION \
  --dataset-hash TRAINING_SNAPSHOT_SHA256 \
  --output benchmarks/results/context-value-qualification/e2e-results.json
```

Every completed row is atomically checkpointed. Repeat the same command with
`--resume` to retain matching completed rows and continue remaining slots; the
selector version and training dataset hash must match the checkpoint.

The promotion gate requires a paired median task-token reduction of at least
15 percent for C versus A, no correctness regression in any pair, and no
critical failure. Missing runs, native usage, correctness, or critical-failure
labels produce `unknown`, which denies promotion. Estimated offline payload
tokens are never substituted into this gate.

Completed native reports bind `source=evaluation_harness`, `estimated=false`,
the candidate selector version, its training-snapshot hash, a separate
evaluation-fixture hash, condition roles, project name, complete run count, and
native `main_tokens`/`auxiliary_tokens`. Candidate version and training dataset
hash are mandatory for execution.

The full experimental matrix remains in `experimental_runs` (144 A/B/C rows).
The top-level promotion view contains only the 96 strictly paired baseline and
candidate rows required by the selector gate; condition B remains available as
the declared control and is never misrepresented as promotion evidence.
`project_variability` reports paired reduction ranges and success rates
separately for the Skill-Hub-shaped and Tellurion-shaped synthetic fixtures.

## Interpretation limits

The offline suites measure evidence selection against a constructed corpus.
They do not measure final answer quality or real project completion. The native
suite is defined but must remain incomplete until a budgeted client run is
explicitly configured and executed. A pending report is evidence of missing
measurement, not a successful result.
