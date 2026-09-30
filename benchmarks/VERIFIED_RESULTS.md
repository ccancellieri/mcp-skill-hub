# Verified local measurements (2026-09-29)

The protocol and context measurements describe the review candidate at commit
`006651208cb3923ccfcaa7695e394feed026e07d`. They use synthetic evidence and
an isolated MCP server. No live project memory, remote selector, or cloud model
was used. The minimal profile is opt-in, and composition requires an explicit
request; these figures do
not establish lower billed usage or better completed development tasks.

| Measurement | Observed result | What it measures |
| --- | --- | --- |
| MCP full → minimal | 87 → 8 tools; 16,362 → 1,242 `o200k_base` tokens (92.41% less) | Compact JSON serialization of actual `tools/list` schemas, including output schemas, and server instructions. |
| Current synthetic context | 32,544 `o200k_base` tokens for deterministic retrieval; 36,792 for manual composition across 144 cases | Supplemental text only. Composition remains larger than direct retrieval on this corpus. |
| Current evidence selection | Both context conditions found every expected marker in 136 answerable cases; eight no-answer cases abstained; zero labeled foreign-scope leaks | Fixture-marker performance, not full semantic equivalence or answer quality. |
| Archived composer envelope comparison | 41,560 → 36,792 `o200k_base` context tokens across 144 paired synthetic cases; 11.49% median reduction among nonempty pairs | Local before/after result, whose before implementation is not publicly commit-pinned. |
| Latest relevance correction | The same 144 marker sets and UTF-8/4 estimates as its exact parent `a15385b8b1a4bb250b51c2731d04427419a284e9` | Regression check on the existing corpus. A separate read-only public-skill probe grew from 559 to 673 context tokens. |

The schema measurement was rerun through the actual MCP protocol in a fresh
process for each profile. The archived serialized payload SHA-256 hashes are
`08e534423cb7c18cee75fd36152d0d8eb300705151d0a0497754f59311814d85`
(full) and `d5665dddb3e7867b1392086a84d22b67fe5e85725a0b03493f09e582ab230030`
(minimal). This run used `tiktoken` 0.12.0 and `o200k_base`. Client runtimes may
cache, defer, omit, or transform these definitions, so the 92.41% figure is a
serialized protocol-payload comparison, not a client token bill.

## Installed stdio integration check

A separate smoke run installed a locally built `mcp-skill-hub` 0.2.0 wheel into
an isolated target, started its minimal MCP server as a **real stdio
subprocess**, and used an isolated home and a synthetic `SKILL.md`. It indexed
the skill with `index_skills_text`, listed eight tools, found its description
without leaking its body, loaded full text on request, expanded a composition
candidate, composed a selected source reference, and preserved the original
multiline prompt. The result was `passed` with no model call requested. The
client environment had FastMCP 3.2.0 and MCP 1.26.0. Setup time varied across
smoke runs and was not treated as a latency benchmark.

To repeat after building the wheel from the same source, install it into a
fresh target with dependencies already available to the invoking Python:

```sh
python -m pip install --no-deps --target "$TARGET" "$WHEEL"
PYTHONPATH="$TARGET" python benchmarks/installed_smoke.py \
  --expected-package-root "$TARGET"
```

The script rejects source-checkout resolution and emits a JSON stage on
failure. It changes no live user configuration. This proves installed-process
integration for the checked path, not coding-task value, whole-task savings,
or real-client tool choice.

The current context measurement used the fixed `synthetic-public-shaped-v1` corpus: 120
retrieval cases plus 24 memory questions. Each of the three conditions produced
144 rows. The canonical corpus SHA-256 in the benchmark manifest is
`e99d952e8fcc2bce09e7a86279eb26668134292be1c35a589473839bd856e407`;
the saved corpus file's byte SHA-256 is
`1ee8c29a7bfc490b678b423d879629dc1f16ceec192a6eeb69cb27d3d4813946`.
Those hashes differ because the manifest hashes canonical JSON. The harness
SHA-256 was `54b84e7b48ec539f007d2beed2490a354fa453ad842b9e39cdd4a1e9d5c3e014`.
The archived before packet came from the prior integrity run; the current packet and
latest relevance regression both sum to 36,792. Historical results and the
relevance correction are documented in the local benchmark reports. The
before implementation was not a separately frozen public release, so this
comparison should be treated as a controlled local result, not an independently
reproduced cross-release benchmark.

To remeasure the current protocol payload, use an environment with FastMCP,
`tiktoken` and a locally cached `o200k_base` tokenizer, then run from the
repository root:

```sh
PYTHONPATH=tests:src python - <<'PY'
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tiktoken
from test_mcp_profiles import _probe

encoding = tiktoken.get_encoding('o200k_base')
with TemporaryDirectory() as root:
    for profile in ('full', 'minimal'):
        result = _probe(Path(root) / profile, '--profile', profile)
        payload = json.dumps(
            {'tools': result['schemas'], 'instructions': result['instructions']},
            ensure_ascii=False, separators=(',', ':'),
        )
        print(profile, len(result['names']), len(payload.encode()),
              len(encoding.encode(payload)))
PY
```

The offline fixture can be rerun without model calls or live data:

```sh
RESULT_DIR=$(mktemp -d)
PYTHONPATH=src python benchmarks/context_value.py --output-dir "$RESULT_DIR"
```

Its `context_token_estimate` field is UTF-8 bytes divided by four, **not** a
tokenizer count. The packet figures above were counted separately with
`o200k_base` over each condition's `context` text. To reproduce the current
token sums from the new output, run:

```sh
python - "$RESULT_DIR/offline-results.json" <<'PY'
import json, sys, tiktoken
rows = json.load(open(sys.argv[1]))['rows']
encoding = tiktoken.get_encoding('o200k_base')
for condition in ('B_build_context', 'C_context_composer'):
    selected = [r for r in rows if r['condition'] == condition]
    print(condition, len(selected),
          sum(len(encoding.encode(r['context'])) for r in selected),
          sum(bool(r['scope_leakage']) for r in selected))
PY
```

Run
`pytest tests/test_mcp_profiles.py tests/test_context_value.py tests/test_compression_integrity.py`
to check profile boundaries, original prompt
preservation, scope and manifest behavior, and lossless JSON/UTF-8 handling.
These checks do not prove semantic fidelity for arbitrary inputs.

The only bounded native development-task qualification to date completed 12 of
144 planned synthetic runs. All 12 fixture tests passed, but the four paired
no-Hub/composer comparisons had a median **-4.80%** task-token reduction
(composer used more). The remaining 132 runs are pending, and the 15% median
promotion gate has not passed. The harness constructs supplemental context in
Python and places it in the client's stdin prompt while ignoring user config;
it does **not** register or exercise the installed MCP server. These older
native runs belong to an earlier source freeze and cannot validate the current
minimal MCP profile or latest composer changes. An installed-client, paired
task result is still required before claiming real-use benefit. Kev, Laya, and
other learned selectors remain unpromoted.
Neither the schema nor packet reductions should be presented as whole-task
savings or production-wide quality gains.
