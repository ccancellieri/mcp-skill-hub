# Roadmap

## Deliver the deterministic core

The product goal is fewer total tokens per completed task at preserved quality.
Smaller protocol payloads are useful evidence, but do not establish task savings.
The [verified results](../benchmarks/VERIFIED_RESULTS.md) separate these measures.

| Work | Current implementation | Remaining qualification |
| --- | --- | --- |
| #154 — Manual composer | Explicit project roots, ranked source descriptions, exact excerpts, stale-source checks, preview and copy. | Native-client workflow qualification is tracked in #129; task effects in #152. |
| #156 — Minimal MCP surface | Eight tools, description-first lookup and explicit source recovery. Full profile stays the default. | Measure discovery overhead and completed tasks in real clients. |
| #157 — Compression fidelity | Lexical JSON whitespace compaction retains strings, duplicate members and numeric spelling. UTF-8 bytes are counted explicitly. | Optional command-output filtering is deferred; it is not a release requirement. |
| #158 — Relevance | Namespace-only skill matches no longer receive a name-match boost. Explicit IDs and relevant companion skills are retained. | Broader bilingual relevance and abstention; current corpus does not demonstrate a new aggregate gain. |
| #128 — Provider routing | Auxiliary calls use the common dispatcher and retain explicit model IDs. | Integrated provider/endpoint smoke checks. |
| #112 — Source integrity | Exact project checks, retained originals, source fingerprints, embedding compatibility checks. | Audit and reindex historically contaminated sources without deleting originals. |

These implementations are delivered through PR #159. Only completed integration
and the corresponding checks justify closing an issue. Current-client model and
effort observations never change configured server providers.

## Qualify actual task value

#152 remains the promotion gate: at least 15% lower median total-task tokens,
without aggregate success degradation or new critical errors. Count auxiliary
calls, rereads, corrections, cache and reasoning when the client exposes them.
The current evidence does not pass that gate.

#129 distinguishes adapter tests from live Claude, Codex, Pi and OpenClaw runs.
Hooks remain optional; failure passes the original prompt through. No L1/L2 call
belongs in the two-second automatic context path.

#131 is a separate maintenance operation: stop the service, back up the live
database, apply the existing offline migration and verify restart. It is not a
startup or context-composition side effect.

## Experiments not promoted

The current OpenJev, Rizzo, Qwen, Kev and Laya comparisons do not justify an
additional automatic model layer. Preserve their negative results. DwarfStar
(#121) and preference-learning product controls (#155) are not planned. Visual
workflows, new orchestration engines and semantic backends require a concrete
need and better evidence before returning to active scope.

## Retired behavior

Generic task interception, automatic continuation, inferred approvals and
hook-driven model switching have been retired. The original prompt, permissions
and reasoning remain the client's responsibility. See the
[context-service migration guide](context-service.md).

The credential vault (#30) shipped and was later removed in `324a544`.
Credentials resolve from provider configuration, environment variables and
opencode credentials; there is no active keyring-backed vault. The direct
keyring dependency was removed; transitive dependencies may still use it.
The managed-agent design document records the historical implementation.
