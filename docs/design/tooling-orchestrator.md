# Tooling Orchestrator — design

**Status:** implemented (P1 shipped; mode control, freshness, and worktree
awareness landed on top of the original slice)
**Date:** 2026-06-22

> **Implementation note.** Sections below marked _“(shipped)”_ describe behaviour
> that has changed since the original approved design. The two material deltas:
> auto-refresh is now gated by an explicit **mode** rather than always-on, and
> readiness now detects **stale** indexes and **git-worktree/branch mismatches**.

## Problem

Skill Hub exposes a large surface of developer tools (its own ~80 MCP tools plus
adjacent servers such as the code-graph indexer and a headless-browser server). In
practice the assistant under-leverages them: it forgets to prefer the indexed
code-graph over a raw text search, it does not notice when an index is missing or
stale for the directory in question, and there is no transparent, reliable mechanism
that steers each turn toward the best-available tool for the active context.

The goal is a **holistic orchestration layer**: Skill Hub detects what a task needs,
ensures the relevant tool is *ready* for the target directory (initializing or
refreshing an index when warranted), and steers the turn toward it — transparently,
without relying on the assistant to remember.

Worked example: the session is rooted at a code workspace and the user asks to
"check the project under `<workspace>/geoid`". If the code-graph index for that
folder is missing or stale, the orchestrator notices, keeps an existing index fresh
automatically, and (for a missing index) surfaces an offer to initialize it — so the
assistant is always pointed at the better tool.

## Goals / non-goals

**Goals**
- Transparent, every-turn steering toward the best-ready tool for the active target.
- Idempotent readiness detection + provisioning (init / refresh) per tool, per target.
- Generalize across developer tools via a declarative registry, not per-tool glue.
- Observable: every routing decision is logged and visible.
- Never break or slow the prompt path.

**Non-goals (this phase)**
- A continuously running freshness daemon (deferred; see Phasing P3).
- Orchestrating external, auth-gated services (calendar, drive, blog) — different
  usage pattern, out of scope.
- Replacing the assistant's judgment: the orchestrator advises and provisions; it
  does not force tool calls.

## Architecture

A new `orchestrator` module inside Skill Hub, consumed by **two entry points that
share one engine**:

1. **The prompt-router path (hook-time).** The existing `UserPromptSubmit` hook
   already calls the CLI `route` subcommand every turn with the current working
   directory and the user message, and injects the returned guidance transparently
   (additional-context channel, sub-10ms tier-1 budget). The orchestrator plugs into
   `route` as the brain — this is the path that fires reliably regardless of whether
   the assistant "remembers".
2. **An explicit MCP tool, `ensure_tooling`.** The same engine exposed as a
   deliberate, debuggable call the assistant (or a human) can invoke directly. This
   is the actual executor for provisioning actions.

### Per-turn data flow (hook path)

```
UserPromptSubmit hook
  └─ CLI: route --cwd <cwd> <message>
       └─ orchestrator.evaluate(cwd, message, session)
            1. Target resolution   — extract candidate paths from the message
                                      (e.g. "<workspace>/geoid") + cwd; resolve to
                                      project roots.
            2. Capability match    — which registered capabilities the task signals
                                      (intent verbs + target type).
            3. Readiness probe      — cheap, cached: is the tool ready for that root?
            4. Plan                 — directives to inject + provisioning actions.
       └─ returns injected directive (additional-context)
  └─ provisioning actions dispatched ASYNC (non-blocking)
  └─ decision logged
```

Provisioning never runs inline on the hook path: refresh/sync actions are dispatched
fire-and-forget so the tier-1 latency budget is preserved.

## Components

### 1. Capability registry

A declarative list, one entry per tool capability. The engine is generic; adding a
tool means adding an entry, not changing the engine.

```
Capability(
  id,                # stable key, e.g. "code-graph"
  signals,           # intent predicate: verbs (explore/trace/map/where/impact/...)
                     #   + target-type test
  scope,             # eligibility predicate over a path (e.g. is-code-project)
  probe,             # (root) -> Readiness{present, fresh, stale_age, detail}
  provision_refresh, # (root) -> command/action for a cheap in-place refresh
  provision_init,    # (root) -> command/action for first-time setup (policy-gated)
  directive_ready,   # template: steer toward the tool
  directive_missing, # template: surface an offer to initialize
  probe_cache_ttl,   # seconds
  sync_ttl,          # min interval between auto-refreshes
)
```

First entry is the code-graph capability (the vertical slice). Later entries
(web-search reachability, embedding-model availability, headless-browser on
URL/UI tasks) drop in without touching the engine.

### 2. Readiness + provisioning engine

- **Probe** results are cached per `(capability, root)` with a TTL to keep the hook
  fast; probes are cheap, **subprocess-free** filesystem/SQLite reads (no
  `codegraph status` call on the hot path).
- **Refresh** of an already-initialized target is dispatched async, TTL-debounced
  via `sync_ttl`. _(shipped)_ It is now **gated by mode** rather than always-on:
  `auto`/`everywhere` auto-run it, `offer` only surfaces a "run sync" directive.
  A *fresh* index queues nothing.
- **Init** of a not-yet-initialized target follows the **mode policy** (below);
  the default (`offer`) surfaces an offer rather than acting.
- All probe/provision calls are wrapped and **non-fatal**: a failure is logged once
  and never breaks the prompt path nor enters a retry loop.

#### Readiness signals _(shipped)_

`probe_codegraph` reports more than present/absent:

- **Present-but-empty rejection.** A `.codegraph/` scaffold with zero nodes (a bare
  `init` with no indexing) reads as *not present*, so the orchestrator offers to
  index rather than steering at an empty graph. Only a *confirmed* zero downgrades
  presence; an unknown count (e.g. a future schema) preserves legacy behaviour.
- **Honest freshness.** Freshness is measured from the **`codegraph.db` build time**,
  not the `.codegraph/` directory mtime (which moves whenever codegraph touches its
  sidecar files, so a days-old graph could read as fresh). In addition, codegraph's
  `.dirty` marker — a millisecond epoch its file-watch hook rewrites on every source
  edit — is compared against the build time: a `.dirty` *newer* than the build means
  source changed since indexing, so the index is **stale** regardless of age.
- **Worktree/branch mismatch.** codegraph resolves a query by walking up to the
  nearest ancestor `.codegraph/`. A linked git worktree (its `.git` is a *file*) with
  no index of its own therefore silently answers from the *parent checkout's* index —
  a different branch's code. The probe detects this (`worktree_mismatch`,
  `ancestor_index`) and the directive warns + offers a **worktree-local** init instead
  of a bare "not indexed" nudge.

### 3. Route injection & directive format

The orchestrator's output rides the additional-context channel the router already
returns, so it is transparent. Four shapes _(shipped)_:

- **Ready (fresh):** `[tooling] <root> is indexed (built <age>) — prefer the indexed
  code-graph queries (search / callers / impact) over raw text search.`
- **Ready (stale):** `[tooling] <root> has a code-graph index but it is STALE (<why>).
  Run \`codegraph sync <root>\` … code-graph results may reflect pre-edit code.`
- **Missing:** `[tooling] <root> is not indexed but the task is about to explore it;
  offer to initialize it (via ensure_tooling) before falling back to text search.`
- **Worktree mismatch:** `[tooling] <root> is a git worktree with no code-graph index
  of its own. The only reachable index is <ancestor> (a different checkout/branch) …
  offer to initialize a worktree-local index (\`codegraph init -i <root>\`).`

The "missing" / "stale" / "mismatch" directives surface *to the assistant*, so it asks
the user — keeping a human in the loop and honoring the established
ask-before-initialize convention.

### 4. `ensure_tooling` MCP tool

Signature (conceptual): `ensure_tooling(path, *, init=False, refresh=True)`.
Idempotent: probes readiness, performs the requested safe actions, returns a
structured readiness report + the steering directive. This is path B — explicit,
debuggable — and the single executor that the hook path also drives for init.

### 5. Observability / feedback

Every decision (`target, capability, probe result, action, directive`) is logged to
the activity log and emitted as a new `orchestrator_decision` event, surfaced in a
dashboard panel and the status summary. This is also the spine for a later learning
loop: correlate "the orchestrator steered toward tool X" with the post-tool observer
("did the assistant then call tool X?") to measure whether steering actually lands.

## Provisioning autonomy policy _(shipped: mode-based)_

Autonomy is now a single **mode** with four settings, configurable from the web
control panel (Settings → Tooling Orchestrator) or `orchestrator_mode` in config.
Both *init* (first-time) and *sync* (refresh of a stale index) follow the same gate:

| Mode | First-time init | Auto-sync of a stale index | Steering directives |
|---|---|---|---|
| `off` | — | — | none (orchestrator silent) |
| `offer` | offer only | offer only | yes |
| `auto` | auto, **only** for a root under a configured parent folder | same | yes |
| `everywhere` | auto, anywhere eligible | auto, anywhere | yes |

- **`offer` (default).** Surface an offer / "run sync" directive; the assistant
  confirms with the user. Honors the ask-before-initialize convention; no surprise cost.
- **`auto`.** Acts automatically, but scoped to `orchestrator_auto_init_roots` — an
  allowlist of **parent folders** (prefix match, with a boundary guard so `/work/code`
  does not match `/work/codex`). Projects outside every parent fall back to *offer*.
- **`everywhere`.** Acts for any eligible project.
- A **fresh** index never triggers a sync in any mode; only a stale one does.

## Configuration

`orchestrator_mode` is the single source of truth. The legacy boolean keys still
derive a mode when `orchestrator_mode` is unset, so pre-existing configs are unchanged
(`enabled=False`→`off`; `auto_init=True`→`everywhere`; non-empty roots→`auto`; else
`offer`).

| Key | Default | Meaning |
|---|---|---|
| `orchestrator_mode` | _unset_ → derived | `off` / `offer` / `auto` / `everywhere`. |
| `orchestrator_auto_init_roots` | `[]` | Parent folders that may auto-init/sync in `auto` mode (prefix match). |
| `orchestrator_sync_ttl_secs` | `300` | Min interval between auto-refreshes; also the age threshold for the freshness fallback. |
| `orchestrator_probe_cache_secs` | `60` | Probe-result cache TTL (keeps the hook fast). |
| `orchestrator_enabled` | `True` | Legacy master switch (derives `off` when false). |
| `orchestrator_auto_init` | `False` | Legacy global auto-init (derives `everywhere` when true). |

## Phasing

- **P1 (shipped):** engine + code-graph capability + route injection + `ensure_tooling`
  tool + decision logging. Extended post-design with: present-but-empty rejection,
  honest freshness (db-build-time + `.dirty`), git-worktree/branch-mismatch detection,
  the four-way `orchestrator_mode` control, and a Settings → Tooling Orchestrator panel.
- **P2:** generalize the registry (web-search reachability, embedding-model
  availability, headless-browser on URL/UI tasks — see
  [browser-automation](../features/browser-automation.md)) + further capability entries.
- **P3:** background freshness daemon (the deferred continuous-watch option) +
  richer dashboard panel + usage-feedback correlation.

## Success criteria

- Pivoting to an eligible-but-unindexed project and asking to explore it results,
  within one turn, in the assistant being steered toward the indexed code-graph and
  offered initialization.
- Existing indexes stay fresh automatically, with no prompt-time latency regression.
- All steering is transparent and every decision is visible in logs / dashboard.

## Error handling

The orchestrator inherits the hook's non-fatal contract: every probe and provision is
guarded; failures are logged once and surfaced at most once, never retried in a loop,
and never break or block the prompt path.

## Testing

- **Unit:** target resolution; registry signal matching; readiness-probe parsing
  (mocked status output); autonomy policy (offer vs. auto vs. refresh); engine plan
  output.
- **Integration:** `route` → injected directive for the ready and missing cases;
  `ensure_tooling` idempotency; async provisioning never blocks and never raises on
  the hook path.

## Implementation decomposition

The build parallelizes cleanly against this spec: the engine + `ensure_tooling`
executor, the registry + route wiring, and the tests + dashboard panel are largely
independent work items, with a dedicated review pass over the non-fatal / async
guarantees before merge.
