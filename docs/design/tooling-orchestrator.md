# Tooling Orchestrator — design

**Status:** approved design (pre-implementation)
**Date:** 2026-06-22

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
  fast; probes are cheap status queries.
- **Refresh** of an already-initialized target is **always automatic** (cheap, safe),
  TTL-debounced via `sync_ttl`, dispatched async.
- **Init** of a not-yet-initialized target follows the **offer policy** (below);
  the default is to surface an offer rather than act.
- All probe/provision calls are wrapped and **non-fatal**: a failure is logged once
  and never breaks the prompt path nor enters a retry loop.

### 3. Route injection & directive format

The orchestrator's output rides the additional-context channel the router already
returns, so it is transparent. Two shapes:

- **Ready:** `[tooling] <root> is indexed (refreshed <age> ago) — prefer the indexed
  code-graph queries (search / callers / impact) over raw text search.`
- **Missing:** `[tooling] <root> is not indexed but the task is about to explore it;
  offer to initialize it (via ensure_tooling) before falling back to text search.`

The "missing" directive surfaces *to the assistant*, so the assistant asks the user —
keeping a human in the loop and honoring the established ask-before-initialize
convention.

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

## Provisioning autonomy policy

- **Refresh / sync of an existing index → always automatic.** This is the
  "keep it updated" behavior.
- **First-time init of a new project → offer, do not auto-run.** The orchestrator
  injects an offer; the assistant confirms with the user. This honors the established
  ask-before-initialize convention and avoids surprise indexing cost.
- **Escape hatch:** `orchestrator_auto_init_roots` — an allowlist of trusted roots
  that may be initialized automatically. Empty by default.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `orchestrator_enabled` | `True` | Master switch. |
| `orchestrator_auto_init` | `False` | Allow automatic first-time init globally. |
| `orchestrator_auto_init_roots` | `[]` | Allowlist of roots that may auto-init. |
| `orchestrator_sync_ttl_secs` | `300` | Min interval between auto-refreshes of a target. |
| `orchestrator_probe_cache_secs` | `60` | Probe-result cache TTL (keeps the hook fast). |

## Phasing

- **P1 (this slice):** engine + code-graph capability + route injection
  (offer-on-missing, auto-refresh) + `ensure_tooling` tool + decision logging.
- **P2:** generalize the registry (web-search reachability, embedding-model
  availability, headless-browser on URL/UI tasks) + the config flags above.
- **P3:** background freshness daemon (the deferred continuous-watch option) +
  dashboard panel + usage-feedback correlation.

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
