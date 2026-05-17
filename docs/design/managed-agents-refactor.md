# Design — Managed-Agents architectural refactor

Status: **design phase**.
Tracking issue: see milestone `M2: Managed Agents`.

## Problem statement

Skill-hub today is a FastMCP stdio server with ~40 tools. Internal state — task DB, teachings, embeddings cache, profile state — is written ad-hoc by each tool. There is no shared event log, no uniform tool envelope, and credentials sit alongside config in plaintext. The system works, but four properties are hard to deliver in the current shape:

1. **Crash recovery.** A killed server can't replay the in-flight work because mutations are not logged before they happen.
2. **Cross-tool observability.** "What did the server do in the last 10 minutes?" requires reading per-tool logs in different shapes.
3. **Cred isolation.** Voyage / Anthropic / GitHub tokens live in `config.json`, so any tool's write of config can in principle touch credentials.
4. **Sandbox boundary.** Tools that execute user-supplied plans (`run_plan`, `execute_plan_step`, `author_plan`) have full process-level privilege.

Anthropic's [Managed Agents engineering post](https://www.anthropic.com/engineering/managed-agents) describes a pattern that addresses all four: separate the **session** (event log), the **harness** (orchestration loop), and the **sandbox** (execution environment) so each can fail or be replaced independently.

This document maps that pattern onto skill-hub.

## Managed Agents principles (paraphrased)

- **Decouple session, harness, sandbox.** Each runs independently; failures don't propagate.
- **Session = durable event log.** Append-only stream outside the harness, accessible via a `getEvents` API.
- **Harness = stateless.** Can be killed and resumed via `wake(sessionId)` — recovers state from the event log.
- **Sandbox = simple `execute(name, input) → string`.** Uniform whether the implementation is a container, MCP server, code execution env, or custom tool.
- **Credentials separate from sandboxes.** Tokens in a vault; sandbox initialization binds the credential by reference, not by value.
- **Context organization happens outside Claude's context window.** Harness fetches event slices and reshapes for cache friendliness.

The above is a paraphrase; refer to the source for the verbatim treatment.

## Current skill-hub mapped onto the principles

| Principle | Current state | Gap |
|---|---|---|
| Session = event log | Per-tool writes to scattered tables | No shared event stream |
| Harness stateless | In-memory caches (embeddings cache, vector cache, model bandit state) | Crash loses in-flight state |
| Uniform tool envelope | Each tool returns dict / str / yields | Inconsistent error reporting; hard to wrap with cross-cutting concerns |
| Cred vault | `config.json` plaintext | Voyage / Anthropic / GitHub tokens in config |
| Sandbox | `subprocess.run` with full PATH inheritance | No execution-time isolation for plan steps |
| Context organization | Implicit via `improve_prompt` rewriters | Working today; design adjacent — not changed by M2 |

## Proposed design — 5 workstreams

Each workstream is independently shippable. They land in this order because each prepares the ground for the next.

### W1 — Event log (`events` table)

Schema (additive — no existing table changes):

```sql
CREATE TABLE events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL,
  ts          REAL NOT NULL,        -- unix seconds, monotonic per session
  kind        TEXT NOT NULL,        -- "tool_invoke" | "tool_result" | "config_change" | "session_start" | "session_end"
  tool_name   TEXT,                 -- nullable; only set on tool_invoke/result
  payload     TEXT NOT NULL,        -- JSON
  source      TEXT NOT NULL DEFAULT 'local'  -- node_id for federation (M4-3)
);
CREATE INDEX events_by_session ON events (session_id, ts);
CREATE INDEX events_by_kind ON events (kind, ts);
```

Tool-invocation flow becomes:

1. Receive request.
2. `events.append(kind="tool_invoke", tool_name, payload=args)` — *before* any mutation.
3. Execute the tool body.
4. `events.append(kind="tool_result", tool_name, payload=result_or_error)` — *after*.

Existing tables (`tasks`, `teachings`, `skills`, …) stay as-is; they become **derived projections** of the event stream. A backfill script for old data is unnecessary — the event log is forward-only.

MCP tool: `get_events(session_id, since=, kind=, limit=) -> list[dict]`.

### W2 — Stateless recovery (`wake_session`)

`wake_session(session_id)`:

1. Read events for that session in order.
2. Reapply projections (rebuild any in-memory caches that derive from events).
3. Resume the most-recent in-flight tool invoke if its `tool_result` was never written.

For W2 to be useful, in-memory caches must be marked rebuildable. Concretely:

- Embedding cache: trivially rebuildable from `tools`/`tasks` rows.
- Model bandit state: replay `record_model_reward` events.
- Vector cache: rebuild from `skills` table.

W2 is mostly **discipline**, not new infrastructure — the events table from W1 is the storage, and the projections already exist.

### W3 — Uniform tool envelope

```python
@dataclass
class ToolResult:
    stdout: str                  # user-facing string (back-compat with existing tools that returned str)
    structured: dict | None      # optional structured payload
    error: str | None            # set when the tool failed; both stdout and error may coexist
    elapsed_ms: int              # observability
    events_emitted: list[int]    # references to events.id for trace
```

Every tool gets wrapped (decorator) so its return value is normalized. Callers (the MCP wire layer, internal callers, tests) see the same shape.

Backwards compatibility: existing MCP responses are derived from `ToolResult.stdout`; clients see no change.

### W4 — Credential vault

- Add `python-keyring` to deps (`keyring` is stdlib-friendly, no compiled extension on macOS).
- `config.json` stores **key names only**: `"voyage_api_key_ref": "skill_hub_voyage"`.
- `vault.get(ref) -> str` reads from OS keyring.
- One-shot migration: on first start after upgrade, if `voyage_api_key` is a literal in config, prompt the user (or via env flag) to move into keyring, then strip from config.

### W5 — Sandbox interface

Optional opt-in via `policy.yml`:

```yaml
sandbox:
  enabled: true
  modes:
    run_plan: subprocess           # cwd-restricted subprocess
    execute_plan_step: subprocess
    author_plan: native            # no sandbox; LLM-only path
```

`provision({resources})` returns a callable that runs the tool body inside the chosen sandbox. Tools that don't need a sandbox call `provision({})` which is a no-op pass-through.

Initial implementation: `subprocess` sandbox uses a per-invoke temp dir as cwd, `PATH` restricted to a configured allowlist, no network unless explicitly granted.

## Migration path

In-place and additive. No existing data is rewritten, no tool surface breaks.

1. **W1 first** (schema + event-emit decorator). Existing tools work; events accumulate.
2. **W2 second** (wake_session). No new schema. Discipline change: caches mark themselves rebuildable.
3. **W3 third** (envelope). Decorator wraps tools. MCP wire layer reads `.stdout`. No client change.
4. **W4 fourth** (vault). Migration runs once on upgrade.
5. **W5 fifth** (sandbox). Opt-in; default = current behavior.

Every step is reversible by toggling the relevant config knob.

## Open questions

1. **Event log retention.** Forever? Per-session prune after N days? Coalesce into snapshots? Initial answer: keep all events for 30 days, then snapshot+prune.
2. **Replay cost.** A long session's event log could be expensive to replay. Snapshot the projections periodically and replay only the tail?
3. **Cross-session linking.** Should a `swarm_launch` parent session link to the child sessions via event references? Probably yes — record the spawn as a `tool_invoke` with the child session_id in payload.
4. **Keyring on headless Linux.** macOS keyring is OS-managed. Linux needs `secret-tool` / `pass`. Should the vault fall back to an encrypted file when no keyring service is available?
5. **Sandbox & MCP stdio.** stdio MCP servers expect stdin/stdout. A subprocess sandbox makes that nested. The sandbox layer only wraps plan-execution tools, not the MCP transport — confirm scope explicitly in W5 design.

## Decision gates before filing sub-issues

- Open questions 1, 4, 5 must have agreed answers before filing W1 / W4 / W5 sub-issues.
- Open questions 2, 3 can wait until W1 ships with real volume data.

## References

- Anthropic, "Managed Agents" engineering post — `https://www.anthropic.com/engineering/managed-agents`
- Tracking issue: see milestone `M2: Managed Agents` in this repo.
- Related work in skill-hub: existing `unattended.md` describes the offline / exhaustion fallback; W2 generalizes that pattern.
