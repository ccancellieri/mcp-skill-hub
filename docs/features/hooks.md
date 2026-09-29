# Hooks: optional client integration

Skill Hub's MCP tools and context composer work without hooks. Hooks automate
specific client events; registering them does not prove that a client supports
or invokes them. The composer does not install additional hooks.

## What to keep

| Hook | Purpose | Needed for MCP? |
|---|---|---|
| `prompt-router.sh` | Adds scoped, deterministic evidence before a prompt | No; keep only for automatic enrichment |
| `post-tool-observer.sh` | Records tool outcomes and task activity | No; optional telemetry |
| `subagent-observer.sh` | Records agent lifecycle events | No; optional telemetry |
| `precompact.sh` | Saves routing and tool-chain state | No; optional state tracking |
| `postcompact.sh` | Runs memory maintenance and can return a report | No; optional maintenance |
| `session-end-real.sh` | Records session closure | No; optional lifecycle tracking |
| `stop-failure.sh` | Records API failures | No; optional diagnostics |
| `auto-approve.sh` | Applies explicitly configured deterministic approval rules | No; a no-op under the default native policy |

`postcompact.sh` can mutate the memory store: the current default for
`postcompact_optimize_apply` is true. Set it false for a dry run. Its report can
add context, so memory maintenance must not be presented as free prompt
compression. Native client permissions remain authoritative; prior successful
commands are not permission to approve future commands.

## Prompt enrichment and cost

The prompt router preserves the original prompt. Its worker retrieves skills
and verified project evidence locally, with no L1/L2 generation or embedding
API. Missing scope suppresses project evidence; global skill lookup remains
available. Disabled enrichment, failures and timeouts add no context.

The default worker deadline is two seconds. A host registration timeout is a
separate outer limit, not a model-generation allowance. Set `hook_enabled`
false to stop the prompt hook while keeping explicit MCP retrieval available.

No model call during retrieval does **not** mean zero token cost: any injected
text is subsequently input to the client model. Measure whole-task token usage,
including tool definitions, context rereads and auxiliary calls. Historical
per-command savings estimates do not establish current savings.

## Retired registrations

Remove `session-start-enforcer.sh`, `intercept-task-commands.sh`,
`session-end.sh` and `auto-proceed.sh` (or their Python equivalents). These are
compatibility no-ops. They no longer create tasks, intercept commands, save a
session after every turn, or continue an agent automatically. The installers
remove their managed registrations while preserving unrelated user hooks.

Custom model-tier reminders are separate from Skill Hub. A reminder naming
Claude models is not evidence of the active client's model or an API call.
Remove or adapt such reminders when using another harness.

See [context service](../context-service.md) for scope and failure semantics,
and [context composer](../context-composer.md) for explicit context selection.
