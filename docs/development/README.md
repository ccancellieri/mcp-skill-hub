# Development Guidance

Use this guide for all changes to Skill Hub. It summarizes established
context-service boundaries; read [the context-service design](../context-service.md)
and the affected code and tests before changing behavior.

Use [the current backlog](../backlog.md) for conclusions and remaining acceptance
gates. Historical benchmark reports describe their frozen source revisions.

## Context Service

- Preserve the original user prompt. Retrieved material is supplemental
  evidence, never a replacement instruction, authorization, or task command.
- Require verified project scope. Do not infer a project from recent activity,
  a basename, or a foreign task/session identity. Report missing or ambiguous
  provenance as an omission.
- The interactive fast path is deterministic and local: no L1/L2 model,
  embedding API, recursive client CLI, task creation, approval action, or
  continuation message. On failure, timeout, disabled enrichment, or missing
  scope, pass the prompt through without project context.
- Missing project scope suppresses project memory and task evidence, but does
  not suppress global skill search.
- L1 is a local model, L2 is a remote API model, and L3 is the client coding
  agent. These are deployment layers, not a reasoning hierarchy. Keep L1/L2
  work optional and outside the foreground context path; never start L3 from a
  prompt hook.
- Native client approvals are the default. Deterministic policies may enforce
  explicit configured rules, but time-based relaxation, semantic guesses, and
  history-derived approval caches must not grant permission.

## Hooks And Integrations

- For `UserPromptSubmit`, new installation registers only the prompt router
  for enrichment. Other observer and lifecycle hooks retain their own
  registrations. Retired task interception, session enforcement, and automatic
  continuation entrypoints remain compatibility no-ops; `auto_approve` is a
  compatibility no-op under the default native approval policy, while an
  explicit deterministic configured policy remains enforceable. Preserve
  unrelated user hooks when repairing managed registrations.
- The prompt hook has a bounded deadline and fails closed as no added context.
  The current adapter boundary is two seconds; reserve client-hook headroom
  rather than increasing it casually.
- Adapters pass verified prompt, scope, and session identity to the common JSON
  CLI. Do not use the host process working directory as scope. Direct process
  execution only: no shell, bounded output, and no host-history mutation.

## Evidence And Tests

- Keep provenance, source coverage, omitted evidence, scope rejection, and
  original prompt handling testable. Tests should cover disabled and unavailable
  context as normal pass-through behavior.
- Treat latency and relevance as measured properties. Compare representative
  prompts against a fixed time and token budget before adding optional ranking
  or dependencies; do not turn a design expectation into a research claim.
- Prefer existing standard-library and workspace dependencies. Add one only
  when it materially improves a demonstrated requirement and document its
  operational cost.
