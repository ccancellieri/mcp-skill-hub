---
description: Execute every step of a plan YAML in dependency order, stopping on first failure.
---

Call `run_plan(plan_path=<plan>, dry_run=<bool>)` via the skill-hub MCP tool.

Parse `$ARGUMENTS` as `<plan_path> [--apply]`:
- `--apply` → `dry_run=False` (writes files, runs acceptance commands).
- omitted → `dry_run=True` (preview only).

Behavior:
- Topologically orders steps by `depends_on`.
- Skips steps already `done` in the sidecar state (idempotent resume).
- Halts on first `failed` / `escalated` step — per the retry-once-on-Sonnet-then-hand-back policy.
- Emits a run summary with per-step outcome, tier used, and guard matches.

If no plan path was provided, ask the user which plan to run (default dir: `~/.claude/plans/`).
