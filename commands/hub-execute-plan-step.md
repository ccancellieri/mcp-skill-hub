---
description: Execute one step of a plan YAML via the right model tier (Sonnet for integration/architecture, Haiku for boilerplate/tests/docs).
---

Call `execute_plan_step(plan_path=<plan>, step_id=<step>, dry_run=<bool>)` via the skill-hub MCP tool.

Parse `$ARGUMENTS` as `<plan_path> <step_id> [--apply]`:
- `--apply` → pass `dry_run=False` (writes files, runs acceptance command).
- omitted → `dry_run=True` (preview only).

Flow the tool performs:
1. Load & validate the plan.
2. Check `depends_on` via the sidecar `<plan>.state.json`.
3. Run file-path guards (escalate on CLAUDE.md, .env, migration filenames, …).
4. Map `step.kind` → tier (Sonnet for integration/architecture; Haiku for the rest).
5. Honor `step.model_hint` if present. Upgrade to Sonnet if a `force_tier_smart` guard matched.
6. Build a scoped context bundle (files + `protocols_ref` + `pattern_ref`).
7. Dispatch to the chosen model. Enforce the scope contract — any write outside `step.files` is rejected.
8. If acceptance fails on `tier_mid`, auto-retry once on `tier_smart`; otherwise mark failed.
9. Record a bandit reward (`task_class=step.kind`, `domain=plan_id`).

If arguments are missing, ask the user for plan path and step id.
