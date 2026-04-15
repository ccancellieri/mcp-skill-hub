---
description: Validate a plan YAML file against the plan-executor schema.
---

Call `validate_plan(plan_path="$ARGUMENTS")` via the skill-hub MCP tool.

Checks performed:
- Top-level required fields (`plan_id`, `goal`, non-empty `steps`).
- Per-step schema: `id`, `kind` (enum), non-empty `files`, `acceptance`.
- `depends_on` references point to known step ids; no cycles.
- `model_hint` (if present) is one of `tier_cheap | tier_mid | tier_smart`.
- `protocols_ref` / `pattern_ref` paths exist on disk.

If no plan path was provided, ask the user which plan they want to validate (default dir: `~/.claude/plans/`).
