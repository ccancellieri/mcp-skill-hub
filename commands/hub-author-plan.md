---
description: Author a plan YAML using the best available Opus runner (Max-plan first, transparent fallback chain).
---

Call `author_plan(goal="$ARGUMENTS", repo_path=<cwd>)` via the skill-hub MCP tool.

The hub will try runners in priority order, transparently, using your Claude Code Max subscription:

1. `in_session` — we're inside Claude Code; the current Opus session authors the YAML directly (zero extra cost).
2. `cli` — spawn `claude -p` headlessly (uses OAuth subscription).
3. `sdk` — `claude_agent_sdk` Python package if installed.
4. `api` — **disabled by default** on the Max plan (config flag `plan_api_runner_enabled=False`).

Output: path to validated YAML at `~/.claude/plans/<slug>.yaml`, or — for the `in_session` runner — a directive for you (the current agent) to write the YAML yourself and then call `validate_plan`.

If no goal was provided, ask the user what they want to plan before calling the tool.
