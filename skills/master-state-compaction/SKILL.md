---
name: master-state-compaction
description: Use when the user asks to "compact memory", "create a master state snapshot", "summarize the project state", "restructure memory", or invokes /hub-compact-master-state, /role-senior-project-architect, or similar. Generates a 4-section snapshot (Architecture / Invariants / Active Working Set / Recent Pivots) for a multi-repo project from auto-memory deltas. Always uses the best configured model (tier_smart) and surfaces inferred assumptions for verification rather than presenting them as facts.
---

# Master State Compaction

Long-running multi-repo systems accumulate per-task auto-memory faster than humans can curate. The `compact_master_state` MCP tool folds those deltas into a single page that captures what the system IS today, what's load-bearing, what's under active work, and what just changed.

## When to use

- User explicitly asks for a snapshot, compaction, or master state.
- Closing a long-running task that shipped meaningful changes (PR merged, version bumped, architectural decision made).
- The project's `.memory/decisions.md` is missing a `## Master Project State` section, or the existing one is > 7 days old.

## How to invoke (correct sequence)

1. **Always dry-run first** — never write without showing the user the proposed snapshot.
   ```
   compact_master_state(project_root="/path/to/project", dry_run=True)
   ```
2. Show the user the rendered preview. Highlight: any new `## Assumptions to Verify` block, what got promoted to invariants, what got dropped from the prior snapshot.
3. **Wait for user approval.** Do not write on your own initiative.
4. On approval, invoke without `dry_run`:
   ```
   compact_master_state(project_root="/path/to/project")
   ```
5. The existing file is automatically backed up to `.memory/.backups/decisions-YYYY-MM-DD-HHMMSS.md`.
6. Assumptions are appended to `.memory/inbox.md` for follow-up verification.

## Quality bar (what makes a good snapshot)

- **Architecture section** is grouped by subsystem, not chronological. Six to eight short paragraphs, not a wall of text.
- **Invariants** are numbered NEVER/ALWAYS rules. Each one says WHY (one phrase) when the why is non-obvious. 8–12 is the sweet spot.
- **Active Working Set** is exactly 3 modules. Each one paragraph, citing recent PR numbers.
- **Recent Pivots** is exactly 3 entries. Format: `date — title`, then trigger / decision / why.
- **Assumptions** are verified-by claims, never silent inferences. If the LLM lists 0 assumptions on a non-trivial input, treat that as suspicious and re-run.

## Anti-patterns

- **Don't skip the dry-run.** Master state edits are high-leverage; surprising the user once burns a lot of trust.
- **Don't fold in noise.** SHIPPED one-shots > 14 days old belong in `_archive/`, not the snapshot.
- **Don't drop user-marked invariants.** If the existing snapshot says "NEVER X" and the recent entries don't contradict it, the rule survives. Use the `always_keep` channel (`<project>/.skillhub/master_state.yaml`) for things that must stay forever.
- **Don't smuggle facts into invariants.** "Asset platform Phase 3 shipped" is an active-modules note, not an invariant.

## Model selection

The tool defaults to `tier_smart` via `get_provider().complete(...)`. With prompt caching enabled, the snapshot prompt is cached as a single ~10KB prefix — repeat invocations on the same project are cheap. If you need to override (e.g., to use a local-only model on an air-gapped host), pass `model="ollama/qwen2.5:14b"` or similar through the lower-level helper in `embeddings.compact_master_state`.

## Installing the teach rule (one-time)

To make the model auto-invoke this skill on natural-language phrases, add a teaching rule via the MCP `teach` tool (run once after installing mcp-skill-hub):

```
teach(
    rule="when the user says 'create a master state snapshot', 'compact memory', "
         "'restructure memory', 'project state snapshot', or invokes "
         "/role-senior-project-architect or /hub-compact-master-state",
    suggest="invoke compact_master_state(project_root=cwd, dry_run=True) first; "
            "show preview to user; on approval call again with dry_run=False"
)
```

The rule lives in the skill-hub teachings store and surfaces via `search_skills` when the user's prompt matches.

## Related skills

- `memory-layer-analysis` — the categorization methodology (Active / Superseded / Archive) the LLM applies internally.
- `superpowers:writing-plans` — for the plan that PRECEDES the compaction, when restructuring is large.
