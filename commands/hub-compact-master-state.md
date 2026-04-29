---
description: Generate or refresh the Master Project State snapshot for the current project
---

Call `compact_master_state(project_root="$ARGUMENTS")` via the skill-hub MCP tool. If `$ARGUMENTS` is empty, default to the current working directory.

Workflow:
1. First call with `dry_run=true` to preview the proposed snapshot.
2. Show the user the rendered preview.
3. Ask for confirmation.
4. On approval, call again with `dry_run=false` to upsert the section under `## Master Project State` at `<project>/.memory/decisions.md`.
5. The existing file is backed up automatically to `<project>/.memory/.backups/` before overwrite.

The tool is a no-op if no auto-memory entries have changed since the last snapshot, or if the project has no auto-memory directory mapped under `~/.claude/projects/`.
