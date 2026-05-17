---
description: Fan out N GitHub issues into N worktree-bound tasks + a parallel Agent dispatch directive.
---

Arguments: `$ARGUMENTS`

Parse `$ARGUMENTS` against the forms below and call exactly ONE skill-hub MCP tool.
If `$ARGUMENTS` is empty, just print the **Help** block at the bottom of this file and stop.

## Argument forms

| Form | Action |
|---|---|
| (empty) or `help` | Print the Help block below verbatim. Do not call any tool. |
| `status <group_id>` | Call `fanout_status(group_id=<group_id>)`. |
| `close <group_id> [summary...]` | Call `fanout_close(group_id=<group_id>, summary="<rest>")`. |
| `dry <project> [filter...]` | Call `fanout_issues(project=<project>, source="gh", filter="<rest>", dry_run=True)`. |
| `text <project> -- <bullets...>` | Call `fanout_issues(project=<project>, source="text", filter="<bullets after -->")`. The bullets are newline-separated `- item` lines. |
| `<project> [filter...]` | Call `fanout_issues(project=<project>, source="gh", filter="<rest>")`. Default source is `gh`. |

After the tool returns:

1. Print the raw tool output (it contains the dispatch directive).
2. **If the output contains `Agent({` blocks, dispatch them yourself in ONE message** — paste every `Agent({...})` block as parallel tool calls in a single response. That is the whole point of fanout: the user pays once for prep, you execute the parallel dispatch in one shot.
3. After dispatch, remind the user they can roll up progress with `/hub-fanout status <group_id>` and close the group with `/hub-fanout close <group_id>`.

For `dry`, `status`, `help`: do NOT dispatch agents — just print the tool output and stop.

## Concrete example

User runs: `/hub-fanout geoid label:bug is:open`

You call:
```
fanout_issues(project="geoid", source="gh", filter="label:bug is:open")
```

Skill-hub creates 3 worktrees + 3 tasks (default `fanout.default_limit=3`), synthesizes a focused prompt per issue from `.github/ISSUE_TEMPLATE/*.yml` + `.github/labels.yml`, and returns a directive containing 3 `Agent({...})` blocks tagged with a `group_id` like `f7a3c2`.

You then paste all 3 `Agent` calls in one message so they run concurrently. When they all return, the user runs `/hub-fanout status f7a3c2` to see progress and `/hub-fanout close f7a3c2 shipped the bug batch` to close them in bulk.

## Help

```
/hub-fanout — parallel issue → worktree dispatch

USAGE
  /hub-fanout <project> [gh filter...]        Fan out (gh source, default 3 issues)
  /hub-fanout dry <project> [gh filter...]    Preview only — no worktrees, no tasks
  /hub-fanout text <project> -- <bullets>     Fan out from a literal bullet list
  /hub-fanout status <group_id>               Roll up progress for a fanout group
  /hub-fanout close  <group_id> [summary]     Close every open task in the group
  /hub-fanout help                            Show this help

EXAMPLES
  /hub-fanout geoid label:bug is:open
  /hub-fanout dry geoid label:good-first-issue
  /hub-fanout text geoid -- - fix login\n- add pagination\n- update README
  /hub-fanout status f7a3c2
  /hub-fanout close  f7a3c2 shipped the bug batch

DEFAULTS
  source = gh           (gh | text | a configured custom adapter)
  limit  = 3            (config: fanout.default_limit)
  branch = cc/<slug>    (config: fanout.naming)
  prompts use the local LLM unless it's unavailable, then a deterministic template

AFTER DISPATCH
  The active Claude must paste every Agent({...}) block from the directive in ONE
  message so the agents run in parallel. /hub-fanout status <group_id> rolls them up.
```
