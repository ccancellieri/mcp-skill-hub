# Skill Chaining & Agent-as-Skill

Local skills support full conditional branching and jump logic — enabling multi-path workflows that react to actual command output.

## Step types

| Step | Description |
|------|-------------|
| `{"run": "cmd", "as": "var"}` | Shell command, store stdout in `{var}` |
| `{"run": "cmd", "as": "var", "if_empty": "fallback_cmd"}` | Run fallback if output is empty |
| `{"run": "cmd", "as": "var", "on_fail": "recovery", "retry": true}` | Recovery + retry on non-zero exit |
| `{"llm": "prompt…", "as": "var", "first_line": true}` | Local LLM generation |
| `{"stop_if_empty": "var", "message": "Nothing to do."}` | Guard: halt if `var` is empty |
| `{"label": "name"}` | Jump target |
| `{"goto": "label"}` | Unconditional jump |
| `{"stop": true, "message": "…"}` | Explicit halt |
| `{"if_contains": "var", "value": "str", "goto": "lbl", "else": "lbl2"}` | Branch on substring |
| `{"if_match": "var", "pattern": "regex", "goto": "lbl", "else": "lbl2"}` | Branch on regex |
| `{"if_empty": "var", "goto": "lbl", "else": "lbl2"}` | Branch when empty |
| `{"if_rc": "var", "eq": 0, "goto": "lbl", "else": "lbl2"}` | Branch on exit code (`eq` / `ne`) |

### Loop protection

The engine counts total iterations and halts at `10 × step_count` to prevent infinite loops.

---

## Example — smart git-push with stash/rebase

```json
{
  "name": "git-push",
  "steps": [
    {"run": "git status --porcelain", "as": "dirty"},
    {"if_empty": "dirty", "goto": "do_push"},

    {"run": "git stash push -m 'auto-stash before push'", "as": "stash_result"},

    {"label": "do_push"},
    {"run": "git push origin {branch}", "as": "push_result", "timeout": 60},
    {"if_rc": "push_result", "eq": 0, "goto": "check_stash"},

    {"run": "git pull --rebase origin {branch}", "as": "rebase_result"},
    {"if_rc": "rebase_result", "ne": 0, "goto": "rebase_failed"},
    {"run": "git push origin {branch}", "as": "push_result"},
    {"goto": "check_stash"},

    {"label": "rebase_failed"},
    {"stop": true, "message": "Rebase conflict — resolve manually.\n{rebase_result}"},

    {"label": "check_stash"},
    {"if_empty": "dirty", "goto": "done"},
    {"run": "git stash pop", "as": "unstash_result"},

    {"label": "done"}
  ]
}
```

### Log output shows every branching decision

```
SKILL [git-push] SHELL  $ git status --porcelain  →  rc=0 (42 chars)  (ok)
SKILL [git-push] IF     dirty empty = no → next
SKILL [git-push] SHELL  $ git stash push -m 'auto-stash before push'  →  rc=0  (ok)
SKILL [git-push] LABEL  do_push
SKILL [git-push] SHELL  $ git push origin main  →  rc=1 (80 chars)  (FAIL)
SKILL [git-push] IF     push_result rc==0 (actual=1) = no → next
SKILL [git-push] SHELL  $ git pull --rebase origin main  →  rc=0  (ok)
SKILL [git-push] IF     rebase_result rc!=0 (actual=0) = no → next
SKILL [git-push] SHELL  $ git push origin main  →  rc=0  (ok)
SKILL [git-push] GOTO   → check_stash
SKILL [git-push] IF     dirty empty = no → next
SKILL [git-push] SHELL  $ git stash pop  →  rc=0  (ok)
```

---

## Agent-as-Skill

Any local skill can delegate to the **L4 agent loop** instead of running linear steps. Add `"type": "agent"` and a `"prompt"` field:

```json
{
  "name": "fix-test",
  "type": "agent",
  "description": "Analyze test failure, read source, suggest fix",
  "triggers": ["fix test", "failing test", "debug test failure"],
  "prompt": "A test is failing. Analyze the test output, find the failure, and suggest a minimal fix.\n1. Run the test suite\n2. Read the failing test file\n3. Read the source being tested\n4. Explain what's wrong",
  "max_turns": 6
}
```

The agent automatically receives `{session_context}` and `{repo_context}` as context. It uses the `level_4` model and has access to `shell`, `read`, `search`, and `list_files` tools.

Agent skills appear alongside step-based skills in `/hub-local-skills`. They're matched by semantic similarity like any other skill — the user never needs to know if execution is a linear script or an agent loop.

---

## Built-in variables injected into every skill

| Variable | Source |
|----------|--------|
| `{session_context}` | Rolling session summary from `~/.claude/mcp-skill-hub/session-context.md` |
| `{tool_examples}` | Recent Claude tool calls (8 most recent) |
| `{repo_context}` | Current branch, dirty files, last 3 commits |
| `{tool_patterns}` | Aggregated command patterns across sessions |

See [advanced/context-bridge.md](context-bridge.md) for how these are populated.

---

## Related

- [features/local-execution.md](../features/local-execution.md) — L3 matching + L4 agent loop
- [features/learning.md](../features/learning.md#5-skill-evolution--shadow-learning) — shadow-learning skill evolution
