# Context Bridge — Local Intelligence from Claude

Every Claude response is captured and stored **locally at zero token cost**. The system tails the session transcript in real-time, extracts tool calls, and builds a growing personal intelligence database that local LLMs can query.

## What gets captured

| What | Stored | Used for |
|------|--------|---------|
| Claude's tool calls (Bash, Grep, Read, Edit…) | `tool_examples` DB table | Informs local skill prompts |
| Working repo state (branch, dirty files, last commits) | `session-context.md` file | Git commit messages, planning |
| Rolling session summary | `session-context.md` file | All local LLM calls |
| Per-repo patterns (commit style, common commands) | `repo_context` DB table | Local persona enrichment |
| Behavioral patterns (PR conventions, search strategy) | `teachings` table | Future local LLM calls |

## Built-in variables available in every local skill

```
{session_context}   — what the user was working on (rolling summary + messages)
{tool_examples}     — recent Claude tool calls from this session
{repo_context}      — branch, dirty files, last 3 commits
{tool_patterns}     — aggregated patterns across sessions (top commands by repo)
```

### Example — `git-commit` with context

```json
{
  "name": "git-commit",
  "steps": [
    {"run": "git diff --staged", "as": "staged_diff"},
    {"llm": "Generate a commit message.\n\nSession context:\n{session_context}\n\nClaude's recent tool calls:\n{tool_examples}\n\nDiff:\n{staged_diff}",
     "as": "commit_msg", "first_line": true}
  ]
}
```

Without the context bridge, the local LLM sees **only the diff**. With it, the LLM knows:
- What you were working on this session
- How Claude named similar commits
- Project conventions captured over many sessions

## Data flow

```
Claude response arrives
    ↓
Stop hook fires → _capture_tool_calls()
    ↓ (incremental, <25ms)
tool_examples table ← stores Bash/Grep/Read/Edit calls
    ↓
session-context.md ← updated with summary + tool calls + repo state
    ↓
Next local skill run → variables injected automatically
```

## Session-end learning

After the conversation closes, two extra passes run at session-end:

| Pass | What it does |
|------|--------------|
| `_update_repo_context()` | LLM summarizes commit style, common commands, project type → `repo_context` table |
| `_extract_teaching_examples()` | Identifies 0–3 reusable behavioral patterns → `teachings` table |

Over time, this shapes local skills to match **your** project conventions — not generic defaults.

## Configure

```
configure(key="context_bridge_enabled", value="true")
configure(key="context_bridge_max_capture_per_hook", value="20")
```

## Related

- [features/learning.md](../features/learning.md) — how the five learning signals combine
- [advanced/skill-chaining.md](skill-chaining.md) — where `{tool_examples}` gets used
- [advanced/fine-tuning.md](fine-tuning.md) — turning the same captured signal into training data
