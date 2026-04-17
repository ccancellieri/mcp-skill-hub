# Local Execution Engine

Run commands, templates, skills, and full agent loops **entirely locally** — no Claude tokens, no network.

## Four escalating levels

Messages are matched through 4 levels. The first match wins.

| Level | What | Model | Example |
|-------|------|-------|---------|
| **L1** | Whitelisted commands | 3b | `"git status"` → `git status` |
| **L2** | Templated commands with params | 7b | `"show last 5 commits"` → `git log --oneline -5` |
| **L3** | Multi-step local skills | embeddings | `"project summary"` → git-summary skill (4 steps) |
| **L4** | Full local agent loop | 14b/32b | `"run tests and summarize"` → plan → approve → execute |

### First-time confirmation

New commands require `y/n` approval. Once approved, the command auto-executes for the rest of the session.

```
User: "show recent git activity"
  → L3 match: git-summary (sim=0.82)
  → [Skill Hub — local execution L3]
    Local skill matched: git-summary
    Steps:
      1. git log --oneline -10
      2. git diff --stat
      3. git status -s
      4. git branch --show-current
    Reply y to run, n to cancel.

User: y
  → ## Branch: main
    ### Recent commits
    1af2a15 Add universal LLM triage…
    …
```

---

## Level 4 — local agent

The L4 agent is a full tool-using loop driven by a local LLM. It has access to `run_skill`, `shell`, `read_file`, `search`, `list_files`, and `done`. **Always shows a plan first** and waits for confirmation before executing.

**Three ways to invoke:**

1. **Explicit:** `/local-agent <task>` — always asks, shows plan
2. **Triage auto-routing:** when LLM triage classifies as `local_agent`
3. **Exhaustion / offline fallback:** when Claude is rate-limited or unreachable

```
User: /local-agent show git status and run tests
  → [Local agent — qwen2.5-coder:14b]
    Plan for: show git status and run tests
    Steps:
      1. Run git status to check working tree
      2. Run git diff --stat for changed files
      3. Execute test suite
    Commands: git status, git diff --stat
    Reply y to execute, n to cancel.

User: y
  → [Local agent — qwen2.5-coder:32b]
    (agent executes plan using tools, returns results)
```

Planning uses `level_3` (14b, fast) while execution uses `level_4` (32b, thorough).

### Management commands

```
/hub-local-status              # show levels, models, commands, skills
/hub-local-skills              # list all local skill definitions
/hub-local-approve git_status  # pre-approve for this session
/local-agent                   # show agent status + available skills
/local-agent <task>            # plan + execute task via L4

/hub-local                     # toggle local mode on/off (bypass Claude)
/hub-local on                  # force on: all messages → L4 agent, session auto-saved
/hub-local off                 # resume Claude

/hub-skill-disable             # disable L1/L2/L3 local execution (persists to config)
/hub-skill-enable              # re-enable local execution
/hub-hook-disable              # disable entire hook pipeline
/hub-hook-enable               # re-enable hook pipeline
```

---

## Local skills (L3)

JSON files in `~/.claude/local-skills/`. **Three skill types:**

### Step-based skill (linear or branching)

```json
{
  "name": "git-summary",
  "description": "Show recent git activity summary",
  "triggers": ["git summary", "recent activity", "what changed"],
  "steps": [
    {"run": "git log --oneline -10", "as": "recent_commits"},
    {"run": "git diff --stat", "as": "changes"},
    {"run": "git status -s", "as": "status"}
  ],
  "output": "## Recent commits\n{recent_commits}\n\n## Changes\n{changes}\n\n## Status\n{status}"
}
```

### Agent-type skill (delegates to L4 loop)

```json
{
  "name": "fix-test",
  "type": "agent",
  "description": "Analyze test failure and suggest fix",
  "triggers": ["fix test", "failing test"],
  "prompt": "A test is failing. Run tests, read the failure, suggest a minimal fix.",
  "max_turns": 6
}
```

### Shadow skill (auto-evolves from Claude's behavior)

```json
{
  "name": "git-commit",
  "shadow": true,
  "description": "Smart git commit with LLM-generated message",
  "triggers": ["commit", "git commit"],
  "steps": [
    {"run": "git diff --staged", "as": "staged_diff"},
    {"llm": "Generate a commit message.\n\nSession context:\n{session_context}\n\nClaude's recent commits:\n{tool_examples}\n\nDiff:\n{staged_diff}",
     "as": "commit_msg", "first_line": true, "fallback": "chore: update"},
    {"run": "git commit -m '{commit_msg}'", "as": "result"}
  ]
}
```

See [advanced/skill-chaining.md](../advanced/skill-chaining.md) for full step-type reference and branching examples.

### Built-in variables (free for every skill)

| Variable | Source |
|----------|--------|
| `{session_context}` | Rolling session summary from `~/.claude/mcp-skill-hub/session-context.md` |
| `{tool_examples}` | Recent Claude tool calls (8 most recent) |
| `{repo_context}` | Current branch, dirty files, last 3 commits |
| `{tool_patterns}` | Aggregated command patterns across sessions |

---

## Dual skill index (Claude + local)

Skills are indexed with a **target**:

| Target | Source | Loaded into | Purpose |
|--------|--------|-------------|---------|
| `claude` | `SKILL.md` from plugins | Claude's context (RAG injection) | Plugin skills for Claude |
| `local` | JSON in `~/.claude/local-skills/` | Local LLM agent (L4) prompt | Skills for local execution |

Claude skills **never** pollute the local LLM prompt. Local skills **never** consume Claude tokens. Both are searchable via `search_skills()` and `list_skills()`.

```
/hub-list-skills                # all skills (grouped by target)
/hub-list-skills local          # only local skills
/hub-list-skills superpowers    # filter by plugin name

index_skills()   # re-index both sides: SKILL.md → target=claude, JSON → target=local
```

---

## Offline & exhaustion

### Offline auto-fallback

When `api.anthropic.com` is unreachable (rate limit, network outage, travel), Skill Hub auto-activates the L4 agent. **No manual intervention.**

```
User: "refactor this function"
  → Hook checks: api.anthropic.com reachable? (2.5s TCP, cached 30s)
  → Unreachable → local mode activated silently
  → L4 agent (qwen2.5-coder:32b) handles the message
  → 0 Claude API calls, full tool access
```

The check is a lightweight TCP connection — no HTTP request, no auth. It runs at most once every 30 s (configurable) so it adds no latency to normal operation.

When connectivity returns, turn off local mode manually:

```
/hub-local off    # resume Claude, session context preserved
```

Or save first:

```
/exhaustion-save  # compact + save
/hub-local off
```

**Configure:**

```
configure(key="offline_auto_fallback", value="false")   # disable auto-detect
configure(key="offline_check_interval", value="60")     # check every 60s
```

### Exhaustion fallback (manual)

When Claude is quota/rate-limited, save your session explicitly:

```
/exhaustion-save                        # auto-save from session context
/exhaustion-save "working on auth API"  # save with explicit description
```

```
=== Exhaustion Auto-Save ===

Task #12 saved: "Auth API middleware rewrite"

Summary: Implemented OAuth token validation middleware. Decided to use
jose library over PyJWT. Next: wire up refresh token rotation.

Next steps when resuming:
  - Implement refresh token rotation in auth_middleware.py
  - Add integration tests for token expiry edge cases

Files modified: auth_middleware.py, token_service.py

To resume later: search_context("Auth API middleware rewrite")
```

The local LLM generates a structured digest. If the LLM is also unavailable, a raw save captures the session text. A memory file is written to `MEMORY.md` so future sessions pick it up.

---

## Standalone REPL

When Claude is rate-limited or the VS Code extension doesn't display hook output, use the standalone REPL:

```bash
skill-hub-repl                     # interactive mode
skill-hub-repl "git status"        # single command
skill-hub-repl "/hub-status"       # any /hub-* command
skill-hub-repl "?"                 # list commands
```

```
╔══════════════════════════════════════════╗
║       Skill Hub — Local REPL             ║
║  All commands run locally via Ollama     ║
║  Type ? for help, Ctrl-C to exit         ║
╚══════════════════════════════════════════╝

  Embed model:  OK (nomic-embed-text)
  Reason model: OK (qwen2.5-coder:7b-instruct-q4_k_m)
  L4 model:     qwen2.5-coder:32b
  Local exec:   ON

skill-hub> git status
  [Local agent plans, asks for confirmation, executes]

skill-hub> /hub-list-tasks
  #8 [open] 4-level local LLM execution — fully wired

skill-hub> ?hub-configure
  View or set config values…
```

The REPL runs the same pipeline as the hook (L1→L2→L3→L4→agent fallback) but directly in your terminal. Messages that would normally pass through to Claude go to the L4 agent instead.

---

## Configure

```
configure(key="local_execution_enabled", value="true")
configure(key="local_models",
  value='{"level_1":"qwen2.5-coder:3b","level_2":"qwen2.5-coder:7b-instruct-q4_k_m","level_3":"qwen2.5-coder:14b","level_4":"qwen2.5-coder:32b"}')
```

Level 4 can also be a remote endpoint:

```
configure(key="local_models", value='{"level_4":"remote:http://your-server:11434"}')
configure(key="remote_llm",
  value='{"base_url":"http://your-server:11434","model":"qwen2.5-coder:32b","timeout":120}')
```

---

## Related

- [advanced/skill-chaining.md](../advanced/skill-chaining.md) — skill step types, branching, loops
- [advanced/context-bridge.md](../advanced/context-bridge.md) — where `{tool_examples}` etc. come from
- [reference/config.md](../reference/config.md) — every local-execution config key
