# Hooks — Zero-Token Interception

Two `UserPromptSubmit` hooks fire **before Claude sees your message**. Together they classify your intent, execute task commands locally, and optionally inject RAG context — all without spending Claude tokens on routing.

## The two hooks

### 1. Session-start enforcer (first)

A lightweight flag-file check that injects a session-start checklist (skill invocation, memory loading) on the **first** message of a new session.

| Aspect | Detail |
|--------|--------|
| Cost | **0 LLM tokens** (file existence check only) |
| Latency | ~10 ms first message, ~1 ms thereafter |
| Depends on | bash, python3 (no Ollama, no skill-hub) |
| Order | Must run **before** `intercept-task-commands.sh` |

Uses a `{tempdir}/claude-session-started-{session_id}` flag. If absent, injects a `systemMessage` with the checklist. Flags >24 h are auto-pruned.

### 2. Task-command interceptor (second)

Catches task commands semantically and executes them locally. **Intercepted patterns:**

- "save to memory" / "save task" / "park this" / "remember this"
- "close task" / "done with this" / "mark as done" / "save and close"
- "what was I working on?" / "show tasks" / "open tasks"
- "what did we discuss about X?" / "find my previous work on Y"

---

## Install (one-time)

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [
        {
          "type": "command",
          "command": "/path/to/mcp-skill-hub/hooks/session-start-enforcer.sh",
          "timeout": 5,
          "statusMessage": "Checking session start protocol..."
        },
        {
          "type": "command",
          "command": "/path/to/mcp-skill-hub/hooks/intercept-task-commands.sh",
          "timeout": 45,
          "statusMessage": "Checking for task commands..."
        }
      ]
    }]
  }
}
```

**Cross-platform:** every hook has both `.sh` (macOS/Linux) and `.py` (Windows/any) versions. The installer auto-selects.

---

## How it saves tokens

| Action | Without hook | With hook |
|--------|-------------|-----------|
| "save to memory" | ~500 tokens (Claude reads, decides, calls tool, reads response) | **0 tokens** |
| "close task 3" | ~800 tokens (Claude + LLM compaction) | **0 tokens** |
| "list open tasks" | ~300 tokens | **0 tokens** |
| Normal messages | Normal cost | Normal cost + ~100 ms if below similarity threshold, ~3–8 s if context injection runs |

Track cumulative savings:

```
token_stats()
```

```
=== Token Savings Report ===
Total intercepted commands: 89
Total tokens saved (est.):  ~52,300
  (~$0.1569 at $3/M tokens, ~$0.7845 at $15/M)

By command type:
  save_task              34x  ~17,000 tokens saved  (avg 500/cmd)
  close_task             28x  ~22,400 tokens saved  (avg 800/cmd)
  list_tasks             18x  ~5,400 tokens saved   (avg 300/cmd)
  search_context          9x  ~3,600 tokens saved   (avg 400/cmd)
```

Disable profiling to skip the DB write per hook call:

```
configure(key="token_profiling", value="false")
```

---

## Context injection — skills loaded/not-loaded

Every auto-injected systemMessage now includes a summary of which skills were loaded and which were found but skipped (budget exhausted):

```
[Skill Hub — auto-injected context | skills loaded: superpowers:brainstorm, feature-dev:code-architect | found-not-loaded: superpowers:writing-plans, hookify:hookify | log: tail -f ~/.claude/mcp-skill-hub/logs/activity.log]
```

Same detail appears as HTML comments in `search_skills()` results:

```
<!-- LOADED (5):      superpowers:brainstorm, feature-dev:code-architect, ... -->
<!-- NOT LOADED (3):  superpowers:writing-plans, hookify:hookify, ... -->
<!-- log: tail -f ~/.claude/mcp-skill-hub/logs/activity.log -->
```

To load more skills:

```
configure(key="hook_context_top_k_skills", value="8")
```

Or call `search_skills(query, top_k=8)` directly.

---

## Universal LLM triage

Every message passes through the local LLM **before** reaching Claude. The triage decides:

| Decision | What happens | Claude tokens |
|----------|-------------|---------------|
| `local_answer` | LLM answers directly (greetings, simple queries) | **0** |
| `local_action` | Routes to a `/hub-*` command (e.g. "what models?" → `/hub-list-models`) | **0** |
| `enrich_and_forward` | LLM adds a hint / analysis for Claude | Reduced |
| `pass_through` | Complex task → Claude handles normally | Normal |

```
User: "what models do I have installed?"
  → Triage: local_action → /hub-list-models
  → 0 Claude tokens

User: "refactor the database to use async"
  → Triage: enrich_and_forward
  → Hint: "Consider async patterns, check existing driver code"
  → RAG: injects matching skills + memory + past tasks
  → Claude gets pre-processed, focused context

User: "hello"
  → Triage: local_answer → "Hello! How can I assist you today?"
  → 0 Claude tokens
```

**Configure:**

```
configure(key="hook_llm_triage", value="true")
configure(key="hook_llm_triage_timeout", value="30")
configure(key="hook_llm_triage_min_confidence", value="0.7")
configure(key="hook_llm_triage_skip_length", value="2000")
```

`/hub-token-stats` shows triage breakdown: how many messages were answered locally vs enriched vs passed through.

---

## How output reaches the user

Skill Hub uses **two** communication paths — understanding which one applies is critical:

| Path | How it works | User sees it? | Example |
|------|--------------|---------------|---------|
| **MCP tool return** | Tool returns a string → into Claude's context → Claude decides what to relay | Only if Claude echoes it | `status()`, `search_skills()`, `configure()` |
| **Hook `block` response** | Hook returns `{"decision": "block", "message": "..."}` → shown **directly** to user | Yes, always | `/hub-status`, `/hub-token-stats`, task interception |
| **Hook `systemMessage`** | Hook returns `{"decision": "allow", "systemMessage": "..."}` → injected into Claude's context as system text | **No** (invisible) | Dynamic context injection, skill loading |

**In practice:**

- **Slash commands** (`/hub-*`) are intercepted and output directly to the user — fast, 0 Claude tokens.
- **MCP tools** (called by Claude via tool use) return results to Claude's context. Claude then summarizes or relays them — costs tokens but gives Claude information to act on.
- **Context injection** enriches Claude's system prompt invisibly — the user only sees Claude's improved response.

If a feature "doesn't seem to do anything", it may be injecting context rather than showing output.

---

## Passive SearXNG RAG

When the hook's skill search returns no results above threshold, it falls back to SearXNG:

1. User sends a message → hook runs skill search
2. If no skills match above threshold → SearXNG web search triggers
3. Top results fetched from `/search?q=…&format=json`
4. Local LLM summarizes into 3-5 sentences
5. Summary injected into Claude's `systemMessage` as `[Web context — SearXNG]`
6. Claude sees pre-digested web context — raw results never sent

See [installation.md](../installation.md#searxng-web-search-optional) for setup.

---

## Related

- [features/local-execution.md](local-execution.md) — what happens when the hook routes to L1–L4
- [features/learning.md](learning.md) — how teaching rules + feedback shape classification
- [reference/logs.md](../reference/logs.md) — debugging hook misbehavior
