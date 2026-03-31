# MCP Skill Hub

A local MCP server that provides **semantic skill search**, **cross-session task memory**, and **zero-token command interception** for Claude Code. Instead of loading all plugin skills into context, Skill Hub indexes them with Ollama embeddings and serves only the relevant ones on demand. It **learns from your usage** to get smarter over time.

## Problem

Claude Code loads every enabled plugin's skills into context at session start. With 20+ plugins and hundreds of skills, this wastes thousands of tokens on definitions you'll never use. The system prompt bloats, response quality degrades, and you pay for tokens that add no value.

## Solution

Skill Hub is a smart skill/plugin router with three layers:

1. **Semantic search** — find skills and past work by meaning, not keywords
2. **Three-signal learning** — teachings, feedback, session history
3. **Zero-token hook interception** — task commands handled locally, Claude never sees them

```
User: "save to memory and close"
         │
    ┌────┴──────────────────┐
    │  UserPromptSubmit     │ ← hook fires BEFORE Claude
    │  Hook                 │
    └────┬──────────────────┘
         │
    ┌────┴──────────────────┐
    │  Local LLM classifies │ ← deepseek-r1 on your machine
    │  "Is this a task      │
    │   command?"           │
    └────┬──────────────────┘
         │
    YES: execute locally          NO: pass through
    save_task() / close_task()    → Claude processes normally
    return {"decision":"block"}
         │
    0 Claude tokens used
```

## Quick Start

```bash
git clone https://github.com/ccancellieri/mcp-skill-hub.git
cd mcp-skill-hub
./install.sh
```

The installer creates a venv, installs the package, pulls `nomic-embed-text` (274 MB), and registers the MCP server in `~/.mcp.json`.

**After install:** restart Claude Code, then:

```
index_skills()     # index all plugin skills
index_plugins()    # index plugin descriptions for suggestions
```

### Optional: Better Models

Pull a reasoning model for re-ranking, compaction, and hook interception:

```bash
# Minimum (1.1 GB) — works on any machine
ollama pull deepseek-r1:1.5b

# Better quality (4.7 GB) — recommended for 16GB+ RAM (e.g. MacBook Pro)
ollama pull deepseek-r1:7b

# Best quality (9 GB) — for 32GB+ RAM
ollama pull deepseek-r1:14b
```

Then configure:

```
configure(key="reason_model", value="deepseek-r1:7b")
```

### Manual Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
ollama pull nomic-embed-text
```

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "skill-hub": {
      "type": "stdio",
      "command": "/absolute/path/to/mcp-skill-hub/.venv/bin/skill-hub"
    }
  }
}
```

## Features

### 1. Semantic Skill Search

Describe your task — get matching skill content:

```
search_skills("build an MCP server in Python")
search_skills("debug a failing pytest", use_rerank=True)  # uses LLM re-ranking
```

Unified search across skills, tasks, and past work:

```
search_context("accessibility audit for a website")
```

### 2. Cross-Session Task Memory

Save open work for future sessions:

```
save_task(title="MCP skill hub dev", summary="Building semantic search...", tags="mcp,ollama")
```

Close with LLM-compacted summary (~200 tokens, processed locally):

```
close_task(task_id=1)
```

Tasks surface automatically in `search_context()` when future queries match.

```
list_tasks()                    # show open tasks
list_tasks(status="closed")     # show completed work
list_tasks(status="all")        # show everything
update_task(3, summary="Added hook interception")
reopen_task(5)                  # reopen a closed task
```

### 3. Zero-Token Hook Interception

A `UserPromptSubmit` hook intercepts task commands **before Claude sees them**. The local LLM classifies your message and executes locally — zero Claude API tokens consumed.

**Intercepted commands** (matched semantically, not just keywords):
- "save to memory" / "save task" / "park this" / "remember this"
- "close task" / "done with this" / "mark as done" / "save and close"
- "what was I working on?" / "show tasks" / "open tasks"
- "what did we discuss about X?" / "find my previous work on Y"

**Install the hook** in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/mcp-skill-hub/hooks/intercept-task-commands.sh",
        "timeout": 45,
        "statusMessage": "Checking for task commands..."
      }]
    }]
  }
}
```

**How it saves tokens:**

| Action | Without hook | With hook |
|--------|-------------|-----------|
| "save to memory" | ~500 tokens (Claude reads, decides, calls tool, reads response) | 0 tokens (local LLM handles, Claude never sees it) |
| "close task 3" | ~800 tokens (Claude + LLM compaction via tool) | 0 tokens (local deepseek-r1 compacts directly) |
| "list open tasks" | ~300 tokens | 0 tokens |
| Normal messages | Normal cost | Normal cost (hook allows through in <50ms) |

### 4. Teaching Rules

Add persistent rules that match semantically:

```
teach(rule="when I give a URL to check", suggest="chrome-devtools-mcp")
teach(rule="working on Terraform infrastructure", suggest="terraform")
teach(rule="debugging CSS or layout issues", suggest="chrome-devtools-mcp")
teach(rule="writing a Telegram bot", suggest="telegram")
```

Future queries like "inspect this page" match "when I give a URL" at ~0.8 similarity.

```
list_teachings()           # see all rules
forget_teaching(2)         # remove rule #2
```

### 5. Plugin Suggestions

Disabled plugins are still suggested when they match:

```
suggest_plugins("take a screenshot of this page and check accessibility")
# → [DISABLED] chrome-devtools-mcp: Browser DevTools...
#   → to enable: toggle_plugin("chrome-devtools-mcp", enabled=True)
```

### 6. Feedback Learning

Rate skills after use — rankings improve for similar future queries:

```
record_feedback(skill_id="superpowers:systematic-debugging", helpful=True)
```

### 7. Configuration

View and change settings without editing files:

```
configure()                                          # show all settings
configure(key="reason_model", value="deepseek-r1:7b")  # upgrade model
configure(key="search_top_k", value="5")             # more results
configure(key="hook_enabled", value="false")          # disable hook
```

Config file: `~/.claude/mcp-skill-hub/config.json`

**Model recommendations by hardware:**

| RAM | Reasoning Model | Embed Model | Total Disk |
|-----|----------------|-------------|------------|
| 8 GB | `deepseek-r1:1.5b` | `nomic-embed-text` | ~1.4 GB |
| 16 GB | `deepseek-r1:7b` | `nomic-embed-text` | ~5 GB |
| 32 GB | `deepseek-r1:14b` | `mxbai-embed-large` | ~10 GB |
| 64 GB+ | `deepseek-r1:32b` | `mxbai-embed-large` | ~20 GB |

## Tools Reference

| Tool | Description |
|------|-------------|
| **Search & Load** | |
| `search_skills(query, top_k, use_rerank)` | Semantic search, returns full skill content |
| `search_context(query, top_k)` | Unified search: skills + tasks + teachings + plugins |
| `suggest_plugins(query)` | Suggest plugins (including disabled) for current task |
| **Tasks** | |
| `save_task(title, summary, context, tags)` | Save open task for future sessions |
| `close_task(task_id, summary)` | Compact via local LLM and close |
| `update_task(task_id, summary, context, tags)` | Update an open task |
| `reopen_task(task_id)` | Reopen a closed task |
| `list_tasks(status)` | List open/closed/all tasks |
| **Learning** | |
| `teach(rule, suggest)` | Add "when X, suggest Y" rule |
| `record_feedback(skill_id, helpful, query)` | Rate a skill/plugin |
| `forget_teaching(teaching_id)` | Remove a teaching rule |
| `list_teachings()` | Show all teaching rules |
| `log_session(tool_name, plugin_id)` | Record tool usage (hooks) |
| **Management** | |
| `index_skills()` | Rebuild skill index |
| `index_plugins()` | Index plugin descriptions |
| `list_skills(plugin)` | List indexed skills |
| `toggle_plugin(plugin_name, enabled)` | Enable/disable plugins |
| `session_stats()` | Plugin usage statistics |
| `configure(key, value)` | View/update config |

## CLI Reference

Direct CLI for use in hooks and scripts (bypasses Claude entirely):

```bash
skill-hub-cli classify "save this to memory"     # classify intent
skill-hub-cli save_task "title" "summary"         # save directly
skill-hub-cli close_task 3                        # close + compact
skill-hub-cli list_tasks open                     # list tasks
skill-hub-cli search_context "my query"           # search
```

## How Learning Works

### Three Signals

1. **Teachings** (explicit): `teach("when I give a URL", "chrome-devtools-mcp")` — embedded as vectors, matched semantically at ~0.6 threshold

2. **Feedback** (semi-explicit): `record_feedback(skill, helpful=True)` — query vector stored, boosts similar future queries by up to 1.5x

3. **Session history** (passive): Stop hook logs which tools were actually called per session, builds usage patterns over time

Plugin suggestions combine all three: `total = embed_sim + teaching_boost + session_boost`

### Task Compaction

When you `close_task()`, the local LLM (deepseek-r1) distills the conversation into:

```json
{
  "title": "MCP Skill Hub development",
  "summary": "Built semantic skill search server with Ollama embeddings...",
  "decisions": ["SQLite over OpenSearch for local use", "nomic-embed-text for embeddings"],
  "tools_used": ["mcp-server-dev", "plugin-dev"],
  "open_questions": ["OpenSearch migration path"],
  "tags": "mcp,ollama,sqlite,skills"
}
```

~200 tokens stored vs ~5000 for the raw conversation. Future `search_context()` matches against the compact vector.

## Architecture

```
src/skill_hub/
├── server.py       # FastMCP tools + MCP protocol
├── store.py        # SQLite: skills, embeddings, feedback, teachings, tasks, session_log
├── indexer.py      # Scan plugin dirs, parse SKILL.md, build index
├── embeddings.py   # Ollama: embed, rerank, compact, rewrite_query
├── config.py       # User-configurable settings (models, thresholds)
└── cli.py          # Direct CLI for hooks (bypasses MCP)

hooks/
├── intercept-task-commands.sh   # UserPromptSubmit: zero-token task interception
└── session-logger.sh            # Stop: passive tool usage logging
```

### Database

Location: `~/.claude/mcp-skill-hub/skill_hub.db`

| Table | Purpose |
|-------|---------|
| `skills` | Skill metadata + full content |
| `embeddings` | Skill vectors |
| `feedback` | (query, skill, helpful) for boost |
| `teachings` | Explicit "when X suggest Y" rules |
| `plugins` | Plugin descriptions |
| `plugin_embeddings` | Plugin vectors |
| `tasks` | Open/closed task digests |
| `session_log` | Per-session tool usage |

### Config

Location: `~/.claude/mcp-skill-hub/config.json`

All settings have sensible defaults. Override only what you need.

## Roadmap

- [ ] Session profiles — predefined plugin sets per work context
- [ ] Auto-profile — predict needed plugins before session start
- [ ] Conversation digest enrichment — embed full conversation summaries
- [ ] OpenSearch backend — for scaling beyond local use
- [ ] Auto-search on every message — UserPromptSubmit hook that also searches skills

## License

Copyright 2026 Carlo Cancellieri

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
