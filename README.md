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
| `index_skills()` | Rebuild skill index (includes extra_skill_dirs) |
| `index_plugins()` | Index plugin descriptions (includes extra_skill_dirs as sources) |
| `list_skills(plugin)` | List indexed skills |
| `toggle_plugin(plugin_name, enabled)` | Enable/disable plugins |
| `session_stats()` | Plugin usage statistics |
| `configure(key, value)` | View/update config |
| `status()` | Health check: MCP, Ollama, models, hook, DB stats |
| `token_stats()` | Token savings report from hook interceptions |

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

### 8. Extra Skill Directories

Index skills from any directory — including archived skill libraries — by adding entries to `extra_skill_dirs` in config:

```
configure(key="extra_skill_dirs", value='[{"path": "~/.claude/skills-archive", "source": "archive", "enabled": true}]')
```

Then re-index:

```
index_skills()    # scans extra_skill_dirs too
index_plugins()   # registers extra_skill_dirs as plugin sources for suggest_plugins()
```

Skills from extra directories get IDs like `archive:skill-name`. They appear in all searches and suggestions alongside plugin skills.

### 9. Status & Token Profiling

Check the health of all components in one call:

```
status()
```

```
=== Skill Hub Status ===
MCP server:      ✓ running
Ollama:          ✓ reachable at http://localhost:11434
Embed model:     ✓ nomic-embed-text
Reason model:    ✓ deepseek-r1:7b
Hook:            ✓ configured and enabled
Token profiling: ✓ on

Database:
  Skills indexed:    1247
  Tasks:             12 (3 open)
  Intercepted cmds:  89 (~52,300 tokens saved)
```

Track cumulative token savings from hook interceptions:

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

### 10. Session Profiles

Switch entire plugin sets per work context — instantly toggle only the plugins you need:

```
/profile              # list available profiles
/profile backend      # activate backend profile (7 plugins)
/profile minimal      # just essentials (3 plugins)
/profile auto build MCP server  # LLM recommends best match
```

**Built-in profiles:**

| Profile | Plugins | Description |
|---------|---------|-------------|
| `minimal` | 3 | superpowers, commit-commands, code-review |
| `backend` | 7 | + code-simplifier, feature-dev, github, security-guidance |
| `frontend` | 7 | + frontend-design, chrome-devtools-mcp, feature-dev, github |
| `mcp-dev` | 8 | + mcp-server-dev, plugin-dev, skill-creator, feature-dev, github |
| `data` | 6 | + data, feature-dev, github |
| `full` | all | Every plugin enabled |

Save your current state as a custom profile:

```
/profile save my-setup "My custom plugin set"
```

Delete a custom profile:

```
/profile delete my-setup
```

Profiles modify `~/.claude/settings.json` — restart Claude Code for changes to take effect.

### 11. Conversation Digest & Auto-Eviction

Every N messages (default 5), the local LLM produces a compact conversation digest:

```
/digest    # force a digest now
```

```
=== Conversation Digest ===

Messages in session: 15
Current focus: implementing session profiles for MCP skill hub

Recent decisions:
  - Use embedding similarity for profile auto-recommendation
  - Store profiles in config.json, not settings.json

Stale topics: CSS debugging, Terraform workspace setup
Suggested profile: mcp-dev
  Activate: /profile mcp-dev
```

The digest is auto-injected as `systemMessage` to keep Claude aware of the conversation's evolution. Stale topics are flagged so irrelevant context doesn't accumulate.

Configure:

```
/configure digest_every_n_messages 10   # less frequent digests
/configure eviction_enabled false       # disable decay tracking
```

### 12. Exhaustion Fallback

When Claude is exhausted (quota/rate limit), the local LLM saves your session:

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

The local LLM generates a structured digest with title, summary, decisions, next steps, and files. If the LLM is also unavailable, a raw save captures the session text.

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
| `interceptions` | Hook-intercepted command log for token profiling |
| `context_injections` | RAG context injection stats |
| `conversation_state` | Periodic conversation digests for relevance tracking |

### Config

Location: `~/.claude/mcp-skill-hub/config.json`

All settings have sensible defaults. Override only what you need.

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_base` | `http://localhost:11434` | Ollama server URL |
| `embed_model` | `nomic-embed-text` | Embedding model |
| `reason_model` | `deepseek-r1:1.5b` | Reasoning model (re-rank, compact, classify) |
| `hook_enabled` | `true` | Enable UserPromptSubmit hook |
| `hook_timeout_seconds` | `45` | Max hook execution time |
| `token_profiling` | `true` | Track estimated token savings |
| `search_top_k` | `3` | Default search results count |
| `search_similarity_threshold` | `0.3` | Minimum cosine similarity |
| `extra_skill_dirs` | `[{skills-archive}]` | Extra skill directories to index |
| `extra_plugin_dirs` | `[]` | Extra plugin directories to index |
| `hook_semantic_threshold` | `0.45` | Min embedding similarity for LLM classify |
| `hook_max_message_length` | `400` | Messages longer than this skip LLM classify |
| `hook_task_command_examples` | `[15 phrases]` | Canonical task phrases for semantic centroid |
| `hook_context_injection` | `true` | Auto-enrich context with RAG + memory |
| `hook_context_max_chars` | `2000` | Max chars injected as systemMessage |
| `hook_precompact_threshold` | `1500` | Messages longer than this get LLM pre-compaction |
| `profiles` | `{6 built-in}` | Session profile definitions |
| `digest_every_n_messages` | `5` | Produce conversation digest every N messages |
| `digest_stale_threshold` | `0.3` | Similarity below this = stale topic |
| `eviction_enabled` | `true` | Enable relevance decay tracking |
| `eviction_min_stale_count` | `3` | Suggest profile switch after N stale detections |
| `exhaustion_fallback` | `true` | Enable exhaustion auto-save |

## Roadmap

- [x] Session profiles — predefined plugin sets per work context
- [x] Auto-profile — LLM recommends best profile for task description
- [x] RAG context injection — auto-enrich Claude's context with relevant skills/tasks/memory
- [x] Auto-eviction — relevance decay tracking + profile switch suggestions
- [x] Context compaction — periodic conversation digest via local LLM
- [x] Exhaustion fallback — local LLM auto-saves session when Claude is unavailable
- [ ] OpenSearch backend — for scaling beyond local use

## License

Copyright 2026 Carlo Cancellieri

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
