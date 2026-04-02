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

Pull a reasoning model for re-ranking, compaction, and hook interception. The hook pipeline uses two focused LLM calls (skill lifecycle + prompt optimization) — **instruct-tuned models give significantly more reliable structured output**:

```bash
# Minimum (1.0 GB) — any machine
ollama pull deepseek-r1:1.5b

# Recommended (4.4 GB) — 16GB+ RAM — instruct tuning, best quality/speed ratio
ollama pull qwen2.5-coder:7b-instruct-q4_k_m

# Best quality (8.4 GB) — 32GB+ RAM
ollama pull qwen2.5-coder:14b-instruct-q4_k_m
```

Then configure:

```
configure(key="reason_model", value="qwen2.5-coder:7b-instruct-q4_k_m")
```

For higher-quality embeddings (better skill retrieval, requires reindex):

```bash
ollama pull mxbai-embed-large
configure(key="embed_model", value="mxbai-embed-large")
index_skills()   # rebuild vectors with new model
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

When a task is closed, **auto-memory** runs automatically: the local LLM evaluates the compacted digest and writes a memory entry to `MEMORY.md` if it judges the content substantive enough (quality threshold 0.4). Low-quality or trivial digests are silently skipped. This keeps your memory index self-maintaining — closed tasks that matter get persisted without manual `/hub-save-memory` calls.

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

Feedback is stored as an **EMA score** on each skill (`feedback_score` column, range 0.5–1.5). Every positive signal nudges the score up; negative nudges it down. The score multiplies the cosine similarity at search time — no extra queries, no per-search cosine scans.

**Implicit feedback** runs automatically at session end — no manual calls needed. After each session, the hook correlates the skills that were injected into context against the tools Claude actually called:

```
Loaded skills: [hub-status, feature-dev:code-architect, superpowers:brainstorm]
Tools used:    [mcp__skill-hub__status, Edit, Read, Write]

→ hub-status domain "status" matches "mcp__skill-hub__status" → +positive EMA nudge
→ feature-dev and superpowers unrelated to actual tools  → -negative EMA nudge
```

Over time this self-tunes the skill index to your actual usage patterns with zero effort.

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

| RAM | Reasoning Model | Embed Model | Total Disk | Notes |
|-----|----------------|-------------|------------|-------|
| 8 GB | `deepseek-r1:1.5b` | `nomic-embed-text` | ~1.4 GB | Fast, basic |
| 16 GB | `qwen2.5-coder:7b-instruct-q4_k_m` | `nomic-embed-text` | ~5 GB | **Recommended** — instruct tuning gives reliable JSON output |
| 32 GB | `qwen2.5-coder:14b-instruct-q4_k_m` | `mxbai-embed-large` | ~9 GB | Best quality |
| 64 GB+ | `qwen2.5-coder:32b` | `mxbai-embed-large` | ~19 GB | Maximum |

> **Why `-instruct` variants?** The hook pipeline now uses two focused LLM calls: `eval_skill_lifecycle` (structured JSON at temp=0) and `optimize_prompt` (free-form rewrite at temp=0.2). Instruct-tuned models follow these single-task prompts reliably — base models frequently drop one task or wrap JSON in markdown. The `7b-instruct-q4_k_m` variant benchmarks at ~3.4s lifecycle + ~1.3s prompt optimization on Apple Silicon — same total latency as the old `3b` single call, with substantially better output quality.

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
├── server.py        # FastMCP tools + MCP protocol
├── store.py         # SQLite: skills (claude+local), embeddings, feedback, teachings, tasks
├── indexer.py       # Scan plugin dirs + local-skills/, parse SKILL.md + JSON, build index
├── embeddings.py    # Ollama: embed, rerank, compact, triage (incl. local_agent action)
├── config.py        # User-configurable settings (models, thresholds, local_models)
├── cli.py           # Hook pipeline: slash commands, L1-L4, triage, ? help, context injection
├── local_agent.py   # Level 4: plan_agent() + run_agent() with tool-calling loop
└── activity_log.py  # File + stderr logging (daily rotation, 50MB cap)

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

### 12. Offline Auto-Fallback

When `api.anthropic.com` is unreachable (rate limit, network outage, travel), Skill Hub detects it and **automatically activates the L4 local agent** — no manual intervention needed.

```
User: "refactor this function"
  → Hook checks: api.anthropic.com reachable? (2.5s TCP, cached 30s)
  → Unreachable → local mode activated silently
  → L4 agent (qwen2.5-coder:32b) handles the message
  → 0 Claude API calls, full tool access
```

The check is a lightweight TCP connection — no HTTP request, no auth. It runs at most once every 30 seconds (configurable) so it adds no latency to normal operation.

When connectivity returns, turn off local mode manually:

```
/hub-local off    # resume Claude, session context preserved
```

Or save the session first:

```
/exhaustion-save  # compact + save before resuming
/hub-local off
```

Configure:

```
/hub-configure offline_auto_fallback false    # disable auto-detection
/hub-configure offline_check_interval 60      # check every 60s instead
```

### 13. Exhaustion Fallback

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

The local LLM generates a structured digest with title, summary, decisions, next steps, and files. If the LLM is also unavailable, a raw save captures the session text. A memory file is also written to MEMORY.md so future sessions pick it up automatically.

### 14. Context Optimization

Analyze your memory files and get recommendations to reduce token usage:

```
/optimize-context
```

```
=== Context Optimization ===

Analyzing 48 memory files (~31,550 tokens total)...

  PRUNE  project_geoid_openapi_schema_cleanup.md — completed task, no longer needed
  COMPACT reference_dynastore_tools_knowledge.md — verbose, can save ~400 tokens
          → "Dynastore tools: enrichment pipeline, query executor, DDLBatch..."
  MERGE  feedback_use_constants_enums.md — overlaps with feedback_use_libraries_over_custom_models.md
  KEEP   project_geoid_core.md

Actions available:
  1 file to prune (~200 tokens saved per session)
  1 file to compact
  1 file to merge
```

### 15. Auto-Save Memory

Generate and save a memory entry from the current session using the local LLM:

```
/save-memory                              # from session context
/save-memory "decided to use SQLite"      # from explicit description
```

The local LLM generates a structured memory file with appropriate type (user/feedback/project/reference) and updates MEMORY.md automatically.

### 16. Universal LLM Triage

Every message passes through the local LLM **before** reaching Claude. The triage decides:

| Decision | What happens | Claude tokens |
|----------|-------------|---------------|
| `local_answer` | LLM answers directly (greetings, simple queries) | **0** |
| `local_action` | Routes to a `/hub-*` command (e.g., "what models?" → `/hub-list-models`) | **0** |
| `enrich_and_forward` | LLM adds a hint/analysis for Claude | Reduced |
| `pass_through` | Complex task → Claude handles normally | Normal |

```
User: "what models do I have installed?"
  → Triage: local_action → /hub-list-models
  → Shows full Ollama model list, 0 Claude tokens

User: "refactor the database to use async"
  → Triage: enrich_and_forward
  → Hint: "Consider async patterns, check existing driver code"
  → RAG: injects matching skills + memory + past tasks
  → Claude gets pre-processed, focused context

User: "hello"
  → Triage: local_answer → "Hello! How can I assist you today?"
  → 0 Claude tokens
```

Configure:

```
/hub-configure hook_llm_triage true           # enable/disable
/hub-configure hook_llm_triage_timeout 30     # max seconds
/hub-configure hook_llm_triage_min_confidence 0.7  # threshold
/hub-configure hook_llm_triage_skip_length 2000    # skip long messages
```

Stats via `/hub-token-stats` show triage breakdown: how many messages were answered locally vs enriched vs passed through.

### 17. Local Execution Engine (Levels 1-4)

Run commands, templates, and multi-step skills **entirely locally** — no Claude tokens, no network. Messages are matched through 4 escalating levels:

| Level | What | Model | Example |
|-------|------|-------|---------|
| **L1** | Whitelisted commands | 3b | "git status" → `git status` |
| **L2** | Templated commands with params | 7b | "show last 5 commits" → `git log --oneline -5` |
| **L3** | Multi-step local skills | embeddings | "project summary" → git-summary skill (4 steps) |
| **L4** | Full local agent loop | 14b/32b | "run tests and summarize" → plan → approve → agent executes |

**First-time confirmation:** New commands require `y/n` approval. Once approved, the command auto-executes for the rest of the session.

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
    1af2a15 Add universal LLM triage...
    ...
```

**Level 4 — Local Agent:**

The Level 4 agent is a full tool-using loop driven by a local LLM. It has access to: `run_skill`, `shell`, `read_file`, `search`, `list_files`, and `done`. The agent always shows a plan first and waits for confirmation before executing.

Three ways to invoke L4:

1. **Explicit:** `/local-agent <task>` — always asks, shows plan
2. **Triage auto-routing:** when the LLM triage classifies a message as `local_agent`, it routes automatically
3. **Exhaustion fallback / local mode:** when Claude is rate-limited, switch with `/hub-local on` — all messages go to the local agent until Claude is back

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

Planning uses the level_3 model (14b, fast) while execution uses level_4 (32b, thorough).

**Management commands:**

```
/hub-local-status              # show levels, models, commands, skills
/hub-local-skills              # list all local skill definitions
/hub-local-approve git_status  # pre-approve for this session
/local-agent                   # show agent status + available skills
/local-agent <task>            # plan + execute task via local agent (L4)

/hub-local                     # toggle local mode on/off (bypass Claude)
/hub-local on                  # force on: all messages → L4 agent, session auto-saved
/hub-local off                 # resume Claude
```

**Local skills** are JSON files in `~/.claude/local-skills/`:

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

See `examples/local-skills/` for more examples. It's recommended to optimize local skills with local models — keep triggers concise and steps focused on shell commands that produce structured output.

Configure:

```
/hub-configure local_execution_enabled true
/hub-configure local_models '{"level_1":"qwen2.5-coder:3b","level_2":"qwen2.5-coder:7b-instruct-q4_k_m","level_3":"qwen2.5-coder:14b","level_4":"qwen2.5-coder:32b"}'
```

The level_4 model can also be a remote endpoint:

```
/hub-configure local_models '{"level_4":"remote:http://your-server:11434"}'
/hub-configure remote_llm '{"base_url":"http://your-server:11434","model":"qwen2.5-coder:32b","timeout":120}'
```

### 18. Resource-Aware LLM Gating

Skill Hub monitors CPU load and memory pressure and skips expensive local LLM operations when the machine is busy. This prevents latency spikes during builds, compiles, or large model loads.

| Operation | IDLE | LOW | MODERATE | HIGH |
|-----------|------|-----|----------|------|
| embed | ✓ | ✓ | ✓ | ✓ |
| triage | ✓ | ✓ | ✓ | skip |
| rerank | ✓ | ✓ | ✓ | skip |
| precompact | ✓ | ✓ | skip | skip |
| digest | ✓ | ✓ | skip | skip |
| optimize\_memory | ✓ | skip | skip | skip |

Pressure is re-evaluated every 10 seconds (cached). Current pressure is always visible in `/hub-status`.

```
/hub-status           # shows: pressure=LOW  cpu=32%  mem=80%  avail=3.3GB
```

Force all operations to run regardless of pressure:

```bash
SKILL_HUB_FORCE_LLM=1 skill-hub-cli classify "..."
```

Disable gating entirely:

```
/hub-configure resource_gating_enabled false
```

### 19. Context Injection — Skills Loaded/Not-Loaded

Every auto-injected system message now includes a one-line summary showing which skills were loaded into context and which were found but skipped (not enough budget):

```
[Skill Hub — auto-injected context | skills loaded: superpowers:brainstorm, feature-dev:code-architect | found-not-loaded: superpowers:writing-plans, hookify:hookify | log: tail -f ~/.claude/mcp-skill-hub/logs/activity.log]
```

The same detail appears as HTML comments in `search_skills()` results:

```
<!-- LOADED (5):      superpowers:brainstorm, feature-dev:code-architect, ... -->
<!-- NOT LOADED (3):  superpowers:writing-plans, hookify:hookify, ... -->
<!-- log: tail -f ~/.claude/mcp-skill-hub/logs/activity.log -->
```

To load more skills: call `search_skills(query, top_k=8)` or raise the default:

```
/hub-configure hook_context_top_k_skills 8
```

### 20. Dual Skill Index (Claude + Local)



Skills are indexed with a **target** that determines where they're loaded:

| Target | Source | Loaded into | Purpose |
|--------|--------|-------------|---------|
| `claude` | SKILL.md from plugins | Claude's context (RAG injection) | Plugin skills for Claude |
| `local` | JSON from `~/.claude/local-skills/` | Local LLM agent (L4) prompt | Skills for local execution |

Claude skills never pollute the local LLM prompt. Local skills never consume Claude tokens. Both are searchable via `search_skills()` and visible in `list_skills()`.

```
/hub-list-skills             # all skills (grouped by target)
/hub-list-skills local       # only local skills
/hub-list-skills superpowers # filter by plugin name
```

Re-indexing picks up both:

```
index_skills()    # indexes SKILL.md as target=claude, JSON as target=local
```

### 21. Inline Help System

Type `?` to discover available commands, or `?command` for detailed usage:

```
?                     # list all commands with descriptions
?hub-list-skills      # detailed usage for /hub-list-skills
?hub-configure        # detailed usage for /hub-configure
?local-agent          # detailed usage for /local-agent
```

Works even when Claude is rate-limited — the `?` system runs entirely in the local hook.

### 22. Standalone REPL

When Claude is rate-limited or the VS Code extension doesn't display hook output, use the standalone REPL:

```bash
skill-hub-repl                     # interactive mode
skill-hub-repl "git status"        # single command mode
skill-hub-repl "/hub-status"       # run any /hub-* command
skill-hub-repl "?"                 # list available commands
```

```
╔══════════════════════════════════════════╗
║       Skill Hub — Local REPL             ║
║  All commands run locally via Ollama     ║
║  Type ? for help, Ctrl-C to exit        ║
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
  View or set config values...
```

The REPL runs the same pipeline as the hook (L1→L2→L3→L4→agent fallback) but directly in your terminal. Messages that would normally pass through to Claude instead go to the Level 4 local agent.

### 23. Training Data Export & Fine-Tuning

Every interaction accumulates signal in the database. Export it as JSONL training data to fine-tune your local models on your specific vocabulary, projects, and routing preferences:

```
/hub-export-training                    # export to default dir
/hub-export-training ~/my-training-data # export to custom dir
```

```
Training data exported to ~/.claude/mcp-skill-hub/training/
  feedback.jsonl:  12 pairs   ← (query, skill, helpful) preference pairs
  triage.jsonl:    43 pairs   ← (message, action) classification pairs
  compact.jsonl:   21 pairs   ← (summary, digest) compaction pairs

Total: 76 training pairs

To fine-tune with mlx-lm (Apple Silicon):
  pip install mlx-lm
  mlx_lm.lora --model mlx-community/deepseek-r1-distill-qwen-1.5b-4bit \
    --train --data ~/.claude/mcp-skill-hub/training --num-layers 8 --iters 200
```

**Three signal types:**

| File | Source | Trains the model to... |
|------|--------|----------------------|
| `feedback.jsonl` | `record_feedback()` + implicit feedback | Identify relevant skills for your queries |
| `triage.jsonl` | `triage_log` table | Route your specific phrasing (`"FAO catalog status"` → `local_action`) |
| `compact.jsonl` | Closed tasks | Produce digests in your style with your terminology |

**Recommended fine-tuning path (Apple Silicon, no GPU needed):**

```bash
pip install mlx-lm

# Fine-tune the triage model (1.5b, fastest)
mlx_lm.lora \
  --model mlx-community/deepseek-r1-distill-qwen-1.5b-4bit \
  --train --data ~/.claude/mcp-skill-hub/training \
  --num-layers 8 --iters 200 --batch-size 4

# After training, fuse and save as Ollama model
mlx_lm.fuse --model mlx-community/deepseek-r1-distill-qwen-1.5b-4bit \
  --adapter-path adapters --save-path ~/my-triage-model

# Create Ollama modelfile
echo 'FROM ~/my-triage-model' > Modelfile
ollama create skill-hub-triage -f Modelfile

# Activate
/hub-configure reason_model skill-hub-triage
```

At ~200+ examples the fine-tuned model will recognize your project names, FAO/geoid/dynastore vocabulary, and preferred routing — making triage significantly more accurate than a generic base model.

### Database

Location: `~/.claude/mcp-skill-hub/skill_hub.db`

| Table | Purpose |
|-------|---------|
| `skills` | Skill metadata + full content + target (claude/local) + `feedback_score` EMA |
| `embeddings` | Skill vectors + pre-stored L2 `norm` (avoids recompute per search) |
| `feedback` | Raw (query, skill, helpful) history; EMA applied to `skills.feedback_score` |
| `teachings` | Explicit "when X suggest Y" rules |
| `plugins` | Plugin descriptions |
| `plugin_embeddings` | Plugin vectors |
| `tasks` | Open/closed task digests |
| `session_log` | Per-session tool usage (source for implicit feedback) |
| `interceptions` | Hook-intercepted command log for token profiling |
| `context_injections` | RAG context injection stats |
| `conversation_state` | Periodic conversation digests for relevance tracking |
| `triage_log` | LLM triage decisions and token savings (source for training export) |
| `session_context` | Per-session rolling summary + loaded skills for dynamic context |

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
| `hook_context_top_k_skills` | `5` | Max skills loaded with full content per message |
| `hook_precompact_threshold` | `1500` | Messages longer than this get LLM pre-compaction |
| `profiles` | `{6 built-in}` | Session profile definitions |
| `digest_every_n_messages` | `5` | Produce conversation digest every N messages |
| `digest_stale_threshold` | `0.3` | Similarity below this = stale topic |
| `eviction_enabled` | `true` | Enable relevance decay tracking |
| `eviction_min_stale_count` | `3` | Suggest profile switch after N stale detections |
| `exhaustion_fallback` | `true` | Enable exhaustion auto-save |
| `offline_auto_fallback` | `true` | Auto-activate L4 agent when Anthropic unreachable |
| `offline_check_interval` | `30` | Seconds between reachability checks |
| `implicit_feedback_enabled` | `true` | Infer skill quality from session tool usage at session-end |
| `auto_memory_on_close_task` | `true` | Auto-write memory entry when a task is closed |
| `hook_context_summary_max_chars` | `800` | Max chars for rolling context summary (prevents unbounded growth) |
| `hook_context_prompt_opt_min_len` | `150` | Min message length to trigger prompt optimization |
| `hook_context_prompt_optimization` | `true` | Enable local LLM prompt rewriting |
| `hook_memory_dir` | *(CWD-derived)* | Override memory directory path |
| `resource_gating_enabled` | `true` | Skip LLM ops under CPU/RAM pressure |
| `resource_cache_ttl_seconds` | `10` | How often to re-check system resources |
| `hook_llm_triage` | `true` | Enable universal LLM triage on all messages |
| `hook_llm_triage_timeout` | `30` | Max seconds for triage LLM call |
| `hook_llm_triage_min_confidence` | `0.7` | Min confidence to act on local answer |
| `hook_llm_triage_skip_length` | `2000` | Messages longer than this skip triage |
| `local_execution_enabled` | `true` | Enable local command execution (L1-L4) |
| `local_models` | `{level_1: 3b, ...}` | Ollama model per execution level |
| `local_commands` | `{git_status: ...}` | Level 1 whitelisted shell commands |
| `local_templates` | `{git_log_n: ...}` | Level 2 templated commands with params |
| `local_skills_dir` | `~/.claude/local-skills` | Directory for Level 3 skill JSON files |
| `remote_llm` | `{}` | Remote LLM endpoint for L4: `{base_url, api_key, model, timeout}` |
| `log_dir` | `~/.claude/mcp-skill-hub/logs` | Activity log directory (daily rotation, 50MB cap) |

## Roadmap

- [x] Session profiles — predefined plugin sets per work context
- [x] Auto-profile — LLM recommends best profile for task description
- [x] RAG context injection — auto-enrich Claude's context with relevant skills/tasks/memory
- [x] Auto-eviction — relevance decay tracking + profile switch suggestions
- [x] Context compaction — periodic conversation digest via local LLM
- [x] Exhaustion fallback — local LLM auto-saves session when Claude is unavailable
- [x] Offline auto-fallback — TCP reachability check → auto-activates L4 agent when Claude unreachable
- [x] Universal LLM triage — local LLM pre-processes all messages, answers locally or enriches
- [x] Local execution engine — 4-level command/template/skill/agent execution with confirmation flow
- [x] Level 4 full agent — plan-first agent loop with tool calling (shell, skills, search, files)
- [x] Dual skill index — skills tagged claude/local, routed to the right LLM context
- [x] Inline help system — `?` lists commands, `?command` shows detailed usage
- [x] Activity logging — file + stderr, daily rotation, configurable log dir
- [x] Remote LLM support — Level 4 can route to a remote Ollama or OpenAI-compatible endpoint
- [x] Split LLM calls — lifecycle (temp=0) + prompt optimization (temp=0.2) as separate focused calls
- [x] DB vector cache — in-process cache eliminates per-search JSON deserialization (~5× faster)
- [x] Feedback EMA — pre-aggregated score on skills table, no per-search O(N) scan
- [x] Implicit feedback — session-end correlates loaded skills vs tool usage, auto-records EMA signal
- [x] Auto memory on close_task — smart_memory_write runs after every task compaction
- [x] Training data export — JSONL export of all signal types for mlx-lm fine-tuning on Apple Silicon
- [ ] OpenSearch backend — for scaling beyond local use

## License

Copyright 2026 Carlo Cancellieri

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
