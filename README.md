# MCP Skill Hub

A local MCP server that provides **semantic skill search** for Claude Code. Instead of loading all plugin skills into context, Skill Hub indexes them with Ollama embeddings and serves only the relevant ones on demand. It **learns from your usage** to get smarter over time.

## Problem

Claude Code loads every enabled plugin's skills into context at session start. With 20+ plugins and hundreds of skills, this wastes thousands of tokens on definitions you'll never use. The system prompt bloats, response quality degrades, and you pay for tokens that add no value.

## Solution

Skill Hub is a smart skill/plugin router with three learning signals:

1. **Explicit teachings** — tell it rules like "when I give a URL, suggest chrome-devtools"
2. **Feedback learning** — rate skills as helpful/not after using them
3. **Passive session learning** — automatically tracks which tools you actually use

```
User: "check this website for accessibility"
         │
         ▼
   ┌─────────────────────┐
   │  Semantic Search     │ ← Ollama nomic-embed-text
   │  + Teaching Rules    │ ← "URL → chrome-devtools"
   │  + Feedback Boost    │ ← past helpful ratings
   │  + Session History   │ ← tools you actually used
   └──────────┬──────────┘
              │
   ┌──────────┴──────────┐
   │  (optional)          │ ← deepseek-r1:1.5b re-ranking
   │  LLM Re-rank        │
   └──────────┬──────────┘
              │
              ▼
   Skills loaded + plugin suggestions
```

## Quick Start

```bash
git clone https://github.com/ccancellieri/mcp-skill-hub.git
cd mcp-skill-hub
./install.sh
```

That's it. The installer:
- Creates a Python venv and installs the package
- Pulls `nomic-embed-text` via Ollama (274 MB)
- Registers the MCP server in `~/.mcp.json`

**After install:** restart Claude Code, then run:

```
index_skills()     # index all plugin skills
index_plugins()    # index plugin descriptions for suggestions
```

### Manual Install

If you prefer manual steps:

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

## Usage

### Search Skills

Describe your task — get back matching skill content:

```
search_skills("build an MCP server in Python")
search_skills("debug a failing pytest", use_rerank=True)  # slower but smarter
```

### Teach It

Add persistent rules that match semantically:

```
teach(rule="when I give a URL to check", suggest="chrome-devtools-mcp")
teach(rule="working on Terraform infrastructure", suggest="terraform")
teach(rule="debugging CSS or layout issues", suggest="chrome-devtools-mcp")
teach(rule="writing a Telegram bot", suggest="telegram")
```

Future queries like "inspect this page" match "check a URL" without exact keywords.

### Plugin Suggestions

Disabled plugins can still be suggested when they match your task:

```
suggest_plugins("take a screenshot of this page")
# → [DISABLED] chrome-devtools-mcp: Browser DevTools...
#   → to enable: toggle_plugin("chrome-devtools-mcp", enabled=True)
```

### Feedback

Rate skills after using them — rankings improve for similar future queries:

```
record_feedback(skill_id="superpowers:systematic-debugging", helpful=True)
```

### Manage Plugins

```
toggle_plugin("chrome-devtools-mcp", enabled=True)   # restart to apply
toggle_plugin("firebase", enabled=False)
list_skills(plugin="superpowers")
session_stats()  # see which plugins you actually use
```

## Tools Reference

| Tool | Description |
|------|-------------|
| **Search & Load** | |
| `search_skills(query, top_k=3, use_rerank=False)` | Semantic search, returns full skill content |
| `suggest_plugins(query="")` | Suggest plugins (including disabled) for current task |
| **Learning** | |
| `teach(rule, suggest)` | Add a persistent "when X, suggest Y" rule |
| `record_feedback(skill_id, helpful, query="")` | Rate a skill to improve future rankings |
| `forget_teaching(teaching_id)` | Remove a teaching rule |
| `list_teachings()` | Show all teaching rules |
| `log_session(tool_name, plugin_id)` | Record tool usage (called by hooks) |
| **Management** | |
| `index_skills()` | Rebuild skill index from plugin directories |
| `index_plugins()` | Index plugin descriptions for suggest_plugins |
| `list_skills(plugin="")` | List indexed skills |
| `toggle_plugin(plugin_name, enabled)` | Enable/disable plugins in settings.json |
| `session_stats()` | Show most-used plugins from session history |

## How Learning Works

### 1. Teachings (Explicit Rules)

```
teach(rule="when I give a URL", suggest="chrome-devtools-mcp")
```

The rule text is embedded as a vector. Future queries are compared by cosine similarity — "inspect this website" matches "when I give a URL" at ~0.8 similarity. Each teaching can also have a weight for fine-tuning.

### 2. Feedback Boost

When you call `record_feedback(skill_id, helpful=True)`, the query vector is stored alongside the skill. Future similar queries (cosine > 0.75) get a boost of up to 1.5x for that skill.

### 3. Session History (Passive)

A Stop hook logs which MCP tools were actually called during each session, associated with the session topic. Over time, this builds a usage map:

```
"debug website" → chrome-devtools (used 8/10 sessions)
"deploy infra"  → terraform (used 5/5 sessions)
```

Plugin suggestions combine all three signals:
`total_score = embedding_similarity + teaching_boost + session_history_boost`

## Architecture

```
src/skill_hub/
├── server.py       # FastMCP tools + MCP protocol
├── store.py        # SQLite: skills, embeddings, feedback, teachings, session_log
├── indexer.py      # Scan plugin dirs, parse SKILL.md, build index
└── embeddings.py   # Ollama embed + deepseek-r1 re-rank

hooks/
└── session-logger.sh   # Stop hook for passive learning
```

### Database

Location: `~/.claude/mcp-skill-hub/skill_hub.db`

| Table | Purpose |
|-------|---------|
| `skills` | Skill metadata + full content |
| `embeddings` | Float vectors per skill |
| `feedback` | (query, skill, helpful) for boost calculation |
| `teachings` | Explicit rules with embedded vectors |
| `plugins` | Plugin descriptions for suggest_plugins |
| `plugin_embeddings` | Float vectors per plugin |
| `session_log` | Per-session tool usage for passive learning |

### Models

| Model | Size | Purpose |
|-------|------|---------|
| `nomic-embed-text` | 274 MB | Embedding (required) |
| `deepseek-r1:1.5b` | 1.1 GB | Re-ranking (optional, reasoning-capable) |

## Roadmap

- [ ] Session profiles — predefined plugin sets per work context
- [ ] Auto-profile — predict needed plugins before session start
- [ ] Conversation digest — embed conversation summaries to enrich the index
- [ ] OpenSearch backend — for scaling beyond local use
- [ ] UserPromptSubmit hook — auto-search skills on every user message

## License

Copyright 2026 Carlo Cancellieri

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
