# MCP Skill Hub

A local MCP (Model Context Protocol) server that provides **semantic skill search** for Claude Code. Instead of loading all plugin skills into context at once, Skill Hub indexes them in a SQLite database with Ollama embeddings and serves only the relevant ones on demand.

## Problem

Claude Code loads every enabled plugin's skills into context at session start. With 20+ plugins and hundreds of skills, this wastes thousands of tokens on skill definitions you'll never use in that session. The system prompt bloats, response quality degrades, and you pay for tokens that add no value.

## Solution

Skill Hub acts as a smart skill router:

1. **Index** all plugin skills once (parses `SKILL.md` files, embeds descriptions via Ollama)
2. **Search** by semantic similarity — describe your task, get back only matching skill content
3. **Learn** from usage feedback — skills that proved helpful for similar queries rank higher over time
4. **Manage** plugins — enable/disable plugins in `settings.json` without leaving the conversation

```
User: "I need to build a REST API with FastAPI"
         │
         ▼
   search_skills(query)
         │
    ┌────┴────┐
    │ Ollama  │ ← embed query with nomic-embed-text
    │ Embed   │
    └────┬────┘
         │
    ┌────┴─────────┐
    │ SQLite        │ ← cosine similarity + feedback boost
    │ Vector Search │
    └────┬─────────┘
         │
    ┌────┴──────────┐
    │ (optional)    │ ← deepseek-r1:1.5b re-ranks top candidates
    │ LLM Re-rank  │
    └────┬──────────┘
         │
         ▼
   Top-K skill content returned inline
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Claude Code

## Installation

```bash
# Clone the repository
git clone https://github.com/ccancellieri/mcp-skill-hub.git
cd mcp-skill-hub

# Create virtual environment and install
python3 -m venv .venv
.venv/bin/pip install -e .

# Pull the embedding model (274 MB)
ollama pull nomic-embed-text

# Optional: pull the re-ranking model (1.1 GB, reasoning-capable)
ollama pull deepseek-r1:1.5b
```

## Configuration

Register the MCP server in `~/.mcp.json`:

```json
{
  "mcpServers": {
    "skill-hub": {
      "type": "stdio",
      "command": "/path/to/mcp-skill-hub/.venv/bin/skill-hub"
    }
  }
}
```

Restart Claude Code. The `skill-hub` tools will appear in your tool list.

## Usage

### 1. Build the Index

Run once after installing or updating plugins:

```
→ index_skills()
Indexed 47 skills.
```

This scans:
- `~/.claude/plugins/cache/` (official plugins)
- `~/.claude/plugins/marketplaces/` (marketplace plugins)
- `~/.claude/skills/` (user-local skills)

### 2. Search Skills

Describe your current task in natural language:

```
→ search_skills(query="build an MCP server in Python")
```

Returns the **full content** of the top matching skills, ready to follow inline. No need to separately load or invoke them.

With optional LLM re-ranking for higher precision:

```
→ search_skills(query="debug a failing pytest", use_rerank=True)
```

### 3. Record Feedback

After using a skill, tell Skill Hub whether it helped:

```
→ record_feedback(skill_id="superpowers:systematic-debugging", helpful=True)
```

This improves future rankings. The boost algorithm:
- Stores (query_vector, skill_id, helpful) tuples
- For future queries, finds past feedback on similar queries (cosine sim > 0.75)
- Applies a boost factor of 1.0x to 1.5x based on positive/negative feedback ratio

### 4. Manage Plugins

List indexed skills:

```
→ list_skills()
→ list_skills(plugin="superpowers")
```

Toggle plugins in `settings.json` (takes effect next session):

```
→ toggle_plugin(plugin_name="firebase", enabled=False)
```

## Tools Reference

| Tool | Description |
|------|-------------|
| `search_skills(query, top_k=3, use_rerank=False)` | Semantic search returning full skill content |
| `record_feedback(skill_id, helpful, query="")` | Record usage feedback for ranking improvement |
| `index_skills()` | Rebuild index from all plugin directories |
| `list_skills(plugin="")` | List indexed skills, optionally filtered by plugin |
| `toggle_plugin(plugin_name, enabled)` | Enable/disable a plugin in settings.json |

## Architecture

```
src/skill_hub/
├── server.py       # FastMCP server — tool definitions and MCP protocol
├── store.py        # SQLite store — skills, embeddings, feedback, cosine similarity
├── indexer.py      # Skill scanner — parses SKILL.md frontmatter, builds index
└── embeddings.py   # Ollama client — embed() and rerank() functions
```

### SQLite Schema

- **skills** — id, name, description, full content, file path, plugin name
- **embeddings** — skill_id, model name, vector (JSON float array)
- **feedback** — query, query_vector, skill_id, helpful flag, timestamp

Database location: `~/.claude/mcp-skill-hub/skill_hub.db`

### Embedding Models

| Model | Size | Purpose |
|-------|------|---------|
| `nomic-embed-text` | 274 MB | Primary embedding model (768-dim vectors) |
| `deepseek-r1:1.5b` | 1.1 GB | Optional re-ranker with chain-of-thought reasoning |

## Roadmap

- **Session profiles** — predefined plugin sets per work context (`skill-hub launch --profile geoid`)
- **Auto-profile** — predict needed plugins from recent feedback patterns before session start
- **OpenSearch backend** — swap SQLite for OpenSearch when scaling beyond local use
- **Hook integration** — `UserPromptSubmit` hook that auto-searches skills based on user message

## License

Copyright 2026 Carlo Cancellieri

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
