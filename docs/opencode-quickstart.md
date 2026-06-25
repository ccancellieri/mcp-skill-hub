# opencode Quick Start

## Installation

The plugin is already installed at `~/.config/opencode/plugins/skill-hub.ts`.

**Restart opencode** to load the plugin:
```bash
# If opencode is running, restart it
opencode
```

## Available Tools

### Convenience Tools (via Plugin)

These tools are lightweight wrappers around `skill-hub-cli`:

| Tool | Description | Example |
|------|-------------|---------|
| `sh-status` | Check Skill Hub status | `sh-status` |
| `sh-tokens` | Show token savings | `sh-tokens` |
| `sh-skills` | List indexed skills | `sh-skills` or `sh-skills "pytest"` |
| `sh-search` | Quick context search | `sh-search "database migration"` |
| `sh-tasks` | List tasks | `sh-tasks open` or `sh-tasks closed` |
| `sh-models` | List Ollama models | `sh-models` |
| `sh-teachings` | List teaching rules | `sh-teachings` |

### MCP Tools (via skill-hub MCP server)

Full-featured tools with prefix `skill-hub_`:

| Tool | Description |
|------|-------------|
| `skill-hub_save_task` | Save a task for future sessions |
| `skill-hub_close_task` | Close and compact a task |
| `skill-hub_list_tasks` | List tasks with filters |
| `skill-hub_search_context` | Search skills + tasks + memory |
| `skill-hub_search_skills` | Semantic skill search |
| `skill-hub_configure` | Update Skill Hub config |
| `skill-hub_status` | Full status check |
| ... | (80+ more tools) |

## Usage Examples

### Check System Status
```
Use sh-status to check if Ollama is running
```

### Save Your Work
```
Use skill-hub_save_task with:
  title: "Implementing auth flow"
  summary: "Working on OAuth integration with Keycloak"
```

### Find Previous Work
```
Use skill-hub_search_context to find work related to "authentication"
```

### List Open Tasks
```
Use sh-tasks to show open tasks
```

### Teach Skill Hub
```
Use skill-hub_teach with:
  rule: "when I mention OGC API"
  suggest: "ogc-api-standards skill"
```

## What's Different from Claude Code

| Feature | Claude Code | opencode |
|---------|-------------|----------|
| Prompt interception | ✅ Zero-token | ❌ Not yet available |
| Context injection | ✅ Automatic | ❌ Not yet available |
| MCP tools | ✅ Full access | ✅ Full access |
| Convenience tools | ✅ Via hooks | ✅ Via plugin |

## Getting Zero-Token Interception

The key feature missing in opencode is the ability to intercept prompts **before** they reach the LLM. This would enable:

- "save this task" → handled locally, 0 tokens
- "show my tasks" → handled locally, 0 tokens
- "what was I working on?" → handled locally, 0 tokens

**Help wanted**: Star/upvote the feature request at `docs/opencode-feature-request.md`

## Troubleshooting

### Plugin Not Loading
```bash
# Check plugin exists
ls ~/.config/opencode/plugins/skill-hub.ts

# Check opencode can find it
opencode --help  # Should show plugins are loaded
```

### MCP Server Not Starting
```bash
# Check MCP server binary
ls /Users/ccancellieri/work/code/mcp-skill-hub/.venv/bin/skill-hub

# Test manually
/Users/ccancellieri/work/code/mcp-skill-hub/.venv/bin/skill-hub --help
```

### Ollama Not Running
```bash
# Start Ollama
ollama serve

# Check models
ollama list
```

## Configuration

Shared config at `~/.config/skill-hub/config.json` (symlinked to `~/.claude/mcp-skill-hub/config.json`).

Key settings:
- `embed_model`: Embedding model (default: `nomic-embed-text`)
- `reason_model`: Reasoning model for classification
- `threshold`: Similarity threshold for skill matching
- `hook_context_top_k_skills`: Number of skills to auto-inject
