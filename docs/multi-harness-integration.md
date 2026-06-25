# Skill Hub Multi-Harness Integration

This document describes the integration of mcp-skill-hub with multiple AI coding harnesses (Claude Code, opencode, and potentially others).

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      mcp-skill-hub                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ MCP Server   │  │ CLI          │  │ Core Library         │ │
│  │ (FastMCP)    │  │ (skill-hub-  │  │ (store, embeddings,  │ │
│  │              │  │  cli)        │  │  indexer, etc.)      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                 │                      │              │
│         └─────────────────┴──────────────────────┘              │
│                           │                                     │
│                    Shared Core Logic                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────┐                   ┌────────▼────────┐
│  Claude Code   │                   │    opencode     │
│                │                   │                 │
│ ✓ MCP Server   │                   │ ✓ MCP Server    │
│ ✓ Hooks        │                   │ ✓ Plugin        │
│   - PreToolUse │                   │   - Tools       │
│   - PostToolUse│                   │   - Events      │
│   - Stop       │                   │   - Session     │
│   - UserPrompt │                   │                 │
│     Submit ✓   │                   │ ✗ No prompt     │
│                │                   │   intercept     │
│ ✨ Zero-token  │                   │   (feature      │
│    intercept   │                   │    request      │
│                │                   │    pending)     │
└────────────────┘                   └─────────────────┘
```

## What's Working Now

### Claude Code (Full Integration)

All features work via:
- **MCP Server**: 80+ tools for task management, skill search, memory, etc.
- **Hooks**: Pre/post tool interception, session lifecycle, **zero-token prompt interception**

Hook types:
- `UserPromptSubmit` - Intercept messages before Claude (zero-token)
- `PreToolUse`/`PostToolUse` - Observe/modify tool calls
- `Stop`/`SessionEnd` - Session memory and cleanup
- `PreCompact`/`PostCompact` - Compaction lifecycle

### opencode (Partial Integration)

Working:
- **MCP Server**: All skill-hub tools available (prefixed with `skill-hub_`)
- **Plugin**: Session tracking + convenience tools

Plugin provides:
- `sh-status` - Quick status check
- `sh-tokens` - Token savings report
- `sh-skills` - List indexed skills
- `sh-search` - Quick context search
- `sh-tasks` - List tasks
- `sh-models` - List Ollama models
- `sh-teachings` - List teaching rules
- Session/task state tracking

Missing:
- **Zero-token interception**: Requires opencode to add `prompt.submit.before` hook

## Files Created

### 1. opencode Plugin
**Location**: `~/.config/opencode/plugins/skill-hub.ts`

Provides convenience tools and session tracking for opencode.

**Usage**: Restart opencode. Tools are automatically available.

### 2. Feature Request
**Location**: `docs/opencode-feature-request.md`

Proposes `prompt.submit.before` hook event for opencode to enable zero-token interception.

**Next Step**: Submit to opencode GitHub issues or discussions.

### 3. Config Symlink
**Location**: `~/.config/skill-hub/config.json` → `~/.claude/mcp-skill-hub/config.json`

Allows both harnesses to share the same configuration.

## Configuration

### opencode Config
Already configured in `~/.config/opencode/config.json`:
```json
{
  "mcp": {
    "skill-hub": {
      "type": "local",
      "command": ["/Users/ccancellieri/work/code/mcp-skill-hub/.venv/bin/skill-hub"],
      "enabled": true
    }
  }
}
```

### Claude Code Config
Already configured in `~/.claude/settings.json`:
```json
{
  "hooks": {
    "UserPromptSubmit": [ /* ... */ ],
    "PreToolUse": [ /* ... */ ],
    "PostToolUse": [ /* ... */ ],
    "Stop": [ /* ... */ ],
    "SessionEnd": [ /* ... */ ]
  }
}
```

## Feature Parity Matrix

| Feature | Claude Code | opencode | Notes |
|---------|-------------|----------|-------|
| MCP tools | ✅ | ✅ | All 80+ tools work |
| Skill search | ✅ | ✅ | Via MCP `search_skills` |
| Task management | ✅ | ✅ | Via MCP `save_task`/`close_task`/etc |
| Session memory | ✅ | ✅ | Via MCP `search_context` |
| Tool hooks | ✅ | ✅ | Pre/post tool execution |
| Session hooks | ✅ | ✅ | Start/stop/idle events |
| **Prompt interception** | ✅ | ❌ | **Zero-token feature** |
| Local command execution | ✅ | ❌ | Depends on prompt interception |
| Context injection | ✅ | ❌ | Depends on prompt interception |

## Next Steps

### For opencode Team

1. Review feature request at `docs/opencode-feature-request.md`
2. Consider adding `prompt.submit.before` hook event
3. This would enable full parity with Claude Code integration

### For Users

1. **Use MCP tools** in opencode: All `skill-hub_*` tools are available
2. **Use convenience tools**: `sh-status`, `sh-tokens`, `sh-search`, etc.
3. **Submit feature request**: Help get prompt interception into opencode

### For Development

1. **Keep core library shared**: Both MCP server and CLI use the same code
2. **Add more plugin features**: Could add compaction hooks, etc.
3. **Consider other harnesses**: Cursor, Continue.dev, etc. could use similar patterns

## Testing

### Verify opencode Plugin
```bash
# Start opencode
opencode

# Try convenience tools
sh-status
sh-tokens
sh-skills
```

### Verify MCP Tools
```
# In opencode prompt:
Use skill-hub_status to check the system
Use skill-hub_list_tasks to show open tasks
```

## Related Documentation

- [MCP Server Tools](../docs/reference/tools.md) - All available MCP tools
- [Hooks Architecture](../docs/features/hooks.md) - How hooks work in Claude Code
- [Architecture](../docs/reference/architecture.md) - Overall system design
