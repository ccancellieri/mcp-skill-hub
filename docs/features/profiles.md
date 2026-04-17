# Session Profiles

Swap entire plugin sets per work context — instantly toggle only the plugins you need.

## Usage

```
/profile                         # list available profiles
/profile backend                 # activate backend profile (7 plugins)
/profile minimal                 # just essentials (3 plugins)
/profile auto build MCP server   # LLM recommends best match
```

## Built-in profiles

| Profile | # plugins | Description |
|---------|-----------|-------------|
| `minimal` | 3 | `superpowers`, `commit-commands`, `code-review` |
| `backend` | 7 | + `code-simplifier`, `feature-dev`, `github`, `security-guidance` |
| `frontend` | 7 | + `frontend-design`, `chrome-devtools-mcp`, `feature-dev`, `github` |
| `mcp-dev` | 8 | + `mcp-server-dev`, `plugin-dev`, `skill-creator`, `feature-dev`, `github` |
| `data` | 6 | + `data`, `feature-dev`, `github` |
| `full` | all | Every plugin enabled |

## Save your own

```
/profile save my-setup "My custom plugin set"
/profile delete my-setup
```

Profiles modify `~/.claude/settings.json` — **restart Claude Code** for changes to take effect.

## Auto-recommendation

```
/profile auto build MCP server
```

The local LLM embeds your description and picks the profile whose plugin set has the highest centroid similarity. Great for ad-hoc context switching.

## Programmatic API

Profiles are also exposed as MCP tools:

| Tool | Purpose |
|------|---------|
| `list_profiles()` | Show all profiles (built-in + custom) |
| `create_profile(name, description, plugins)` | Save a custom profile |
| `switch_profile(name)` | Activate profile (updates settings.json) |
| `auto_curate_plugins(query)` | LLM recommends a profile for a task |
| `delete_profile(name)` | Delete a custom profile |

## Drift advisory

At `SessionStart`, if your current plugin set doesn't match any known profile closely, Skill Hub notes the drift and suggests either:

1. Saving your current state as a custom profile
2. Switching to the nearest built-in match
