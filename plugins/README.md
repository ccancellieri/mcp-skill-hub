# Bundled Plugins

First-party plugins shipped inside the `mcp-skill-hub` repo. Auto-discovered at startup when `bundled_plugins_enabled` is true (default) in `~/.claude/mcp-skill-hub/config.json`.

Each subdirectory is a self-contained plugin with its own `plugin.json`, optional `web/`, `hooks/`, `storage/`, and `tests/`. Bundled plugins respect `~/.claude/settings.json["enabledPlugins"]` exactly like external plugins listed in `extra_plugin_dirs`.

## Current bundled plugins

| Name | Mount | Purpose |
|---|---|---|
| `memory-export` | `/memory-export` | Export and import the full hub memory (DB + project memories) as portable tar.gz, with optional LLM-driven conflict resolution |
