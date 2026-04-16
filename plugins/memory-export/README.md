# memory-export

Bundled mcp-skill-hub plugin. Exports and re-imports the **full hub memory**:
- Selected SQLite tables from `~/.claude/mcp-skill-hub/skill_hub.db`
- `~/.claude/mcp-skill-hub/config.json`
- The `enabledPlugins` slice of `~/.claude/settings.json`
- User-selectable project memories from `~/.claude/projects/<key>/memory/` (private/ subdirs always excluded)
- Optionally `~/.claude/local-skills/*.json`

Snapshot is a single `tar.gz`. Default import mode is **merge / skip on conflict** (`INSERT OR IGNORE`). Per-target the user can switch to **override** (`INSERT OR REPLACE`) or **LLM-merge** — the latter routed through `skill_hub.llm.get_provider()` with a tier dropdown (Claude / external / local Ollama).

## Routes

- `GET  /memory-export/`              — overview page (project picker + per-target mode controls)
- `GET  /memory-export/preview`       — JSON preview (counts, PII offenders, estimated bytes)
- `GET  /memory-export/snapshot.tar.gz` — streams a fresh snapshot
- `POST /memory-export/import`        — multipart upload + JSON restore plan; returns a per-table report

## Auto-registration

Discovered at startup when `bundled_plugins_enabled: true` in `~/.claude/mcp-skill-hub/config.json` (the default). No `extra_plugin_dirs` entry needed.

## Tests

```bash
cd /Users/ccancellieri/work/code/mcp-skill-hub
pytest -q plugins/memory-export/tests/
```

All tests are stdlib-only (no network); the LLM provider is monkeypatched in `test_intelligent_merge.py`.
