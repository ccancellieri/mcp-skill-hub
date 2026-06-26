# Cross-Project Knowledge Bridge

Federates wiki knowledge across related repos, showing cross-project knowledge graphs and syncing decisions/patterns between projects.

## Features

- **Knowledge Graph Visualization**: Interactive D3.js graph showing connections between projects, entities, and shared tags
- **Cross-Project Context**: Automatically surfaces relevant decisions and patterns from related projects during sessions
- **Wiki Sync**: Bidirectional sync between project `.memory/` directories and a shared wiki
- **Tag-Based Relations**: Discovers relationships between projects based on shared tags

## Installation

The plugin is bundled with mcp-skill-hub. Enable it via:

```bash
# Via settings.json
echo '{"enabledPlugins": {"cross-project-knowledge-bridge": true}}' > ~/.claude/settings.json

# Or via MCP tool
skill-hub_toggle_plugin plugin_name="cross-project-knowledge-bridge" enabled=true
```

## Configuration

Configure projects in `plugin.json`:

```json
{
  "config": {
    "projects": [
      {"name": "geoid", "path": "~/work/code/geoid", "tags": ["ogc", "stac", "dynastore"]},
      {"name": "dynastore", "path": "~/work/code/dynastore", "tags": ["dynastore"]},
      {"name": "fao-aip-catalog", "path": "~/work/code/fao-aip-catalog", "tags": ["stac", "catalog"]}
    ],
    "sync_interval_seconds": 300,
    "wiki_roots": {
      "geoid": "~/.claude/projects/geoid/wiki",
      "shared": "~/.claude/mcp-skill-hub/wiki"
    }
  }
}
```

### Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `projects` | array | List of project configurations |
| `projects[].name` | string | Unique project identifier |
| `projects[].path` | string | Path to project root (supports `~`) |
| `projects[].tags` | array | Tags for relationship detection |
| `sync_interval_seconds` | number | Auto-sync interval (default: 300) |
| `wiki_roots` | object | Per-project wiki paths |

## Extension Points

The plugin implements:

### A1: Web Mount

- Mounted at `/knowledge-bridge`
- Interactive knowledge graph visualization
- Project list with sync status

### A3: Hooks

- `on_session_start`: Loads relevant cross-project context based on current working directory

### A8: Namespaced Vectors

- `knowledge:federated` namespace for cross-project knowledge embeddings

### A9: Watch Paths

- Watches `.memory/*.md` files for changes

## Usage

### Web Interface

Navigate to `/knowledge-bridge` in the skill-hub webapp to:
- View the knowledge graph
- See project relationships
- Trigger manual sync
- Explore decisions and patterns

### Session Context

When working in a project directory, the plugin automatically:
1. Detects the current project
2. Finds related projects via shared tags
3. Injects relevant decisions and patterns into session context

### Manual Sync

```bash
# Via web UI
POST /knowledge-bridge/sync

# Via command line
python plugins/cross-project-knowledge-bridge/scripts/sync_wiki.py
```

### Extract Decisions

```bash
# Extract from all projects
python plugins/cross-project-knowledge-bridge/scripts/extract_decisions.py

# Extract from specific project
python plugins/cross-project-knowledge-bridge/scripts/extract_decisions.py geoid
```

## File Structure

```
plugins/cross-project-knowledge-bridge/
├── plugin.json           # Plugin manifest and configuration
├── web/
│   ├── app.py           # FastAPI sub-app
│   └── templates/
│       └── graph.html   # D3.js visualization template
├── hooks/
│   └── on_session_start.py  # Context injection hook
├── scripts/
│   ├── sync_wiki.py     # Bidirectional wiki sync
│   └── extract_decisions.py  # Parse decisions.md
├── storage/
│   └── schema.sql       # Cross-project mapping tables
└── README.md
```

## Memory Format

The plugin expects `.memory/decisions.md` files with this format:

```markdown
# Architecture Decision: Use Protocol Pattern

Date: 2024-01-15

## Context
When implementing cross-project knowledge sharing...

## Decision
We will use the Protocol pattern...

## Consequences
- Benefits: ...
- Trade-offs: ...
```

And `.memory/patterns.md`:

```markdown
## Protocol Resolution Pattern

Always access functionality via `get_protocol(ProtocolName)` instead of
concrete Manager classes. This ensures proper abstraction and caching.

Example:
```python
# Correct
protocol = get_protocol(DatabaseProtocol)
engine = protocol.engine

# Incorrect
manager = DatabaseManager()
engine = manager.engine
```
```

## Storage Schema

The plugin creates tables for:

- `cpkb_projects`: Registered projects
- `cpkb_entities`: Extracted entities mapped to wiki
- `cpkb_relations`: Cross-project relationships
- `cpkb_sync_log`: Sync history
- `cpkb_graph_edges`: Precomputed graph edges

Run schema migration:

```python
from skill_hub.store import SkillStore
import sqlite3

with open('plugins/cross-project-knowledge-bridge/storage/schema.sql') as f:
    schema = f.read()

store = SkillStore()
store._conn.executescript(schema)
store._conn.commit()
```
