# Plugin Extension Points

Canonical reference for the protocols a plugin directory (registered via
`extra_plugin_dirs` in `~/.claude/mcp-skill-hub/config.json`) may implement.
All keys live in the plugin's top-level `plugin.json`.

## A1 — Web mount

```json
{
  "web_mount": {"mount": "/myplugin", "title": "My Plugin", "icon": "cube", "nav": true}
}
```

Plugin ships `web/app.py` exposing `get_app() -> FastAPI`. Skill-hub mounts
the sub-app at `cfg.mount`, appending to `app.state.plugin_nav` if
`nav: true`.

Manual override: top-level config key `extra_web_mounts`:

```json
"extra_web_mounts": [
  {"plugin_path": "/abs/path", "mount": "/foo", "title": "Foo", "nav": true}
]
```

## A2 — Dashboard sections

Plugin ships `web/dashboard_sections.py`:

```python
def get_sections() -> list[dict]:
    return [
        {"id": "foo-kpi", "title": "Foo", "html": "<p>42</p>", "order": 50},
        # or: {"id": "...", "title": "...", "template": "rel/path.html", "context": {...}, "order": 60}
    ]
```

Rendered after core KPI cards in `/` dashboard.

## A7 — Plugin DB (scoped sqlite)

```json
{
  "storage": {"schema": "storage/schema.sql", "namespace": "mynamespace"}
}
```

`storage/schema.sql` must contain only
`CREATE TABLE IF NOT EXISTS plugin_{namespace}_*` (and matching `CREATE INDEX`)
statements. Applied on first `store.plugin_db("mynamespace")` call per-process;
reapplied when the file's sha256 changes.

Usage from inside the plugin:

```python
from skill_hub.store import SkillStore
db = SkillStore().plugin_db("mynamespace")
db.execute("INSERT INTO plugin_mynamespace_events (kind, at) VALUES (?, ?)", (...))
rows = db.fetch_all("SELECT * FROM plugin_mynamespace_events")
# Read-only correlation with core tables is allowed:
rows = db.fetch_all("SELECT id, title FROM tasks WHERE status='open'")
# But writes to non-plugin_* tables raise PermissionError.
```

## A8 — Namespaced vectors

```python
store.upsert_vector("mynamespace", doc_id="draft-2024-01-01",
                    text="body…", metadata={"kind": "draft"})
hits = store.search_vectors("release announcement",
                            namespaces=["mynamespace"], top_k=5)
```

Underlying table: `vectors(namespace, doc_id UNIQUE per ns, model, vector, norm, metadata, indexed_at)`.

## A9 — Watcher paths

```json
{"watch_paths": ["state/drafts/**/*.md", "state/portfolio/**/*.md"]}
```

Watcher dispatches changes to `{plugin_path}/indexer.py`:
- `index_doc(store, path)` — per-file re-index (preferred).
- `index_docs(store)` — full corpus fallback.

Debounce 2s, cooldown 120s (shared with core reindex).

## A11 — Shared template macros + CSS tokens

Plugin sub-apps whose `web/app.py:get_app()` exposes
`app.state.templates = Jinja2Templates(...)` automatically gain access to the
shared macros dir via a `ChoiceLoader` fallback. Usage:

```jinja
{% from "_macros/kpi.html"   import kpi_card %}
{% from "_macros/table.html" import data_table %}
{% from "_macros/form.html"  import form_field %}
{% from "_macros/toast.html" import toast %}
```

CSS design tokens (`--color-*`, `--space-*`, `--radius-*`, `--font-size-*`,
`--font-family`) are available globally via `/static/app.css`.

---

## A3 — Plugin hook registry

Plugins register event handlers in `plugin.json`:

```json
"hooks": {
  "on_session_start":   "hooks/on_session_start.py",
  "on_session_end":     "hooks/on_session_end.py",
  "on_tool_call":       "hooks/on_tool_call.py",
  "on_skill_activated": {"script": "hooks/on_skill.py", "async": true}
}
```

Each handler script receives a JSON payload on stdin and may emit JSON on
stdout. Sync handlers run with a 10-second timeout; async handlers fire on a
background thread and their output is discarded. Non-zero exits are logged
but never propagate. Kill-switch: `PLUGIN_HOOKS_DISABLED=1`. Dispatch is
centralized in `skill_hub.plugin_hooks.dispatch(event, payload)`.

Current emit sites:
- `log_session` tool → `on_tool_call` (synchronous, vetoes supported)
- `close_session(summary)` → `on_session_end` (synchronous)

## A4 — Memory adapter

Plugins declare memory roots to include in the shared search corpus. Two
declaration styles are supported:

**M2 — per-index routing (preferred):**

```json
"memory": {
  "indexes": [
    {"name": "career:profile",   "level": "L3", "reads": ["refs/**/*.md"]},
    {"name": "career:narrative", "level": "L2", "reads": ["state/drafts/**/*.md",
                                                           "state/portfolio/**/*.md"]},
    {"name": "career:private-signal", "level": "L2",
     "reads": ["~/.claude/.../memory/private/**/*.md"]}
  ],
  "writes": ["~/.claude/mcp-skill-hub/my_plugin/**"]
}
```

Each index entry embeds its glob-matched files into a named namespace at the
declared level. Chunk settings come from the matching `vector_index_config` row
(auto-created via `vector_indexes`).

**Legacy — flat reads (backward-compat):**

```json
"memory": {
  "reads":  ["/abs/path/**/*.md", "~/relative/**"],
  "writes": ["~/.claude/mcp-skill-hub/my_plugin/**"]
}
```

Legacy reads land in `memory:<plugin_name>` at level L2.

In both styles, `search_context(query, include_plugin_memory=True)` merges
results into the "Plugin Memory" section. `writes` is advisory (audit /
dashboard). Size cap: 200 kB per file. Indexing runs inside `index_plugins()`
via `skill_hub.memory_index.index_plugin_memory(store)`.

## A5 — Scheduled task templates

Plugins declare cron-style templates:

```json
"scheduled_tasks": [
  {
    "name": "daily-report",
    "cron": "30 8 * * 1-5",
    "prompt_template": "tasks/daily_report.md",
    "enabled_default": false
  }
]
```

New MCP tools:

- `list_plugin_tasks(plugin="")` — inventory of declared + currently-enabled tasks
- `enable_plugin_task(plugin, name)` — materializes the prompt into
  `~/.claude/mcp-skill-hub/scheduled_tasks/{plugin}__{name}.json`
- `disable_plugin_task(plugin, name)` — reverses enablement

State persists in the `plugin_task_state` SQLite table. All tasks are opt-in;
`enabled_default: true` in the manifest is a suggestion, not an auto-enable.

## A10 — Plugin-scoped memory optimizer

Plugins declare compaction / relevance prompts for their memory roots:

```json
"memory_optimizer": {
  "roots": ["~/.claude/mcp-skill-hub/my_plugin/**"],
  "compaction_prompt": "prompts/compaction.md",
  "relevance_filter":  "prompts/relevance.md"
}
```

`optimize_plugin_memory(plugin, dry_run=True)` runs the plugin's compaction
prompt against its roots via `embeddings.optimize_context` and returns
KEEP/PRUNE/COMPACT/MERGE decisions. `exhaustion_save(context, namespace)`
routes per-plugin digests to the right memory tree. `search_context` honors
per-plugin relevance filters when surfacing private memory matches.

## M2 — Named vector indexes (per-plugin config)

Plugins declare the indexing configuration for each namespace they use:

```json
"vector_indexes": [
  {
    "name":           "career:profile",
    "default_level":  "L3",
    "half_life_days": 365,
    "chunk_size":     4000,
    "chunk_overlap":  400
  },
  {
    "name":           "career:narrative",
    "default_level":  "L2",
    "half_life_days": 60,
    "chunk_size":     3000,
    "chunk_overlap":  300
  }
]
```

These rows are upserted into `vector_index_config` when `index_plugins()` runs.
They control: default level for new vectors, recency decay half-life, and
chunking for large files. Missing rows fall back to global defaults
(L2, 30d, 4000/400).

---

## Memory levels L0–L4

Every vector has a `level` tag that drives retrieval weighting and TTL:

| Level | Scope | Half-life | Weight | Examples |
|-------|-------|-----------|--------|---------|
| L0 | ephemeral | 6h | 0.3 | scratch notes, single-turn tool outputs |
| L1 | session | 7d | 0.8 | active task state, open decisions |
| L2 | working | 30d | 1.0 | drafts, recent bugs, in-progress refs |
| L3 | stable | 180d | 1.1 | decisions, patterns, feedback rules |
| L4 | identity | 10y | 1.3 | role, career direction, preferences |

Retrieval score formula: `score = cosine × level_weight × exp(-age_days / half_life)`.

Promotion path: L1 (age>7d, access≥2) → L2; L2 (age>30d, access≥5) → L3.
Run `promote_memory(dry_run=True)` to preview, `promote_memory(dry_run=False)`
to apply. Core tasks `enable_core_task("promote_memory")` schedules this weekly.

L4 is write-only via `remember_identity(fact, tag)` — never auto-promoted or
auto-pruned.

---

## Hello plugin — minimal walkthrough

```
my-plugin/
├── plugin.json                # declares at least one extension point
├── skills/hello/SKILL.md      # name/description YAML frontmatter
├── web/app.py                 # optional: get_app() -> FastAPI
├── hooks/on_tool_call.py      # optional: stdin JSON → stdout JSON
└── storage/schema.sql         # optional: plugin_<namespace>_* tables
```

Minimal `plugin.json`:

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Hello plugin exercising every extension point.",
  "web_mount": {"mount": "/hello", "title": "Hello", "nav": true},
  "hooks": {"on_tool_call": "hooks/on_tool_call.py"},
  "memory": {"reads": ["~/notes/**/*.md"]},
  "scheduled_tasks": [
    {"name": "daily", "cron": "0 9 * * *",
     "prompt_template": "tasks/daily.md", "enabled_default": false}
  ],
  "storage": {"schema": "storage/schema.sql", "namespace": "hello"},
  "memory_optimizer": {
    "roots": ["~/notes/**"],
    "compaction_prompt": "prompts/compaction.md"
  }
}
```

Install:

```python
configure(
  key="extra_plugin_dirs",
  value='[{"path": "/path/to/my-plugin", "source": "hello", "enabled": true}]'
)
index_plugins()
index_skills()
```

Flagship example: see
[`/Users/ccancellieri/work/code/career-skill-hub-plugin`](../../career-skill-hub-plugin)
— a consumer plugin that exercises every A1–A11 extension point end-to-end
(seven skills, web sub-app with four routes, two hooks, OAuth integration,
plugin-scoped SQL schema, namespaced vectors, watcher, scheduled tasks,
memory optimizer with compaction + relevance prompts).
