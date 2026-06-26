# Wiki-to-Skill Generator Plugin

Auto-generates actionable skills from wiki entity/concept pages with high access counts.

## Purpose

Transforms accumulated wiki knowledge into reusable skills that can be suggested during sessions. When a wiki page is accessed frequently, it likely contains patterns worth codifying as a skill.

## Extension Points

- **A3 (hooks)**: `on_session_start` - Suggests relevant wiki-derived skills to active sessions
- **A5 (scheduled_tasks)**: Nightly scan for undistilled wiki entities
- **A7 (storage)**: Tracks generated skills and source wiki pages

## Installation

The plugin is bundled with mcp-skill-hub. Enable via:

```python
configure("extra_plugin_dirs", '[{"path": "/path/to/mcp-skill-hub/plugins/wiki-to-skill-generator", "enabled": true}]')
index_plugins()
index_skills()
```

## Usage

### Automatic Generation

Enable the scheduled task to run nightly:

```
enable_plugin_task("wiki-to-skill-generator", "generate-skills")
```

### Manual Generation

Run the generation script:

```bash
cd /path/to/mcp-skill-hub/plugins/wiki-to-skill-generator
python scripts/generate_skills.py
```

With dry-run preview:

```bash
python scripts/generate_skills.py --dry-run
```

### Filtering by Type

```bash
python scripts/generate_skills.py --types entity,concept
```

### Custom Access Threshold

```bash
python scripts/generate_skills.py --min-access 5
```

## Configuration

Edit `plugin.json` to customize:

```json
{
  "config": {
    "min_access_count": 3,
    "exclude_types": ["source", "overview"],
    "llm_tier": "tier_mid"
  }
}
```

## Generated Skills

Skills are written to `skills/generated/{slug}.md` with:

- YAML frontmatter with name, description, triggers
- Structured sections: Context, Procedure, Examples, Gotchas
- Back-reference to source wiki page

## Storage Schema

The `plugin_wiki_skills` table tracks:

- `wiki_slug`: Source wiki page
- `skill_path`: Path to generated skill
- `access_count`: Wiki access count at generation time
- `use_count`: Times skill was suggested

## Hook Behavior

On session start, the hook:

1. Queries `plugin_wiki_skills` for all registered skills
2. Optionally filters by relevance to current context
3. Returns skill suggestions to the session

## Dependencies

- `skill_hub.store.SkillStore` for database access
- `skill_hub.llm.get_provider()` for pattern extraction
- `skill_hub.wiki` for querying wiki pages
- `jinja2` for skill templating
