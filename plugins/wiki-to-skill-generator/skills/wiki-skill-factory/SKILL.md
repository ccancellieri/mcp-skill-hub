---
name: wiki-skill-factory
description: Meta-skill for generating actionable skills from wiki knowledge pages. Use when you need to understand how wiki-to-skill generation works or want to manually trigger skill generation.
triggers:
  - "generate skill from wiki"
  - "wiki skill factory"
  - "extract patterns from wiki"
---

# Wiki Skill Factory

This skill documents how the `wiki-to-skill-generator` plugin transforms wiki knowledge into actionable skills.

## How It Works

1. **Discovery**: Finds wiki entity/concept pages with high `access_count` (>=3 by default)
2. **Extraction**: LLM analyzes page content for actionable patterns
3. **Generation**: Creates a `SKILL.md` file with structured triggers and procedures
4. **Registration**: Records mapping in `plugin_wiki_skills` table
5. **Suggestion**: `on_session_start` hook suggests relevant skills to active sessions

## Configuration

Plugin config in `plugin.json`:

- `min_access_count`: Minimum wiki page access count to consider (default: 3)
- `exclude_types`: Wiki page types to skip (default: ["source", "overview"])
- `llm_tier`: LLM tier for pattern extraction (default: "tier_mid")
- `skill_output_dir`: Where to write generated skills (default: "skills/generated/")

## Manual Generation

Run the scheduled task:

```
enable_plugin_task("wiki-to-skill-generator", "generate-skills")
```

Or call the script directly:

```bash
python scripts/generate_skills.py --dry-run
```

## Generated Skill Structure

Generated skills follow this template:

- `name`: Derived from wiki page title
- `description`: LLM-extracted summary
- `triggers`: Pattern-matched phrases from content
- `procedure`: Step-by-step guidance extracted from wiki
- `examples`: Concrete use cases
- `gotchas`: Common pitfalls or edge cases

## Maintenance

- Generated skills are stored in `skills/generated/`
- Re-running generation overwrites existing skills for the same wiki page
- Use `index_skills()` to refresh the skill index after generation

---
*Source: wiki-to-skill-generator plugin*
