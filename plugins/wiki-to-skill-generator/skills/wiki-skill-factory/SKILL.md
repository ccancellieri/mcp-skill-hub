---
name: wiki-skill-factory
description: Meta-skill for generating project-specialized skills from the memory of work done. Use when you need to understand how memory-to-skill generation works or want to manually trigger it.
triggers:
  - "generate skills from memory"
  - "wiki skill factory"
  - "specialized skills from lessons learned"
---

# Skill Factory — memory of work done → specialized skills

This skill documents how the `wiki-to-skill-generator` plugin turns the
**memory of work done** into **project-specialized** skills. The source is the
`memory:user-project` store — the per-project auto-memory (lessons, decisions,
patterns) under `~/.claude/projects/<project>/memory/*.md` — so skills come out
specialized per project, not generic.

## What makes a *proper, specialized* skill

The authoring bar (enforced at the L3 stage below):

- **Specialized, not generic.** Encode the concrete lesson — the real names,
  paths, decisions, and gotchas of *this* project. Never flatten into vague
  advice that would apply to any codebase.
- **Trigger-optimized `description`.** One line that names the project/domain so
  the skill triggers accurately.
- **Actionable body.** Overview / When to use / Procedure / Gotchas, with the
  specifics kept intact.

## The pipeline (L1/L2/L3 escalation ladder)

| Stage | Tier | Job |
|-------|------|-----|
| 0 — Rearrange upfront | — | Read memory files, group by project, dedup in RAM. Never mutates stored memory (unless `--reindex`). |
| 1 — Triage & cluster | **L1** cheap | Cluster a project's lessons into themes; keep only skill-worthy ones. |
| 2 — Draft | **L2** mid | Draft a project-specialized skill per surviving cluster. |
| 3 — Author | **L3** smart | Polish into a proper `SKILL.md` following *this* guidance. Only skill-worthy clusters reach L3 (cost control). |

## Manual generation

Dry run (L1 triage only — cheap, writes nothing):

```bash
python scripts/generate_skills.py --dry-run
```

Target one project, or refresh embeddings first:

```bash
python scripts/generate_skills.py --project dynastore
python scripts/generate_skills.py --reindex        # slow: reloads embed model
```

Or via the scheduled task / dashboard button (`/wiki-skill-generator/`).

## Output

- Per project: `skills/generated/<project-slug>/<skill-name>/SKILL.md`
- Frontmatter: `name`, `description`, `triggers`, `source: memory-generated`, `project`
- Mapping recorded in the `plugin_wiki_skills` table; `on_session_start` suggests
  relevant skills to active sessions.

---
*Source: wiki-to-skill-generator plugin*
