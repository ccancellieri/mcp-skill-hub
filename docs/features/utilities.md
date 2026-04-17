# Utilities & Quality-of-Life

A grab-bag of smaller features that make daily use nicer.

## Status & token profiling

Check the health of every component in one call:

```
status()
```

```
=== Skill Hub Status ===
MCP server:      ✓ running
Ollama:          ✓ reachable at http://localhost:11434
Embed model:     ✓ nomic-embed-text
Reason model:    ✓ qwen2.5-coder:7b-instruct-q4_k_m
Hook:            ✓ configured and enabled
Token profiling: ✓ on

Database:
  Skills indexed:    1247
  Tasks:             12 (3 open)
  Intercepted cmds:  89 (~52,300 tokens saved)
```

Track cumulative savings → see [hooks.md](hooks.md#how-it-saves-tokens).

---

## Resource-aware LLM gating

Skill Hub monitors CPU load and memory pressure, and **skips expensive local LLM operations when the machine is busy**. This prevents latency spikes during builds, compiles, or large model loads.

| Operation | IDLE | LOW | MODERATE | HIGH |
|-----------|------|-----|----------|------|
| embed | ✓ | ✓ | ✓ | ✓ |
| triage | ✓ | ✓ | ✓ | skip |
| rerank | ✓ | ✓ | ✓ | skip |
| precompact | ✓ | ✓ | skip | skip |
| digest | ✓ | ✓ | skip | skip |
| optimize_memory | ✓ | skip | skip | skip |

Pressure is re-evaluated every 10 s (cached). Current pressure is visible in `/hub-status`:

```
/hub-status
# pressure=LOW  cpu=32%  mem=80%  avail=3.3GB
```

Force all operations to run regardless of pressure:

```bash
SKILL_HUB_FORCE_LLM=1 skill-hub-cli classify "..."
```

Disable gating:

```
configure(key="resource_gating_enabled", value="false")
```

---

## Context optimization

Analyze your memory files and get recommendations to reduce token usage:

```
/optimize-context
```

```
=== Context Optimization ===

Analyzing 48 memory files (~31,550 tokens total)…

  PRUNE  project_geoid_openapi_schema_cleanup.md — completed task, no longer needed
  COMPACT reference_dynastore_tools_knowledge.md — verbose, can save ~400 tokens
          → "Dynastore tools: enrichment pipeline, query executor, DDLBatch…"
  MERGE  feedback_use_constants_enums.md — overlaps with feedback_use_libraries_over_custom_models.md
  KEEP   project_geoid_core.md

Actions available:
  1 file to prune (~200 tokens saved per session)
  1 file to compact
  1 file to merge
```

---

## Extra skill directories

Index skills from any directory — including archived libraries — by adding entries to `extra_skill_dirs`:

```
configure(key="extra_skill_dirs",
  value='[{"path": "~/.claude/skills-archive", "source": "archive", "enabled": true}]')
```

Then re-index:

```
index_skills()    # scans extra_skill_dirs too
index_plugins()   # registers them as plugin sources for suggest_plugins()
```

Skills from extra dirs get IDs like `archive:skill-name`. They appear alongside plugin skills in all searches and suggestions.

---

## Inline help system

Type `?` to discover commands or get detailed usage:

```
?                     # list all commands with descriptions
?hub-list-skills      # detailed usage for /hub-list-skills
?hub-configure        # detailed usage for /hub-configure
?local-agent          # detailed usage for /local-agent
```

Works even when Claude is rate-limited — the `?` system runs entirely in the local hook.

---

## Tooltips & help modals

See [../TOOLTIPS_HELP_GUIDE.md](../../TOOLTIPS_HELP_GUIDE.md) for the full UI layer — interactive tooltips (hover/keyboard), help modals, expandable dashboard cards, cost breakdown, pricing reference.

---

## Related

- [reference/logs.md](../reference/logs.md) — debugging when something goes wrong
- [reference/config.md](../reference/config.md) — every tuneable knob
