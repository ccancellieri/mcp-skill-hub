# Skill Hub Documentation

The [root README](../README.md) is the elevator pitch. Everything else lives here — split so Claude (and you) only load what's needed.

## 🗺️ Map

```
docs/
├── installation.md              Install modes, models, SearXNG, remote VPS
│
├── features/                    What Skill Hub does, feature by feature
│   ├── web-control-panel.md     The /control FastAPI suite
│   ├── semantic-search.md       search_skills, search_context, tasks, digest
│   ├── hooks.md                 Zero-token interception + context injection
│   ├── learning.md              Teachings, feedback EMA, implicit learning, evolution
│   ├── local-execution.md       L1-L4 engine, agent, offline, exhaustion, triage
│   ├── profiles.md              Session plugin profiles + auto-recommendation
│   └── utilities.md             Extra dirs, status, gating, REPL, tooltips, inline help
│
├── reference/                   Stable contracts — tools, config, schema
│   ├── tools.md                 Every MCP tool + CLI command
│   ├── config.md                All config keys, defaults, descriptions
│   ├── architecture.md          Source layout + dual skill index + output paths
│   ├── database.md              SQLite schema
│   └── logs.md                  Log streams + troubleshooting
│
├── advanced/                    For power users building on top of the hub
│   ├── skill-chaining.md        Local skill branching, labels, agent-as-skill
│   ├── context-bridge.md        Claude → local skill intelligence flow
│   └── fine-tuning.md           Training export + mlx-lm fine-tuning
│
├── plugin-extension-points.md   Third-party plugin extension contract
├── unattended.md                Run Claude Code overnight
└── roadmap.md                   Shipped + upcoming milestones
```

## 🎯 Quick paths

| I want to… | Read |
|------------|------|
| **Install** on my machine | [installation.md](installation.md) |
| Understand **how the hook saves tokens** | [features/hooks.md](features/hooks.md) |
| **Keep working when Claude is rate-limited** | [features/local-execution.md](features/local-execution.md) |
| **Teach** the hub my vocabulary | [features/learning.md](features/learning.md) |
| See every **MCP tool** I can call | [reference/tools.md](reference/tools.md) |
| Tune a **config value** | [reference/config.md](reference/config.md) |
| Build a **custom local skill** | [advanced/skill-chaining.md](advanced/skill-chaining.md) |
| **Fine-tune** on my own data | [advanced/fine-tuning.md](advanced/fine-tuning.md) |
| **Debug** a hook that isn't firing | [reference/logs.md](reference/logs.md) |
| **Run overnight** without prompts | [unattended.md](unattended.md) |

## 📝 Editing guidance

- Keep each doc single-purpose. If it starts overflowing, split it.
- Link across docs rather than duplicating. The goal is **minimum context while editing**.
- Code examples: prefer the actual tool names (`search_skills`, `configure`) over invented phrasing.
- Tables beat prose for config and tool references.
- When adding a feature, also update [roadmap.md](roadmap.md) and this index.
