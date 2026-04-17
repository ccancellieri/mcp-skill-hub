# Architecture

## Source layout

```
src/skill_hub/
├── server.py           FastMCP tools + MCP protocol
├── store.py            SQLite: skills (claude+local), embeddings, feedback, teachings, tasks
├── indexer.py          Scan plugin dirs + local-skills/, parse SKILL.md + JSON, build index
├── embeddings.py       Ollama: embed, rerank, compact, triage (incl. local_agent action)
├── config.py           User-configurable settings (models, thresholds, local_models)
├── cli.py              Hook pipeline: slash commands, L1-L4, triage, ? help, context injection
├── local_agent.py      Level 4: plan_agent() + run_agent() with tool-calling loop
├── activity_log.py     File + stderr logging (daily rotation, 50MB cap)
├── resource_monitor.py CPU/memory pressure gating for LLM operations
├── searxng.py          SearXNG web search fallback integration
├── watcher.py          Watchdog auto-reindex on plugin file changes
└── repl.py             Interactive REPL for local debugging

hooks/
├── session-start-enforcer.sh    UserPromptSubmit: session-start protocol reminder (0 LLM tokens)
├── session_start_enforcer.py    ↳ cross-platform Python equivalent
├── intercept-task-commands.sh   UserPromptSubmit: zero-token task interception
├── intercept_task_commands.py   ↳ cross-platform Python equivalent
├── session-end.sh               Stop: session memory + stats
├── session_end.py               ↳ cross-platform Python equivalent
└── session-logger.sh            Stop: passive tool usage logging
```

## Dual skill index

Skills are stored with a **target** field that routes them to the correct LLM context.

| Target | Source | Loaded into | Purpose |
|--------|--------|-------------|---------|
| `claude` | `SKILL.md` in plugins | Claude context (RAG injection) | Plugin skills for Claude |
| `local` | JSON in `~/.claude/local-skills/` | Local LLM agent (L4) prompt | Skills for local execution |

- Claude skills **never** pollute the local LLM prompt
- Local skills **never** consume Claude tokens
- Both are searchable via `search_skills()` and visible in `list_skills()`

```
index_skills()    # SKILL.md → target=claude, JSON → target=local
```

## Output paths

Skill Hub uses three distinct communication paths — which one applies determines what the user sees.

| Path | How it works | Visible? | Example |
|------|--------------|----------|---------|
| MCP tool return | Tool returns a string → into Claude's context → Claude decides what to relay | Only if Claude echoes it | `status()`, `search_skills()` |
| Hook `block` response | `{"decision": "block", "message": "…"}` → shown directly to user | **Always** | `/hub-status`, task interception |
| Hook `systemMessage` | `{"decision": "allow", "systemMessage": "…"}` → injected into Claude's context as system text | **No** (invisible) | Dynamic context injection |

## Request flow

```
User message
     │
     ▼
┌────────────────────────────────────────┐
│ session-start-enforcer.sh              │
│   flag file check → inject checklist?  │
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ intercept-task-commands.sh             │
│   → skill-hub-cli classify             │
│     ↳ semantic + LLM triage            │
│                                        │
│   local_answer / local_action?         │
│     → block + reply locally  (0 tokens)│
│                                        │
│   L1/L2/L3/L4 match?                   │
│     → show plan, block, execute        │
│                                        │
│   enrich_and_forward?                  │
│     → allow + systemMessage RAG        │
│                                        │
│   pass_through?                        │
│     → allow, no injection              │
└──────────────────┬─────────────────────┘
                   ▼
              Claude Code
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stop hook                              │
│   → session-end.sh                     │
│     ↳ update tool_examples             │
│     ↳ rolling session summary          │
│     ↳ implicit feedback EMA            │
│     ↳ skill evolution (shadow)         │
└────────────────────────────────────────┘
```

## Web control panel

Separate process at `http://localhost:8765/control` — a FastAPI suite with a 2-second mtime reconciler daemon that watches `config.json` and aligns OS state to match. See [features/web-control-panel.md](../features/web-control-panel.md).

## Related

- [reference/database.md](database.md) — SQLite schema
- [reference/config.md](config.md) — every config key
- [reference/logs.md](logs.md) — log streams + troubleshooting
