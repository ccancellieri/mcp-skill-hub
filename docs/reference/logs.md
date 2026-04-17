# Logs & Troubleshooting

Two log streams show what the local LLM is doing. Both rotate daily and are capped at **50 MB total**.

## Hook debug log

Raw hook I/O, timing, decisions:

```bash
# macOS / Linux
tail -f ~/.claude/mcp-skill-hub/logs/hook-debug.log

# Windows (PowerShell)
Get-Content -Wait $HOME\.claude\mcp-skill-hub\logs\hook-debug.log
```

### Sample

```
[14:32:01] [  0.0s] ENFORCER   NEW SESSION  id=abc123
[14:32:01] [  0.0s] ENFORCER   injecting session-start checklist
[14:32:03] [  0.0s] INTERCEPT  fired  session=abc123  len=42  msg="refactor the auth middleware"
[14:32:03] [  0.1s] INTERCEPT  cli  exit=0  time=1823ms  stdout_len=2450
[14:32:03] [  1.9s] INTERCEPT  ALLOW  enriched=yes  systemMsg=1890chars  cli_time=1823ms
[14:32:03] [  1.9s] INTERCEPT  done  total_time=1924ms
```

Each line shows: `[time] [elapsed] HOOK_TYPE  details`.

| Prefix | Source |
|--------|--------|
| `ENFORCER` | `session-start-enforcer` — session detection, checklist injection |
| `INTERCEPT` | `intercept-task-commands` — classify, block/allow, context injection |
| `STOP` | `session-end` — memory save, session stats |

## Activity log

MCP tool calls, LLM invocations, skill searches:

```bash
# macOS / Linux
tail -f ~/.claude/mcp-skill-hub/logs/activity.log

# Windows (PowerShell)
Get-Content -Wait $HOME\.claude\mcp-skill-hub\logs\activity.log
```

Written by the MCP server process (not hooks). Shows `TOOL`, `HOOK`, `LLM`, and `EVENT` entries with structured key-value pairs.

---

## Common issues

| Symptom | Check |
|---------|-------|
| **Hooks not firing** | `~/.claude/settings.json` → verify `hooks.UserPromptSubmit` entries exist |
| **"CLI failed" in hook log** | Run `skill-hub-cli classify "test"` manually to see the error |
| **SearXNG not found** | `curl "http://localhost:8989/search?q=test&format=json"` |
| **Remote VPS unreachable** | `curl http://your-vps:11434/api/tags` |
| **Ollama models not loading** | `ollama list` (check it's pulled) and `ollama ps` (check it's running) |
| **`search_skills` returns nothing** | Did you run `index_skills()` after install? Check `status()` DB stats. |
| **Local skill never matches** | Raise `local_skill_threshold` trigger in `configure`, or check `/hub-list-skills local` |
| **Context never injected** | Check `hook_context_injection=true`; look at HOOK messages in `activity.log` |
| **Evolution writes bad skills** | Disable with `configure(key="skill_evolution_enabled", value="false")`; restore via `skill_versions` table |

## Diagnostic calls

```
status()               # full health check
token_stats()          # did interception actually run?
session_stats()        # plugin usage
list_teachings()       # are rules what you expect?
/hub-local-status      # L1-L4 engine state
```

## Related

- [reference/architecture.md](architecture.md) — understand where each log entry comes from
- [features/hooks.md](../features/hooks.md) — how the hook decides what to log
