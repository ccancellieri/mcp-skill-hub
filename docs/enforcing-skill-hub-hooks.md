# Keeping mcp-skill-hub in the request loop

Claude Code (and other tools) periodically rewrite `~/.claude/settings.json`.
When they do, they often drop the skill-hub hook block — so the router,
session enforcer, compression, and observers silently stop running and the MCP
falls out of the request loop. This is hard to notice: nothing errors, the
optimizations just quietly disappear.

`skill_hub.base_config` defines the **canonical base hook configuration** and an
**idempotent merger** that repairs settings.json without clobbering anything
else. (Ref: <https://code.claude.com/docs/en/hooks>.)

## Re-apply the hooks

From inside a Claude Code session (the hook pipeline routes these as slash
commands — they work even when the hooks themselves are missing, because the
slash-command fast path is in the MCP server, not the hooks):

```
/hub-install-hooks          # re-apply any missing skill-hub hooks (idempotent)
/hub-install-hooks check    # report what's missing, change nothing
/hub-install-hooks dry-run  # preview the changes, write nothing
```

Aliases: `/hub-enforce`, `/hub-reapply-hooks`.

From a shell (e.g. after a tool clobbered settings and the MCP is fully gone):

```bash
python -m skill_hub.base_config            # re-apply
python -m skill_hub.base_config check      # exit 1 if any hook is missing
python -m skill_hub.base_config --dry-run  # preview
```

After re-applying, **restart Claude Code** for the hooks to take effect.

## What gets installed

The base config is the single source of truth for "the MCP in the loop". It
covers these events (matched and de-duplicated by hook *script basename*, so
re-running never duplicates and a moved checkout is repaired, not doubled):

| Event | Hook script | Purpose |
|---|---|---|
| `UserPromptSubmit` | `session-start-enforcer.sh` | session-start protocol |
| `UserPromptSubmit` | `intercept-task-commands.sh` | context enrichment |
| `UserPromptSubmit` | `prompt-router.sh` | model/skill routing |
| `PreCompact` / `PostCompact` | `precompact.sh` / `postcompact.sh` | routing snapshot + memory compression |
| `Stop` | `session-end.sh`, `auto-proceed.sh` | memory save + plan continuation |
| `PreToolUse` | `auto-approve.sh` | allow-list (Bash) |
| `PostToolUse` / `PostToolUseFailure` | `post-tool-observer.sh` | command observation |
| `StopFailure` | `stop-failure.sh` | API-error logging |
| `SessionEnd` | `session-end-real.sh` | session close |
| `SubagentStart` / `SubagentStop` | `subagent-observer.sh` | subagent observation |

The merger:

- **preserves** unrelated hooks (codegraph, `~/.claude/hooks/*`, model-tier
  enforcement) and every other settings key;
- **backs up** `settings.json` → `settings.json.bak` before writing;
- **creates** settings.json if absent;
- is a **no-op** when all hooks are already present.

Hook command paths are resolved relative to this repo (`<repo>/hooks/*.sh`);
`SKILL_HUB_HOOKS_DIR` overrides the location for non-standard checkouts.

> Note: this re-applies the **hooks**. Registering the `skill-hub` MCP *server*
> itself lives in the separate Claude Code MCP config and is out of scope here.
