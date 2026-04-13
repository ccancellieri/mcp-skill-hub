# Claude Code — Model Selection & Token Optimization Guide

## Overview

This guide covers how to choose the right model, use plan mode effectively, switch models mid-session, and reduce token consumption in Claude Code.

---

## Models Available in Claude Code

| Alias | Model | Best For |
|---|---|---|
| `sonnet` | Claude Sonnet 4.6 | Default, everyday coding tasks |
| `opus` | Claude Opus 4.6 | Complex reasoning, architecture, debugging |
| `haiku` | Claude Haiku 4.5 | Fast, lightweight tasks |
| `opusplan` | Opus (plan) + Sonnet (execute) | Best balance of quality and cost |
| `sonnet[1m]` | Sonnet with 1M context | Large codebases |
| `opus[1m]` | Opus with 1M context | Large codebases + complex reasoning |
| `best` | Most capable available | When quality is top priority |
| `default` | Account recommended model | Reverts to subscription default |

### Default Models by Subscription Tier

- **Max / Team Premium**: Defaults to Opus 4.6
- **Pro**: Defaults to Sonnet 4.6
- **Team Standard**: Defaults to Sonnet 4.6

---

## How to Switch Models

### During a session
```
/model sonnet
/model opus
/model opusplan
```

### At CLI startup
```bash
claude --model opusplan
claude --model sonnet
```

### Persist across all sessions (settings.json)
```json
{
  "model": "opusplan"
}
```
Settings file location: `~/.claude/settings.json`

> **Note:** `opusplan` does not appear in UI dropdowns — it is a routing alias. Always set it via command or config, not a picker.

---

## Plan Mode

Press **Shift+Tab** to enter plan mode. Claude explores the codebase and drafts an approach before making any edits.

- Prevents costly re-work if initial direction is wrong
- With `opusplan`, plan mode automatically uses Opus; execution uses Sonnet
- Approve the plan to begin execution

---

## opusplan — What It Actually Does

`opusplan` is a routing alias, not a separate model. It:
- Uses **Opus tokens** during the planning phase (Shift+Tab / plan mode)
- Uses **Sonnet tokens** during the execution phase (file edits, commands)

### opusplan vs. Manual Switching

They are **equivalent in cost and behavior**. `opusplan` just automates what you would do manually:

1. `/model opus` → enter plan mode → review plan
2. `/model sonnet` → execute the plan

The only difference is convenience. There is no hidden optimization in `opusplan` beyond what manual switching provides.

### Cost Comparison

| Approach | Token Cost | Output Quality |
|---|---|---|
| Full Sonnet | Lowest | Good |
| opusplan / manual switch | Medium | Better planning quality |
| Full Opus | Highest (~5x Sonnet) | Best |

**Use `opusplan` when:** you have complex planning needs but straightforward execution.  
**Use full Sonnet when:** tasks are routine and cost matters most.  
**Use full Opus when:** quality is the only priority.

---

## Model Switching — Does It Cost Extra Tokens?

**No switching overhead.** When you run `/model sonnet`, the full conversation context is carried forward as-is. Claude Code does not resend or re-process prior messages just because the model changed.

Each new response sends the accumulated context to whichever model is currently active — same as any normal message. There is no "switching tax."

### Prompt Caching on Switch

Claude Code automatically caches repeated content:
- System prompts
- CLAUDE.md contents
- Tool definitions
- Prior conversation messages

When Sonnet takes over from Opus, cached context is reused at reduced cost. The cache carries over across model switches.

### One Minor Caveat

Different models may interpret the same context slightly differently. In practice this is negligible since plan text is explicit, but be aware that Sonnet re-reading an Opus-generated plan could occasionally miss nuance in highly ambiguous instructions.

---

## Token Reduction Best Practices

### Context Management

| Technique | How to Use |
|---|---|
| `/compact` | Manually summarize context before it grows large |
| Auto-compaction | Automatic — triggers near context limits |
| Keep CLAUDE.md short | Loaded on every turn; keep under 200 lines |

### Session Design

| Technique | Details |
|---|---|
| Use plan mode before execution | Prevents expensive re-work from wrong direction |
| Delegate verbose tasks to subagents | Isolates noisy output (logs, tests) from main context |
| Disable unused MCP servers | Tool schemas only load when server is active — reduces overhead |
| `/effort low` or `/effort medium` | Reduces thinking (reasoning) tokens for routine tasks |

### Model Selection

| Technique | Details |
|---|---|
| Use `opusplan` for complex + routine mixed sessions | Saves Opus tokens on execution |
| Use full Sonnet for routine sessions | Cheapest option |
| Monitor usage with `/cost` | Track token consumption per session |

---

## Recommended Workflow for Complex Tasks

1. Set model: `/model opusplan` (or configure in settings.json)
2. Press **Shift+Tab** to enter plan mode
3. Claude (Opus) explores the codebase and drafts a plan
4. Review the plan — request changes if needed
5. Approve the plan — Claude automatically switches to Sonnet for execution
6. Run `/compact` between major tasks if the session is long

---

## Settings Reference

### ~/.claude/settings.json

```json
{
  "model": "opusplan",
  "env": {
    "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-5-20251001"
  }
}
```

### Key CLI Flags

```bash
claude --model opusplan          # Set model at startup
claude --max-turns 10            # Limit agentic turns
claude --output-format json      # Machine-readable output
```

---

## Quick Reference

```
/model opusplan     → Opus for planning, Sonnet for execution (recommended)
/model sonnet       → Switch to Sonnet mid-session
/model opus         → Switch to Opus mid-session
/compact            → Summarize context to reduce token usage
/cost               → Show token usage for current session
Shift+Tab           → Enter plan mode
``` 