# Web Control Panel — `/control`

A FastAPI control suite at `http://localhost:8765/control` that owns **service lifecycle end-to-end** — not a dashboard, a controller.

## What it does

| Tab | What you can do |
|-----|-----------------|
| **Services** | Start / Stop Ollama daemon, router model, embedding model, SearXNG Docker container, file watcher. Toggling physically reclaims RAM/VRAM/CPU — not a soft bypass. Each card has `auto_start` and opt-in `auto_disable_under_pressure` checkboxes. State persists across reboots. |
| **Plugins** | Every Claude Code plugin in one grid — enable/disable + one-click profile activation. Layers on top of `toggle_plugin()`. |
| **Monitor bar** | Live RAM / CPU readings via `psutil`. Sticky red banner + per-card hints when a conservative pressure threshold is sustained for 30 s. Suggests which service to stop. |
| **Install helpers** | Missing Ollama / SearXNG container / pulled-but-not-loaded models → Install button streams `brew install ollama`, `docker compose up -d`, or `ollama pull <model>` into the card log. |
| **Sticky banner** | When any service is disabled, every page shows `⚠ Disabled: …` so you never forget a feature is off. |

## How reconciliation works

A 2-second reconciler daemon thread reads `~/.claude/mcp-skill-hub/config.json` on mtime change and aligns OS state to match the `services` dict. Toggles take effect live — **no restart**.

## Pressure thresholds

The panel tracks two "conservative" pressure thresholds that trigger UI warnings:

| Signal | Trigger |
|--------|---------|
| RAM pressure | Available RAM drops below a conservative band for **30 s sustained** |
| CPU pressure | 1-minute load average above threshold for **30 s sustained** |

When triggered, cards with `auto_disable_under_pressure = true` self-disable to reclaim resources. See [utilities.md](utilities.md) for the full gating matrix.

## Relationship to other features

- Triggers live MCP tools under the hood (`toggle_plugin`, `pull_model`, `configure`)
- Respects the [resource gating](utilities.md#resource-aware-llm-gating) matrix
- Works with [session profiles](profiles.md) for one-click plugin swaps
