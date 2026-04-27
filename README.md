# MCP Skill Hub

> **A local MCP server that makes Claude Code smarter, cheaper, and offline-capable.**
> Semantic skill search, zero-token hook interception, and a full 4-level local execution engine — all running on your machine with Ollama.

<p align="center">
  <a href="#-quick-start"><img alt="Quick Start" src="https://img.shields.io/badge/Quick_Start-3_minutes-brightgreen?style=flat-square"></a>
  <a href="docs/"><img alt="Docs" src="https://img.shields.io/badge/Docs-→_docs/-blue?style=flat-square"></a>
  <a href="#-license"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-lightgrey?style=flat-square"></a>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-macOS_·_Linux_·_Windows-black?style=flat-square">
  <img alt="Offline" src="https://img.shields.io/badge/Works-Offline-purple?style=flat-square">
</p>

---

## Why Skill Hub?

Claude Code loads **every enabled plugin's skills** into your context at session start. With 20+ plugins, that's thousands of wasted tokens on definitions you'll never use. Responses degrade, latency grows, bills rise.

Skill Hub fixes this with three layers that work together:

| Layer | What it does | Saved |
|------|--------------|-------|
| 🔎 **Semantic search** | Finds skills by meaning, not keywords — only loads what matters | ~80% context |
| 🎯 **Zero-token hooks** | Task commands (`save task`, `close task`, `list tasks`) are caught **before** Claude sees them | **100%** of those tokens |
| 🤖 **Local execution (L1–L4)** | Whitelisted commands → templates → skills → full local agent; all on Ollama | API calls avoided |

Add teaching rules, feedback learning, shadow skill evolution, offline auto-fallback, and a web control panel — and the hub **learns your vocabulary** over time.

---

## 🚀 Quick Start

```bash
git clone https://github.com/ccancellieri/mcp-skill-hub.git
cd mcp-skill-hub
./install.sh          # macOS / Linux
python install.py     # cross-platform
```

The installer pulls a 274 MB embedding model, registers the MCP server, and merges hooks into `~/.claude/settings.json` (idempotent — safe to re-run).

**Then restart Claude Code and run:**

```
index_skills()      # index all plugin skills
index_plugins()     # index plugin descriptions
```

👉 Full installation options (SearXNG, remote VPS, model picks per RAM budget): **[docs/installation.md](docs/installation.md)**

---

## 🎬 In 30 Seconds

```
User: "save to memory and close"
         │
    ┌────┴──────────────────┐
    │  UserPromptSubmit     │ ← hook fires BEFORE Claude
    │  Hook                 │
    └────┬──────────────────┘
         │
    ┌────┴──────────────────┐
    │  Local LLM classifies │ ← qwen2.5-coder:7b on your machine
    │  "Is this a task      │
    │   command?"           │
    └────┬──────────────────┘
         │
    YES: execute locally           NO: pass through
    save_task() / close_task()     → Claude processes normally
    return {"decision":"block"}
         │
    0 Claude tokens used
```

---

## ✨ Feature Highlights

<table>
<tr>
<td width="33%" valign="top">

### 🔎 Semantic Search
Describe the task in natural language — get matching skills ranked by cosine similarity + your feedback history.
```python
search_skills("debug failing pytest")
```
**→ [docs/features/semantic-search.md](docs/features/semantic-search.md)**

</td>
<td width="33%" valign="top">

### 🎯 Zero-Token Hooks
Task commands are intercepted **before** Claude. Each interception saves 300–800 tokens.
```
"save this as task"      →  0 tokens
"list my open tasks"     →  0 tokens
"what was I working on?" →  0 tokens
```
**→ [docs/features/hooks.md](docs/features/hooks.md)**

</td>
<td width="33%" valign="top">

### 🤖 Local Execution
4 escalating levels: whitelisted commands → templates → multi-step skills → full L4 agent loop.
```
L1: "git status"
L2: "show last 5 commits"
L3: "project summary"  (4-step skill)
L4: "run tests and summarize"
```
**→ [docs/features/local-execution.md](docs/features/local-execution.md)**

</td>
</tr>
<tr>
<td valign="top">

### 🧠 Learning
Teaching rules, feedback EMA, session history, shadow evolution — the hub gets **measurably smarter** over time.
**→ [docs/features/learning.md](docs/features/learning.md)**

</td>
<td valign="top">

### 🖥️ Web Control Panel
FastAPI suite at `http://localhost:8765/control` — start/stop Ollama, SearXNG, models; live RAM/CPU pressure; plugin toggles; profile switching.
**→ [docs/features/web-control-panel.md](docs/features/web-control-panel.md)**

</td>
<td valign="top">

### 🧭 Offline & Fallback
TCP probes Anthropic every 30 s. Unreachable? L4 agent silently takes over. Rate-limited? Exhaustion-save compacts your session.
**→ [docs/features/local-execution.md#offline--exhaustion](docs/features/local-execution.md#offline--exhaustion)**

</td>
</tr>
<tr>
<td valign="top">

### 🗂️ Session Profiles
Swap entire plugin sets per context: `minimal`, `backend`, `frontend`, `mcp-dev`, `data`, `full` — or save your own.
**→ [docs/features/profiles.md](docs/features/profiles.md)**

</td>
<td valign="top">

### 🪶 Context Bridge
Captures Claude's tool calls in real-time → `{session_context}`, `{tool_examples}`, `{repo_context}` injected into every local skill.
**→ [docs/advanced/context-bridge.md](docs/advanced/context-bridge.md)**

</td>
<td valign="top">

### 🎓 Fine-Tuning
Export JSONL training data from your own feedback, triage, and compact history. Fine-tune on Apple Silicon via `mlx-lm`.
**→ [docs/advanced/fine-tuning.md](docs/advanced/fine-tuning.md)**

</td>
</tr>
</table>

---

## 📚 Documentation

Everything lives in [docs/](docs/). Start with the index below — it's kept in sync with code.

### 🧭 [**Documentation Index →**](docs/README.md)

| Area | Doc | When to read |
|------|-----|--------------|
| **Getting Started** | [installation.md](docs/installation.md) | First install, model picks, SearXNG, remote VPS |
| **Features** | [web-control-panel.md](docs/features/web-control-panel.md) | Manage services + plugins from a browser |
|  | [semantic-search.md](docs/features/semantic-search.md) | `search_skills`, `search_context`, tasks, digest |
|  | [hooks.md](docs/features/hooks.md) | How zero-token interception + context injection work |
|  | [learning.md](docs/features/learning.md) | Teachings, feedback, implicit learning, evolution |
|  | [local-execution.md](docs/features/local-execution.md) | L1–L4, offline fallback, exhaustion save, triage |
|  | [profiles.md](docs/features/profiles.md) | Plugin profile packs + auto-recommendation |
|  | [utilities.md](docs/features/utilities.md) | Extra skill dirs, status, resource gating, REPL, tooltips |
| **Reference** | [reference/tools.md](docs/reference/tools.md) | Every MCP tool + CLI command |
|  | [reference/config.md](docs/reference/config.md) | All config keys, defaults, description |
|  | [reference/architecture.md](docs/reference/architecture.md) | Source layout, dual skill index, output paths |
|  | [reference/database.md](docs/reference/database.md) | SQLite schema + table purposes |
|  | [reference/logs.md](docs/reference/logs.md) | Log streams, common issues, troubleshooting |
| **Advanced** | [advanced/skill-chaining.md](docs/advanced/skill-chaining.md) | Local skill branching, labels, `agent` type |
|  | [advanced/context-bridge.md](docs/advanced/context-bridge.md) | How Claude's tool calls flow into local skills |
|  | [advanced/fine-tuning.md](docs/advanced/fine-tuning.md) | Exporting JSONL, training with `mlx-lm` |
| **Ops** | [unattended.md](docs/unattended.md) | Run Claude Code overnight without prompts |
|  | [plugin-extension-points.md](docs/plugin-extension-points.md) | How third-party plugins extend Skill Hub |
|  | [roadmap.md](docs/roadmap.md) | Shipped milestones + what's next |

---

## 🏃 Common Workflows

```bash
# Search past work
search_context("accessibility audit for a website")

# Save & close tasks (zero Claude tokens via hooks)
save_task(title="MCP skill hub dev", summary="Building semantic search…")
close_task(task_id=1)    # compacts to ~200 tokens, writes memory entry

# Teach the hub
teach(rule="when I give a URL", suggest="chrome-devtools-mcp")

# Switch profiles
/profile backend
/profile auto build MCP server

# Check token savings
token_stats()   # → e.g. "52,300 tokens saved across 89 interceptions"
```

---

## 🌳 Worktree-Driven Parallel Sessions

Spawn a Claude session inside an isolated git worktree as part of saving a task,
and resume it later — the worktree outlives the task by default.

```bash
# Cold start from a non-repo dir like ~/work/code/
cwt geoid es-pr2c                              # opens iTerm tab in a fresh worktree
cwt geoid swarm-3 --mode background            # headless agent, output to logfile
cwt --resume 47                                # focus alive session, or relaunch
cwt --list                                     # open tasks + worktree liveness
```

From inside Claude (auto-saves the task and spawns the session):
```python
save_task("ES PR-2c retarget", "...", project="geoid", mode="terminal")
reopen_task(47)                                # alive → focus, dead → relaunch
close_task(47, remove_worktree=True)           # also tears down the worktree
```

**Layout:**
- Worktree: `<repo>/.claude/worktrees/<slug>` (per-repo, gitignored)
- Branch: `cc/<slug>` (local-only convention for AI-tooling work)
- Liveness: `<worktree>/.claude/session.pid` (cleaned up by a Stop hook)

**Modes:** `terminal` (macOS iTerm/Terminal tab), `tmux` (window in `$TMUX`),
`background` (headless `claude --print` to a logfile).

**Config** (`~/.claude/mcp-skill-hub/config.json`):
```json
{
  "worktree": {
    "repo_roots": ["~/work/code"],
    "default_mode": "terminal"
  }
}
```

---

## 🔧 Requirements

- **Python 3.10+**, **Ollama**, **~5 GB disk** for models (more for larger reasoning models)
- macOS / Linux / Windows — cross-platform installer picks the right hooks
- Optional: **Docker** (for SearXNG), **remote Ollama VPS** (offload heaviest model)

---

## 📄 License

Copyright © 2026 Carlo Cancellieri — Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).
