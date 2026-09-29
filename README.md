# MCP Skill Hub

Scoped memory and skill context for Claude, Codex, Pi and OpenClaw.
The primary goal is fewer total tokens per completed task through deterministic
compression and explicit context selection.
Quality and whole-task savings are measured separately from payload estimates.
The interactive path preserves the original prompt and works without a local
model or API key. Client-specific adapters use the same context service.

**Migration:** generic auto-proceed, automatic task interception, model switching
in the prompt hook and inferred approvals have been retired. See
[context retrieval and migration](docs/context-service.md) and
[client integrations](integrations/README.md). Existing task and memory data are preserved.

<p align="center">
  <a href="#-quick-start"><img alt="Quick Start" src="https://img.shields.io/badge/Quick_Start-3_minutes-brightgreen?style=flat-square"></a>
  <a href="docs/"><img alt="Docs" src="https://img.shields.io/badge/Docs-→_docs/-blue?style=flat-square"></a>
  <a href="#-license"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-lightgrey?style=flat-square"></a>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-macOS_·_Linux_·_Windows-black?style=flat-square">
  <img alt="Offline" src="https://img.shields.io/badge/Works-Offline-purple?style=flat-square">
</p>

---

## Why Skill Hub?

Keep the current task's evidence available without filling every prompt with
unrelated tasks or the full skill library. The Context page previews retrieved
sources and the exact supplemental text before it reaches a coding client.

Start at `/context`: choose projects, review ranked sources, set a budget,
and copy the composed packet. [Composer guide](docs/context-composer.md) ·
[Minimal MCP profile](docs/mcp-profiles.md) ·
[Evaluation protocol](benchmarks/CONTEXT_VALUE.md).

Use `skill-hub --profile minimal` to expose eight everyday context tools with
description-first skill lookup. The full profile remains the compatibility
default. Local selector learning stays an offline experiment, not an ordinary
workflow requirement.

| Component | Responsibility |
|-----------|----------------|
| Context service | Deterministic, scoped retrieval with size limits and source references |
| Skill library | Import, inspect and load relevant skill content on demand |
| Task and memory tools | Explicit updates and background curation |
| Client | Reasoning, model choice, continuation and permissions |

L1 is a local model, L2 a remote API model, and L3 the coding agent in the client.
Those deployment roles are separate from legacy execution-level names below.

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
Original prompt + project identity
                |
      bounded context retrieval
                |
Original prompt + sourced evidence -> client reasoning

Timeout or missing context -> original prompt passes through
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

### Prompt Context
The hook adds bounded evidence without intercepting task commands or rewriting
the request. Clients without an installed adapter can call `prepare_context`
through MCP. Permission and continuation decisions remain with the client.

**[Context contract](docs/context-service.md)**

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
Confirmed context selections can train a small local ranker. New versions stay
inactive until explicitly promoted with qualifying task-level evidence; improvement
is not assumed.
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

# Master State compaction (folds task auto-memory into project's decisions.md)
compact_master_state(project_root="~/work/code/geoid", dry_run=True)
# After preview + approval:
compact_master_state(project_root="~/work/code/geoid")
# Or wire into close_task:
close_task(task_id=1, compact_master_state=True)

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

## /team — specialized orchestration

Skill Hub is the **intelligence layer** on top of Claude Code's native agent primitives — subagents, agent teams, and the Workflow tool. It does not re-implement orchestration; it supplies specialized role definitions, a model·effort policy, and an upfront prompt-refactor step, then delegates execution entirely to the native substrate. The `/team` command is the single entry point for all of this.

Before spawning a single agent, `/team` calls `improve_prompt` to sharpen the working brief. It then calls `team_plan` to resolve the full roster: which agent types run, at what model tier, in what order, with how many verification loops. The roster is deterministic given `(kind, effort)` — pass `--estimate` to see it without executing anything.

| `/team <kind>` | Task shape | Substrate | Notes |
|---|---|---|---|
| `review` | adversarial — 4 lens reviewers challenge each other | agent team | requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`; falls back to parallel subagents |
| `arch` | adversarial — competing hypotheses + devil's-advocate debate | agent team | same fallback |
| `issues` | deterministic triage pipeline | Workflow tool | fetch → classify → draft; resumable; cheap |
| `implement` | deterministic build pipeline | Workflow tool | design → build → verify → clean → PR |

The `--effort` flag sets both the model floor and the number of verification loops (default `xhigh`):

| Effort | Model floor | Verification loops |
|---|---|---|
| `low` | haiku / sonnet | 0 |
| `medium` | sonnet | 1 |
| `high` | sonnet / opus | 2 |
| `xhigh` (default) | opus | 3 |

Accuracy-critical roles (`team-arch-analyst`, `team-reviewer`, `team-human-voice-writer`) reach Opus at `xhigh`. Mechanical roles (`team-code-implementer`, `team-mechanical-refactorer`, `team-github-operator`) cap at sonnet or haiku — they do not need Opus-level judgement.

The six agent types and their assignments:

- **team-arch-analyst** (opus) — read-only deep architecture and code analysis; cites `file:line`; never edits
- **team-reviewer** (opus at xhigh) — adversarial review; refute-by-default; severity ratings
- **team-code-implementer** (sonnet) — implements a clear spec following existing patterns; no scope creep
- **team-mechanical-refactorer** (sonnet) — behavior-preserving rename/simplify; smallest diff
- **team-human-voice-writer** (opus) — first-person engineer prose; no AI attribution, no emoji, no "recommendations" tables
- **team-github-operator** (haiku) — `gh` inspect/fetch/triage/post; posts only pre-written prose, never authors it

```
/team review 142                                  # adversarial 4-lens review of PR 142
/team arch src/skill_hub/router                   # architecture analysis with devil's advocate
/team issues mcp-skill-hub label:bug --estimate   # preview triage plan + cost, no execution
/team implement 49 --effort high                  # build pipeline, high effort (2 verify loops)
```

---

## 🔧 Requirements

- **Python 3.10+**, **Ollama**, **~5 GB disk** for models (more for larger reasoning models)
- macOS / Linux / Windows — cross-platform installer picks the right hooks
- Optional: **Docker** (for SearXNG), **remote Ollama VPS** (offload heaviest model)

---

## 📄 License

Copyright © 2026 Carlo Cancellieri — Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).
