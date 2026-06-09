# Comparison: mcp-skill-hub vs ruflo (claude-flow)

## Why this document exists

`ruflo` (npm: `@claude-flow/cli`) and `mcp-skill-hub` overlap in roughly 20% of their surfaces — task tracking, semantic search, session persistence, hooks. The rest is disjoint. This page documents the comparison so a reader can decide whether to install both, neither, or just one.

**Project position**: skill-hub is the consolidated tool. It went through two moves on the overlapping surface:

1. **Absorbed from ruflo (milestone M4).** The ruflo features the maintainer valued — swarm-lite, autopilot-lite, federation-lite, and the skill/agent importers — were reimplemented as native skill-hub primitives, with zero runtime dependency on ruflo. After M4, ruflo could be uninstalled.
2. **Then superseded by the native `/team` layer (PR #52).** Once Claude Code shipped first-class **subagents** (`Agent` with `isolation: "worktree"`), **agent teams**, and the **Workflow tool**, skill-hub's home-grown in-process orchestration engines (swarm, autopilot, the `author_plan`/`run_plan`/`execute_plan_step` stepper, and the W5 sandbox) became redundant and were **retired**. skill-hub no longer runs its own agent loop. It is now the *intelligence layer* over those native primitives: role definitions, a model·effort policy (`team_plan`), and upfront prompt refactoring (`improve_prompt`), all driven by the `/team` command.

What remains from the ruflo absorption: **federation-lite** (`federation_view`) and the **importers** (`scripts/import_ruflo_{skills,agents}.py`).

**Hard constraint**: skill-hub never runtime-depends on ruflo. Read the [no-ruflo-dep gate](#no-ruflo-runtime-dependency) below.

---

## What each is actually for

|  | **mcp-skill-hub** | **ruflo (claude-flow)** |
|---|---|---|
| Core purpose | Make a single interactive Claude Code session faster and smarter; orchestrate Claude Code's *native* agents via `/team` when parallel work is needed | Orchestrate many Claude/LLM agents at once with its own engine |
| Scale | One session (you ↔ Claude), fanning out to native subagents / agent teams on demand | Swarm of N agents working in parallel |
| Mode | Passive infrastructure + policy layer over native orchestration | Active coordinator with its own runtime |
| Surface area | ~40 MCP tools, focused | 300+ MCP tools, broad |
| Required deps | Python + FastMCP + optional Ollama | Node 20+, optional API keys |
| Local-LLM-first | Yes — Ollama is a first-class provider | No — provider, but one of many |
| Cost to run | Free, local | Free for state plumbing; LLM calls = API bill |

---

## Side-by-side on overlapping features

| Feature | skill-hub | ruflo | Verdict |
|---|---|---|---|
| **Task save/reopen** | `save_task` / `list_tasks` — tied to *conversation* | `claims_claim` / `claims_board` — tied to *work item across agents* | Skill-hub for solo bookmarks; claims semantics absorbed into the task table |
| **Session restore** | `save_task` resumes context | `session_save` / `session_restore` resumes any agent | Both fine; skill-hub simpler |
| **Semantic search** | `search_skills`, `search_context` — over skills + memory | `memory_search_unified`, `embeddings_search` — over agent DB | Skill-hub faster (indexed locally); both LLM-optional |
| **Teaching rules** | `teach()` / `list_teachings` / `forget_teaching` — Claude-only | No equivalent | Skill-hub only |
| **Plugin routing** | `suggest_plugins`, `toggle_plugin`, profiles | No equivalent | Skill-hub only |
| **Token-saving** | `token_stats`, `optimize_context`, hook interception | Indirect via `agentdb_consolidate` | Skill-hub only |
| **Local LLM (Ollama)** | First-class: `list_models`, `pull_model`, dashboard | One provider among many | Skill-hub first-class |
| **Dashboard** | `/control` FastAPI suite | No native dashboard | Skill-hub only |
| **Parallel agents** | **Native** — `/team` orchestrates Claude Code subagents (`Agent` with `isolation: "worktree"`), agent teams, and the Workflow tool, with a model·effort policy via `team_plan`. (Briefly shipped as in-process `swarm_launch`; retired in PR #52.) | `agent_spawn` × N, `swarm_init`, `hive-mind_*` | Native via `/team`; skill-hub adds the role + model·effort policy on top |
| **Worktree isolation** | **Native** — `isolation: "worktree"` on `Agent` calls + `worktree_preflight` collision check | `isolation: worktree` on `Agent` calls | Parity; skill-hub adds the preflight |
| **Headless autopilot** | **Native** — `/loop` + the Workflow tool for unattended runs. (Briefly shipped as in-process `autopilot_run`; retired in PR #52.) | `autopilot_enable` — work overnight | Native via `/loop` + Workflow |
| **Cross-machine federation** | **Native (thin)** — `federation_view` ATTACHes a peer SQLite DB read-only (WAL + `node_id`) | `ruflo federation init` | Skill-hub keeps a thin WAL+sync layer, not a protocol |
| **Witness / fix manifest** | **Native** — `record_witness` / `list_witness` append-only fix manifest per repo | `witness` skill (ADR-103) | Parity |
| **Performance / neural training** | None | `performance_*`, `neural_train`, `daa_*` | Out of scope for skill-hub |

---

## Where they actively conflict

- **Memory store**: running both means two indexes drifting apart. Pick skill-hub as source of truth. If a user has ruflo installed, the importers (`scripts/import_ruflo_skills.py`, `scripts/import_ruflo_agents.py`) move the content into skill-hub once, after which ruflo can be uninstalled.
- **Task tracking**: `list_tasks` vs `claims_board` diverge. skill-hub's task table is the single source of truth; the claims semantics live there.

---

## Why skill-hub absorbed rather than bridged

A bridge means skill-hub runtime-depends on ruflo. That couples two release cycles, adds Node-on-Python install pain, and gives users no clean uninstall path. Absorbing means:

1. The ruflo capability becomes a native skill-hub primitive (SQLite WAL for federation, importer scripts for skills/agents).
2. The user runs the importer once during migration.
3. The user uninstalls `@claude-flow/cli`.
4. Nothing in skill-hub still references ruflo.

**Post-script (PR #52).** The orchestration half of the absorption — swarm-lite and autopilot-lite, which had been reimplemented as in-process engines — was then retired outright. Claude Code's native subagents, agent teams, and Workflow tool do the orchestration; skill-hub keeps only the policy and intelligence layer (`/team`, `team_plan`, `improve_prompt`) on top of them. Running your own agent loop inside an MCP server is strictly worse than letting the harness do it — fewer moving parts, no subprocess bookkeeping, native worktree isolation. Federation-lite and the importers, which are not orchestration, were kept.

---

## No-ruflo runtime dependency

**Enforced by CI**:

```bash
# Must succeed (no ruflo packages):
! grep -Eqi 'claude-flow|ruflo' pyproject.toml
# Must succeed (no ruflo imports):
! grep -rE 'import claude_flow|from claude_flow|import ruflo|from ruflo' src/
```

The only place ruflo is allowed to appear is inside one-shot importer scripts under `scripts/import_ruflo_*.py`, which read a ruflo install from disk and never load it as Python.

---

## Migration path (ruflo → skill-hub)

```bash
# 1. Install / update skill-hub
pip install -U mcp-skill-hub
skill_hub --version

# 2. Run the importers (read-only against your ruflo install)
python scripts/import_ruflo_skills.py        # → ~/.skill_hub/skills/imported_ruflo/
python scripts/import_ruflo_agents.py        # → ~/.skill_hub/agents/

# 3. Smoke-test in a Claude Code session
#    - Parallel work: `/team implement <issue>` (orchestrates native subagents)
#    - An imported agent: Agent({ subagent_type: "ruflo-core:coder" })
#    - Shared state across machines: `federation_view(remote_db_path=...)`

# 4. Uninstall ruflo
npm uninstall -g @claude-flow/cli
tar -czf ~/.claude-flow-archive.tar.gz ~/.claude-flow/ && rm -rf ~/.claude-flow/
```

---

## When you might still want ruflo

- Cross-installation federation with strong identity / cryptographic signing (skill-hub's `federation_view` is intentionally a thin WAL+sync layer, not a protocol).
- Neural training / Q-Learning routing / multi-armed bandit at the swarm tier (skill-hub keeps the bandit at single-session model-routing only).
- Plugin marketplace with IPFS distribution (skill-hub's `suggest_plugins` is local-first).

If those aren't on your list, skill-hub plus Claude Code's native orchestration covers everything you'll use.
