# Comparison: mcp-skill-hub vs ruflo (claude-flow)

## Why this document exists

`ruflo` (npm: `@claude-flow/cli`) and `mcp-skill-hub` overlap in roughly 20% of their surfaces — task tracking, semantic search, session persistence, hooks. The rest is disjoint. This page documents the comparison so a reader can decide whether to install both, neither, or just one.

**Project position**: skill-hub is the consolidated tool. The roadmap (milestone **M4**: `m4-ruflo-absorb`) reimplements the ruflo features the maintainer values as native skill-hub primitives. After M4 ships, ruflo is no longer needed alongside.

**Hard constraint**: skill-hub never runtime-depends on ruflo. Read the [no-ruflo-dep gate](#no-ruflo-runtime-dependency) below.

---

## What each is actually for

|  | **mcp-skill-hub** | **ruflo (claude-flow)** |
|---|---|---|
| Core purpose | Make a single interactive Claude Code session faster and smarter | Orchestrate many Claude/LLM agents at once |
| Scale | One session (you ↔ Claude) | Swarm of N agents working in parallel |
| Mode | Passive infrastructure | Active coordinator |
| Surface area | ~40 MCP tools, focused | 300+ MCP tools, broad |
| Required deps | Python + FastMCP + optional Ollama | Node 20+, optional API keys |
| Local-LLM-first | Yes — Ollama is a first-class provider | No — provider, but one of many |
| Cost to run | Free, local | Free for state plumbing; LLM calls = API bill |

---

## Side-by-side on overlapping features

| Feature | skill-hub | ruflo | Verdict |
|---|---|---|---|
| **Task save/reopen** | `save_task` / `list_tasks` — tied to *conversation* | `claims_claim` / `claims_board` — tied to *work item across agents* | Skill-hub for solo bookmarks; M1-4 adds claims-semantics natively |
| **Session restore** | `save_task` resumes context | `session_save` / `session_restore` resumes any agent | Both fine; skill-hub simpler |
| **Semantic search** | `search_skills`, `search_context` — over skills + memory | `memory_search_unified`, `embeddings_search` — over agent DB | Skill-hub faster (indexed locally); both LLM-optional |
| **Teaching rules** | `teach()` / `list_teachings` / `forget_teaching` — Claude-only | No equivalent | Skill-hub only |
| **Plugin routing** | `suggest_plugins`, `toggle_plugin`, profiles | No equivalent | Skill-hub only |
| **Token-saving** | `token_stats`, `optimize_context`, hook interception | Indirect via `agentdb_consolidate` | Skill-hub only |
| **Local LLM (Ollama)** | First-class: `list_models`, `pull_model`, dashboard | One provider among many | Skill-hub first-class |
| **Dashboard** | `/control` FastAPI suite | No native dashboard | Skill-hub only |
| **Parallel agents** | None today; **planned M4-1 (swarm-lite)** | `agent_spawn` × N, `swarm_init`, `hive-mind_*` | Closing in M4 |
| **Worktree isolation** | None today; **planned M1-6 (worktree-aware tasks) + M3-1 (preflight)** | `isolation: worktree` on `Agent` calls | Closing in M1 + M3 |
| **Headless autopilot** | None today; **planned M4-2 (autopilot-lite)** | `autopilot_enable` — work overnight | Closing in M4 |
| **Cross-machine federation** | None today; **planned M4-3 (federation-lite via WAL+sync)** | `ruflo federation init` | Closing in M4 |
| **Witness / fix manifest** | None today; **planned M1-5 (witness-log)** | `witness` skill (ADR-103) | Closing in M1 |
| **Performance / neural training** | None | `performance_*`, `neural_train`, `daa_*` | Out of scope for skill-hub |

---

## Where they actively conflict

- **Memory store**: running both means two indexes drifting apart. Pick skill-hub as source of truth. If a user has ruflo installed today, importers M4-4 (skills) and M4-5 (agents) move the content into skill-hub once, after which ruflo can be uninstalled.
- **Task tracking**: `list_tasks` vs `claims_board` diverge. M1-4 absorbs claims semantics into skill-hub's task table.

---

## Why skill-hub absorbs rather than bridges

A bridge means skill-hub runtime-depends on ruflo. That couples two release cycles, adds Node-on-Python install pain, and gives users no clean uninstall path. Absorbing means:

1. The ruflo capability becomes a native skill-hub primitive (subprocess.Popen for swarm, SQLite WAL for federation, etc.).
2. The user runs the importer once during migration.
3. The user uninstalls `@claude-flow/cli`.
4. Nothing in skill-hub still references ruflo.

This pattern is documented in milestone M4 (`m4-ruflo-absorb` label).

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

## Migration path (after M4 ships)

```bash
# 1. Confirm skill-hub has M4 features
pip install -U mcp-skill-hub
skill_hub --version

# 2. Run the importers (read-only against your ruflo install)
skill_hub import-ruflo-skills        # → ~/.skill_hub/skills/imported_ruflo/
skill_hub import-ruflo-agents        # → ~/.skill_hub/agents/

# 3. Smoke-test in a Claude Code session
#    - The ruflo-style swarm: `swarm_launch(...)` via skill-hub
#    - An imported agent: Agent({ subagent_type: "ruflo-core:coder" })
#    - Claims board: `claim_task` / `list_tasks --claimed`

# 4. Uninstall ruflo
npm uninstall -g @claude-flow/cli
tar -czf ~/.claude-flow-archive.tar.gz ~/.claude-flow/ && rm -rf ~/.claude-flow/
```

---

## When you might still want ruflo

- Cross-installation federation with strong identity / cryptographic signing (skill-hub's M4-3 federation is intentionally a thin WAL+sync layer, not a protocol).
- Neural training / Q-Learning routing / multi-armed bandit at the swarm tier (skill-hub keeps the bandit at single-session model-routing only).
- Plugin marketplace with IPFS distribution (skill-hub's `suggest_plugins` is local-first).

If those aren't on your list, M4 absorbs everything you'll use.
