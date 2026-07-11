# Roadmap

## Shipped ✅

### Core

- [x] Session profiles — predefined plugin sets per work context
- [x] Auto-profile — LLM recommends best profile for task description
- [x] RAG context injection — auto-enrich Claude's context with relevant skills / tasks / memory
- [x] Auto-eviction — relevance decay tracking + profile switch suggestions
- [x] Context compaction — periodic conversation digest via local LLM
- [x] Exhaustion fallback — local LLM auto-saves session when Claude is unavailable
- [x] Offline auto-fallback — TCP reachability check → auto-activates L4 agent when Claude unreachable
- [x] Universal LLM triage — local LLM pre-processes all messages, answers locally or enriches
- [x] Local execution engine — 4-level command / template / skill / agent execution with confirmation flow
- [x] Level 4 full agent — plan-first agent loop with tool calling (shell, skills, search, files)
- [x] Dual skill index — skills tagged `claude` / `local`, routed to the right LLM context
- [x] Inline help system — `?` lists commands, `?command` shows detailed usage
- [x] Activity logging — file + stderr, daily rotation, configurable log dir
- [x] Remote LLM support — Level 4 can route to a remote Ollama or OpenAI-compatible endpoint
- [x] Split LLM calls — lifecycle (temp=0) + prompt optimization (temp=0.2) as separate focused calls

### Performance

- [x] DB vector cache — in-process cache eliminates per-search JSON deserialization (~5× faster)
- [x] Feedback EMA — pre-aggregated score on skills table, no per-search O(N) scan
- [x] Implicit feedback — session-end correlates loaded skills vs tool usage, auto-records EMA signal
- [x] Auto memory on close_task — `smart_memory_write` runs after every task compaction
- [x] Training data export — JSONL export of all signal types for mlx-lm fine-tuning on Apple Silicon

### Sprints S1–S6

- [x] **S1 F-INDEX** — sqlite-vec binary-quant + float32 rerank (7.5× search speedup, 97.3% recall@5) + incremental hash-skip indexer
- [x] **S2 F-LLM** — `LLMProvider` Protocol + litellm adapter; unified 14+ call sites (embeddings, haiku, searxng, local_agent, ollama_router) behind one surface; `/control/llm` dashboard model picker + tier-aware pull form
- [x] **S3 F-SELECT** — named plugin profiles (`list_profiles`, `create_profile`, `switch_profile`, `auto_curate_plugins`) with SessionStart drift advisory
- [x] ~~**S4 F-ROUTE** — ε-greedy bandit over `tier_cheap` / `tier_mid` / `tier_smart` with Laplace smoothing; `route_to_model`, `record_model_reward`, `bandit_stats` MCP tools~~ — **removed in the #130 zero-usage purge** (issue #120 decided REMOVE: zero recorded rewards, nothing ever called `record_model_reward` outside crash-recovery replay). The `model_rewards` table was left behind as dead DDL and dropped offline via the #131 migration script.
- [x] **S5 F-PROMPT** — pluggable prompt rewriters (`add_skill_context`, `add_recent_tasks`, `normalize_language`); `improve_prompt` + `list_prompt_rewriters` MCP tools; opt-in hook integration
- [x] **S6 F-MEM** — unified sqlite-vec store for tasks + teachings (same binary-KNN + float32 rerank path as skills); mirror-on-write, delete-clean, startup backfill

---

## Upcoming

### M1 — Useful Without LLM

Visibility + pure-stdlib tools so skill-hub is obviously useful even with no Ollama / embedding backend.

- [x] #6 — `no-llm-mode`: explicit flag with visible status
- [x] #7 — tool-capability-matrix: every tool declares its dependency tier
- [x] #8 — degraded-search: FTS5 keyword fallback when embeddings unavailable
- [x] ~~#9 — claims-board: claim / handoff / steal on tasks (no LLM needed)~~ — **removed in the #130 zero-usage purge** (`claim_task` / `handoff_task` / `release_task` / `steal_task` + `src/skill_hub/team/` claims code had zero calls; `team_plan` itself was kept — it is a separate, live tool). Dead `claimed_by` / `claim_token` / `claimed_at` / `stealable_at` columns dropped offline in #131.
- [x] ~~#10 — witness-log: append-only fix manifest per repo~~ — **removed in the #130 zero-usage purge** (`record_witness` / `list_witness`, zero calls).
- [x] #11 — worktree-aware tasks: capture branch + worktree path on save
- [x] #12 — PII gate: regex scan before `save_task` / `teach` on public repos
- [x] #13 — dashboard: `/status/capabilities` view

### M2 — Managed-Agents architectural refactor (design phase)

Selectively apply patterns from Anthropic's Managed Agents post — durable event log, stateless recovery, uniform tool envelope, credential vault, optional sandbox.

- [x] #14 — tracking issue + `docs/design/managed-agents-refactor.md` (Q1/Q4/Q5 resolved)
- [x] #27 — W1 event log: `events` table + emit decorator + `get_events` / `events_prune`
- [x] #28 — W2 stateless recovery: `wake_session` + cache-rebuild discipline
- [x] #29 — W3 uniform tool envelope: `ToolResult` + wrapping decorator
- [x] #30 — W4 credential vault: keyring + 3-tier backend + config→vault migration
- [x] ~~#31 — W5 sandbox interface: `provision()` + subprocess backend for plan-execution tools~~ — **shipped, then removed (PR #52).** It guarded the in-process plan-execution stepper (`author_plan` / `run_plan` / `execute_plan_step`), which has itself been retired in favour of Claude Code's native Workflow tool and `/team` subagents — those run in their own harness-managed worktrees, so an in-process sandbox no longer has anything to wrap. `validate_plan` (lint-only) survived PR #52, but was itself removed in the later #130 zero-usage purge (`plan_executor/validator.py`, zero calls) — no plan-lint tool remains today.

### M3 — Worktree + multi-repo policy enforcement

Move maintainer feedback rules from memory into callable skill-hub primitives.

- [x] #15 — `worktree_preflight`: collision check tool (3-axis: worktree + branch + open PR)
- [x] ~~#16 — `sync_check`: cross-repo stale-import detector (git diff + grep, no false positives)~~ — **removed in the #130 zero-usage purge** (`federation_view` / `sync_check` + `sync_check.py`, zero calls).
- [x] ~~#17 — `lint_canary`: rotate through ruff selectors (core_task + MCP tool, witness-log JSONL)~~ — **removed in the #130 zero-usage purge** alongside the witness log it logged to.
- [x] ~~#18 — memory-rule export: feedback files → per-repo `POLICY.md` (`export_policies` MCP tool)~~ — **removed in the #130 zero-usage purge** (`policy_export.py` + `team/policy.py`, zero calls).
- [x] #19 — cross-project task federation: per-repo filter on every task tool

### Other

- [ ] **OpenSearch backend** — for scaling beyond local use
