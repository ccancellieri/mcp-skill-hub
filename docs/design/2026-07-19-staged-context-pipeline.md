# Staged context pipeline: prepare → enrich → compress → final

## Context
The hub prepares each prompt on cheap model tiers before the client's heavy/selected
model produces the final answer. The intended shape is a staged pipeline: **prepare** the
prompt, **enrich** it (skills, plugins, memory, tasks), **compress** the enriched payload,
then hand off to the **final** model (the client's own). This document specifies how to
realize that over the existing semantic-tier failover ladder.

## Decisions
1. **Stages run over the semantic-tier ladder.** Each LLM-backed stage asks for a cheap
   `tier_intent`; the escalation ladder picks whatever provider is reachable, so a stage
   still runs when the local backend is down. Stages are NOT pinned to physical providers —
   this keeps the semantic-tier model (cheap/mid/smart/planner) rather than reintroducing a
   fixed level-1/2/3 provider taxonomy.
2. **Add the missing compress stage** before handoff.
3. **Consolidate the two overlapping enrichment paths** into one canonical pipeline.
4. **Finish routing all LLM consumers** through the single `request(tier_intent)` entry.

## Current state (gap analysis)
| Stage | Status | Location |
|---|---|---|
| Prepare | partial | `router/rewriters.py`, `pipeline.py` rewrite stage |
| Enrich | done | `router/preloader.py` + `route()` assembly |
| Compress | **missing** | `compression/*` exists, never wired into assembly |
| Final | done (client model) | escalation ladder + `llm/request.request()` |

## WS-A — Compress stage
- **Insertion**: after the canonical enrichment assembly builds the system/user message,
  before returning it to the hook.
- **Trigger**: estimate tokens via `len(text)//4` (repo convention; no tokenizer exists).
  If over a configurable budget → compress.
- **Method**: deterministic `maybe_compress()` first (prod standard). Optionally the async
  digest (`compression/digest.digest_or_squeezed`, cached) for large prose blocks. No lossy
  ML (retired).
- **Config**: new `router_compress_context_enabled` (default true),
  `router_compress_budget_tokens` (start ~1500, tune). Reuse `compression_enabled` /
  `compression_min_tokens` gates.
- Reuses existing primitives only — **no new compression code**.
- **Tests**: over-budget block compresses; under-budget untouched; disabled → no-op;
  deterministic-only (never lossy); reversible markers intact.

## WS-B — Pipeline consolidation
The two enrichment paths fire on **different** hook events, so they are complementary, not
redundant — but they duplicate classification and one is dormant:
- `route()` (`router/route.py`) — runs on **every** UserPromptSubmit, default **ON**
  (`router_enabled=True`). Canonical: model routing, skill preload, thin-prompt enrich,
  compact advisor, orchestrator integration.
- `Pipeline` (`pipeline.py`) — runs only at **SessionStart**, gated by
  `pre_conversation_pipeline_enabled` (default **False** → dormant). Legacy module; its
  unique stage is context *synthesis* (+ session-task creation with embeddings).

**Decision**: `route()` is the single canonical pipeline. **Retire the dormant Pipeline A**
(`pipeline.py` + its `_run_pipeline()` call at `hooks/session_start_enforcer.py:738`) after
verifying nothing depends on `PipelineResult` or its SessionStart session-task creation. If
the session-task creation IS relied on (wake_session / session memory bootstrap), fold that
one piece into `route()`/session bootstrap rather than keeping the whole module.

The mandate's stages then live in `route()`: **prepare** (rewriters — make first-class
rather than thin-prompt-only) → **enrich** (preloader, done) → **compress** (WS-A, new) →
**final** (client model). Each LLM-backed stage requests a cheap `tier_intent` via the
ladder, so a stage still runs when the local backend is down.

**Risk**: retiring A only affects `pre_conversation_pipeline_enabled=True` setups (default
off). Confirm session-task-creation coverage before deleting.

## WS-C — Consumer unification (single entry point)
Entry: `llm/request.request(tier_intent, prompt, *, local_only=False, model, op, timeout,
temperature, max_tokens, cache, complexity, domain, ...)` → wraps `get_provider().complete()`
via the escalation ladder; returns `""` on `LLMError`. Already adopted by embeddings,
skill_evolution, plugin_curation, memory_supersede, compression/digest.

~12 bypass sites + 2 provider-direct. **Two migration patterns** (the key point — most
bypasses are deliberately local-only calls to the local backend in the command path, so
they must NOT silently escalate to paid tiers when it is down):
- **Local-fast, command-path → `request(..., local_only=True)`** (keep local-only, no
  escalation; preserve latency/cost): `cli.py` `_classify_intent`, `_match_local_command`,
  `_match_local_template`, `_llm_split`, `_hydrate_passthrough`; `router/ollama_client.py`
  `classify`.
- **Background enrichment/synthesis → `request(..., local_only=False)`** (walk the ladder,
  survive local-down — aligns with the mandate): `cli.py` `_update_repo_context`,
  `_extract_teaching_examples`, `_cmd_optimize_claude_md`, `_execute_local_skill`;
  `server.py` optimize_memory direct-complete.
- **Tricky multi-turn/iterative (may not fit single-shot `request()`)**: `local_agent.py`
  agentic loop, `_evolve_skills`, `_distill_tool_chains`. Migrate only if they fit cleanly;
  otherwise open a follow-up issue and leave consumer-unification partially done.

## Rollout (worktree lanes)
- **Lane 1 (core + router area)**: WS-B + WS-A + WS-D + the router's own classify migration
  + this design doc. Files: `router/route.py`, `router/ollama_client.py`, `pipeline.py`
  (delete), `hooks/session_start_enforcer.py`, `config.py`, compression usage, tests.
- **Lane 2 (cli/server area)**: WS-C consumer migration outside the router. Files: `cli.py`,
  `server.py`, `local_agent.py`, tests. Tricky agentic sites → follow-up issue if they don't
  fit single-shot `request()`.
- Non-overlapping files → parallel worktrees. **Merge Lane 1 first**, then rebase Lane 2.
  Each lane: worktree off `main`, full test suite green, draft PR.
