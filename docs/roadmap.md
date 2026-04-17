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
- [x] **S4 F-ROUTE** — ε-greedy bandit over `tier_cheap` / `tier_mid` / `tier_smart` with Laplace smoothing; `route_to_model`, `record_model_reward`, `bandit_stats` MCP tools
- [x] **S5 F-PROMPT** — pluggable prompt rewriters (`add_skill_context`, `add_recent_tasks`, `normalize_language`); `improve_prompt` + `list_prompt_rewriters` MCP tools; opt-in hook integration
- [x] **S6 F-MEM** — unified sqlite-vec store for tasks + teachings (same binary-KNN + float32 rerank path as skills); mirror-on-write, delete-clean, startup backfill

---

## Upcoming

- [ ] **OpenSearch backend** — for scaling beyond local use
