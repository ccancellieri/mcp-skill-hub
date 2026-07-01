"""Configuration management for Skill Hub.

Config file: ~/.claude/mcp-skill-hub/config.json

Users can override model selection based on their hardware.
Default config is optimized for minimal resource usage.
"""

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "config.json"

# Defaults — minimal footprint, works on any machine
_DEFAULTS = {
    # Ollama connection
    "ollama_base": "http://localhost:11434",

    # Multi-endpoint Ollama — priority-ordered list of endpoints.
    # Each entry: {name, url, priority, enabled, auth_header (optional)}
    # When empty or absent, ollama_base is used as the single endpoint.
    "ollama_endpoints": [],

    # Per-model endpoint routing: model name → list of endpoint names (priority order).
    # Empty dict = use global endpoint priority for all models.
    "ollama_model_routing": {},

    # No-LLM mode (issue #6) — explicit opt-in to short-circuit every embed
    # / Ollama / reason-LLM probe. With this on, ``embed_available()`` and the
    # capability matrix report the LLM-dependent backends as missing without
    # any network/disk probing, so the user sees the same "29/40 tools
    # available" verdict shown in `status` and on the dashboard banner.
    "no_llm_mode": False,

    # Embedding backend cascade
    "embedding_backend": "auto",  # auto | ollama | sentence_transformers
    "embedding_backend_priority": ["ollama", "sentence_transformers"],
    # embedding_fallback_on_error removed — cascade always raises RuntimeError on total failure
    "sentence_transformers_model": "all-MiniLM-L6-v2",

    # LLM backend config (for pipeline tiers, replacing the old Ollama-only approach)
    "classify_backend": "haiku_json",    # haiku_json | yake_keywords | ollama_qwen
    "synthesis_backend": "haiku",        # haiku | sonnet | ollama_local
    "rewrite_backend": "sonnet",         # sonnet | haiku | none
    "rerank_backend": "none",            # none | cohere | jina_local | ollama

    # Bundled (in-tree) plugins under <repo>/plugins/* are auto-discovered
    # at startup unless this is set to False.
    "bundled_plugins_enabled": True,

    # Embedding model — used for all vector operations
    # Options: nomic-embed-text (274MB), mxbai-embed-large (669MB), all-minilm (45MB)
    "embed_model": "nomic-embed-text",

    # Reasoning model — used for re-ranking, compaction, intent classification
    # Options by hardware:
    #   - 8GB RAM:  deepseek-r1:1.5b (1.1GB) — fast, basic reasoning
    #   - 16GB RAM: deepseek-r1:7b (4.7GB) or qwen2.5:7b — good balance
    #   - 32GB RAM: deepseek-r1:14b (9GB) or qwen2.5:14b — best quality
    #   - 64GB+:    deepseek-r1:32b or qwen2.5-coder:32b — excellent
    "reason_model": "deepseek-r1:1.5b",

    # Hook behavior
    "hook_timeout_seconds": 45,         # max time for hook LLM classification
    "hook_enabled": True,               # enable/disable UserPromptSubmit hook
    "hook_semantic_threshold": 0.45,    # min embedding similarity to trigger LLM classify
                                        # lower = more messages reach LLM (more sensitive)
                                        # raise to 0.55+ if you get false positives with small models
    "hook_max_message_length": 2000,    # messages longer than this skip LLM classify entirely
    "token_profiling": True,            # track estimated token savings per interception
    "log_tool_usage": True,             # log each tool Claude calls to activity.log (TOOL lines)
    "tool_usage_codegraph_hint": True,  # flag grep/rg use in .codegraph-indexed repos

    # Task command examples — used to build the semantic centroid for the
    # embedding prefilter. Add your own phrases in any language.
    # The centroid is recomputed whenever the list changes.
    "hook_task_command_examples": [
        "save to memory",
        "save this task",
        "park this for later",
        "remember this discussion",
        "save and close",
        "close task",
        "done with this",
        "mark as done",
        "I'm done here",
        "what was I working on",
        "show my open tasks",
        "list tasks",
        "what did we discuss about",
        "find my previous work on",
        "search my past work",
    ],

    # Context injection — auto-enrich Claude's context with relevant skills/tasks/memory
    "hook_context_injection": True,     # enable RAG + auto-memory injection
    "hook_context_max_chars": 40000,    # total budget for systemMessage (~10k tokens)
    "hook_context_max_skill_chars": 8000,  # max chars per skill (truncated if larger)
    "hook_context_top_k_skills": 5,     # max skills to load with full content per message
    "hook_context_min_skills": 3,      # min skills to load (auto-fill from RAG if LLM picks fewer)
    "hook_precompact_threshold": 1500,  # messages longer than this get pre-compacted
    # Pre-compact a long input deterministically (extractive Kompress, no local LLM)
    # by default — the condensed view is consumed by the cloud model, which tolerates
    # extractive text. Flip True to restore the legacy abstractive local-LLM digest.
    "precompact_use_llm": False,

    # Search defaults
    "search_top_k": 3,                  # default number of results
    "search_similarity_threshold": 0.3, # minimum cosine similarity
    "feedback_boost_max": 1.5,          # maximum feedback boost factor
    "teaching_min_similarity": 0.6,     # minimum sim for teaching rule match

    # CodeGraph enrichment in search_context — deterministic, on by default.
    # search_context appends a bounded symbol block from the project's CodeGraph
    # index when one exists at the active repo root. The lookup is LLM-free
    # (pure index query), bounded (top-k + char cap), cached, and never raises,
    # so it is a no-op for repos without a ``.codegraph/`` index.
    "search_context_use_codegraph": True,
    # Repos whose CodeGraph index the ``codegraph-sync`` cron job keeps fresh
    # (incremental ``codegraph sync``). Empty → the job is a cheap no-op. Add
    # absolute repo roots (each must already be ``codegraph init``-ed).
    "codegraph_reindex_roots": [],
    # Per-repo wall-clock cap for one incremental sync in the cron handler.
    "codegraph_reindex_timeout_seconds": 120,

    # Compaction
    "compact_max_input_chars": 4000,

    # Deterministic compression (optional `headroom-ai` dependency) — structure-aware
    # reduction of tool outputs / logs / JSON / search results as a free, fast,
    # offline alternative to the local-LLM compaction tax and to blunt char-truncation.
    # Auto-no-ops when `headroom-ai` is not installed. Prose/code are left for the LLM.
    "compression_enabled": True,        # master switch for the compression pre-stage
    "compression_min_tokens": 200,      # skip payloads below ~this token count
    "compression_context_aware": True,  # pass the user query as relevance context
    # Context-window headroom awareness — scale the effective compression threshold
    # based on current system pressure (from resource_monitor.snapshot()).  When ON,
    # HIGH pressure lowers the threshold (compress more eagerly); IDLE/LOW keeps the
    # configured value.  OFF by default: behaviour is byte-identical to the previous
    # release until the user opts in.
    "compression_headroom_aware": False,
    # Lossy ML path — Kompress (ModernBERT) extractive prose compression. ON by
    # default after eval (scripts/compression_eval.py): avg ratio 0.60, avg
    # embedding-fidelity 0.87. It deletes low-salience tokens — LOSSY and (for prose)
    # irreversible. Requires the `compression_full` extra (headroom-ai[ml]); auto-
    # no-ops silently without it. To install: `uv pip install mcp-skill-hub[compression_full]`
    # or `pip install headroom-ai[ml]`. token_stats() will report "no-op" when the
    # extra is missing even though this flag is True.
    # NOTE: search-result/memory-like text compresses at ~0.79 fidelity (marginal);
    # raise compression_ml_target_ratio toward 1.0 for higher fidelity / less saving,
    # or set compression_ml_enabled=False to revert to lossless-only.
    "compression_ml_enabled": True,         # Kompress (ModernBERT) prose compression
    # code-aware (tree-sitter AST) stays OFF: the eval showed it never fires on real
    # tool output (headroom routes code to Kompress first), so it adds no value here.
    "compression_code_aware_enabled": False,  # tree-sitter AST code compression
    "compression_ml_target_ratio": 0.6,     # Kompress target size (compressed/original)

    # Local LLM metering — record latency + token throughput for every Ollama call.
    # Surfaces in token_stats() and the System Health dashboard card.
    "llm_metering_enabled": True,           # master switch for per-call LLM metering

    # Extended prompt-cache TTL — request Anthropic's ~1h cache window (vs the
    # default ~5m) for long-lived reused prefixes (session memory, master state).
    # Off by default: the extended tier needs a recent litellm/SDK; enabling it
    # on an unsupported stack could reject the cache_control. Opt in once
    # confirmed. Stabilizing the prefix itself (deterministic ordering of
    # injected skills/plugins) is always on and independent of this flag.
    "llm_cache_extended_ttl": False,

    # Conversation digest — periodic context compaction.
    # Deterministic-first: by default the periodic digest condenses recent messages
    # extractively (Kompress, no local LLM). The richer abstractive digest — which
    # also infers stale topics + a profile-switch suggestion — runs only when the
    # eviction feature needs it (eviction_enabled) or when forced via digest_use_llm.
    "digest_every_n_messages": 5,       # produce a digest every N messages
    "digest_stale_threshold": 0.3,      # similarity below this = "stale" topic
    "digest_use_llm": False,            # force the abstractive local-LLM digest

    # Auto-eviction — relevance decay tracking. OFF by default: profile-switch
    # suggestions require the abstractive local-LLM digest (it infers the target
    # profile), so keeping eviction opt-in preserves the deterministic-first /
    # cost-cut posture. Flip True to re-enable stale-topic + profile-switch hints.
    "eviction_enabled": False,          # enable relevance decay tracking
    "eviction_min_stale_count": 3,      # suggest profile switch after N stale detections

    # Plan-executor API fallback — when authoring plans, the runner chain
    # normally tries in_session → claude -p → SDK first (all via Max/OAuth
    # subscription, no token cost). Set this to True ONLY if you explicitly
    # want the litellm/API-token fallback as a last resort. Requires
    # ANTHROPIC_API_KEY to also be set.
    "plan_api_runner_enabled": False,   # keep disabled on Max plan

    # Exhaustion fallback — local LLM takes over when Claude is unavailable
    "exhaustion_fallback": True,        # enable exhaustion auto-save

    # Offline auto-fallback — automatically activate L4 local agent when
    # api.anthropic.com is unreachable (rate limit, network outage, travel)
    "offline_auto_fallback": False,       # disabled: L4 agent unreliable with small models
    "offline_check_interval": 30.0,    # seconds between reachability checks

    # Implicit feedback — infer skill quality from session tool usage at session-end
    # Skills whose domain keywords match the tools Claude actually used → positive
    # Skills loaded but completely unrelated to what Claude did → negative
    "implicit_feedback_enabled": True,

    # Auto memory on close_task — run smart_memory_write after every task compaction
    # Silently skips if the local LLM judges the digest too thin to be worth saving
    "auto_memory_on_close_task": True,

    # Continuous teaching — Phase G.2
    # When enabled: feedback_*.md file writes auto-teach rules, and session-start
    # messages matching "remember X" / "never do X" / "always do X" are auto-taught.
    "continuous_teaching_enabled": False,

    # Semantic response cache — reuse cached answers for near-identical questions
    "response_cache_enabled": True,
    "response_cache_min_sim": 0.88,   # min similarity to serve from cache
    "response_cache_verify": False,   # embedding similarity alone is sufficient at 0.93+

    # Task decomposition — break complex multi-part requests into ordered subtasks
    "task_decomposition_enabled": False,  # disabled: LLM decomposition unreliable with small models
    "task_decomposition_min_len": 300,  # only decompose messages longer than this

    # Prompt pattern tracking — learn from recurring message shapes
    # When a pattern recurs ≥ threshold times, auto-generate a local skill
    "pattern_tracking_enabled": True,
    "pattern_auto_skill_threshold": 5,  # recurrences before auto-generating a skill

    # Context Bridge — capture AI tool calls + build local intelligence
    # Captures Claude's (or any AI's) tool calls from the transcript, stores
    # them in DB, surfaces them to local LLMs via {session_context},
    # {tool_examples}, {repo_context}, {tool_patterns} built-in variables.
    "context_bridge_enabled": True,
    "context_bridge_max_capture_per_hook": 20,   # max tool calls per Stop hook
    "context_bridge_prune_days": 30,             # prune examples older than this
    "context_bridge_prune_max_rows": 5000,       # max total rows in tool_examples
    "context_bridge_teaching_extraction": False, # disabled: LLM extraction unreliable with small models
    "context_bridge_repo_context": True,         # maintain per-repo context summaries

    # Skill Evolution — shadow learning from Claude (or any AI)
    # At session end, compares local skill output with Claude's tool usage.
    # If Claude's approach is better, evolves the skill's prompts/steps.
    # Old versions are stored in skill_versions table for rollback.
    "skill_evolution_enabled": True,
    "skill_evolution_auto": False,              # auto-evolve ALL skills (not just shadow:true)
    "skill_evolution_max_per_session": 3,       # max skills to evolve per session
    "skill_evolution_min_session_msgs": 5,      # min session messages before evolving
    "skill_evolution_feed_memory": True,        # include project memory in evolution context
    "skill_evolution_cross_pollinate": True,    # reference official Claude skills during evolution
    "skill_sync_on_index": True,               # check for plugin updates when indexing

    # Pre-conversation pipeline — 4-tier enrichment (L1-L4)
    "pre_conversation_pipeline_enabled": False,  # opt-in
    "pipeline_tier1_timeout_ms": 500,
    "pipeline_tier2_timeout_ms": 400,
    "pipeline_tier3_timeout_ms": 1200,
    "pipeline_tier4_timeout_ms": 1500,
    "pipeline_tier4_min_complexity": "medium",   # low | medium | high
    "task_similarity_threshold": 0.75,
    "task_auto_create_min_chars": 0,             # every conversation
    "pipeline_synthesis_max_sentences": 5,

    # Session → task auto-bind (resume-or-create on every new session)
    "session_task_auto_create_enabled": True,          # master kill switch
    "session_task_match_strategy": "hybrid",           # hybrid | cwd_branch | semantic | off
    "session_task_match_window_days": 7,               # only resume tasks touched in last N days
    "session_task_semantic_threshold": 0.75,           # min cosine for semantic match

    # Model/effort recommendation — inject hints based on task complexity
    "model_recommendation_enabled": True,       # inject model/effort hints in systemMessage
    "always_forward_to_claude": True,           # NEVER block — always forward to Claude

    # Resource-aware LLM gating — skip expensive local LLM ops under pressure
    # Pressure levels: idle(0), low(1), moderate(2), high(3)
    # Each operation has a max pressure level at which it still runs.
    # Set to false to disable resource gating entirely (always run all LLM ops).
    "resource_gating_enabled": True,
    "resource_cache_ttl_seconds": 10,   # how often to re-check system resources

    # PostCompact memory optimisation — triggered by the PostCompact hook after
    # /compact summarises the context.  Compaction is a natural pruning moment.
    # ``postcompact_optimize_apply`` True = mutate the store; False = dry-run only.
    # ``postcompact_pressure_max``   ceiling for the postcompact path (LOW is more
    #   permissive than the background IDLE gate used by nightly promote_memory).
    "postcompact_optimize_apply": True,
    "postcompact_pressure_max": "LOW",

    # Router teachings — consult matching teachings when preloading skills so
    # learned "when X → do Y" rules are surfaced to the model.
    "router_use_teachings": True,

    # Session-end L0→L1 promotion — run a lightweight memory-promotion pass
    # at session end (guards: config flag + HIGH-pressure skip).
    "session_end_promote": True,

    # LLM triage — local LLM pre-processes ALL messages before Claude
    "hook_llm_triage": False,           # disabled: small models can't classify reliably, blocks real work
    "hook_llm_triage_timeout": 30,      # max seconds for triage LLM call
    "hook_llm_triage_min_confidence": 0.7,  # min confidence to act on local_answer
    "hook_llm_triage_skip_length": 2000,    # messages longer than this skip triage

    # ── Local execution levels ──────────────────────────────────────
    # Level 1: Safe shell whitelist — predefined commands, no params
    # Level 2: Templated shell — commands with LLM-extracted parameters
    # Level 3: Local skill execution — LLM follows multi-step skill scripts
    # Level 4: Full local agent — tool-using agent loop (local or remote LLM)
    "local_execution_enabled": False,    # disabled: blocks commands from reaching Claude

    # Model per level — heavier models for harder tasks
    # Values: Ollama model name, or "remote:<base_url>" for external APIs
    "local_models": {
        "level_1": "qwen2.5-coder:3b",           # simple command mapping
        "level_2": "qwen2.5-coder:7b-instruct-q4_k_m",  # parameter extraction
        "level_3": "qwen2.5-coder:14b",           # skill following
        "level_4": "qwen2.5-coder:32b",           # or "remote:http://host:11434"
    },

    # Level 1: whitelisted shell commands (name → command)
    "local_commands": {
        "git_status": "git status",
        "git_log": "git log --oneline -20",
        "git_diff_stat": "git diff --stat",
        "git_diff": "git diff",
        "git_branch": "git branch -a",
        "git_stash_list": "git stash list",
        "git_remote": "git remote -v",
        "ls": "ls -la",
        "pwd": "pwd",
        "df": "df -h",
        "uptime": "uptime",
    },

    # Level 2: templated commands (name → template with {param} placeholders)
    # Allowed params are extracted by LLM from the user message
    "local_templates": {
        "git_log_n": {"cmd": "git log --oneline -{n}", "params": {"n": "int"}},
        "git_diff_file": {"cmd": "git diff {file}", "params": {"file": "path"}},
        "git_show": {"cmd": "git show {ref}", "params": {"ref": "str"}},
        "git_checkout": {"cmd": "git checkout {branch}", "params": {"branch": "str"}},
        "git_add": {"cmd": "git add {file}", "params": {"file": "path"}},
        "cat_file": {"cmd": "cat {file}", "params": {"file": "path"}},
        "grep_pattern": {"cmd": "grep -rn {pattern} .", "params": {"pattern": "str"}},
    },

    # Level 4: remote model endpoint (used when local_models.level_4 starts with "remote:")
    "remote_llm": {
        "base_url": "",            # e.g. "http://myserver:11434" or OpenAI-compatible URL
        "api_key": "",             # for authenticated endpoints
        "model": "",               # model name at the remote endpoint
        "timeout": 120,
    },

    # Level 3: local skills directory — step-based .json skills executed by local LLM
    # Defaults to ~/.claude/local-skills/ — can be separated from Claude skills
    # to optimize local skills for local models (recommended)
    "local_skills_dir": str(Path.home() / ".claude" / "local-skills"),

    # Local LLM persona — optional static bio seed prepended to all local LLM calls.
    # The full dynamic persona is assembled from this + teachings + closed tasks + memory.
    # Example: "Python/FastAPI developer. Projects: geoid (STAC catalog), glicemia (T1D bot)."
    "local_system_prompt": "",
    "local_persona_ttl_seconds": 120,   # seconds before persona is rebuilt from store
    "local_persona_max_chars": 600,     # hard cap on assembled persona string

    # SearXNG web search — Stage 4.1 RAG fallback when skill search returns nothing
    # Auto-detects localhost:8989 if searxng_url is empty.
    # The VPS SearXNG URL can be set explicitly (e.g. "http://vps-ip:8989").
    "searxng_url": "",         # explicit URL (VPS or custom) — empty = auto-detect
    # searxng_enabled moved to services.searxng.enabled (see bottom of file)
    "searxng_top_k": 3,
    "searxng_timeout": 5,      # seconds for URL probe (is SearXNG reachable?)
    "searxng_search_timeout": 15,  # seconds for actual search (engines need time)
    # How to condense fetched web results before injecting them as context.
    # Default "kompress": fast, light, LLM-free extractive compression (ModernBERT) —
    # grounded (no hallucination), no Ollama dependency. Set True to use the legacy
    # local-LLM (Ollama) abstractive summary instead (fluent but slow + can invent facts).
    "searxng_use_llm_summary": False,

    # Activity log — daily rotation, 50 MB cap
    # Set to a custom path to redirect logs (e.g. "/tmp/skill-hub-logs")
    "log_dir": str(Path.home() / ".claude" / "mcp-skill-hub" / "logs"),

    # hook-debug.log — size-based rotation (the many hook writers append to it
    # directly, so it is not covered by the activity-log TimedRotatingFileHandler).
    # Rotated at MCP server start / session end when over the cap; oldest pruned.
    "hook_log_max_bytes": 10 * 1024 * 1024,  # rotate when over 10 MB
    "hook_log_keep": 3,                       # keep this many rotated files

    # ── Prompt Router ───────────────────────────────────────────────────────
    # Three-tier classifier: Tier-1 heuristics → Tier-2 Ollama → Tier-3 Haiku
    # Fired on every UserPromptSubmit via hooks/prompt-router.sh

    # Master kill-switch (env SKILL_HUB_ROUTER_ENABLED=0 also works)
    "router_enabled": True,

    # Confidence threshold above which the router hard-switches the model
    # in settings.json. Below: soft suggestion only.
    "router_hard_switch_threshold": 0.9,

    # JSONL audit log — one line per prompt, for analyze_router_log tool
    "router_log": str(Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"),

    # Tier 2: local Ollama classifier — moved to services.ollama_router.model
    # env SKILL_HUB_ROUTER_OLLAMA_MODEL still overrides it at runtime
    # Tier-1 confidence below this triggers Tier-2 escalation
    "router_tier2_confidence_gate": 0.85,
    # Max seconds for Tier-2 Ollama call (routing must be fast)
    "router_tier2_timeout": 10.0,

    # S4 F-ROUTE — ε-greedy bandit over tier_cheap/tier_mid/tier_smart
    "router_bandit_enabled": True,
    "router_bandit_epsilon": 0.1,

    # S5 F-PROMPT — pluggable prompt rewriters. When enabled, the UserPromptSubmit
    # hook runs the default_chain and appends the enrichment to the userMessage.
    "router_improve_prompt_enabled": False,
    "improve_prompt_default_chain": ["add_skill_context", "add_recent_tasks"],
    "improve_prompt_skill_top_k": 3,
    "improve_prompt_tasks_limit": 2,

    # G2 — proactive tool steering. When the working repo is codegraph-indexed
    # and the prompt shows search/grep intent, nudge Claude toward codegraph
    # queries + compact shell output (saves context vs. raw grep dumps).
    # Intent-gated, so it never fires on unrelated prompts (no per-prompt noise).
    "tool_steering_enabled": True,

    # Tier 3: Claude Haiku 4.5 batched call (opt-in)
    # Enable via /control or env SKILL_HUB_ROUTER_HAIKU=1; stored as services.haiku_router.enabled
    # Requires ANTHROPIC_API_KEY in environment
    # Tier-2 confidence below this escalates to Tier-3 Haiku
    "router_haiku_threshold": 0.7,
    # Individually toggle each Haiku batch task (all default on when Haiku fires)
    "router_haiku_classify": True,        # classification: complexity/ambiguity/scope
    "router_haiku_settings_opt": True,    # config optimisation suggestion
    "router_haiku_compact_hint": True,    # /compact recommendation
    "router_haiku_subtask_decomp": True,  # multi-part prompt decomposition

    # Per-session routing-verdict cache (issue #88).
    # When enabled, the stable skills/plugins block is persisted to disk per
    # session and replayed byte-identical each turn, forming a cacheable prefix
    # for the provider's prompt cache.  Default OFF — zero behaviour change.
    "router_verdict_cache_enabled": False,
    # Maximum number of messages before the cached block is recomputed.
    "router_verdict_cache_max_messages": 20,
    # Wall-clock TTL (seconds) after which the cached block is recomputed.
    "router_verdict_cache_ttl_secs": 1800,

    # Auto-compact advisor: inject /compact suggestion when context is estimated
    # to be ≥ this fraction full (based on session message count)
    "router_compact_threshold": 0.70,

    # Thin-prompt enrichment: for very short messages (<60 chars), prepend
    # context from the active task to help Claude respond without clarifying
    "router_enrich_thin_prompts": True,

    # User memory bridge — auto-index ~/.claude/projects/*/memory/*.md into
    # the memory:user-project namespace so search_context surfaces user notes
    # alongside skills and tasks. Set to False to disable.
    "user_memory_enabled": True,

    # Session memory compaction — background 6-section summary of each Claude
    # Code session (ported from anthropic/claude-cookbooks
    # misc/session_memory_compaction.ipynb). Persists to
    # ~/.claude/mcp-skill-hub/session-memory/<session_id>.md and is injected
    # as systemMessage on resume (survives /compact).
    "session_memory_enabled": True,
    # Minimum message count to trigger the first build (cheap guard — avoids
    # summarising 2-message test sessions).
    "session_memory_min_messages": 6,
    # Cap the transcript slice handed to the LLM (bytes). ~200 KB covers a
    # long session without blowing the Haiku 4.5 context window.
    "session_memory_max_transcript_bytes": 200_000,
    # Tier for the summariser — tier_mid = Haiku 4.5 (cheap + structured).
    "session_memory_tier": "tier_mid",
    # Inject the stored memory on the first prompt of a resumed session.
    "session_memory_inject_on_resume": True,
    # Cap how many characters of memory are injected as systemMessage.
    "session_memory_inject_max_chars": 8000,

    # Tooling orchestrator — per-turn tool-readiness steering.
    # ``orchestrator_mode`` is the single source of truth for behaviour:
    #   "off"        — disabled; evaluate() returns nothing.
    #   "offer"      — surface offer/steer directives; never auto-provision.
    #   "auto"       — auto-init projects whose root is under an auto-init folder.
    #   "everywhere" — auto-init any unindexed code project explored.
    # When the key is absent it is derived from the legacy booleans below, so
    # existing configs keep their behaviour. ``orchestrator_auto_init_roots`` is
    # the list of parent folders enabling auto-init in "auto" mode (prefix match).
    "orchestrator_mode": None,              # off | offer | auto | everywhere (None → derive from legacy keys)
    "orchestrator_enabled": True,           # (legacy) master switch — derives mode when mode unset
    "orchestrator_auto_init": False,        # (legacy) global auto-init — derives mode when mode unset
    "orchestrator_auto_init_roots": [],     # parent folders that may auto-init (prefix match in "auto" mode)
    "orchestrator_sync_ttl_secs": 300,      # min interval between auto-refreshes (seconds)
    "orchestrator_probe_cache_secs": 60,    # probe-result cache TTL (keeps the hook fast)

    # Issue #37 — task↔issue bidirectional sync writeback mode.
    # Controls whether skill-hub writes back to GitHub when a locally-closed task
    # has an open linked issue.  Valid values:
    #   "off"     — (default) never write to GitHub; safe for read-only use.
    #   "comment" — post a completion comment on the linked issue (idempotent).
    #   "close"   — post a comment and close the linked issue.
    "task_issue_writeback": "off",

    # M2 W1 — event log retention.
    # Raw events for closed sessions older than this many days are coalesced
    # into a single session_snapshot row and deleted.  Set to 0 to keep forever.
    "event_log_retention_days": 30,

    # Discussions write path (issue #87) — opt-in only, default OFF.
    # When False, create_discussion() is a no-op that returns {"status": "disabled"}.
    "discussions_write_enabled": False,
    # Category name to use when creating discussions via create_discussion().
    # Resolved to a category id at write time via the GraphQL discussionCategories query.
    "discussions_category": "General",
    # Repo used by the discussions-sync-nightly cron job ("owner/name").
    # Empty string = resolve from current working directory at run time.
    "discussions_repo": "",

    # Base-config self-heal — automatically re-apply missing hooks / MCP
    # registration / base-roles block at session start.  Disable only if you
    # manage those surfaces manually.
    "auto_repair_base_config": True,

    # Continuous memory sweep — periodic background promote_memory pass.
    # Runs only when the machine is IDLE (Pressure.IDLE); skipped under any load.
    # Default OFF: enable explicitly once the machine's idle profile is understood.
    "continuous_sweep_enabled": False,
    # Minimum minutes between sweep runs.  A sweep that ran (or was skipped due
    # to pressure) within this window will not attempt again until it expires.
    "continuous_sweep_interval_minutes": 60,

    # Cron scheduler — background jobs driven by cron_jobs table
    "cron_jobs_enabled": True,

    # Background job queue — deferred work via subagent / litellm / Ollama
    "background_via_subagent_enabled": False,       # opt-in
    "background_worker_priority": ["subagent", "litellm", "ollama", "defer"],
    "background_subagent_idle_threshold_ms": 3000,
    "background_max_jobs_per_prompt": 1,
    "background_job_retry_max": 3,

    # Task activity state thresholds — used by get_task_activity_state()
    "task_activity_active_seconds": 60,    # last_activity_at within 60s → "active"
    "task_activity_idle_seconds": 3600,    # within 60min → "idle", else "open"

    # Vector engine — "sqlite-vec" uses the native ANN extension with binary
    # quantization + float32 rerank; any other value falls back to the legacy
    # in-Python cosine path.
    "vec_engine": "sqlite-vec",
    "binary_quant_enabled": True,
    # Binary KNN candidate pool size before float32 rerank. 60 yields ~97%
    # recall@5 parity vs float32 on 1914 skills; tune up for more recall.
    "rerank_top_k": 60,

    # S2 — pluggable LLM providers. Model strings use litellm syntax:
    #   "ollama/<model>" for local Ollama
    #   "anthropic/<model>" (requires ANTHROPIC_API_KEY)
    #   "openai/<model>"    (requires OPENAI_API_KEY)
    # tier_cheap: used for low-stakes reasoning (rerank, classify, summarise)
    # tier_mid:   used for batched router escalation
    # tier_smart: used for hard multi-step planning / final synthesis
    "llm_providers": {
        "tier_cheap":   "ollama/qwen2.5-coder:3b",
        "tier_mid":     "anthropic/claude-haiku-4-5",
        "tier_smart":   "anthropic/claude-sonnet-4-6",
        # tier_planner: strongest model for authoring/design. Used by
        # plan_executor's API-fallback runner; in-session runner prefers
        # the active Claude Code agent (user's chosen model).
        "tier_planner": "anthropic/claude-opus-4-8",
        "embed":        "ollama/nomic-embed-text",
    },
    # Default tier when code doesn't specify one.
    "llm_default_tier": "tier_cheap",

    # S2b — auxiliary LLM escalation ladder (issue #117). Ordered provider
    # registry; the escalation engine walks it by order → availability →
    # model complexity → remaining quota. SECRET-FREE defaults: api_base is
    # empty and api_key references an opencode provider id or an env var — the
    # actual endpoint/key live only in the user's local opencode config / env.
    #
    # ADDING A REMOTE PROVIDER — no code changes needed; add a row here (or via
    # the /providers page). Any endpoint reachable over HTTP works:
    #
    #   Remote Ollama (kind: "ollama"):
    #     {"name": "my-remote-ollama", "level": "L2", "kind": "ollama",
    #      "api_base": "http://ollama.example.internal:11434",
    #      "api_key": {},
    #      "enabled": true, "order": 20,
    #      "models": [{"id": "ollama/qwen2.5-coder:7b", "complexity": "heavy",
    #                  "tags": ["programming"]}]}
    #
    #   OpenAI-compatible gateway (kind: "openai_compatible"):
    #     api_key source options: "env" (ref = env-var name holding the key),
    #     "opencode" (ref = opencode provider id), or "inline" (ref = key value,
    #     NOT for tracked configs). Model ids must start with "openai/" so
    #     litellm routes through the configured api_base.
    #
    #   Anthropic or Anthropic-compatible (kind: "anthropic"):
    #     Set api_key.source="env", api_key.ref="<ENV_VAR_NAME>".
    #     api_base may be left empty for the public endpoint.
    #
    # For embeddings, a remote Ollama endpoint is also added to the separate
    # ollama_endpoints list so the multi-endpoint client can serve them:
    #   {"name": "remote", "url": "http://ollama.example.internal:11434",
    #    "priority": 2, "enabled": true}
    "llm_provider_registry": [
        {"name": "local-ollama", "level": "L1", "kind": "ollama",
         "api_base": "", "api_key": {}, "enabled": True, "order": 10,
         "models": [{"id": "ollama/qwen2.5-coder:3b", "complexity": "light",
                     "tags": ["fast", "digest", "programming"]}]},
        # An OpenAI-compatible gateway. Leave models empty here (secret-free
        # default); configure base/key/models per-user via the /providers page
        # or by pointing api_key.ref at an opencode provider id. Per-model
        # "tags" drive specialisation routing (e.g. ["python"], ["web", "ui-ux"]).
        {"name": "work-gateway", "level": "L3", "kind": "openai_compatible",
         "api_base": "", "api_key": {"source": "opencode", "ref": ""},
         "enabled": False, "order": 30, "models": []},
        {"name": "personal-claude", "level": "personal", "kind": "anthropic",
         "api_base": "", "api_key": {"source": "env", "ref": "ANTHROPIC_API_KEY"},
         "enabled": True, "order": 90,
         "models": [{"id": "anthropic/claude-haiku-4-5", "complexity": "light",
                     "tags": ["fast", "classify", "rerank", "query-rewrite"]},
                    {"id": "anthropic/claude-sonnet-4-6", "complexity": "heavy",
                     "tags": ["programming", "python", "git", "implementation"]}]},
    ],
    # Cooldown (seconds) before re-probing a model that returned a quota/429
    # signal or hit its monthly cap. Assumption (no value given): 1h re-probe.
    "llm_cooldown_seconds": 3600,
    # How long (seconds) to cache a "daemon is down" probe result. A confirmed-down
    # result is cached much longer than the "up" TTL (30s) so that rapid call bursts
    # do not repeatedly attempt and fail against a stopped daemon.
    "ollama_down_probe_ttl_seconds": 120,
    # When the local LLM (Ollama) is down, the per-prompt hot path can't run the
    # rich skill-lifecycle + rolling-summary enrichment inside its budget. With
    # this on, it instead fires a detached worker that runs the same work on the
    # remote escalation ladder off the critical path; the refreshed summary +
    # skill set land in session state for the NEXT turn. Deterministic FTS skills
    # still surface synchronously for the current turn regardless.
    "hook_async_escalation": True,
    # Hard cap (USD/day) on personal-Claude *auxiliary* spend; null = no cap
    # (opt-in). Over cap → auxiliary tasks degrade to L0/L1.
    "llm_personal_daily_usd_cap": None,

    # Tier used by optimize_memory for LLM file classification.
    # cheap | mid | smart  (maps to llm_providers.tier_*)
    "optimize_memory_tier": "smart",

    # Max completion tokens for the optimize_memory classification call. Must be
    # generous: reasoning models on the escalation ladder spend tokens thinking
    # before emitting the per-file JSON, so a low cap (e.g. 2000) truncates the
    # response to nothing and the report comes back empty.
    "optimize_memory_max_tokens": 4000,

    # Extra skill directories — indexed alongside the plugin cache
    # Each entry: {"path": "/abs/path", "source": "label", "enabled": true}
    # Default OFF: skills-archive holds retired skills that must not be indexed
    # or surfaced in router injections. Enabling it re-pollutes the skill index.
    "extra_skill_dirs": [
        {
            "path": str(Path.home() / ".claude" / "skills-archive"),
            "source": "archive",
            "enabled": False,
        }
    ],

    # Extra plugin directories — each directory is indexed as a plugin source.
    # The directory may contain subdirs with plugin.json or README.md manifests.
    # Each entry: {"path": "/abs/path", "source": "label", "description": "...", "enabled": true}
    # If omitted, extra_skill_dirs entries are auto-registered as plugin sources too.
    "extra_plugin_dirs": [],

    # Session profiles — predefined plugin sets per work context.
    # Each profile lists the plugins to ENABLE; everything else is disabled.
    # Use short names (before @); the system resolves to full keys.
    # /profile <name> to activate, /profile save <name> to capture current state.
    "profiles": {
        "minimal": {
            "description": "Bare minimum — just git and code quality",
            "plugins": ["superpowers", "commit-commands", "code-review"]
        },
        "backend": {
            "description": "Python/FastAPI backend development",
            "plugins": [
                "superpowers", "commit-commands", "code-review",
                "code-simplifier", "feature-dev", "github",
                "security-guidance"
            ]
        },
        "frontend": {
            "description": "Frontend/UI development with browser tools",
            "plugins": [
                "superpowers", "commit-commands", "code-review",
                "frontend-design", "chrome-devtools-mcp", "feature-dev",
                "github"
            ]
        },
        "mcp-dev": {
            "description": "MCP server and plugin development",
            "plugins": [
                "superpowers", "commit-commands", "code-review",
                "mcp-server-dev", "plugin-dev", "skill-creator",
                "feature-dev", "github"
            ]
        },
        "data": {
            "description": "Data engineering and pipelines",
            "plugins": [
                "superpowers", "commit-commands", "code-review",
                "data", "feature-dev", "github"
            ]
        },
        "full": {
            "description": "All plugins enabled",
            "plugins": "__all__"
        },
    },

    # LLM Wiki knowledge layer — interlinked markdown wiki as source of truth.
    # The SQLite vector index is rebuilt FROM the wiki pages (derived accelerator).
    # wiki_preload_enabled: inject a compact wiki excerpt into thin-prompt context.
    # Set False to skip the wiki source in _gather_context (zero cost, hot path).
    "wiki_preload_enabled": True,
    "wiki_enabled": True,
    "wiki_root": str(Path.home() / ".claude" / "mcp-skill-hub" / "wiki"),
    # Per-project private-scope authorization: {project: [authorized_scope, ...]}
    # e.g. {"career": ["career"], "glicemia": ["glicemia"]}
    "wiki_private_scopes": {"glicemia": ["glicemia"], "career": ["career"]},
    # Whether to include wiki-private pages in memory-export bundles.
    "wiki_export_private": False,
    # Max approved source pages distilled per batch wiki_ingest run (cost cap).
    "wiki_ingest_batch_limit": 10,
}


# ---------------------------------------------------------------------------
# Services + monitor — control-panel schema
# ---------------------------------------------------------------------------

_SERVICE_DEFAULTS = {
    "auto_reconcile": True,
    "ollama_daemon":   {"enabled": True, "auto_start": True, "auto_disable_under_pressure": False},
    "ollama_router":   {"enabled": True, "auto_start": True, "auto_disable_under_pressure": False,
                         "model": "qwen2.5:3b", "approx_ram_mb": 2000},
    "ollama_embed":    {"enabled": True, "auto_start": True, "auto_disable_under_pressure": False,
                         "model": "nomic-embed-text", "approx_ram_mb": 500},
    "searxng":         {"enabled": True, "auto_start": True, "auto_disable_under_pressure": False,
                         "container": "skill-hub-searxng"},
    "watcher":         {"enabled": True, "auto_start": True, "auto_disable_under_pressure": False},
    "haiku_router":    {"enabled": False, "auto_start": False, "auto_disable_under_pressure": False},
}

_MONITOR_DEFAULTS = {
    "ram_free_mb_min": 2048,    # pressure threshold
    "cpu_load_pct_max": 0.80,   # pressure threshold
}

_DEFAULTS["services"] = _SERVICE_DEFAULTS
_DEFAULTS["monitor"] = _MONITOR_DEFAULTS

# System-health panel + background watcher (skill_hub.system_health).
_SYSTEM_HEALTH_DEFAULTS = {
    "watcher_enabled": True,    # run the advisory background sampler
    "auto_cleanup": False,      # also auto-run safe remediations (kill stale, purge)
    "interval_seconds": 120,    # sampling cadence
    "swap_pct_trigger": 85.0,   # auto-purge when swap crosses this (auto_cleanup only)
}
_DEFAULTS["system_health"] = _SYSTEM_HEALTH_DEFAULTS

# Worktree-driven parallel sessions (skill_hub.worktree).
_WORKTREE_DEFAULTS = {
    "repo_roots":   ["~/work/code"],
    "default_mode": "terminal",          # terminal | tmux | background
    # Seed each new worktree with a codegraph index (copy the parent repo's index,
    # then `codegraph sync` to the branch state) so fanned-out agents use codegraph
    # instead of grep on complex code. The index lives in the worktree dir, so
    # teardown removes it and frees the disk.
    "codegraph_provision": True,
    "codegraph_provision_timeout": 180,  # max seconds for the copy+sync step
}
_DEFAULTS["worktree"] = _WORKTREE_DEFAULTS


# Fan-out: parallel-issue worktree dispatch (skill_hub.fanout).
_FANOUT_DEFAULTS = {
    "default_limit":  3,
    "prompt_model":   "",             # blank → use llm provider's default
    "prompt_timeout": 60,
    "sources": [
        "skill_hub.fanout.sources:GitHubSource",
        "skill_hub.fanout.sources:TextSource",
    ],
}
_DEFAULTS["fanout"] = _FANOUT_DEFAULTS


# M4-3 federation-lite — schema convention for multi-host shared state.
# ``node_id`` tags every event/task with the authoring host so a synced DB
# replica (Syncthing / rsync / git-annex) can be filtered cross-host. Empty
# string → store.py falls back to $SKILL_HUB_NODE_ID then socket.gethostname().
_FEDERATION_DEFAULTS = {
    "node_id": "",
}
_DEFAULTS["federation"] = _FEDERATION_DEFAULTS


# Legacy top-level keys folded into the services/monitor dicts on load.
# Tuple shape: (legacy_key, service_name, field) — service_name=None means
# top-level is preserved (only delete the old key).
_LEGACY_SERVICE_MAP = (
    ("router_haiku_enabled", "haiku_router", "enabled"),
    ("searxng_enabled",      "searxng",      "enabled"),
    ("router_ollama_model",  "ollama_router", "model"),
    ("embed_model_for_router", "ollama_embed", "model"),
)


def _migrate_legacy(cfg: dict) -> dict:
    """Fold legacy top-level service keys into ``services.<svc>.<field>``.

    Idempotent: if the services dict already carries the field, the legacy
    key is simply deleted without overwriting.
    """
    services = cfg.setdefault("services", {})
    if not isinstance(services, dict):
        return cfg

    changed = False
    for legacy_key, svc_name, field in _LEGACY_SERVICE_MAP:
        if legacy_key not in cfg:
            continue
        svc = services.setdefault(svc_name, {})
        if not isinstance(svc, dict):
            continue
        # Only adopt the legacy value if the new location is absent — lets the
        # user edit the new shape without being overridden by a stale top-level.
        if field not in svc:
            svc[field] = cfg[legacy_key]
        del cfg[legacy_key]
        changed = True

    if changed:
        # Persist the migrated shape so the next load is a no-op.
        try:
            save_config(cfg)
        except OSError:
            pass
    return cfg


def _merge_defaults(cfg: dict) -> dict:
    """Deep-merge ``_DEFAULTS`` into ``cfg`` so nested keys (services.*) are filled in."""
    for key, default in _DEFAULTS.items():
        if key not in cfg:
            cfg[key] = default if not isinstance(default, (dict, list)) else (
                dict(default) if isinstance(default, dict) else list(default)
            )
            continue
        if isinstance(default, dict) and isinstance(cfg[key], dict):
            for sub_key, sub_default in default.items():
                if sub_key not in cfg[key]:
                    cfg[key][sub_key] = (
                        dict(sub_default) if isinstance(sub_default, dict)
                        else list(sub_default) if isinstance(sub_default, list)
                        else sub_default
                    )
                elif isinstance(sub_default, dict) and isinstance(cfg[key][sub_key], dict):
                    for leaf_key, leaf_default in sub_default.items():
                        cfg[key][sub_key].setdefault(leaf_key, leaf_default)
    return cfg


def load_config() -> dict:
    """Load config from file, fold legacy keys, and fill defaults."""
    config: dict = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}
    config = _migrate_legacy(config)
    config = _merge_defaults(config)
    return config


def save_config(config: dict) -> None:
    """Save config to file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _coerce_to_default_type(key: str, value: Any) -> Any:
    """Repair a value that was persisted as a JSON string for a structured key.

    UIs/CLIs that pass every field as text can store a key like
    ``llm_providers`` as a JSON *string* instead of a dict. Consumers then do
    ``providers.get(...)`` or ``dict(providers)`` and crash — which silently
    disables tier resolution and the whole L0→LN escalation ladder. When the
    key's default is a dict/list and the stored value is a matching JSON
    string, parse it back to the structured form.
    """
    default = _DEFAULTS.get(key)
    if isinstance(default, (dict, list)) and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
        if isinstance(parsed, type(default)):
            return parsed
    return value


def get(key: str) -> Any:
    """Get a single config value.

    Returns ``Any`` because values span scalars and lists; callers coerce with
    ``int()`` / ``float()`` / ``str()`` at the use site. Structured keys stored
    as JSON strings are coerced back so consumers never see a stringified dict.
    """
    value = load_config().get(key, _DEFAULTS.get(key))
    return _coerce_to_default_type(key, value)


def set(key: str, value) -> None:  # noqa: A001
    """Persist a single top-level config key.

    Coerces JSON-string values for structured keys so the file self-heals on
    the next write instead of persisting a stringified dict/list.
    """
    cfg = load_config()
    cfg[key] = _coerce_to_default_type(key, value)
    save_config(cfg)


def service_field(service_name: str, field: str, default=None):
    """Read a field from ``services.<service_name>.<field>`` with default fallback."""
    cfg = load_config()
    svc = (cfg.get("services") or {}).get(service_name) or {}
    if field in svc:
        return svc[field]
    base_default = (_SERVICE_DEFAULTS.get(service_name) or {})
    return base_default.get(field, default)


def is_service_enabled(service_name: str) -> bool:
    return bool(service_field(service_name, "enabled", True))
