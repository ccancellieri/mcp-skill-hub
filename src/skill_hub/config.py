"""Configuration management for Skill Hub.

Config file: ~/.claude/mcp-skill-hub/config.json

Users can override model selection based on their hardware.
Default config is optimized for minimal resource usage.
"""

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "config.json"

# Defaults — minimal footprint, works on any machine
_DEFAULTS = {
    # Ollama connection
    "ollama_base": "http://localhost:11434",

    # Multi-endpoint Ollama — priority-ordered list of endpoints.
    # Each entry: {name, url, priority, enabled, auth_header (optional)}
    # When empty or absent, ollama_base is used as the single endpoint.
    "ollama_endpoints": [
        {
            "name": "local",
            "url": "http://localhost:11434",
            "priority": 1,
            "enabled": False,  # off by default (user's machine keeps Ollama off)
            "auth_header": None,
        }
    ],

    # Per-model endpoint routing: model name → list of endpoint names (priority order).
    # Empty dict = use global endpoint priority for all models.
    "ollama_model_routing": {},

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
    "hook_precompact_threshold": 1500,  # messages longer than this get LLM pre-compaction

    # Search defaults
    "search_top_k": 3,                  # default number of results
    "search_similarity_threshold": 0.3, # minimum cosine similarity
    "feedback_boost_max": 1.5,          # maximum feedback boost factor
    "teaching_min_similarity": 0.6,     # minimum sim for teaching rule match

    # Compaction
    "compact_max_input_chars": 4000,

    # Conversation digest — periodic context compaction
    "digest_every_n_messages": 5,       # produce a digest every N messages
    "digest_stale_threshold": 0.3,      # similarity below this = "stale" topic

    # Auto-eviction — relevance decay tracking
    "eviction_enabled": True,           # enable relevance decay tracking
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

    # Model/effort recommendation — inject hints based on task complexity
    "model_recommendation_enabled": True,       # inject model/effort hints in systemMessage
    "always_forward_to_claude": True,           # NEVER block — always forward to Claude

    # Resource-aware LLM gating — skip expensive local LLM ops under pressure
    # Pressure levels: idle(0), low(1), moderate(2), high(3)
    # Each operation has a max pressure level at which it still runs.
    # Set to false to disable resource gating entirely (always run all LLM ops).
    "resource_gating_enabled": True,
    "resource_cache_ttl_seconds": 10,   # how often to re-check system resources

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

    # Activity log — daily rotation, 50 MB cap
    # Set to a custom path to redirect logs (e.g. "/tmp/skill-hub-logs")
    "log_dir": str(Path.home() / ".claude" / "mcp-skill-hub" / "logs"),

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
        "tier_cheap": "ollama/qwen2.5-coder:3b",
        "tier_mid":   "anthropic/claude-haiku-4-5",
        "tier_smart": "anthropic/claude-sonnet-4-6",
        "embed":      "ollama/nomic-embed-text",
    },
    # Default tier when code doesn't specify one.
    "llm_default_tier": "tier_cheap",

    # Extra skill directories — indexed alongside the plugin cache
    # Each entry: {"path": "/abs/path", "source": "label", "enabled": true}
    "extra_skill_dirs": [
        {
            "path": str(Path.home() / ".claude" / "skills-archive"),
            "source": "archive",
            "enabled": True,
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
    "sustain_seconds": 30,      # time before auto-disable triggers
}

_DEFAULTS["services"] = _SERVICE_DEFAULTS
_DEFAULTS["monitor"] = _MONITOR_DEFAULTS


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


def get(key: str) -> str | int | float | bool:
    """Get a single config value."""
    return load_config().get(key, _DEFAULTS.get(key))


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
