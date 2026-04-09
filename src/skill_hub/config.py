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

    # Exhaustion fallback — local LLM takes over when Claude is unavailable
    "exhaustion_fallback": True,        # enable exhaustion auto-save

    # Offline auto-fallback — automatically activate L4 local agent when
    # api.anthropic.com is unreachable (rate limit, network outage, travel)
    "offline_auto_fallback": True,
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
    "response_cache_verify": True,    # verify freshness with LLM when sim ≥ 0.93

    # Task decomposition — break complex multi-part requests into ordered subtasks
    "task_decomposition_enabled": True,
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
    "context_bridge_teaching_extraction": True,  # extract teaching examples at session end
    "context_bridge_repo_context": True,         # maintain per-repo context summaries

    # Skill Evolution — shadow learning from Claude (or any AI)
    # At session end, compares local skill output with Claude's tool usage.
    # If Claude's approach is better, evolves the skill's prompts/steps.
    # Old versions are stored in skill_versions table for rollback.
    "skill_evolution_enabled": True,
    "skill_evolution_auto": False,              # auto-evolve ALL skills (not just shadow:true)
    "skill_evolution_max_per_session": 3,       # max skills to evolve per session
    "skill_evolution_min_session_msgs": 5,      # min session messages before evolving

    # Resource-aware LLM gating — skip expensive local LLM ops under pressure
    # Pressure levels: idle(0), low(1), moderate(2), high(3)
    # Each operation has a max pressure level at which it still runs.
    # Set to false to disable resource gating entirely (always run all LLM ops).
    "resource_gating_enabled": True,
    "resource_cache_ttl_seconds": 10,   # how often to re-check system resources

    # LLM triage — local LLM pre-processes ALL messages before Claude
    "hook_llm_triage": True,            # enable universal LLM triage
    "hook_llm_triage_timeout": 30,      # max seconds for triage LLM call
    "hook_llm_triage_min_confidence": 0.7,  # min confidence to act on local_answer
    "hook_llm_triage_skip_length": 2000,    # messages longer than this skip triage

    # ── Local execution levels ──────────────────────────────────────
    # Level 1: Safe shell whitelist — predefined commands, no params
    # Level 2: Templated shell — commands with LLM-extracted parameters
    # Level 3: Local skill execution — LLM follows multi-step skill scripts
    # Level 4: Full local agent — tool-using agent loop (local or remote LLM)
    "local_execution_enabled": True,

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
    "searxng_enabled": True,
    "searxng_top_k": 3,
    "searxng_timeout": 5,      # seconds for URL probe (is SearXNG reachable?)
    "searxng_search_timeout": 15,  # seconds for actual search (engines need time)

    # Activity log — daily rotation, 50 MB cap
    # Set to a custom path to redirect logs (e.g. "/tmp/skill-hub-logs")
    "log_dir": str(Path.home() / ".claude" / "mcp-skill-hub" / "logs"),

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


def load_config() -> dict:
    """Load config from file, falling back to defaults for missing keys."""
    config = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            user_config = json.loads(CONFIG_PATH.read_text())
            config.update(user_config)
        except (json.JSONDecodeError, OSError):
            pass
    return config


def save_config(config: dict) -> None:
    """Save config to file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def get(key: str) -> str | int | float | bool:
    """Get a single config value."""
    return load_config().get(key, _DEFAULTS.get(key))
