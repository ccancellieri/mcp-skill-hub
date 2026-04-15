"""Settings route — live config editor grouped by prefix buckets."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import config as _config

router = APIRouter()

# Ordered list of (prefix, label).  The last entry is always the catch-all.
_BUCKETS = [
    # ── Automation ────────────────────────────────────────────────────────────
    ("auto_approve", "Auto-approve"),
    ("auto_proceed", "Auto-proceed"),
    ("adaptive_windows", "Adaptive windows"),
    ("prefix_bundles", "Prefix bundles"),
    ("task_type_bundles", "Task-type bundles"),
    # ── Hook ─────────────────────────────────────────────────── more-specific first
    ("hook_context", "Context injection"),   # hook_context_* RAG enrichment
    ("hook_llm", "LLM triage"),         # hook_llm_* pre-triage
    ("hook", "Hook behavior"),          # remaining hook_* core
    # ── Router ────────────────────────────────────────────────────────────────
    ("router_haiku", "Haiku router"),   # router_haiku_* — must precede "router"
    ("router_bandit", "Router bandit"), # router_bandit_* — must precede "router"
    ("improve_prompt", "Prompt rewriters"),  # improve_prompt_*
    ("router", "Router core"),          # remaining router_*
    # ── Execution ─────────────────────────────────────────────────────────────
    ("local_persona", "Local persona"), # local_persona_* — must precede "local"
    ("local", "Local execution"),
    # ── Models & search ───────────────────────────────────────────────────────
    ("llm", "LLM providers"),
    ("vec", "Vector & embeddings"),
    ("searxng", "SearXNG search"),
    ("search", "Search & memory"),
    # ── Session & memory ──────────────────────────────────────────────────────
    ("session_memory", "Session memory"),
    ("context_bridge", "Context bridge"),
    ("skill_evolution", "Skill evolution"),
    ("digest", "Digest & eviction"),
    ("response_cache", "Response cache"),
    ("pattern", "Patterns & decomp"),
    # ── Config ────────────────────────────────────────────────────────────────
    ("services", "Services & monitor"),
    ("profiles", "Profiles"),
    # ── Legacy / plugin-provided (may be empty) ───────────────────────────────
    ("vector", "Vector"),
    ("dashboard", "Dashboard"),
    ("embedding", "Embedding"),
    ("chrome", "Chrome intents"),
    ("questions", "Questions"),
    # ── Catch-all ─────────────────────────────────────────────────────────────
    ("other", "Other"),
]

_BUCKET_HELP = {
    "auto_approve": "Controls the auto-approve hook — which tools/commands skip the confirm prompt.",
    "auto_proceed": "Multi-signal auto-proceed after Stop hook fires (clarifying questions, timers, etc.).",
    "adaptive_windows": "Tiered time windows that relax/tighten auto-approve based on recent outcomes.",
    "prefix_bundles": "Command-prefix bundles granted as a group once any member is verdict-allowed.",
    "task_type_bundles": "Per-task-type bundles: e.g. editing tasks unlock read-only bash by default.",
    "hook": "Core hook toggles — enabled/disabled, timeout, message-length guard, semantic threshold.",
    "hook_context": "RAG context injection into systemMessage — skills, tasks, precompact budget.",
    "hook_llm": "Local LLM pre-triage of every prompt before Claude — confidence gating and timeout.",
    "router_haiku": "Haiku 4.5 batched escalation tasks — classify, compact-hint, subtask decomp.",
    "router_bandit": "ε-greedy bandit over cheap/mid/smart tiers — exploration vs exploitation.",
    "improve_prompt": "Prompt-rewriter chain — skill-context enrichment and recent-task injection.",
    "router": "Router core — enable/disable, tier-2 Ollama gate, compact advisor, thin-prompt fill.",
    "local_persona": "Local LLM identity — static bio seed, TTL, and max assembled persona length.",
    "local": "Local execution levels 1–4: commands, templates, skill runner, agent, remote endpoint.",
    "llm": "LLM provider tiers (cheap/mid/smart/embed), Ollama base URL, reasoning model.",
    "vec": "Vector engine (sqlite-vec), binary quantization, rerank pool size.",
    "searxng": "SearXNG web search — URL, timeouts, and result count.",
    "search": "Skill/task/memory search — top-k, similarity threshold, feedback boost.",
    "session_memory": "Per-session transcript compaction, injection on resume, size limits.",
    "context_bridge": "Capture AI tool calls into local DB for context enrichment and pattern learning.",
    "skill_evolution": "Shadow-learn from Claude and evolve local skills over time.",
    "digest": "Conversation digest, stale-topic detection, compact budget, and auto-eviction.",
    "response_cache": "Semantic response cache — reuse answers for near-identical questions.",
    "pattern": "Prompt-pattern tracking, auto-skill generation, task decomposition.",
    "services": "Background services (Ollama, SearXNG, watcher, Haiku) and resource monitor.",
    "profiles": "Named plugin sets — activate a profile to switch context quickly.",
    "vector": "Embedding store thresholds, index sizes, and vector-search knobs.",
    "dashboard": "FastAPI webapp host, port, and feature toggles.",
    "embedding": "Ollama embedding model, dimension, batch size.",
    "chrome": "chrome-devtools MCP intent queue (URL targets, default action).",
    "questions": "Clarifying-question queue tuning (poll interval, TTL).",
    "other": "Uncategorized keys.",
}

# Field-level hints extracted from config.py comments
_FIELD_HINTS = {
    # Connection
    "ollama_base": "Ollama server URL (default: localhost:11434)",
    # Models
    "embed_model": "Embedding model for all vector ops. Options: nomic-embed-text (274MB, default), mxbai-embed-large (669MB), all-minilm (45MB)",
    "reason_model": "Reasoning model for re-ranking/compaction. 8GB: deepseek-r1:1.5b, 16GB: deepseek-r1:7b, 32GB+: deepseek-r1:14b",
    "llm_providers": "LLM endpoints by tier (tier_cheap/mid/smart/embed). Use litellm syntax: ollama/<model>, anthropic/<model>, openai/<model>",
    "llm_default_tier": "Default LLM tier when code doesn't specify one",
    # Hook core
    "hook_enabled": "Enable/disable UserPromptSubmit hook entirely",
    "hook_timeout_seconds": "Max seconds for hook LLM classification",
    "hook_semantic_threshold": "Min similarity to trigger LLM classify (0–1). Lower = more sensitive, higher = fewer false positives",
    "hook_max_message_length": "Messages longer than this bypass LLM classify entirely",
    "token_profiling": "Track estimated token savings from each interception",
    "always_forward_to_claude": "NEVER block prompts — always forward to Claude (safety switch)",
    # Hook context
    "hook_context_injection": "Enable RAG + auto-memory injection into systemMessage",
    "hook_context_max_chars": "Total char budget for context injection (~10k tokens)",
    "hook_context_max_skill_chars": "Max chars per skill (truncated if larger)",
    "hook_context_top_k_skills": "Max skills to load with full content per message",
    "hook_context_min_skills": "Min skills to load (auto-fill from RAG if LLM picks fewer)",
    "hook_precompact_threshold": "Messages longer than this get LLM pre-compaction",
    "hook_task_command_examples": "Phrases to build semantic centroid for task-detection prefilter",
    # Hook LLM triage
    "hook_llm_triage": "Enable local LLM pre-triage of all messages (off by default: small models unreliable)",
    "hook_llm_triage_timeout": "Max seconds for triage LLM call",
    "hook_llm_triage_min_confidence": "Min confidence threshold to act on local LLM triage result",
    "hook_llm_triage_skip_length": "Messages longer than this skip triage entirely",
    # Router core
    "router_enabled": "Enable/disable the three-tier prompt router (heuristics → Ollama → Haiku)",
    "router_hard_switch_threshold": "Confidence above this triggers hard model switch in settings.json",
    "router_tier2_confidence_gate": "Tier-1 confidence below this escalates to Tier-2 Ollama",
    "router_tier2_timeout": "Max seconds for Tier-2 Ollama call",
    "router_log": "Path to JSONL audit log (one line per prompt)",
    "router_compact_threshold": "Inject /compact suggestion when context is ≥ this fraction full",
    "router_enrich_thin_prompts": "Prepend task context for very short messages (<60 chars)",
    # Router bandit
    "router_bandit_enabled": "Enable ε-greedy bandit optimization over tier_cheap/mid/smart",
    "router_bandit_epsilon": "Exploration probability (0–1): higher = more exploration",
    # Router Haiku
    "router_haiku_threshold": "Tier-2 confidence below this escalates to Tier-3 Haiku",
    "router_haiku_classify": "Enable Haiku complexity/ambiguity/scope classification",
    "router_haiku_settings_opt": "Enable Haiku config optimization suggestions",
    "router_haiku_compact_hint": "Enable Haiku /compact recommendations",
    "router_haiku_subtask_decomp": "Enable Haiku multi-part prompt decomposition",
    # Prompt rewriters
    "router_improve_prompt_enabled": "Enable prompt enrichment via rewriter chain",
    "improve_prompt_default_chain": "List of rewriter names to run (e.g., add_skill_context, add_recent_tasks)",
    "improve_prompt_skill_top_k": "Max skills to include in prompt enrichment",
    "improve_prompt_tasks_limit": "Max recent tasks to include in prompt enrichment",
    # Search
    "search_top_k": "Default number of search results",
    "search_similarity_threshold": "Min cosine similarity (0–1) for search matches",
    "feedback_boost_max": "Maximum feedback boost multiplier (1.0–3.0)",
    "teaching_min_similarity": "Min similarity threshold for teaching rule matches",
    # Compaction
    "compact_max_input_chars": "Max chars to feed to compaction LLM per call",
    # Digest
    "digest_every_n_messages": "Produce conversation digest every N messages",
    "digest_stale_threshold": "Similarity below this marks a topic as stale",
    # Eviction
    "eviction_enabled": "Enable relevance decay tracking for inactive topics",
    "eviction_min_stale_count": "Suggest profile switch after N stale topic detections",
    # Memory
    "user_memory_enabled": "Auto-index user project memory files into memory:user-project namespace",
    "session_memory_enabled": "Enable per-session transcript compaction (6-section summary)",
    "session_memory_min_messages": "Min message count to trigger first session memory build",
    "session_memory_max_transcript_bytes": "Max transcript bytes (~200KB covers long sessions)",
    "session_memory_tier": "LLM tier for session memory summarization",
    "session_memory_inject_on_resume": "Inject stored memory as systemMessage on session resume",
    "session_memory_inject_max_chars": "Cap memory chars injected into systemMessage",
    "auto_memory_on_close_task": "Run smart_memory_write after task compaction",
    # Response cache
    "response_cache_enabled": "Cache answers for near-identical questions",
    "response_cache_min_sim": "Min similarity to serve from cache (0–1, typical 0.88+)",
    "response_cache_verify": "When enabled, adds extra verification step (slower but safer)",
    # Context bridge
    "context_bridge_enabled": "Capture Claude's tool calls for local intelligence",
    "context_bridge_max_capture_per_hook": "Max tool calls captured per Stop hook",
    "context_bridge_prune_days": "Prune tool examples older than this",
    "context_bridge_prune_max_rows": "Max total rows in tool_examples table",
    "context_bridge_teaching_extraction": "Extract teaching rules from tool calls (off: unreliable with small models)",
    "context_bridge_repo_context": "Maintain per-repo context summaries",
    # Skill evolution
    "skill_evolution_enabled": "Shadow-learn from Claude and evolve local skills",
    "skill_evolution_auto": "Auto-evolve all skills (off: only shadow:true skills)",
    "skill_evolution_max_per_session": "Max skills to evolve per session",
    "skill_evolution_min_session_msgs": "Min session messages before attempting evolution",
    "skill_evolution_feed_memory": "Include project memory in skill evolution context",
    "skill_evolution_cross_pollinate": "Reference official Claude skills during evolution",
    "skill_sync_on_index": "Check for plugin updates when indexing skills",
    # Task decomposition
    "task_decomposition_enabled": "Break multi-part requests into ordered subtasks (off: unreliable with small models)",
    "task_decomposition_min_len": "Only decompose messages longer than this",
    # Pattern tracking
    "pattern_tracking_enabled": "Learn from recurring message patterns",
    "pattern_auto_skill_threshold": "Auto-generate skill after pattern recurs this many times",
    # Local execution
    "local_execution_enabled": "Enable local execution levels 1–4 (off: blocks commands from reaching Claude)",
    "local_models": "Ollama model per execution level (level_1/2/3/4). Use 'remote:<url>' for external APIs",
    "local_commands": "Whitelisted shell commands for level 1 execution",
    "local_templates": "Templated commands with {param} placeholders for level 2",
    "local_skills_dir": "Directory for local skills (~/.claude/local-skills by default)",
    "local_system_prompt": "Optional bio seed prepended to all local LLM calls",
    "local_persona_ttl_seconds": "Rebuild persona from store after this many seconds",
    "local_persona_max_chars": "Hard cap on assembled persona string",
    "remote_llm": "Remote LLM endpoint config (base_url, api_key, model, timeout)",
    "plan_api_runner_enabled": "Use litellm API-token fallback for plan execution (keep disabled on Max plan)",
    # Fallbacks
    "exhaustion_fallback": "Save context when Claude becomes unavailable",
    "offline_auto_fallback": "Auto-activate L4 agent when api.anthropic.com unreachable (off: L4 unreliable with small models)",
    "offline_check_interval": "Seconds between reachability checks",
    # Implicit feedback
    "implicit_feedback_enabled": "Infer skill quality from session tool usage",
    # Vector / search
    "vec_engine": "Vector engine (sqlite-vec for ANN + binary quantization, else fallback to cosine)",
    "binary_quant_enabled": "Enable binary quantization for faster ANN",
    "rerank_top_k": "Binary KNN candidate pool before float32 rerank (~60 yields 97% recall@5)",
    # Services
    "services": "Background services config (Ollama, SearXNG, watcher, Haiku router)",
    "monitor": "Resource monitor thresholds (RAM, CPU, pressure sustain time)",
    # Search
    "searxng_url": "SearXNG server URL (empty = auto-detect localhost:8989)",
    "searxng_top_k": "Max search results from SearXNG",
    "searxng_timeout": "Seconds to wait for SearXNG probe (is it reachable?)",
    "searxng_search_timeout": "Seconds for actual SearXNG search (engines need time)",
    # Logging
    "log_dir": "Activity log directory (daily rotation, 50MB cap)",
    # Profiles
    "profiles": "Named plugin sets per work context (minimal, backend, frontend, mcp-dev, data, full)",
    # Extra dirs
    "extra_skill_dirs": "Extra skill directories to index alongside plugin cache",
    "extra_plugin_dirs": "Extra plugin directories with plugin.json or README.md manifests",
}

# Keys whose names don't share the prefix of their logical bucket.
_BUCKET_OVERRIDES: dict[str, str] = {
    # llm
    "embed_model": "llm",
    "reason_model": "llm",
    "ollama_base": "llm",
    # hook core
    "token_profiling": "hook",
    "always_forward_to_claude": "hook",
    # hook_context: keys that don't start with "hook_context"
    "hook_precompact_threshold": "hook_context",
    "hook_task_command_examples": "hook_context",
    # improve_prompt: router_improve_prompt_enabled doesn't start with "improve_prompt"
    "router_improve_prompt_enabled": "improve_prompt",
    # router core: doesn't start with "router"
    "model_recommendation_enabled": "router",
    # local_persona: local_system_prompt doesn't start with "local_persona"
    "local_system_prompt": "local_persona",
    # local
    "remote_llm": "local",
    "offline_auto_fallback": "local",
    "offline_check_interval": "local",
    "plan_api_runner_enabled": "local",
    "exhaustion_fallback": "local",
    # llm
    "embed_model": "llm",
    "reason_model": "llm",
    "ollama_base": "llm",
    # vec
    "binary_quant_enabled": "vec",
    "rerank_top_k": "vec",
    # search
    "feedback_boost_max": "search",
    "teaching_min_similarity": "search",
    # session_memory
    "auto_memory_on_close_task": "session_memory",
    "user_memory_enabled": "session_memory",
    # skill_evolution
    "skill_sync_on_index": "skill_evolution",
    "implicit_feedback_enabled": "skill_evolution",
    "learn_from_claude_sessions": "skill_evolution",
    # digest
    "compact_max_input_chars": "digest",
    "eviction_enabled": "digest",
    "eviction_min_stale_count": "digest",
    # pattern
    "task_decomposition_enabled": "pattern",
    "task_decomposition_min_len": "pattern",
    # services
    "monitor": "services",
    "resource_gating_enabled": "services",
    "resource_cache_ttl_seconds": "services",
    # profiles
    "profile_auto_switch_enabled": "profiles",
    "profile_auto_switch_window": "profiles",
    "extra_skill_dirs": "profiles",
    "extra_plugin_dirs": "profiles",
}

# Nav group headers: maps the first bucket of each visual group to a label.
_NAV_GROUPS: dict[str, str] = {
    "auto_approve": "Automation",
    "hook_context": "Hook",
    "router_haiku": "Router",
    "local_persona": "Execution",
    "llm": "Models & Search",
    "session_memory": "Session & Memory",
    "services": "Config",
    "vector": "Legacy / Plugins",
    "other": "Misc",
}


def _bucket_for(key: str) -> str:
    if key in _BUCKET_OVERRIDES:
        return _BUCKET_OVERRIDES[key]
    for prefix, _ in _BUCKETS[:-1]:
        if key.startswith(prefix):
            return prefix
    return "other"


def _field_type(val: Any) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, dict):
        return "dict"
    if isinstance(val, list):
        return "list"
    return "str"


def _group_config(cfg: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {b: [] for b, _ in _BUCKETS}
    for key in sorted(cfg.keys()):
        val = cfg[key]
        bucket = _bucket_for(key)
        entry = {"key": key, "value": val, "type": _field_type(val), "hint": _FIELD_HINTS.get(key, "")}
        if entry["type"] == "dict":
            # Nested dicts (e.g. task_type_bundles with list values) don't
            # flatten well — render them as a JSON textarea instead.
            if any(isinstance(v, (dict, list)) for v in val.values()):
                entry["json_text"] = json.dumps(val, indent=2)
                entry["json_mode"] = True
            else:
                entry["json_mode"] = False
                entry["children"] = [
                    {
                        "key": f"{key}.{k}",
                        "value": v,
                        "type": _field_type(v),
                        "hint": _FIELD_HINTS.get(f"{key}.{k}", ""),
                    }
                    for k, v in val.items()
                ]
        elif entry["type"] == "list":
            # Flat (primitive) lists stay comma-separated; complex lists
            # (e.g. adaptive_windows) become editable JSON textareas.
            if all(isinstance(x, (str, int, float, bool)) for x in val):
                entry["display"] = ", ".join(str(x) for x in val)
                entry["flat_list"] = True
            else:
                entry["json_text"] = json.dumps(val, indent=2)
                entry["flat_list"] = False
        groups[bucket].append(entry)
    return groups


def _coerce(original: Any, raw: str | None) -> Any:
    t = _field_type(original)
    if t == "bool":
        return raw == "on"
    if raw is None:
        return original
    raw = raw.strip()
    if t == "int":
        try:
            return int(raw)
        except ValueError:
            return original
    if t == "float":
        try:
            return float(raw)
        except ValueError:
            return original
    if t == "list":
        if not raw:
            return []
        items = [p.strip() for p in raw.split(",") if p.strip()]
        # Preserve element type if original was numeric
        if original and isinstance(original[0], int):
            out = []
            for p in items:
                try:
                    out.append(int(p))
                except ValueError:
                    pass
            return out
        if original and isinstance(original[0], float):
            out = []
            for p in items:
                try:
                    out.append(float(p))
                except ValueError:
                    pass
            return out
        return items
    return raw


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> Any:
    cfg = _config.load_config()
    groups = _group_config(cfg)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "groups": groups,
            "buckets": _BUCKETS,
            "bucket_help": _BUCKET_HELP,
            "nav_groups": _NAV_GROUPS,
            "field_hints": _FIELD_HINTS,
            "active_tab": "settings",
        },
    )


@router.post("/settings/save", response_class=HTMLResponse)
async def settings_save(request: Request) -> HTMLResponse:
    form = await request.form()
    form_keys = set(form.keys())
    # __bool__ marks every bool field rendered (checkbox, even if unchecked).
    rendered_bools = set(form.getlist("__bool__")) if hasattr(form, "getlist") else set()
    cfg = _config.load_config()
    # Track every top-level key we actually rendered on the page. Bool fields
    # are absent from the form when unchecked, so we need the "page rendered it"
    # signal separately. We use a hidden marker: every rendered top-level key
    # emits a "__rendered__:<key>" form key? Simpler: detect "touched" by
    # presence of any sub-field for dicts, or presence-of-key-or-known-bool.
    changed = 0
    rendered_roots: set[str] = set()
    for fk in form_keys:
        # __json__.<key> -> treat <key> as the rendered root.
        if fk.startswith("__json__."):
            rendered_roots.add(fk.split(".", 1)[1].split(".", 1)[0])
        else:
            rendered_roots.add(fk.split(".", 1)[0])
    for rb in rendered_bools:
        rendered_roots.add(rb.split(".", 1)[0])

    for key, orig in list(cfg.items()):
        t = _field_type(orig)
        if key not in rendered_roots:
            # Field wasn't on the page (e.g. complex list) — skip.
            continue
        # JSON-textarea override (complex lists / nested dicts).
        json_key = f"__json__.{key}"
        if json_key in form_keys:
            raw = str(form.get(json_key) or "").strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if parsed != orig:
                        cfg[key] = parsed
                        changed += 1
                except json.JSONDecodeError:
                    pass
            continue
        if t == "dict":
            new_dict = dict(orig)
            for sub_key, sub_val in orig.items():
                form_key = f"{key}.{sub_key}"
                if _field_type(sub_val) == "bool":
                    if form_key in rendered_bools:
                        new_dict[sub_key] = form.get(form_key) == "on"
                else:
                    raw = form.get(form_key)
                    if raw is not None:
                        new_dict[sub_key] = _coerce(sub_val, str(raw))
            if new_dict != orig:
                cfg[key] = new_dict
                changed += 1
            continue
        if t == "list" and orig and not all(
            isinstance(x, (str, int, float, bool)) for x in orig
        ):
            continue
        if t == "bool":
            if key not in rendered_bools:
                continue
            new_val = form.get(key) == "on"
        else:
            raw = form.get(key)
            if raw is None:
                continue
            new_val = _coerce(orig, str(raw))
        if new_val != orig:
            cfg[key] = new_val
            changed += 1
    try:
        _config.save_config(cfg)
        msg = f"Saved ✓ ({changed} changed)"
        cls = "ok"
    except OSError as e:
        msg = f"Error: {e}"
        cls = "err"
    return HTMLResponse(f'<span class="status {cls}">{msg}</span>')
