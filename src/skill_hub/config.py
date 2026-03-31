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
    "hook_max_message_length": 400,     # messages longer than this skip LLM classify entirely
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
    "hook_context_max_chars": 2000,     # max chars injected as systemMessage (~500 tokens)
    "hook_precompact_threshold": 1500,  # messages longer than this get LLM pre-compaction

    # Search defaults
    "search_top_k": 3,                  # default number of results
    "search_similarity_threshold": 0.3, # minimum cosine similarity
    "feedback_boost_max": 1.5,          # maximum feedback boost factor
    "teaching_min_similarity": 0.6,     # minimum sim for teaching rule match

    # Compaction
    "compact_max_input_chars": 4000,    # max chars sent to LLM for compaction

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
