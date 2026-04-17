# Config Reference

**Location:** `~/.claude/mcp-skill-hub/config.json`

All settings have sensible defaults. Override only what you need.

```
configure()                                          # show all settings
configure(key="reason_model", value="deepseek-r1:7b")
configure(key="search_top_k", value="5")
configure(key="hook_enabled", value="false")
```

---

## Ollama & models

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_base` | `http://localhost:11434` | Ollama server URL |
| `embed_model` | `nomic-embed-text` | Embedding model |
| `reason_model` | `deepseek-r1:1.5b` | Reasoning model (re-rank, compact, classify). **Recommended:** `qwen2.5-coder:7b-instruct-q4_k_m` |
| `local_models` | `{level_1: 3b, …}` | Ollama model per execution level |
| `remote_llm` | `{}` | Remote LLM endpoint for L4: `{base_url, api_key, model, timeout}` |

## Hook & interception

| Key | Default | Description |
|-----|---------|-------------|
| `hook_enabled` | `true` | Enable UserPromptSubmit hook |
| `hook_timeout_seconds` | `45` | Max hook execution time |
| `hook_semantic_threshold` | `0.45` | Min embedding similarity for LLM classify |
| `hook_max_message_length` | `2000` | Messages longer than this skip LLM classify |
| `hook_task_command_examples` | *(15 phrases)* | Canonical task phrases for semantic centroid |
| `token_profiling` | `true` | Track estimated token savings |

## Context injection

| Key | Default | Description |
|-----|---------|-------------|
| `hook_context_injection` | `true` | Auto-enrich context with RAG + memory |
| `hook_context_max_chars` | `40000` | Max chars injected as systemMessage (~10k tokens) |
| `hook_context_top_k_skills` | `5` | Max skills loaded with full content per message |
| `hook_context_summary_max_chars` | `800` | Max chars for rolling context summary |
| `hook_context_prompt_opt_min_len` | `150` | Min message length to trigger prompt optimization |
| `hook_context_prompt_optimization` | `true` | Enable local LLM prompt rewriting |
| `hook_precompact_threshold` | `1500` | Messages longer than this get LLM pre-compaction |
| `hook_memory_dir` | *(CWD-derived)* | Override memory directory path |

## Universal LLM triage

| Key | Default | Description |
|-----|---------|-------------|
| `hook_llm_triage` | `true` | Enable universal LLM triage on all messages |
| `hook_llm_triage_timeout` | `30` | Max seconds for triage LLM call |
| `hook_llm_triage_min_confidence` | `0.7` | Min confidence to act on local answer |
| `hook_llm_triage_skip_length` | `2000` | Messages longer than this skip triage |

## Search

| Key | Default | Description |
|-----|---------|-------------|
| `search_top_k` | `3` | Default search results count |
| `search_similarity_threshold` | `0.3` | Minimum cosine similarity |

## Skill directories

| Key | Default | Description |
|-----|---------|-------------|
| `extra_skill_dirs` | `[{skills-archive}]` | Extra skill directories to index |
| `extra_plugin_dirs` | `[]` | Extra plugin directories to index |

## Local execution (L1–L4)

| Key | Default | Description |
|-----|---------|-------------|
| `local_execution_enabled` | `true` | Enable local command execution (L1–L4) |
| `local_commands` | `{git_status: …}` | Level 1 whitelisted shell commands |
| `local_templates` | `{git_log_n: …}` | Level 2 templated commands with params |
| `local_cmd_similarity_threshold` | `0.55` | Min task-centroid similarity to try L1/L2 matching |
| `local_skill_threshold` | `0.85` | Min embedding similarity for L3 local skill match |
| `local_skills_dir` | `~/.claude/local-skills` | Directory for Level 3 skill JSON files |

## Skill evolution (shadow learning)

| Key | Default | Description |
|-----|---------|-------------|
| `skill_evolution_enabled` | `true` | Master switch |
| `skill_evolution_auto` | `false` | Evolve **all** skills vs only `shadow:true` |
| `skill_evolution_max_per_session` | `3` | Max skills evolved per session |
| `skill_evolution_min_session_msgs` | `5` | Min messages before evolution runs |

## Profiles & digest

| Key | Default | Description |
|-----|---------|-------------|
| `profiles` | `{6 built-in}` | Session profile definitions |
| `digest_every_n_messages` | `5` | Produce conversation digest every N messages |
| `digest_stale_threshold` | `0.3` | Similarity below this = stale topic |
| `eviction_enabled` | `true` | Enable relevance decay tracking |
| `eviction_min_stale_count` | `3` | Suggest profile switch after N stale detections |

## Offline & exhaustion

| Key | Default | Description |
|-----|---------|-------------|
| `exhaustion_fallback` | `true` | Enable exhaustion auto-save |
| `offline_auto_fallback` | `true` | Auto-activate L4 agent when Anthropic unreachable |
| `offline_check_interval` | `30` | Seconds between reachability checks |

## Learning & auto-memory

| Key | Default | Description |
|-----|---------|-------------|
| `implicit_feedback_enabled` | `true` | Infer skill quality from session tool usage at session-end |
| `auto_memory_on_close_task` | `true` | Auto-write memory entry when a task is closed |

## Resource gating

| Key | Default | Description |
|-----|---------|-------------|
| `resource_gating_enabled` | `true` | Skip LLM ops under CPU/RAM pressure |
| `resource_cache_ttl_seconds` | `10` | How often to re-check system resources |

## Context bridge

| Key | Default | Description |
|-----|---------|-------------|
| `context_bridge_enabled` | `true` | Capture Claude tool calls for local skills |
| `context_bridge_max_capture_per_hook` | `20` | Max tool calls captured per Stop hook |

## SearXNG

| Key | Default | Description |
|-----|---------|-------------|
| `searxng_url` | `""` *(auto-detect)* | Explicit SearXNG URL; empty = probe `localhost:8989` |
| `searxng_enabled` | `true` | Enable SearXNG web search (active + passive) |
| `searxng_top_k` | `3` | Number of search results to fetch + summarize |
| `searxng_timeout` | `5` | Seconds for reachability probe |
| `searxng_search_timeout` | `15` | Seconds for actual search (engines need time) |

## Logging

| Key | Default | Description |
|-----|---------|-------------|
| `log_dir` | `~/.claude/mcp-skill-hub/logs` | Activity log directory (daily rotation, 50 MB cap) |

---

## Patterns to remember

- Keys containing URLs or JSON blobs: pass the **whole JSON as a string** to `configure(value=...)`
- Changes take effect **live** — no restart — because the control panel has a 2-second mtime reconciler
- Use `configure()` with no args to dump the current full config

## Related

- [reference/architecture.md](architecture.md) — what each subsystem does
- [reference/tools.md](tools.md) — every MCP tool
- [features/web-control-panel.md](../features/web-control-panel.md) — GUI for most config keys
