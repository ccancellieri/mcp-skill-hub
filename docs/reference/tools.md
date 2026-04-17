# Tools Reference

Everything callable from Claude (as MCP tools) or from a terminal (via `skill-hub-cli`).

## MCP tools

### Search & load

| Tool | Description |
|------|-------------|
| `search_skills(query, top_k, use_rerank)` | Semantic search, returns full skill content |
| `search_context(query, top_k)` | Unified search: skills + tasks + teachings + plugins |
| `search_web(query, top_k)` | Web search via SearXNG + local LLM summary |
| `suggest_plugins(query)` | Suggest plugins (including disabled) for current task |

### Tasks

| Tool | Description |
|------|-------------|
| `save_task(title, summary, context, tags)` | Save an open task for future sessions |
| `close_task(task_id, summary)` | Compact via local LLM and close |
| `update_task(task_id, summary, context, tags)` | Update an open task |
| `reopen_task(task_id)` | Reopen a closed task |
| `list_tasks(status)` | List open / closed / all tasks |

### Learning

| Tool | Description |
|------|-------------|
| `teach(rule, suggest)` | Add "when X, suggest Y" rule |
| `record_feedback(skill_id, helpful, query)` | Rate a skill or plugin |
| `forget_teaching(teaching_id)` | Remove a teaching rule |
| `list_teachings()` | Show all teaching rules |
| `log_session(tool_name, plugin_id)` | Record tool usage (used by hooks) |

### Profiles

| Tool | Description |
|------|-------------|
| `list_profiles()` | Show all profiles (built-in + custom) |
| `create_profile(name, description, plugins)` | Save a custom profile |
| `switch_profile(name)` | Activate profile (updates `settings.json`) |
| `delete_profile(name)` | Delete a custom profile |
| `auto_curate_plugins(query)` | LLM recommends a profile for a task |

### Routing (bandit)

| Tool | Description |
|------|-------------|
| `route_to_model(query, tier)` | Pick a model via ε-greedy bandit (`tier_cheap`/`tier_mid`/`tier_smart`) |
| `record_model_reward(model, reward)` | Record outcome quality for bandit learning |
| `bandit_stats()` | Per-model pull counts + estimated reward |

### Prompt improvement

| Tool | Description |
|------|-------------|
| `improve_prompt(text)` | Run opt-in prompt rewriters |
| `list_prompt_rewriters()` | Show available rewriters (`add_skill_context`, `add_recent_tasks`, `normalize_language`, …) |

### Management

| Tool | Description |
|------|-------------|
| `index_skills()` | Rebuild skill index (includes `extra_skill_dirs`) |
| `index_plugins()` | Index plugin descriptions (incl. `extra_skill_dirs` as sources) |
| `list_skills(plugin)` | List indexed skills |
| `toggle_plugin(plugin_name, enabled)` | Enable/disable plugins |
| `session_stats()` | Plugin usage statistics |
| `configure(key, value)` | View / update config |
| `status()` | Health check: MCP, Ollama, models, hook, DB stats |
| `token_stats()` | Token savings report from hook interceptions |
| `list_models()` | Installed Ollama models with role markers |
| `pull_model(model)` | Download a new Ollama model via Ollama API |
| `optimize_memory(dry_run)` | LLM analysis of memory files with prune/compact recommendations |
| `exhaustion_save(context)` | Auto-save session when Claude is rate-limited |

---

## CLI — `skill-hub-cli`

Direct CLI for use in hooks and scripts (bypasses Claude entirely):

```bash
skill-hub-cli classify "save this to memory"    # classify intent
skill-hub-cli save_task "title" "summary"        # save directly
skill-hub-cli close_task 3                       # close + compact
skill-hub-cli list_tasks open                    # list tasks
skill-hub-cli search_context "my query"          # search
```

Every `/hub-*` slash command and every hook ultimately shell out to `skill-hub-cli` — it's the stable, scriptable surface.

---

## REPL — `skill-hub-repl`

Interactive local shell that runs the same hook pipeline directly in your terminal:

```bash
skill-hub-repl                     # interactive mode
skill-hub-repl "git status"        # single command
skill-hub-repl "/hub-status"       # any /hub-* command
skill-hub-repl "?"                 # list commands
```

See [features/local-execution.md](../features/local-execution.md#standalone-repl) for examples.
