# Unattended Claude Code

Two hooks make Claude Code run long / overnight jobs without stopping for
permission prompts or "proceed" nudges. Works identically in the VS Code
extension and the CLI — you don't launch anything special.

## 1. Per-project auto-approve (PreToolUse)

The installer seeds `~/.claude/skill-hub-allow.yml` from
[`examples/skill-hub-allow.yml`](../examples/skill-hub-allow.yml).
The hook reads that file plus an optional per-project
`<project>/.claude/skill-hub-allow.yml` (merged; project wins on duplicates).

Keys:
- `safe_bash_prefixes` — Bash commands starting with any of these auto-approve.
- `safe_tools` — tool names (e.g. `Read`, `Grep`) always auto-approved.
- `deny_patterns` — Python regex; matching Bash commands are blocked.

Deny wins over allow. Unknown commands fall through to the normal prompt.

Logs: `~/.claude/mcp-skill-hub/logs/hook-debug.log` (grep `AUTO_APPROVE`).

## 2. Auto-proceed on long plans (Stop)

Default config (seeded by `install.py`):

```json
{
  "auto_proceed": true,
  "auto_proceed_window": {"start_hour": 0, "end_hour": 24},
  "auto_proceed_max": 20
}
```

When Claude stops, the hook re-feeds `proceed` automatically if ANY of these
three signals match (logged as `AUTO_PROCEED signal=...`):

1. **`plan_checklist`** — newest plan file under `~/.claude/plans/` contains
   `- [ ]` items (existing behavior).
2. **`open_task`** — the skill-hub SQLite DB has an `open` task whose
   `session_id` matches the current Stop-hook session. Created by
   `mcp__skill-hub__save_task`; cleared by `close_task`.
3. **`recent_marker`** — a plan file was modified in the last 60 minutes and
   contains the literal marker `<!-- auto-proceed -->` on any line. Use this
   for prose-only plans without checklists.

Counter per session is tracked in
`~/.claude/mcp-skill-hub/state/auto_proceed.json`; once the cap is reached,
normal Stop resumes. The active open task is also mirrored at
`~/.claude/mcp-skill-hub/state/active_task.json` for hooks that need it
without touching the DB.

Env overrides (if you want it on only for a single session):
`SKILL_HUB_AUTO_PROCEED=1`, `SKILL_HUB_MAX_PROCEEDS=50`.

## 3. Adaptive auto-approve — learning from Claude's usage

`auto-approve.sh` consults a verdict cache at
`~/.claude/mcp-skill-hub/command_verdicts.db` in addition to the static
allow-list.

- **Write path (always on):** `post-tool-observer.sh` (PostToolUse) records
  every successful Bash run as `source=user_approved`. Next time that command
  (or a near-canonical variant — paths, hashes, numbers stripped) appears,
  the PreToolUse hook auto-approves it from cache with zero latency.
- **LLM fallback (opt-in):** set `"auto_approve_llm": true` in
  `~/.claude/mcp-skill-hub/config.json` to let the local Ollama model classify
  uncached commands. Few-shot examples are pulled from recent user-approved
  entries. Model can only upgrade "unknown" → "allow"; it **never denies**.

**Priority:** static deny > user_approved cache > llm cache > static allow > prompt user.

Config knobs (all in `config.json`):

```json
{
  "auto_approve_learn": true,
  "auto_approve_llm": false,
  "auto_approve_confidence": 0.85,
  "auto_approve_verdict_ttl_days": 30,
  "auto_approve_timeout_s": 4.0
}
```

## 4. Benefit/cost dashboard

Every `close_task()` (and any manual `render_dashboard()` MCP call) refreshes
`~/.claude/mcp-skill-hub/reports/dashboard.html`. It combines DB counters
(tokens saved, tasks, skills, embeddings, feedback ratio, triage log) with a
tail-parse of `hook-debug.log` for hook-specific metrics (auto-approve
decisions, LLM latency p50/p95, cache hits, resume events). The page is
self-contained — no CDN, works offline.

A "Net estimated savings" tile subtracts local-LLM wall time converted at a
conservative 50 Sonnet-equivalent tokens per second (see tooltip) so you can
judge whether the hooks are net positive over time.

## 5. Auto-resume after API outage (SessionEnd + SessionStart)

Hooks: `session_end.py` + `session_start_enforcer.py`.

If the session transcript tail contains `api_error`, `Internal server error`,
`overloaded_error`, or `rate_limit`, `session_end` writes a marker at
`~/.claude/mcp-skill-hub/state/needs_resume.json` (session id, timestamp, newest
plan under `~/.claude/plans/`).

On the next session start, `session_start_enforcer` consumes the marker and
injects a `systemMessage` telling Claude to re-read the plan and continue the
interrupted work. Marker is single-shot: deleted after one consume.

No wrapper, no retries in shell — the next session you open in VS Code
automatically picks up where the last one died.

## 6. Dashboard v3 (FastAPI suite)

`render_dashboard()` (and every `close_task`) boots a FastAPI + HTMX +
Alpine.js webapp via uvicorn in a daemon thread on loopback
(default `http://127.0.0.1:8765/`). Singleton per MCP process; bind
failure returns `None` and falls back to the static HTML snapshot at
`~/.claude/mcp-skill-hub/reports/dashboard.html`. No Node, no CDN —
HTMX/Alpine vendored locally.

Tabs:
- **Dashboard** — KPIs, sparklines, auto-approve breakdown, LLM
  latency, cache hit stats.
- **Settings** — live-edit `config.json` grouped by section; writes
  through on form POST.
- **Verdicts** — filter/sort, delete, flip allow↔deny, pin/unpin,
  bulk-promote to `~/.claude/skill-hub-allow.yml`.
- **Tasks** — open/closed panels, rename, edit, close, reopen, merge,
  delete, teach-from-task modal, text + semantic search. Per-task
  **auto-approve override** toggle (force on/off for this task only).
  Expanding a row reveals a per-task **Logs** section
  (`GET /tasks/{id}/logs`, lazy-loaded via HTMX, optional WebSocket
  live tail at `/tasks/{id}/logs/ws`). Lines are filtered by the
  `task=<id>` token that hook `log()` helpers prepend whenever
  `~/.claude/mcp-skill-hub/state/active_task.json` points at a task —
  see `hooks/verdict_cache.py::task_tag()`. Untagged legacy lines
  remain in the global `/logs` view but are not shown per-task.
- **Skills** — most-used table, plugin filter, details drawer.
- **Teachings** — list, add, delete, search.
- **Logs** — WebSocket live tail of `hook-debug.log` with level/source
  filters, pause/resume, regex highlight, download.
- **Vector** — 2D random-projection scatter over skills / tasks /
  teachings / verdicts; similarity halo on click.
- **Intents** — chrome-devtools intent queue at
  `~/.claude/mcp-skill-hub/state/chrome_intents.jsonl`; enqueue/list/
  mark-done. Stop hook injects a `systemMessage` telling Claude to
  drain via chrome-devtools MCP.
- **Questions** — SSE stream for hook-initiated prompts; toast
  notifications with answer buttons. `auto_approve.py` short-polls
  (3s) on ambiguous commands when enabled.

New config keys (`~/.claude/mcp-skill-hub/config.json`):
```
"dashboard_server_enabled": true,
"dashboard_server_port": 8765,
"dashboard_auto_open_browser": false,
"ask_user_on_ambiguous": false,
"ask_user_timeout_s": 10.0,
"ask_on_deny": true,
"auto_approve_night_mode": false
```

`ask_on_deny` (default `true`) changes `deny_pattern` matches from hard
blocks into user prompts: the hook POSTs a question to
`/questions/ask` and short-polls `/questions/list` for up to
`ask_user_timeout_s` (default 10s). Answering `allow` overrides the
deny (and caches a `user_approved` verdict so the next run is instant);
`deny` or timeout produces a hard block. A small hardcoded set of
**catastrophic** patterns (fork bombs `:(){ :|:& };:`, `dd if=... of=/dev/...`)
still block immediately without asking — those aren't recoverable. If the
dashboard server is unreachable the hook falls back to the legacy block
behavior.

`auto_approve_night_mode` = auto-accept any question that times out,
scoped to the configured `auto_proceed_window`. `ask_user_on_ambiguous`
routes uncached/unclassified commands to the Questions tab instead of
falling through to the normal prompt. `dashboard_auto_open_browser`
opens the URL in the default browser on first boot.

Target idle RSS <60MB. Loopback-only bind; no auth.

## 7. Adaptive allowance (time tiers + task-type bundles)

The binary 23:00-07:00 night window is deprecated in favour of tiered
`adaptive_windows`. Each entry names a `prefix_bundle` active during its
hour range; the first matching window wins. Built-in bundles:

- `read_only` — `sed -n`, `head`, `tail`, `grep`, `rg`, `find`, `wc`,
  `file`, `stat`, `ls`, `cat`, `tree`, `which`, `echo`, `pwd`, and
  `git status|diff|log|branch|show`.
- `build` — `pytest`, `uv run pytest`, `npm test|run`, `make`,
  `ruff`, `mypy`, `pyright`, `uv run`, `python -m pytest`.
- `deploy` — `docker build`, `docker compose`, `kubectl get|describe`,
  `gh pr`, `gh run`.
- `all_non_denied` — sentinel: approve anything not matching a
  `deny_pattern` (legacy night-mode behavior).

Config (seeded by `install.py`):

```json
{
  "adaptive_windows": [
    {"name": "evening", "start_hour": 18, "end_hour": 23, "prefix_bundle": "read_only"},
    {"name": "night",   "start_hour": 23, "end_hour": 7,  "prefix_bundle": "all_non_denied"}
  ],
  "prefix_bundles": { "my_custom": ["npm run lint", "npm run build"] },
  "task_type_bundles": { "research": "read_only", "build": "build", "deploy": "deploy" }
}
```

**Task-type bundles.** The active task marker at
`~/.claude/mcp-skill-hub/state/active_task.json` may carry a `task_type`
field. When the per-task permissive override is ON
(`set_task_auto_approve(enabled=True)`), the bundle mapped by
`task_type_bundles[task_type]` is added **additively** to the task's own
`task_safe_prefixes`. It never reduces safety.

**Deny-pattern scoping (bug fix).** `deny_patterns` are now matched only
against *unquoted* fragments of the Bash command. The parser is a
direct stateful walk of the raw command (not `shlex.split`), so nested
escaped quotes such as `uv run python -c "print('rm -rf /')"` or
`echo "outer \"rm -rf /\" inner"` no longer leak deny strings into the
scan haystack. The pattern still blocks actual invocations like
`rm -rf /` and `cd /tmp && rm -rf /`. Covered by
`tests/test_auto_approve_adaptive.py`.

Priority chain (updated):
**static deny (scoped) > pinned > user_approved exact >
vector-similar (≥ threshold) > adaptive window bundle >
llm cache > static allow > prompt**.

Edit any of the above live from the **Settings** tab of the dashboard.

## 8. Vector-similarity classifier

Between the exact verdict-cache lookup and the optional LLM classifier,
`auto_approve.py` now runs a pure-python cosine search over
`user_approved` verdict rows that have stored embeddings. If the best
match is above `vector_autoapprove_threshold` (default 0.88), the
command is approved with `source=vector` and re-cached under its own
hash (so the next exact match is O(1)).

Priority chain: **static deny > pinned > user_approved exact >
vector-similar (≥ threshold) > llm cache > static allow > prompt**.

Safety: the vector classifier never denies — it can only upgrade
"unknown" to "allow", same guarantee as the LLM classifier. Static
`deny_patterns` remain the only path to a deny.

Config keys:
```
"vector_autoapprove_enabled": true,
"vector_autoapprove_threshold": 0.88,
"learn_from_claude_sessions": false
```

Typical latency: ~10ms embed + ~1ms/1000 rows cosine, vs 150-500ms for
the Ollama LLM fallback.
