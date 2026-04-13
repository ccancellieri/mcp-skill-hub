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

Enable once in `~/.claude/mcp-skill-hub/config.json`:

```json
{
  "auto_proceed": true,
  "auto_proceed_max": 20
}
```

When Claude stops and the newest plan under `~/.claude/plans/` still has
`- [ ]` items, the hook re-feeds `proceed` automatically. Counter per session
is tracked in `~/.claude/mcp-skill-hub/state/auto_proceed.json`; once the cap
is reached, normal Stop resumes.

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

## 6. Dashboard v2 (interactive)

`render_dashboard()` (and every `close_task`) now boots a tiny stdlib
`http.server.ThreadingHTTPServer` on loopback and returns its URL
(default `http://127.0.0.1:8765/`). The server is lazy-started, single
instance per MCP process, shuts down when MCP exits.

Features in the UI:
- **Verdicts tab** — browse, filter, delete, flip (allow↔deny), pin,
  bulk-promote selected commands to `~/.claude/skill-hub-allow.yml`.
- **Tasks tab** — rename, delete, merge (concatenates summaries into a
  new open task and closes originals), teach-from-task.
- **Skills tab** — usage stats (injections, helpful %, feedback score).
- **Vector Viz** — 2D random-projection scatter of skills / tasks /
  teachings / verdicts. Click a point for details.
- **Classifier** — paste a command to see its nearest verdict-cache
  neighbor and whether it would auto-approve at the current threshold.

Config keys (in `~/.claude/mcp-skill-hub/config.json`):
```
"dashboard_server_enabled": true,
"dashboard_server_port": 8765
```

If the port is busy or the server fails to bind, we silently fall back
to the static HTML snapshot at
`~/.claude/mcp-skill-hub/reports/dashboard.html` — `close_task` never
fails because of the dashboard.

## 7. Vector-similarity classifier

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
