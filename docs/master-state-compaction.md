# Master Project State Compaction

## What this solves

Long-running multi-repo systems accumulate per-task auto-memory faster than humans can curate. The `compact_master_state` MCP tool folds those deltas into a single page that captures what the system IS today, what's load-bearing, what's under active work, and what just changed — so cold-start sessions get the load-bearing rules before they touch a file, and superseded entries don't drown out current ones.

## Architecture

```
            ┌─────────────────────────────────┐
            │  /hub-compact-master-state      │  user invokes (or close_task wires it)
            │  close_task(compact_master_state=True)
            └────────────┬────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────┐
            │  server.compact_master_state    │  MCP tool — entry point, dry-run gate
            └────────────┬────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────┐
            │  master_state.compact_to_master_state   │  orchestrator
            │  ─ resolves project → auto-memory dir   │
            │  ─ lists recent project_*.md / feedback_*.md
            │  ─ reads existing snapshot section      │
            │  ─ MTIME NO-OP CHECK (skip LLM if fresh)│
            │  ─ summarizes memory entries            │
            │  ─ calls LLM helper                     │
            │  ─ renders to Markdown                  │
            │  ─ atomic upsert + backup + prune       │
            │  ─ appends assumptions to inbox.md      │
            └────────────┬────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────┐
            │  embeddings.compact_master_state │  LLM call
            │  ─ get_provider().complete(      │  routes to BEST configured model
            │      tier="tier_smart",          │  (Anthropic Opus/Sonnet > local Qwen)
            │      cache=True,                 │  prompt caching for cheap reruns
            │      timeout=240,                │
            │    )                             │
            │  ─ structured JSON output:       │
            │    architecture / invariants /   │
            │    active_modules / recent_pivots│
            │    / assumptions[]               │
            └─────────────────────────────────┘
```

## Files

| Path | Role |
|------|------|
| `src/skill_hub/server.py::compact_master_state` | MCP tool surface |
| `src/skill_hub/server.py::close_task` | Optional wiring via `compact_master_state=True` kwarg |
| `src/skill_hub/master_state.py` | File IO, slug resolution, rendering, upsert, backup, mtime no-op |
| `src/skill_hub/embeddings.py::compact_master_state` | LLM helper (routes through best-tier provider) |
| `src/skill_hub/llm/prompts/master_state.yaml` | Structured-JSON prompt template |
| `commands/hub-compact-master-state.md` | Slash command |
| `skills/master-state-compaction/SKILL.md` | When-to-use skill (documentation; not auto-indexed unless added to `extra_skill_dirs`) |
| `skills/memory-layer-analysis/SKILL.md` | Categorization methodology skill |
| `tests/test_master_state.py` | 23 unit tests |

## Safety properties

1. **Atomic writes.** `_atomic_write` uses `tempfile.mkstemp(dir=parent) + os.replace`. Same-directory tempfile guarantees the rename is a single inode swap on POSIX — a crash mid-write leaves either the old file intact or the new file complete, never a truncated one.
2. **Bounded backup retention.** `_prune_backups` keeps the 10 most-recent backups per file (`_BACKUP_RETENTION = 10`). Prevents unbounded growth in `.memory/.backups/`.
3. **MTIME no-op.** When `dry_run=False` and the existing snapshot's mtime ≥ newest auto-memory entry's mtime, return `{"status": "noop"}` WITHOUT calling the LLM. Saves cost; idempotent reruns are free.
4. **Dry-run bypass.** `dry_run=True` always renders, even if the snapshot is fresh — the user explicitly asked for a preview.
5. **Fail-soft in `close_task`.** If `compact_master_state=True` and the compaction errors, `close_task` logs a warning and returns the error in the response string but does NOT fail the task close. The dashboard-render error policy is the model.
6. **Assumptions surfaced, not buried.** The LLM output includes an `assumptions[]` field. These are written to `<project>/.memory/inbox.md` (the canonical staging area for inferred-not-confirmed claims) AND rendered in the snapshot under a clearly-marked "Assumptions to Verify" section. Inferences never silently masquerade as facts.

## Cost / quality trade-offs

- **`tier_smart`** routes to Claude Opus/Sonnet when an `ANTHROPIC_API_KEY` is set, falling back to the local "smart" Ollama model otherwise. Synthesis quality matters here because the snapshot is consumed at every cold start; the per-call premium is small relative to that compounding leverage.
- **Prompt caching (`cache=True`)** is a no-op on non-Anthropic providers, so it's safe to leave on always. On Anthropic, the second invocation on the same project pays ~0.1× the first because the existing snapshot + memory entries form a stable cacheable prefix.
- **MTIME no-op** is the cheapest quality-preserving optimization: zero token cost when nothing has changed.

## When `tier_smart` is the wrong call

Tactical helpers (rerank, query-rewrite, classify) stay on `tier_cheap` / `tier_mid` because their failure mode is "slightly worse search results", not "wrong cold-start context". Don't blanket-promote the rest of the codebase to `tier_smart`.

## Slug resolution

`_project_to_memory_dir(project_root)` walks parent directories to find the longest matching `~/.claude/projects/<slug>/memory/` where `<slug>` is the path-as-slug encoding (`-Users-ccancellieri-work-code-geoid` → walks to `-Users-ccancellieri-work-code` if the leaf has no `memory/` subdir). This matches Claude Code's actual layout where session JSONLs may live under `<project>-<subdir>/` slugs but the canonical `memory/` lives under the parent. Verified by `test_project_to_memory_dir_walks_up_to_parent_slug`.

## Dogfooding workflow

```bash
# 1. Always preview first
compact_master_state(project_root="~/work/code/geoid", dry_run=True)

# 2. Inspect the rendered snapshot, especially the "Assumptions to Verify" block
#    — these are the LLM's inferences, not observed facts. Verify each.

# 3. On approval, write
compact_master_state(project_root="~/work/code/geoid")

# 4. Or wire into task close (one-shot, automatic at end of shipped work)
close_task(task_id=42, compact_master_state=True)
```

## Verification

23 unit tests in `tests/test_master_state.py`:
- File IO: `_read_existing_section`, `_list_recent_memory`, `_summarize_memory_entries`
- Rendering: full payload + fallback marker
- Upsert: replace existing, prepend when missing, create file, idempotent
- Atomic writes: create + overwrite + no orphan tempfiles
- Backup retention: prune to N newest, no-op when under threshold, integration with upsert
- MTIME no-op: returns noop when snapshot fresher; dry_run bypasses
- Slug walk-up: parent-slug fallback when leaf has no `memory/`
- End-to-end: dry-run no-write, written-with-backup-and-inbox-append

`pytest tests/test_master_state.py -v` → 23/23 pass.
`pytest tests/test_close_task_intent.py tests/test_master_state.py tests/test_pipeline.py` → 51/51 pass (no adjacent regressions).
