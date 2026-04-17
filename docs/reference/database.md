# Database Schema

**Location:** `~/.claude/mcp-skill-hub/skill_hub.db` (SQLite with `sqlite-vec` extension).

All persistent state — skills, embeddings, tasks, teachings, triage decisions, tool examples — lives in this single file. Back it up, delete it to reset, rsync it between machines.

## Tables

| Table | Purpose |
|-------|---------|
| `skills` | Skill metadata + full content + `target` (`claude` / `local`) + `feedback_score` EMA |
| `embeddings` | Skill vectors + pre-stored L2 `norm` (avoids recompute per search) |
| `feedback` | Raw `(query, skill, helpful)` history; EMA applied to `skills.feedback_score` |
| `teachings` | Explicit "when X suggest Y" rules |
| `plugins` | Plugin descriptions |
| `plugin_embeddings` | Plugin vectors |
| `tasks` | Open / closed task digests |
| `session_log` | Per-session tool usage (source for implicit feedback) |
| `interceptions` | Hook-intercepted command log for token profiling |
| `context_injections` | RAG context injection stats |
| `conversation_state` | Periodic conversation digests for relevance tracking |
| `triage_log` | LLM triage decisions and token savings (source for training export) |
| `session_context` | Per-session rolling summary + loaded skills for dynamic context |
| `tool_examples` | Real-time capture of Claude's tool calls (fed to local skills) |
| `repo_context` | Per-repo commit style, common commands, project type |
| `skill_versions` | Versioned snapshots of local skills before evolution — rollback safety |

## Vector indexing

- **Binary quantization** via sqlite-vec for fast KNN — 7.5× search speedup
- **Float32 rerank** over top-N candidates — preserves 97.3% recall@5
- **Hash-skip** on re-index: unchanged skills don't re-embed
- **Unified vec0 store** shared by skills + tasks + teachings — same ranking path for all

## Typical maintenance

```bash
# Full rebuild (e.g. after changing embed_model)
index_skills()
index_plugins()

# Inspect evolution history
sqlite3 ~/.claude/mcp-skill-hub/skill_hub.db \
  "SELECT skill_name, version, change_reason, created_at
     FROM skill_versions ORDER BY created_at DESC LIMIT 20;"

# Clean out old session logs (manual)
sqlite3 ~/.claude/mcp-skill-hub/skill_hub.db \
  "DELETE FROM session_log WHERE created_at < datetime('now', '-30 days');"
```

## Backup / restore

```bash
# Online backup while running (SQLite respects it)
sqlite3 ~/.claude/mcp-skill-hub/skill_hub.db ".backup /path/backup.db"

# Restore
cp /path/backup.db ~/.claude/mcp-skill-hub/skill_hub.db
```

## Related

- [reference/architecture.md](architecture.md) — how the DB plugs into the rest of the system
- [advanced/fine-tuning.md](../advanced/fine-tuning.md) — exporting accumulated signal as JSONL
