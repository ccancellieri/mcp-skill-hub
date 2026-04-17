# Semantic Search & Task Memory

## Skill search

Describe the task — get matching skill content, ranked by cosine similarity × your feedback EMA.

```python
search_skills("build an MCP server in Python")
search_skills("debug a failing pytest", use_rerank=True)  # LLM re-ranking
```

### Unified search

Search across **skills + tasks + teachings + plugins** in one call:

```python
search_context("accessibility audit for a website")
```

Under the hood, the same embedding is compared against each store and the results are merged and scored consistently — one threshold, one ranking.

---

## Cross-session task memory

Save open work so future sessions pick it up automatically.

### Save / close

```python
save_task(title="MCP skill hub dev",
          summary="Building semantic search…",
          tags="mcp,ollama")

close_task(task_id=1)   # LLM-compacted to ~200 tokens, processed locally
```

When you call `close_task()`, the local LLM (configured `reason_model`) distills the conversation into:

```json
{
  "title": "MCP Skill Hub development",
  "summary": "Built semantic skill search server with Ollama embeddings…",
  "decisions": ["SQLite over OpenSearch for local use", "nomic-embed-text for embeddings"],
  "tools_used": ["mcp-server-dev", "plugin-dev"],
  "open_questions": ["OpenSearch migration path"],
  "tags": "mcp,ollama,sqlite,skills"
}
```

~200 tokens stored vs ~5,000 for the raw conversation. The compact vector is matched on future `search_context()` calls.

### Auto-memory on close

When `auto_memory_on_close_task = true` (default), the local LLM evaluates the compacted digest and writes a memory entry to `MEMORY.md` if it judges the content substantive enough (quality threshold 0.4). Low-quality or trivial digests are silently skipped.

Closed tasks that matter get persisted — no manual `/hub-save-memory` calls.

### Browse

```python
list_tasks()                    # open tasks
list_tasks(status="closed")     # completed work
list_tasks(status="all")        # everything

update_task(3, summary="Added hook interception")
reopen_task(5)                  # revive a closed task
```

---

## Conversation digest & auto-eviction

Every N messages (default 5), the local LLM produces a compact digest:

```
/digest    # force a digest now
```

```
=== Conversation Digest ===

Messages in session: 15
Current focus: implementing session profiles for MCP skill hub

Recent decisions:
  - Use embedding similarity for profile auto-recommendation
  - Store profiles in config.json, not settings.json

Stale topics: CSS debugging, Terraform workspace setup
Suggested profile: mcp-dev
  Activate: /profile mcp-dev
```

The digest is auto-injected as `systemMessage` to keep Claude aware of the conversation's evolution. Stale topics are flagged so irrelevant context doesn't accumulate.

**Configure:**

```
configure(key="digest_every_n_messages", value="10")   # less frequent
configure(key="eviction_enabled", value="false")       # disable decay tracking
```

---

## Auto-save memory

Generate and save a memory entry from the current session:

```
/save-memory                               # from session context
/save-memory "decided to use SQLite"       # from explicit description
```

The local LLM picks a type (user / feedback / project / reference), writes the file, and updates `MEMORY.md` automatically.

---

## Related

- [hooks.md](hooks.md) — how context injection decides which skills to load
- [reference/tools.md](../reference/tools.md) — full MCP tool reference
