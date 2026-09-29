# MCP tool profiles

The `skill-hub` server starts with the **full** MCP tool profile by default.
Existing client configurations therefore keep their current tool surface.
Select the smaller surface explicitly in the server command:

```sh
skill-hub --profile minimal
```

The profile is fixed for that server process. Restart the MCP server after
changing its command. `skill-hub --profile full` explicitly selects the default;
an unknown profile exits with an argument error. This setting is separate from
saved plugin profiles and does not change client permissions or model providers.
Minimal startup does not launch the dashboard, service reconciler, memory sweep,
or reindex sweep. Run indexing or the dashboard separately when needed; the
minimal process consumes the existing local index.

## Prepare an index without a model

From a package installation, explicitly choose the skill directories to index:

```sh
skill-hub-cli index_skills_text --skill-dir /absolute/path/to/skills
```

Repeat `--skill-dir` to include additional directories. This command scans only
their `SKILL.md` files, stores text for keyword search, and makes no embedding
call. It does not index project memory, install hooks or modify providers.
Existing vector-backed rows are not replaced with changed text and stale
vectors: those updates require the full indexing path and report an error.
Invalid directories and indexing errors return a nonzero exit status.

`--db /absolute/path/index.db` allows isolated indexing checks. A server launched
normally still uses its default store, so omit this option when preparing the
index for your configured MCP server. An empty index legitimately returns no
matches; it does not trigger a model download or fallback.

For the web composer without automatic background startup, separately run
`skill-hub-dashboard --no-services` and open `/context`. This skips service
reconciliation, scheduled maintenance and the health watcher. It does not stop
already-running services or disable explicitly requested operations in the UI.

## Tool surface

The minimal profile advertises these eight tools:

| Tool | Use |
| --- | --- |
| `prepare_context` | Retrieve bounded evidence for the original prompt and verified project path. |
| `prepare_composition` | Prepare deterministic candidates for manual selection. |
| `compose_context` | Compose selected, verified sources without training feedback. |
| `expand_context_candidate` | Recover a candidate's original indexed source. |
| `search_skills` | Search indexed skill descriptions with local keyword search. |
| `get_skill_content` | Load a chosen skill's full text explicitly. |
| `retrieve_compressed` | Recover an original behind a reversible compression marker. |
| `optimize_prompt_deterministic` | Compare conservative prompt compression explicitly. |

Minimal `search_skills` does not invoke embeddings, reranking, or response
compression. It returns descriptions and references to `get_skill_content`;
`use_rerank=True` and `include_content=True` are rejected. Minimal composition
accepts only `mode="manual"` and rejects `confirmed=True`, so it cannot create
training feedback. The full profile retains the original `search_skills`
full-content default and explicitly selected composition research modes.
Selector learning functions remain available for offline research, outside
the MCP tool surface. Both profiles preserve the original prompt and the
same verified project-scope rules in context retrieval.

To compare the actual advertised schemas in a client, call MCP `tools/list`
against each process. Removed tools are absent from that list and cannot be
called through MCP `tools/call` in the minimal process.
