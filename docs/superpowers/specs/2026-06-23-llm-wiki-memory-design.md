# LLM Wiki Knowledge Layer — Design Spec

**Date:** 2026-06-23
**Status:** Approved design, first slice (core loop)
**Scope:** Add an "LLM Wiki" persistent-knowledge layer to mcp-skill-hub, inverting today's vector-DB-as-source-of-truth memory model.

---

## 1. Context & Problem

Today's memory model is the RAG pattern Karpathy's *LLM Wiki* note contrasts against:

- **Source of truth = the SQLite `vectors` table** (`store.py`): flat rows of `namespace` + `doc_id` + JSON embedding, tiered L0–L4 with recency decay. Markdown is *input* (chunked → embedded one-way via `memory_index.index_user_memory`/`index_plugin_memory`) or *output* (`master_state.py` renders `.memory/decisions.md`) — never a living artifact.
- **Retrieval re-derives knowledge every query** (`search_context`, `server.py` ~1827 = vector KNN + level weight + recency). No persistent synthesis, no cross-references followed.
- **No links between entries** (only a one-way `task_issue_links` table). No graph.
- **No knowledge-base lint** — nothing checks contradictions, orphans, stale claims, missing cross-refs. (`lint_canary.py` lints *code*; `sync_check.py` is cross-repo grep.)
- Ingest sources (Discussions #41, Issues, sessions, activity #90) dump raw text + metadata. There is no "page" concept that gets *updated* across many files per source.

The **LLM Wiki pattern**: the LLM incrementally builds and maintains a persistent, interlinked markdown wiki that sits between the user and raw sources. Knowledge is compiled once and kept current, not re-derived per query. Three layers — raw sources (immutable) / wiki (LLM-owned interlinked markdown) / schema (a conventions doc). Operations: **ingest** (a source touches 10–15 pages, updating cross-refs), **query** (cited synthesis, answers filed back as pages), **lint** (find contradictions/orphans/stale). `index.md` = content catalog, `log.md` = append-only chronological journal. Obsidian is the human UI; the wiki is a git repo of markdown.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Invert source of truth**: interlinked markdown wiki pages become canonical; the SQLite vector index is rebuilt *from* the pages and demoted to a derived search accelerator. | True to the LLM Wiki pattern; keeps all existing search infra; lets us retire flat-memory-as-truth code. |
| D2 | **Graph = `[[wikilinks]]` in pages + a derived `wiki_edges` SQLite table** parsed from them. No graph DB. Obsidian renders the visual graph. | Graph semantics without operating Neo4j; markdown stays the single source of truth for links. |
| D3 | **Single global vault across all projects**, refined by D3a. | One cross-project Obsidian graph; cross-project pages without duplication. |
| D3a | **Two access scopes, not exclusion**: a public scope (`wiki` namespace, exported) and a per-project **private** scope (`wiki-private` namespace, `_private/<project>/` subtrees) that is **indexed and usable by the MCP for authorized projects** (e.g. career plugin → `_private/career/`; glicemia → `_private/glicemia/`) but **access-gated at query time** and **not exported by default**. | User requirement: index private data into a separate index so the MCP can use it for boundary projects, while honoring the hard glicemia/sovereign/PII rules. Reduces duplication (private knowledge lives once). |
| D4 | **First slice = core loop only**: privacy fix + page schema + ingest→pages + `index.md`/`log.md` + migration. Lint, query-file-back, Obsidian polish, webapp view, router/hook injection are deferred follow-ups. | YAGNI; keep the first PR reviewable. |

## 3. Architecture

### 3.1 Module & naming
- New module **`src/skill_hub/wiki.py`** (file IO, page model, edge derivation, ingest/query/reindex/migrate orchestration). **NB:** `src/skill_hub/vault.py` already exists and is the *credential* vault (keyring/age) — the wiki must never be called "vault" in code. Use `wiki_root`.
- LLM JSON producer lives in **`embeddings.py`** as `wiki_ingest(...)`, a twin of `compact_master_state` (~line 300), routed through `get_provider().complete(tier="tier_smart", cache=True)`.
- Reuse, do not re-implement: `master_state._atomic_write`, `_upsert_section`, `_prune_backups`, `_backup`; `memory_index._split_text` (chunker); `indexer._content_hash` (change detection); `store.upsert_vector`/`search_vectors`.

### 3.2 Vault layout (D3 + D3a)
```
~/.claude/mcp-skill-hub/wiki/          ← wiki_root (new config key)
  schema.md                            ← conventions doc (LLM-owned, CLAUDE.md-style)
  index.md                             ← content catalog: page → one-line summary, by type
  log.md                               ← append-only chronological journal
  pages/
    entity/   <slug>.md
    concept/  <slug>.md
    source/   <slug>.md                ← source-summary pages
    overview/ <slug>.md
    project/  <project>.md             ← one project-index landing page per project
  _private/
    <project>/<slug>.md                ← private scope; namespace wiki-private; access-gated
  .obsidian/                           ← gitignored workspace state
```
- **Public pages** foldered by *type*, scoped by frontmatter `projects: [...]`. Type-foldering keeps `[[slug]]` directory-agnostic and lets one page span N projects without duplication.
- **Private pages** foldered by *project* under `_private/`, because the privacy/authorization boundary is per-project. Indexed into the `wiki-private` namespace.
- **Slugs are globally unique** across the whole vault (one flat slug namespace) — required for `[[wikilink]]` resolution. Migration slugifier detects collisions and suffixes by project (`pipeline-geoid`).

### 3.3 Config keys (`config.py` `_DEFAULTS`)
```python
"wiki_enabled": True,
"wiki_root": str(Path.home() / ".claude" / "mcp-skill-hub" / "wiki"),
"wiki_private_scopes": {},   # {project: [authorized_reader_scope, ...]}; e.g. {"career": ["career"], "glicemia": ["glicemia"]}
"wiki_export_private": False, # private scope excluded from export unless explicitly opted in
```

### 3.4 Page model & frontmatter
Real YAML (`yaml.safe_load` — `pyyaml>=6.0` is already declared in `pyproject.toml`; the flat-regex `indexer._FM_FIELD_RE` parser stays for SKILL.md only).

```yaml
---
id: 01HX...            # stable ULID; never changes; join key across rename/retitle
slug: vectors-table    # globally-unique kebab slug; the [[wikilink]] target
title: "vectors table"
type: entity           # entity | concept | source | overview | project
projects: [skill-hub]  # [] illegal; [_global] = genuinely cross-cutting
scope: public          # public | private  (private ⇒ lives under _private/<project>/)
tags: [storage, sqlite]
aliases: ["vector store"]
source_refs:           # provenance to the immutable raw layer
  - "~/.claude/projects/-Users-..-mcp-skill-hub/memory/foo.md"
created: 2026-06-23
updated: 2026-06-23
---
Body with [[wikilinks]].
```
Required: `id, slug, title, type, projects, scope, created, updated`. Optional: `tags, aliases, source_refs`.
Page types: `entity` (a named thing), `concept` (idea/pattern/decision), `source` (one ingested raw input distilled), `overview` (synthesis of many pages), `project` (per-project landing page, replaces today's per-project `MEMORY.md`/`.memory/index.md` role).
Wikilinks: `[[slug]]`, `[[slug|display]]`, embeds `![[slug]]`. Filenames are `<slug>.md` so Obsidian resolution and our parser agree with zero config. Aliases emitted under Obsidian's own `aliases:` key for free interop.

### 3.5 Derived tables — `wiki_pages` + `wiki_edges`
Added to `SkillStore._ensure_schema` (`store.py` DDL block). **Fully rebuildable from markdown — markdown is SoT.**

```sql
CREATE TABLE IF NOT EXISTS wiki_pages (
    slug        TEXT PRIMARY KEY,        -- globally unique
    id          TEXT NOT NULL,           -- ULID, stable across rename
    title       TEXT NOT NULL,
    type        TEXT NOT NULL,
    scope       TEXT NOT NULL DEFAULT 'public',  -- public | private
    projects    TEXT NOT NULL,           -- JSON array
    tags        TEXT, aliases TEXT,       -- JSON arrays
    rel_path    TEXT NOT NULL,           -- relative to wiki_root
    updated     TEXT,
    indexed_at  TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_pages_id ON wiki_pages (id);

CREATE TABLE IF NOT EXISTS wiki_edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    src_slug  TEXT NOT NULL,
    dst_slug  TEXT NOT NULL,             -- post-alias-resolution
    dst_raw   TEXT NOT NULL,             -- exactly what was inside [[ ]]
    edge_kind TEXT NOT NULL DEFAULT 'wikilink',  -- wikilink | alias | embed
    project   TEXT,                      -- src page's projects[0]
    resolved  INTEGER NOT NULL DEFAULT 1,-- 0 = dangling (dst not in wiki_pages)
    UNIQUE(src_slug, dst_slug, edge_kind)
);
CREATE INDEX IF NOT EXISTS idx_wiki_edges_dst ON wiki_edges (dst_slug);
CREATE INDEX IF NOT EXISTS idx_wiki_edges_src ON wiki_edges (src_slug);
CREATE INDEX IF NOT EXISTS idx_wiki_edges_resolved ON wiki_edges (resolved);
```
- Edge extraction: regex `\[\[([^\]]+)\]\]` per body; split on first `|`; `![[...]]` ⇒ `edge_kind='embed'`. Resolve `dst_raw` → slug via `wiki_pages.slug` then `aliases`; unresolved ⇒ `resolved=0` (feeds deferred orphan/lint).
- Rebuild = `DELETE FROM wiki_edges; DELETE FROM wiki_pages;` then re-walk `pages/**/*.md` + `_private/**/*.md`. Idempotent.
- Queries: backlinks (`WHERE dst_slug=? AND resolved=1`), orphans (LEFT JOIN no inbound), dangling (`resolved=0`), traversal (recursive CTE), project subgraph (`WHERE project=?`).

### 3.6 Vector index (derived accelerator)
- Two namespaces seeded in `_DEFAULT_VECTOR_INDEXES` (`store.py` ~96): `"wiki"` and `"wiki-private"`, both `{"default_level": "L3", "half_life_days": 365.0}` (curated knowledge barely decays). **Must be added** or `upsert_vector` defaults to L2 and decays curated pages.
- `doc_id = "<page_id>#<section-anchor>"` (use stable `id`, not slug, so renames don't orphan vectors). Per-`##`-section chunking, then size-split oversized sections (reuse `memory_index._split_text`).
- `metadata = {slug, title, type, scope, projects[], section, rel_path, page_id}`; `source = "wiki"`.
- **`promote_memory` guarded with `namespace NOT IN ('wiki','wiki-private')`** — destructive promote/prune must never touch a derived-from-SoT index (would silently diverge it from markdown). **Load-bearing; ships in the same slice as the namespaces.**
- **#35 dim-mismatch guard:** record the embedding dim used for the `wiki`/`wiki-private` namespaces; `wiki_reindex` validates the active model's dim against the stored expectation and **fails loud** (raises/structured error) rather than silently no-op'ing.

### 3.7 Privacy & access-control model (D3a) — load-bearing
1. **Step 0 leak fix (blocking, independent bug):** `iter_user_memory_files` (`memory_index.py:239`) does `mem_dir.rglob("*.md")` with no filtering, embedding `private/` files into `memory:user-project` whenever `user_memory_enabled` (default `True`, `config.py:424`). Add `if "private" in f.parts: continue`. Regression test asserts a `private/foo.md` is never embedded. Export already excludes private (`scope.py:189`); only indexing leaks.
2. **Private scope indexed but gated:** `_private/<project>/` pages → `wiki-private` namespace with `scope=private`, `projects=[<project>]`.
3. **Query-time authorization:** `wiki_query`/`search_context` include `wiki-private` rows only for authorized scopes. Authorization derives from `wiki_private_scopes` config + the calling context (the active project / profile / plugin). A career-plugin context is authorized for `_private/career/`; a geoid session is authorized for neither glicemia nor career. Reuses the existing plugin-`reads` + `search_context_profile` (`profiles.py`) machinery rather than inventing new ACL.
4. **Cross-scope links allowed, retrieval gated:** a public page may `[[link]]` a private page; the edge is stored, but surfacing the private target's content is access-gated.
5. **Export:** `wiki-private` excluded from the memory-export bundle unless `wiki_export_private=True`. PII gate (`pii_gate.py`) guards every write path (public and private).

## 4. Operations

### 4.1 Ingest — `wiki.ingest_source(descriptor, dry_run=True)`
`SourceDescriptor = {kind, id, title, text, url, occurred_at, target_scope, target_project}`. Single normalized entrypoint for all current and future sources.

1. **Candidate discovery (deterministic, no LLM):** embed source text, `search_vectors(text, namespaces=["wiki"(+"wiki-private" if authorized target_scope)], top_k=15)` → candidate pages to touch. The derived index IS the "which pages to touch" oracle. Read candidate bodies (char-budgeted like `_summarize_memory_entries`).
2. **LLM call (`tier_smart`, `cache=True`, prompt `llm/prompts/wiki_ingest.yaml`):** returns JSON `{source_page:{slug,title,body}, page_updates:[{slug,title,type,scope,new_body,reason,is_new}], index_entries:[{slug,title,one_line}], assumptions:[...]}`. `new_body` is the *full rewritten page* (keeps the writer deterministic). LLM instructed to preserve existing `[[wikilinks]]` and add cross-refs.
3. **Deterministic write phase:** for each page, atomic write to `pages/<type>/<slug>.md` (or `_private/<project>/<slug>.md`) with **backup-before-overwrite**; code (not LLM) sets frontmatter `updated`/`source_refs`/`scope`; update `index.md`; append `log.md`; re-derive `wiki_edges` for touched pages; re-index touched pages into the right namespace. `assumptions[]` → `<wiki>/inbox.md` (mirrors master-state's inbox discipline).
4. **Idempotence:** `source_hash` in the source page frontmatter; unchanged source ⇒ skip the LLM call entirely.
5. **Dry-run default + fail-soft:** `dry_run=True` returns `{status:"dry_run", pages:[{slug,action,diff}], index_diff, log_line}` and writes nothing. As a hook side-effect, errors append to a log, never raise (per `.memory/patterns.md`).

**Rewiring existing sources** to build a `SourceDescriptor` → `ingest_source` instead of raw `upsert_vector`:
| Source | Rewire |
|---|---|
| `discussions_sync.sync_discussions` | `kind="discussion"` → `ingest_source`; retire the raw `discussions` namespace. Tool signature unchanged. |
| `issue_sync.reconcile` | unchanged for task-reconcile; add optional `ingest=True` (deferred ON) feeding resolved issues as `kind="issue"`. |
| session close | KEEP `session:log` (L1 short-term). Add `close_session(to_wiki=False)` seam; do not auto-route this slice. |

### 4.2 Query — `wiki_query(query, top_k=5, _file_back=False)`
Hybrid: read `index.md` for the curated candidate universe + grouping, rank with `search_vectors(namespaces=["wiki"(+authorized "wiki-private")])`, union top-K with index.md entries whose title lexically hits. Return ranked page bodies + `source_refs` provenance. Synthesis-with-citations left to the caller model this slice. `_file_back` is a reserved stub (`NotImplementedError`) → deferred query-file-back.

### 4.3 Reindex — `wiki_reindex(dry_run=False)`
`DELETE FROM vectors WHERE namespace IN ('wiki','wiki-private')`, then walk pages, re-derive `wiki_pages`+`wiki_edges`, re-`upsert_vector` every page section. Enforces D1 (pages canonical, index derived). Idempotent. Runs the #35 dim guard.

### 4.4 Status — `wiki_status()`
Stdlib-only (works in `no_llm_mode`): counts pages/edges/orphans (deterministic, not lint analysis), last `log.md` entry, **drift** (pages on disk vs namespace row counts). If drift > 0, `wiki_reindex` is authoritative.

## 5. MCP Tool Surface
Four tools, registered in `server.py` (`@mcp.tool` + `@requires_capability`) with matching `ToolSpec` rows in `capabilities.py` (tier auto-derived by `tier_from_spec` from the `hard` tuple):
- `wiki_ingest(source_kind, source_id, text="", url="", target_scope="public", target_project="", dry_run=True)` — cap `llm`, hard `(DB, EMBED, REASON_LLM)`.
- `wiki_query(query, top_k=5)` — cap `embedding`, hard `(DB, EMBED)`.
- `wiki_reindex(dry_run=False)` — cap `embedding`, hard `(DB, EMBED)`.
- `wiki_status()` — cap `none`, hard `()`.
Reserved seams (not built): `wiki_lint`, `wiki_file_answer`.

## 6. index.md & log.md
Both **deterministic code-written** (LLM only proposes `index_entries`; `wiki.py` owns file structure → consistent every ingest).
- `index.md`: grouped by type, alphabetized within group, one line `- [[slug]] — one-line summary` per page. Private entries listed only in an access-gated section (or omitted from the shared index; private index is per-scope).
- `log.md`: append-only `## [YYYY-MM-DD] <op> | <title> (<n> pages)`. The `## [` prefix is the parse contract (`grep "^## \[" log.md`). One line per op, never rewritten.

## 7. Migration (one-time, mechanical)
New `wiki migrate` command (idempotent — keyed on `source_refs`; an already-converted source is skipped). Writes pages only; operator then runs `wiki reindex` (vectors are regenerated, never hand-migrated).
- (a) Per-project `<project>/.memory/*.md` + `MEMORY.md` → `project/<project>.md` index page + one `source` page per file (`source_refs` to originals). Reuse `master_state._project_to_memory_dir`, `_strip_frontmatter` for reading; derive `projects:` from the project dir.
- (b) Global auto-memory `~/.claude/projects/<slug>/memory/*.md` (via `iter_user_memory_files`, **post-leak-fix**) → one `source` page each; cross-cutting files (e.g. `karpathy-coding-guidelines.md`) → `projects:[_global]`. **`private/**` files → `_private/<project>/`, `scope=private`** (indexed into `wiki-private`, not exported).
- (c) Operational vectors (`session:log`, `logs`, `habits:*`, `discussions`, `issues`, `tasks`) — **not migrated**; stay derived-only operational indexes. A finding worth permanence is promoted into a page via normal ingest, not migration.
- **Not migrated:** `inbox.md` (unconfirmed inferences). Originals kept on disk as the immutable raw layer.
- LLM entity/concept fan-out is left to live ingest — migration does the cheap 1-file→1-`source`-page mapping only (keeps the migration commit small and reviewable).

## 8. Wiring (minimal this slice)
- **Manual ingest only.** Master-state precedent (`decisions.md`): expensive curation ops default OFF in hooks. Wiki ingest is more expensive (10–15 page rewrites + LLM). Do **not** wire `session_end → wiki_ingest`.
- **One seeded-DISABLED cron job** `("wiki-reindex-nightly", "0 5 * * *", "wiki_reindex", enabled=0)` + handler (mirrors `log-digest-snapshot`). Inert until #96 starts the scheduler **and** the user enables it.
- **No prompt-router/hook changes.** `preloader._gather_context` consulting the wiki and `search_context` folding the `wiki` namespace are deferred — don't touch the hot path before the wiki has mass.

## 9. Drop / Keep
**Deprecate (not delete this slice — after migration is proven):**
- `master_state` `.memory/decisions.md` render path (superseded by ingest→`project/` pages; the `compact_to_master_state` LLM call may survive as a tool).
- `memory_index.index_user_memory` + `USER_MEMORY_NAMESPACE` (auto-memory now canonicalized into the wiki). Keep gated by `user_memory_enabled` until wiki ingest covers the same files.

**Keep (load-bearing — do not touch):** `compression/`, `router/`, `store.SkillStore` (`vectors`/`upsert_vector`/`search_vectors`), `pii_gate` (more important now), `indexer` (SKILL.md), `watcher` (extend later), `log_insights` (`logs` namespace), `discussions_sync` (rewired, not dropped), `memory-export` plugin (fix the project-key scoping, don't drop).

**Risky to remove — flag, don't touch:** `_DEFAULT_VECTOR_INDEXES["memory:user-project"]` seed (existing installs have data), `memory_index.index_plugin_memory` (plugins declare `reads` globs for a reason).

## 10. Risks & Mitigations (from adversarial review)
1. **Slug collisions across projects** → migration slugifier suffixes by project on collision; deferred lint enforces uniqueness.
2. **YAML vs legacy regex frontmatter** → wiki uses `yaml.safe_load`; `pyyaml>=6.0` confirmed in `pyproject.toml`.
3. **`promote_memory` corrupting the derived index** → namespace guard ships same slice (§3.6).
4. **`log.md`/`index.md` append conflicts in one global vault** → single-writer this slice (manual ingest); if concurrency arrives, switch `log.md` to per-day files under `log/`. Documented, not built now.
5. **Migration fan-out cost / large first commit** → mechanical 1-file→1-page only; LLM fan-out via live ingest.
6. **#35 dim-mismatch silent no-op** → loud-fail dim guard in `wiki_reindex` (§3.6).
7. **Consistency window / crash mid-ingest** → per-page atomic write + per-`upsert_vector` commit; SoT (markdown) survives; `wiki_status` drift + `wiki_reindex` reconcile.
8. **Rename breaks inbound links** → stable `id` join key; deferred lint rewrites `[[oldslug]]`→`[[newslug]]`. This slice: dangling edges captured as `resolved=0`.
9. **AI-context git boundary** → the wiki under `~/.claude/` is inside the never-commit AI-context tree. It is its own optional git repo (`.obsidian/` gitignored) and must never ride into a project commit. The mcp-skill-hub repo tracks only code + this spec, never the vault contents.

## 11. First-Slice Build Sequence (each step independently verifiable)
0. **Leak fix** in `iter_user_memory_files` + regression test. *Verify:* `private/foo.md` never embedded.
1. Add `wiki`/`wiki-private` namespaces (`store.py`); `wiki_pages`/`wiki_edges` DDL; `promote_memory` namespace guard; config keys. *Verify:* schema migrates; promote skips wiki namespaces.
2. `wiki.py` — YAML frontmatter parse, page render, edge derivation (regex). *Verify:* round-trip a page; extract edges incl. dangling.
3. `wiki_reindex` + `wiki_status` + tools. *Verify:* seed 2 hand pages → reindex → status reports 2 pages / N edges / 0 drift; dim guard fails loud on mismatch.
4. `embeddings.wiki_ingest` LLM producer + `wiki_ingest.yaml`. *Verify:* mock provider → assert JSON shape.
5. `wiki.ingest_source` (candidate discovery → LLM → deterministic write) + `wiki_ingest` tool (`dry_run=True` default) + access-gating. *Verify:* dry-run returns diffs, writes nothing; non-dry writes pages + index + log + backups; private target gated for unauthorized scope.
6. `wiki_query` (hybrid) + file-back stub. *Verify:* returns ranked pages w/ provenance; private excluded for unauthorized scope.
7. Rewire `discussions_sync` → `ingest_source`; retire `discussions` namespace. *Verify:* `discussions_sync(dry_run=True)` produces page diffs, not raw upserts.
8. `wiki migrate` (mechanical) + seeded-disabled cron job/handler. *Verify:* migrate is idempotent; `list_core_tasks` shows job disabled.

## 12. Deferred → Follow-up Issues
1. `wiki_lint` — contradictions / orphans / stale / missing-cross-refs (LLM, `tier_smart`).
2. query-file-back (`wiki_file_answer`).
3. `search_context` folds the `wiki` namespace + index.md-first; namespace priority above `logs`/`discussions`.
4. `preloader._gather_context` consults the wiki for prompt-router injection.
5. `watcher` watches the vault → incremental re-embed; `wiki_edges` reconciliation pass.
6. Cron scheduler live (**depends on #96**) → nightly `wiki_reindex` enabled.
7. `session_end` optional auto-ingest (`close_session(to_wiki=True)` seam exists).
8. Obsidian polish (graph defaults, tag colors, three-repo canvas); webapp wiki view.
9. memory-export: teach `intelligent_merge`/`build_snapshot` the wiki scope + `wiki_export_private` opt-in.

**Relationship to existing issues:** overlaps #87 (Discussions write-path), #42 (event-sourced umbrella — wiki pages could later be event projections), #90 (log indexing — unaffected, `logs` stays first-class), #96 (cron — blocks follow-up 6). Closes none outright; the core loop supersedes the ad-hoc `discussions` namespace.

## 13. Out of Scope (this slice)
Lint, query-file-back, router/hook injection, live cron, session auto-ingest, Obsidian/webapp views, export integration for the private scope, LLM-driven entity/concept fan-out during bulk migration.
