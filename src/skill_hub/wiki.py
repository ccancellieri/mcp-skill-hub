"""LLM Wiki knowledge layer — core file IO, page model, edge derivation, reindex, status.

The wiki is a directory of interlinked Markdown pages that serve as the
canonical source of truth. The SQLite tables (wiki_pages, wiki_edges) and the
vector namespace (wiki / wiki-private) are fully derived from the pages and
rebuilt by ``reindex``. Never edit the derived tables directly.

Vault layout:
    <wiki_root>/
        pages/<type>/<slug>.md    — public pages, foldered by type
        _private/<project>/<slug>.md — private pages, gated at query time

NB: "Vault" here means the wiki content directory, not credentials.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page model
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 100


@dataclass
class WikiPage:
    """In-memory representation of one wiki page."""
    id: str                          # stable ULID; join key across renames
    slug: str                        # globally-unique kebab slug
    title: str
    type: str                        # entity|concept|source|overview|project
    projects: list[str]              # [] illegal; [_global] = cross-cutting
    scope: str                       # public | private
    body: str                        # body text (everything after frontmatter)
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    created: str = ""                # ISO date string
    updated: str = ""                # ISO date string
    source_hash: str = ""            # sha256 of the source text at last ingest


@dataclass
class WikiEdge:
    """One wikilink extracted from a page body."""
    src_slug: str
    dst_raw: str    # exactly what was inside [[ ]]
    dst_slug: str   # resolved (may equal dst_raw if no alias)
    edge_kind: str  # wikilink | embed


# ---------------------------------------------------------------------------
# Frontmatter parsing and rendering
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a ``---\\n...\\n---\\n`` YAML block from the rest of the text.

    Returns ``(frontmatter_dict, body)``. If the file has no frontmatter block
    the dict is empty and body is the full text.
    """
    if not text.startswith("---"):
        return {}, text

    # Skip the opening "---" plus its trailing newline.
    rest = text[3:]
    nl = rest.find("\n")
    if nl == -1:
        return {}, text
    # Allow trailing spaces/CR on the opening fence line.
    if rest[:nl].strip():
        return {}, text  # "---" must be alone on its line
    rest = rest[nl + 1:]     # content after the opening fence line

    # Find the closing "---" fence: either at the very start (empty body)
    # or preceded by "\n".
    if rest.startswith("---"):
        close = -1
        fm_text = ""
        after = rest[3:]
    else:
        close = rest.find("\n---")
        if close == -1:
            return {}, text
        fm_text = rest[:close]
        after = rest[close + 4:]  # skip "\n---"

    # The body may start with "\n" — strip exactly one leading newline.
    if after.startswith("\n"):
        after = after[1:]

    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        return {}, text

    if not isinstance(fm, dict):
        return {}, text

    return fm, after


def render_page(page: WikiPage) -> str:
    """Serialize a WikiPage back to ``---``-fenced YAML frontmatter + body.

    Round-trips with ``parse_frontmatter``.
    """
    fm: dict[str, Any] = {
        "id": page.id,
        "slug": page.slug,
        "title": page.title,
        "type": page.type,
        "projects": page.projects,
        "scope": page.scope,
        "created": page.created,
        "updated": page.updated,
    }
    if page.tags:
        fm["tags"] = page.tags
    if page.aliases:
        fm["aliases"] = page.aliases
    if page.source_refs:
        fm["source_refs"] = page.source_refs
    if page.source_hash:
        fm["source_hash"] = page.source_hash

    fm_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True,
                        sort_keys=False)
    return f"---\n{fm_text}---\n{page.body}"


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_LINK_RE = re.compile(r"(?<!!)(\[\[([^\]]+)\]\])")


def extract_edges(slug: str, body: str) -> list[WikiEdge]:
    """Extract all ``[[wikilinks]]`` and ``![[embeds]]`` from a page body.

    Splits on the first ``|`` to separate slug from display alias.
    ``![[target]]`` produces ``edge_kind='embed'``.
    ``[[target]]`` / ``[[target|display]]`` produce ``edge_kind='wikilink'``.
    """
    edges: list[WikiEdge] = []

    # Process embeds first so they don't match the plain-link regex.
    seen: set[tuple[str, str]] = set()

    for m in _EMBED_RE.finditer(body):
        inner = m.group(1)
        dst_raw = inner.split("|")[0].strip()
        key = (dst_raw, "embed")
        if key not in seen:
            seen.add(key)
            edges.append(WikiEdge(
                src_slug=slug, dst_raw=dst_raw,
                dst_slug=dst_raw, edge_kind="embed",
            ))

    for m in _LINK_RE.finditer(body):
        inner = m.group(2)
        dst_raw = inner.split("|")[0].strip()
        key = (dst_raw, "wikilink")
        if key not in seen:
            seen.add(key)
            edges.append(WikiEdge(
                src_slug=slug, dst_raw=dst_raw,
                dst_slug=dst_raw, edge_kind="wikilink",
            ))

    return edges


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def page_path(wiki_root: Path, page: WikiPage) -> Path:
    """Return the canonical filesystem path for a page.

    Public  → ``<wiki_root>/pages/<type>/<slug>.md``
    Private → ``<wiki_root>/_private/<projects[0]>/<slug>.md``
    """
    if page.scope == "private":
        project = page.projects[0] if page.projects else "_global"
        return wiki_root / "_private" / project / f"{page.slug}.md"
    return wiki_root / "pages" / page.type / f"{page.slug}.md"


# ---------------------------------------------------------------------------
# Page-from-file loading
# ---------------------------------------------------------------------------

def _load_page(path: Path) -> WikiPage | None:
    """Parse a markdown file into a WikiPage. Returns None on error."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("wiki: cannot read %s: %s", path, exc)
        return None

    fm, body = parse_frontmatter(text)

    slug = fm.get("slug") or path.stem
    page_id = fm.get("id") or slug
    title = fm.get("title") or slug
    page_type = fm.get("type") or "entity"
    projects = fm.get("projects") or []
    if isinstance(projects, str):
        projects = [projects]
    scope = fm.get("scope") or "public"
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    aliases = fm.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    source_refs = fm.get("source_refs") or []
    if isinstance(source_refs, str):
        source_refs = [source_refs]

    created = str(fm.get("created") or "")
    updated = str(fm.get("updated") or "")
    source_hash = str(fm.get("source_hash") or "")

    # Derive scope from directory if not in frontmatter.
    if not fm.get("scope") and "_private" in path.parts:
        scope = "private"

    return WikiPage(
        id=page_id, slug=slug, title=title, type=page_type,
        projects=list(projects), scope=scope, body=body,
        tags=list(tags), aliases=list(aliases),
        source_refs=list(source_refs), created=created, updated=updated,
        source_hash=source_hash,
    )


# ---------------------------------------------------------------------------
# Reindex (deterministic, no LLM)
# ---------------------------------------------------------------------------

def reindex(store: Any, wiki_root: Path, dry_run: bool = False) -> dict:
    """Walk wiki pages, rebuild wiki_pages/wiki_edges, re-embed into vector namespaces.

    Delete-then-reinsert makes this fully idempotent.

    #35 dim-guard: if the active embedding model's dimension differs from
    what the store expects, raises ``ValueError`` loudly rather than
    silently no-op'ing.

    Returns counts: {pages, edges, vectors, dry_run}.
    """
    wiki_root = Path(wiki_root)

    # Discover page files.
    public_files: list[Path] = []
    private_files: list[Path] = []

    pages_root = wiki_root / "pages"
    private_root = wiki_root / "_private"

    if pages_root.exists():
        public_files = [p for p in pages_root.rglob("*.md") if p.is_file()]
    if private_root.exists():
        private_files = [p for p in private_root.rglob("*.md") if p.is_file()]

    all_files = public_files + private_files

    if dry_run:
        return {
            "dry_run": True,
            "pages": len(all_files),
            "edges": 0,
            "vectors": 0,
        }

    # ---- dim guard ----
    _check_dim_guard(store)

    conn = store._conn

    # Clear derived tables and wiki vectors.
    conn.execute("DELETE FROM wiki_edges")
    conn.execute("DELETE FROM wiki_pages")
    conn.execute("DELETE FROM vectors WHERE namespace IN ('wiki','wiki-private')")
    conn.commit()

    # Load pages.
    pages: list[WikiPage] = []
    for path in all_files:
        p = _load_page(path)
        if p is None:
            continue
        # Infer scope from location if needed.
        if "_private" in path.parts:
            p.scope = "private"
        pages.append(p)

    # Build slug → id and alias → slug lookup for edge resolution.
    slug_set: set[str] = {p.slug for p in pages}
    alias_to_slug: dict[str, str] = {}
    for p in pages:
        for alias in p.aliases:
            alias_to_slug.setdefault(alias, p.slug)

    # Insert wiki_pages rows.
    for p in pages:
        rel_path = str(_page_rel_path(wiki_root, p))
        conn.execute(
            """
            INSERT OR REPLACE INTO wiki_pages
                (slug, id, title, type, scope, projects, tags, aliases, rel_path, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.slug, p.id, p.title, p.type, p.scope,
                json.dumps(p.projects),
                json.dumps(p.tags) if p.tags else None,
                json.dumps(p.aliases) if p.aliases else None,
                rel_path,
                p.updated or None,
            ),
        )
    conn.commit()

    # Extract and insert edges.
    edge_count = 0
    for p in pages:
        raw_edges = extract_edges(p.slug, p.body)
        project = p.projects[0] if p.projects else None
        for e in raw_edges:
            # Resolve dst_raw → slug via slug set then alias map.
            if e.dst_raw in slug_set:
                dst_slug = e.dst_raw
                resolved = 1
            elif e.dst_raw in alias_to_slug:
                dst_slug = alias_to_slug[e.dst_raw]
                resolved = 1
            else:
                dst_slug = e.dst_raw
                resolved = 0
            try:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO wiki_edges
                        (src_slug, dst_slug, dst_raw, edge_kind, project, resolved)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (p.slug, dst_slug, e.dst_raw, e.edge_kind, project, resolved),
                )
                edge_count += cur.rowcount
            except Exception as exc:  # noqa: BLE001
                _log.warning("wiki: edge insert failed for %s→%s: %s",
                             p.slug, e.dst_raw, exc)
    conn.commit()

    # Embed page sections into vector namespaces.
    vec_count = 0
    for p in pages:
        namespace = "wiki-private" if p.scope == "private" else "wiki"
        sections = _split_page_sections(p)
        rel_path = _page_rel_path(wiki_root, p)
        for section_anchor, section_text in sections:
            doc_id = f"{p.id}#{section_anchor}" if section_anchor else p.id
            meta = {
                "slug": p.slug,
                "title": p.title,
                "type": p.type,
                "scope": p.scope,
                "projects": p.projects,
                "section": section_anchor,
                "rel_path": str(rel_path),
                "page_id": p.id,
            }
            try:
                store.upsert_vector(
                    namespace=namespace,
                    doc_id=doc_id,
                    text=section_text,
                    metadata=meta,
                    source="wiki",
                    project=p.projects[0] if p.projects else None,
                    tags=p.tags or None,
                )
                vec_count += 1
            except Exception as exc:  # noqa: BLE001
                _log.warning("wiki: embed failed for %s %s: %s",
                             p.slug, doc_id, exc)

    return {
        "dry_run": False,
        "pages": len(pages),
        "edges": edge_count,
        "vectors": vec_count,
    }


def _remove_page_by_rel_path(store: Any, wiki_root: Path, rel_path_str: str) -> bool:
    """Delete one page's wiki_pages row, outgoing wiki_edges, and vector chunks.

    Looks up the page by its ``rel_path`` column (the canonical relative path
    stored at index time).  Returns True when a row was found and deleted,
    False when the path was not in the index (already gone or never indexed).
    """
    conn = store._conn
    row = conn.execute(
        "SELECT slug, id, scope FROM wiki_pages WHERE rel_path=?",
        (rel_path_str,),
    ).fetchone()
    if row is None:
        return False
    slug = row["slug"] if not isinstance(row, tuple) else row[0]
    page_id = row["id"] if not isinstance(row, tuple) else row[1]
    scope = row["scope"] if not isinstance(row, tuple) else row[2]
    namespace = "wiki-private" if scope == "private" else "wiki"

    conn.execute("DELETE FROM wiki_edges WHERE src_slug=?", (slug,))
    conn.execute("DELETE FROM wiki_pages WHERE slug=?", (slug,))
    conn.execute(
        "DELETE FROM vectors WHERE namespace=? AND (doc_id=? OR doc_id LIKE ?)",
        (namespace, page_id, f"{page_id}#%"),
    )
    conn.commit()
    return True


def reindex_paths(
    store: Any,
    wiki_root: Path,
    *,
    changed: set[Path] | None = None,
    deleted: set[Path] | None = None,
) -> dict:
    """Incremental re-embed for a known set of changed/deleted vault paths.

    Called by the vault watcher instead of a full ``reindex`` when the exact
    set of touched ``.md`` files is known.

    - **deleted**: for each path, look up the page by ``rel_path`` in the DB,
      remove its ``wiki_pages`` row, outgoing ``wiki_edges``, and all vector
      chunks.  Also reconcile *inbound* edges: any edge whose ``dst_slug``
      matched the deleted page is marked unresolved (``resolved=0``).
    - **changed** (created or modified): load the page from disk; if the file
      no longer exists (race), treat it as a delete instead.  Call
      :func:`_index_pages` to upsert the page row, rebuild its outgoing edges,
      and re-embed its sections.

    Returns ``{pages_updated, pages_deleted, vectors, fallback=False}``.

    Raises ``ValueError`` on a dim-guard conflict (mirrors ``reindex``).
    """
    wiki_root = Path(wiki_root)
    _check_dim_guard(store)

    pages_deleted = 0
    pages_updated = 0
    vec_count = 0

    def _rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(wiki_root.resolve()))
        except ValueError:
            return str(path)

    # --- handle deletions ---
    for del_path in (deleted or set()):
        rel = _rel(del_path)
        removed = _remove_page_by_rel_path(store, wiki_root, rel)
        if removed:
            pages_deleted += 1
            # Mark inbound edges dangling (dst no longer exists).
            slug_row = None  # already deleted — reconstruct slug from stem
            # Best-effort: use the file stem as the likely slug.
            slug_guess = del_path.stem
            store._conn.execute(
                "UPDATE wiki_edges SET resolved=0 WHERE dst_slug=?",
                (slug_guess,),
            )
            store._conn.commit()

    # --- handle creates / modifies ---
    # Also catch changed paths where the file was already gone (race → treat as delete).
    for chg_path in (changed or set()):
        if not chg_path.exists():
            # File vanished between event and timer fire — treat as deletion.
            rel = _rel(chg_path)
            removed = _remove_page_by_rel_path(store, wiki_root, rel)
            if removed:
                pages_deleted += 1
            continue
        page = _load_page(chg_path)
        if page is None:
            continue
        if "_private" in chg_path.parts:
            page.scope = "private"
        vec_count += _index_pages(store, wiki_root, [page])
        pages_updated += 1

    return {
        "pages_updated": pages_updated,
        "pages_deleted": pages_deleted,
        "vectors": vec_count,
        "fallback": False,
    }


def _check_dim_guard(store: Any) -> None:
    """Raise ValueError if the stored vec_dim conflicts with the active model's dim.

    Mirrors the check in store._ensure_vec_dim: if the store has already
    recorded a dimension (via _meta_get('vec_dim')) and the active embedding
    model returns a different dimension, the wiki index would be silently
    corrupt. Fail loud instead (#35).
    """
    stored_dim = store._meta_get("vec_dim")
    if stored_dim is None:
        # No dimension recorded yet — first embed call will set it.
        return
    stored_dim_int = int(stored_dim)

    # Detect active embedding dim without embedding a long text.
    active_dim = store._detect_embedding_dim()
    if active_dim is None:
        # Embedding backend not available; skip the guard.
        return

    if active_dim != stored_dim_int:
        raise ValueError(
            f"wiki_reindex: embedding dim mismatch — stored={stored_dim_int}, "
            f"active={active_dim}. Run with the same embedding model used for "
            "the existing index, or wipe the vector store and reindex."
        )


# ---------------------------------------------------------------------------
# Status (stdlib-only — no embedding required)
# ---------------------------------------------------------------------------

def status(store: Any, wiki_root: Path) -> dict:
    """Return a summary of the wiki's current state.

    Stdlib-only: works in no_llm_mode. Counts pages, edges, orphans (pages
    with no inbound resolved edge), dangling edges (resolved=0), last log.md
    line, and drift (pages on disk vs wiki/wiki-private vector row counts).
    """
    wiki_root = Path(wiki_root)
    conn = store._conn

    page_count: int = conn.execute(
        "SELECT COUNT(*) FROM wiki_pages"
    ).fetchone()[0]
    edge_count: int = conn.execute(
        "SELECT COUNT(*) FROM wiki_edges"
    ).fetchone()[0]
    dangling: int = conn.execute(
        "SELECT COUNT(*) FROM wiki_edges WHERE resolved = 0"
    ).fetchone()[0]
    orphan_count: int = conn.execute(
        """
        SELECT COUNT(*) FROM wiki_pages wp
        WHERE NOT EXISTS (
            SELECT 1 FROM wiki_edges we
            WHERE we.dst_slug = wp.slug AND we.resolved = 1
        )
        """
    ).fetchone()[0]

    # Vector row counts for drift detection.
    vec_count: int = conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE namespace IN ('wiki','wiki-private')"
    ).fetchone()[0]

    # Disk page count (files on disk, independent of DB).
    disk_count = 0
    pages_root = wiki_root / "pages"
    private_root = wiki_root / "_private"
    if pages_root.exists():
        disk_count += sum(1 for _ in pages_root.rglob("*.md"))
    if private_root.exists():
        disk_count += sum(1 for _ in private_root.rglob("*.md"))

    # Last log.md line.
    last_log: str = ""
    log_path = wiki_root / "log.md"
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if line.strip():
                    last_log = line.strip()
                    break
        except OSError:
            pass

    return {
        "pages_db": page_count,
        "pages_disk": disk_count,
        "edges": edge_count,
        "orphans": orphan_count,
        "dangling_edges": dangling,
        "vec_rows": vec_count,
        "drift": abs(page_count - disk_count),
        "last_log": last_log,
    }


# ---------------------------------------------------------------------------
# Structural lint (deterministic, stdlib-only, no LLM)
# ---------------------------------------------------------------------------

# TODO: LLM-based contradiction detection across pages is intentionally
# deferred — it requires a full pass over 1170+ pages and is expensive.

_LINT_SLUG_CAP = 50  # max slugs to include per offending list


def lint(store: Any, wiki_root: Path) -> dict:
    """Deterministic structural lint of the wiki vault. No LLM, stdlib-only.

    Checks:
    - **dangling_edges**: wiki_edges rows whose dst_slug resolves to no
      known page (resolved=0 in the DB). Keyed by src_slug → dst_raw.
    - **orphans**: pages with no inbound resolved edge. A page is an orphan
      when no other page links to it (dst_slug == slug, resolved=1).
    - **stale_pages**: source pages whose on-disk source file hash differs
      from the stored ``source_hash`` frontmatter field, meaning the source
      changed since the last ingest.

    Returns a dict with counts and capped offending-slug lists.
    """
    wiki_root = Path(wiki_root)
    conn = store._conn

    # --- Dangling edges (resolved=0) -----------------------------------------
    dangling_rows = conn.execute(
        "SELECT src_slug, dst_raw FROM wiki_edges WHERE resolved = 0"
    ).fetchall()
    dangling_pairs = [(r[0], r[1]) for r in dangling_rows]
    dangling_count = len(dangling_pairs)
    dangling_sample = [
        f"{src}→{dst}" for src, dst in dangling_pairs[:_LINT_SLUG_CAP]
    ]

    # --- Orphans (pages with no inbound resolved edge) -----------------------
    orphan_rows = conn.execute(
        """
        SELECT wp.slug FROM wiki_pages wp
        WHERE NOT EXISTS (
            SELECT 1 FROM wiki_edges we
            WHERE we.dst_slug = wp.slug AND we.resolved = 1
        )
        ORDER BY wp.slug
        """
    ).fetchall()
    orphan_slugs = [r[0] for r in orphan_rows]
    orphan_count = len(orphan_slugs)
    orphan_sample = orphan_slugs[:_LINT_SLUG_CAP]

    # --- Stale pages (source_hash drift) -------------------------------------
    # Walk disk pages (the source of truth). Compare their frontmatter
    # source_hash against the hash of the source_refs[0] file.
    stale_slugs: list[str] = []
    pages_root = wiki_root / "pages"
    private_root = wiki_root / "_private"
    candidate_files: list[Path] = []
    if pages_root.exists():
        candidate_files += list(pages_root.rglob("*.md"))
    if private_root.exists():
        candidate_files += list(private_root.rglob("*.md"))

    for path in candidate_files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm, _ = parse_frontmatter(text)
        stored_hash = fm.get("source_hash") or ""
        refs = fm.get("source_refs") or []
        if not stored_hash or not refs:
            continue
        source_path = Path(str(refs[0]))
        if not source_path.is_file():
            continue
        try:
            live_hash = _hash_text(source_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if live_hash != stored_hash:
            slug = fm.get("slug") or path.stem
            stale_slugs.append(slug)

    stale_count = len(stale_slugs)
    stale_sample = sorted(stale_slugs)[:_LINT_SLUG_CAP]

    return {
        "dangling_edges": dangling_count,
        "dangling_sample": dangling_sample,
        "orphans": orphan_count,
        "orphan_sample": orphan_sample,
        "stale_pages": stale_count,
        "stale_sample": stale_sample,
        "total_issues": dangling_count + orphan_count + stale_count,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _page_rel_path(wiki_root: Path, page: WikiPage) -> Path:
    """Relative path of a page from wiki_root (mirrors page_path logic)."""
    full = page_path(wiki_root, page)
    try:
        return full.relative_to(wiki_root)
    except ValueError:
        return full


def _slugify_heading(heading: str) -> str:
    """Convert a markdown heading to a URL-safe anchor slug."""
    h = heading.strip().lstrip("#").strip().lower()
    h = re.sub(r"[^a-z0-9]+", "-", h)
    return h.strip("-") or "section"


def _split_page_sections(page: WikiPage) -> list[tuple[str, str]]:
    """Split a page body into (anchor, text) chunks for embedding.

    Splits on ``##`` headings, then size-splits oversized sections
    (reusing memory_index._split_text). A single-chunk page returns
    ``[("", full_body)]`` so doc_id = ``<page_id>`` with no anchor.
    """
    from .memory_index import _split_text

    body = page.body.strip()
    if not body:
        return [("", f"{page.title}\n\n{page.slug}")]

    # Split on ## headings.
    heading_re = re.compile(r"^(##[^#][^\n]*)", re.MULTILINE)
    parts = heading_re.split(body)

    if len(parts) <= 1:
        # No ## headings — treat whole body as a single section.
        chunks = _split_text(body, _CHUNK_SIZE, _CHUNK_OVERLAP)
        if len(chunks) == 1:
            return [("", chunks[0])]
        return [(f"chunk-{i}", c) for i, c in enumerate(chunks)]

    sections: list[tuple[str, str]] = []
    # parts[0] is pre-heading content; parts[1::2] are headings, parts[2::2] are bodies.
    pre = parts[0].strip()
    if pre:
        sections.append(("", pre))

    heading_texts = parts[1::2]
    body_texts = parts[2::2]
    for heading, section_body in zip(heading_texts, body_texts):
        anchor = _slugify_heading(heading)
        text = f"{heading}\n{section_body}".strip()
        sub_chunks = _split_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
        if len(sub_chunks) == 1:
            sections.append((anchor, sub_chunks[0]))
        else:
            for i, chunk in enumerate(sub_chunks):
                sections.append((f"{anchor}-{i}", chunk))

    return sections if sections else [("", body)]


# ---------------------------------------------------------------------------
# Migration — mechanical one-file→one-source-page conversion
# ---------------------------------------------------------------------------

# Source-directory-name fragments that map to a private scope.
# Keys are substrings matched against the encoded project dir name (case-insensitive).
_PRIVATE_DIR_KEYWORDS: dict[str, str] = {
    "diabete": "glicemia",
    "glicemia": "glicemia",
}

# Filenames to skip unconditionally during migration.
_SKIP_FILENAMES: frozenset[str] = frozenset({"inbox.md"})


def _stable_page_id(source_path: Path) -> str:
    """Return a stable, sortable page id derived deterministically from the source path.

    Using the absolute path ensures the same file always gets the same id across
    re-runs (idempotency).  The format is ``src-<hex12>`` so it sorts stably and
    is clearly machine-generated.
    """
    digest = hashlib.sha1(str(source_path).encode()).hexdigest()
    return f"src-{digest[:16]}"


def _slugify(text: str) -> str:
    """Convert arbitrary text to a URL-safe kebab slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "page"


def _extract_h1_title(text: str) -> str | None:
    """Return the first # heading from markdown text, or None."""
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+)", line)
        if m:
            return m.group(1).strip()
    return None


def _humanize(stem: str) -> str:
    """Turn a filename stem like ``karpathy-coding-guidelines`` into a title."""
    return stem.replace("-", " ").replace("_", " ").title()


def _file_iso_date(path: Path) -> str:
    """Return the file's mtime as an ISO date string."""
    try:
        mtime = path.stat().st_mtime
        import datetime
        return datetime.date.fromtimestamp(mtime).isoformat()
    except OSError:
        return date.today().isoformat()


def _resolve_scope_and_project(
    source_path: Path,
    mem_dir: Path,
    project_label: str,
    private_scopes: dict[str, list[str]],
) -> tuple[str, str]:
    """Return (scope, project_name) for a source file.

    Private conditions (in priority order):
    1. File is under a ``private/`` subdirectory within mem_dir.
    2. The project_label (encoded dir name) contains a private-dir keyword
       (e.g. ``diabete`` → scope ``glicemia``).
    3. A key from ``private_scopes`` appears as a substring of project_label
       (handles encoded paths like ``-Users-user-work-career-skill-hub-plugin``
       matching the key ``career``).

    Returns:
        scope: "private" or "public"
        project_name: the scope name (from keyword map or matched key)
    """
    label_lower = project_label.lower()

    def _match_scope() -> str | None:
        """Return matched scope name, or None for public."""
        # Keyword override (e.g. diabete → glicemia).
        for keyword, scope_name in _PRIVATE_DIR_KEYWORDS.items():
            if keyword in label_lower:
                return scope_name
        # Explicit private_scopes: key is a substring of the encoded dir name.
        for proj_key in private_scopes:
            if proj_key.lower() in label_lower:
                return proj_key
        return None

    # Check if path is inside a private/ subdir of mem_dir.
    try:
        rel = source_path.relative_to(mem_dir)
    except ValueError:
        rel = source_path
    if any(part == "private" for part in rel.parts):
        scope_name = _match_scope() or project_label
        return "private", scope_name

    # Check project-level private override (no private/ subdir needed).
    scope_name = _match_scope()
    if scope_name is not None:
        return "private", scope_name

    return "public", project_label


def _atomic_write_page(path: Path, content: str) -> None:
    """Write content to path atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _collect_existing_source_refs(wiki_root: Path) -> dict[str, Path]:
    """Scan existing wiki pages and return {source_ref: page_path}.

    Used by migrate() to skip files already converted.
    """
    result: dict[str, Path] = {}
    for md_path in wiki_root.rglob("*.md"):
        if not md_path.is_file():
            continue
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm, _ = parse_frontmatter(text)
        for ref in (fm.get("source_refs") or []):
            result.setdefault(str(ref), md_path)
    return result


def _unique_slug(
    base_slug: str,
    project: str,
    used: set[str],
) -> str:
    """Return a slug that is not in ``used``.

    Collision resolution order:
    1. base_slug (no suffix)
    2. base_slug-<project>
    3. base_slug-<project>-1, -2, ... (numeric suffix)
    """
    if base_slug not in used:
        return base_slug
    candidate = f"{base_slug}-{_slugify(project)}"
    if candidate not in used:
        return candidate
    i = 1
    while True:
        candidate = f"{base_slug}-{_slugify(project)}-{i}"
        if candidate not in used:
            return candidate
        i += 1


# ---------------------------------------------------------------------------
# Ingest — distill one source into pages (LLM) + deterministic write
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    """Stable short hash of source text for idempotence/staleness checks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _find_page_by_slug(store: Any, wiki_root: Path, slug: str) -> WikiPage | None:
    """Locate an existing page by slug via the derived index, then disk fallback.

    The DB (``wiki_pages``) is authoritative when current. When the vault has
    been written/migrated but not yet reindexed, fall back to the conventional
    on-disk locations (filename == slug). Page writes always use ``<slug>.md``.
    """
    try:
        row = store._conn.execute(
            "SELECT rel_path FROM wiki_pages WHERE slug=?", (slug,)
        ).fetchone()
    except Exception:
        row = None
    if row:
        rel = row["rel_path"] if not isinstance(row, tuple) else row[0]
        page = _load_page(wiki_root / rel)
        if page is not None and page.slug == slug:
            return page

    # Disk fallback: known layouts, then a cheap filename glob.
    candidates = [wiki_root / "pages" / "source" / f"{slug}.md"]
    candidates += list((wiki_root / "pages").glob(f"*/{slug}.md"))
    candidates += list((wiki_root / "_private").glob(f"*/{slug}.md"))
    root_resolved = wiki_root.resolve()
    for path in candidates:
        # Containment guard: never read a file that resolves outside the vault
        # (defends against path traversal via a crafted slug, e.g. "../..").
        try:
            if not path.resolve().is_relative_to(root_resolved):
                continue
        except (OSError, ValueError):
            continue
        if path.is_file():
            page = _load_page(path)
            if page is not None and page.slug == slug:
                return page
    return None


def _build_candidate_context(
    store: Any, wiki_root: Path, source_text: str,
    namespaces: list[str], top_k: int = 15, char_budget: int = 14000,
) -> tuple[str, list[str]]:
    """Deterministic candidate discovery: vector-search related pages, read bodies.

    The derived index IS the "which pages to touch" oracle. Returns
    ``(formatted_context, candidate_slugs)``. Fails soft to ``("", [])``.
    """
    try:
        hits = store.search_vectors(source_text, namespaces=namespaces, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        _log.warning("wiki ingest: candidate search failed: %s", exc)
        return "", []

    parts: list[str] = []
    slugs: list[str] = []
    used = 0
    for h in hits:
        meta = h.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:  # noqa: BLE001
                meta = {}
        slug = meta.get("slug")
        rel = meta.get("rel_path")
        if not slug or slug in slugs or not rel:
            continue
        page = _load_page(wiki_root / rel)
        if page is None:
            continue
        block = (f"## [[{page.slug}]] — {page.title} (type={page.type}, "
                 f"scope={page.scope})\n{page.body}\n")
        if used + len(block) > char_budget:
            break
        parts.append(block)
        slugs.append(slug)
        used += len(block)
    return "\n".join(parts), slugs


def _page_from_update(
    upd: dict, existing: WikiPage | None,
    target_scope: str, target_project: str, today: str,
    source_refs: list[str] | None = None, source_hash: str = "",
) -> WikiPage:
    """Build a WikiPage from an LLM page dict, preserving stable id/created/projects."""
    slug = upd.get("slug") or _slugify(upd.get("title") or "page")
    scope = upd.get("scope") or (existing.scope if existing else target_scope)
    ptype = upd.get("type") or (existing.type if existing else "entity")
    title = upd.get("title") or (existing.title if existing else _humanize(slug))
    body = upd.get("new_body") or upd.get("body") or (existing.body if existing else "")
    projects = (existing.projects if existing and existing.projects
                else [target_project or "_global"])
    page_id = existing.id if existing else _stable_page_id(Path(scope) / slug)
    created = existing.created if (existing and existing.created) else today
    refs = source_refs if source_refs is not None else (
        existing.source_refs if existing else [])
    return WikiPage(
        id=page_id, slug=slug, title=title, type=ptype,
        projects=list(projects), scope=scope, body=body,
        source_refs=list(refs), created=created, updated=today,
        source_hash=source_hash or (existing.source_hash if existing else ""),
    )


def _backup_page(path: Path) -> None:
    """Keep a single rolling backup of a page before overwrite."""
    try:
        backup_dir = path.parent / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / f"{path.name}.bak").write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.warning("wiki ingest: backup failed for %s: %s", path, exc)


def _append_inbox(wiki_root: Path, assumptions: list, today: str,
                  source_id: str) -> None:
    """Append LLM assumptions to <wiki_root>/inbox.md (mirrors master-state inbox)."""
    if not assumptions:
        return
    lines: list[str] = []
    for a in assumptions:
        if isinstance(a, dict):
            claim = a.get("claim") or ""
            verify = a.get("verify_by") or ""
            lines.append(f"- [{today}] ({source_id}) {claim} — verify: {verify}")
        else:
            lines.append(f"- [{today}] ({source_id}) {a}")
    inbox = wiki_root / "inbox.md"
    try:
        existing = (inbox.read_text(encoding="utf-8") if inbox.exists()
                    else "# Inbox — unconfirmed inferences\n\n")
        _atomic_write_page(inbox, existing + "\n".join(lines) + "\n")
    except Exception as exc:  # noqa: BLE001
        _log.warning("wiki ingest: inbox append failed: %s", exc)


def _index_pages(store: Any, wiki_root: Path, pages: list[WikiPage]) -> int:
    """Incrementally update wiki_pages/wiki_edges/vectors for touched pages only.

    Bounded by the touched set (~10-15 pages). Edge dst-resolution uses the full
    slug set so new cross-links resolve; previously-dangling inbound edges are
    reconciled by a later full ``reindex`` (spec risk #8). Embedding is fail-soft:
    the markdown SoT is already on disk, ``wiki_status`` drift + ``reindex`` are
    the reconcile authority. Returns the count of vectors written.
    """
    conn = store._conn
    existing_slugs = {r[0] for r in conn.execute("SELECT slug FROM wiki_pages")}
    slug_set = existing_slugs | {p.slug for p in pages}
    alias_to_slug: dict[str, str] = {}
    for p in pages:
        for a in p.aliases:
            alias_to_slug.setdefault(a, p.slug)

    for p in pages:
        rel_path = str(_page_rel_path(wiki_root, p))
        conn.execute(
            """
            INSERT OR REPLACE INTO wiki_pages
                (slug, id, title, type, scope, projects, tags, aliases, rel_path, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (p.slug, p.id, p.title, p.type, p.scope, json.dumps(p.projects),
             json.dumps(p.tags) if p.tags else None,
             json.dumps(p.aliases) if p.aliases else None,
             rel_path, p.updated or None),
        )
        conn.execute("DELETE FROM wiki_edges WHERE src_slug=?", (p.slug,))
        project = p.projects[0] if p.projects else None
        for e in extract_edges(p.slug, p.body):
            if e.dst_raw in slug_set:
                dst, resolved = e.dst_raw, 1
            elif e.dst_raw in alias_to_slug:
                dst, resolved = alias_to_slug[e.dst_raw], 1
            else:
                dst, resolved = e.dst_raw, 0
            conn.execute(
                """
                INSERT OR IGNORE INTO wiki_edges
                    (src_slug, dst_slug, dst_raw, edge_kind, project, resolved)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (p.slug, dst, e.dst_raw, e.edge_kind, project, resolved),
            )
    conn.commit()

    vec_count = 0
    for p in pages:
        namespace = "wiki-private" if p.scope == "private" else "wiki"
        try:
            conn.execute(
                "DELETE FROM vectors WHERE namespace=? AND (doc_id=? OR doc_id LIKE ?)",
                (namespace, p.id, f"{p.id}#%"),
            )
            conn.commit()
        except Exception:  # noqa: BLE001
            pass
        rel_path = _page_rel_path(wiki_root, p)
        for anchor, text in _split_page_sections(p):
            doc_id = f"{p.id}#{anchor}" if anchor else p.id
            meta = {
                "slug": p.slug, "title": p.title, "type": p.type,
                "scope": p.scope, "projects": p.projects, "section": anchor,
                "rel_path": str(rel_path), "page_id": p.id,
            }
            try:
                store.upsert_vector(
                    namespace=namespace, doc_id=doc_id, text=text, metadata=meta,
                    source="wiki", project=p.projects[0] if p.projects else None,
                    tags=p.tags or None,
                )
                vec_count += 1
            except Exception as exc:  # noqa: BLE001
                _log.warning("wiki ingest: embed failed for %s: %s", doc_id, exc)
    return vec_count


# ---------------------------------------------------------------------------
# Automatic source selection — deterministic scanner + approval queue
# ---------------------------------------------------------------------------

def scan_candidates(store: Any, wiki_root: Path) -> list[dict]:
    """Select ``source`` pages needing distillation. Deterministic, no LLM.

    Reads the vault from disk (works in ``no_llm_mode``). A source page is a
    candidate when it is:
    - **stale**: its underlying file (``source_refs[0]``) content hash differs
      from the recorded ``source_hash``; or
    - **undistilled**: no non-source page yet references any of its
      ``source_refs`` (i.e. it has never been distilled into entity/concept
      pages). This is the migration backfill set.

    Returns ``[{slug, title, scope, reason, source_refs, est_calls}]`` ranked
    stale-before-undistilled, then alphabetically by slug.
    """
    wiki_root = Path(wiki_root)
    source_pages: list[WikiPage] = []
    distilled_refs: set[str] = set()

    for md in wiki_root.rglob("*.md"):
        if not md.is_file() or md.name == "index.md":
            continue
        page = _load_page(md)
        if page is None:
            continue
        if page.type == "source":
            source_pages.append(page)
        else:
            for r in page.source_refs:
                distilled_refs.add(str(r))

    candidates: list[dict] = []
    for p in source_pages:
        reason: str | None = None
        if p.source_hash and p.source_refs:
            orig = Path(str(p.source_refs[0]))
            if orig.is_file():
                try:
                    if _hash_text(orig.read_text(encoding="utf-8")) != p.source_hash:
                        reason = "stale"
                except OSError:
                    pass
        if reason is None:
            refs = {str(r) for r in p.source_refs}
            if not (refs & distilled_refs):
                reason = "undistilled"
        if reason:
            candidates.append({
                "slug": p.slug, "title": p.title, "scope": p.scope,
                "reason": reason, "source_refs": p.source_refs, "est_calls": 1,
            })

    rank = {"stale": 0, "undistilled": 1}
    candidates.sort(key=lambda c: (rank.get(c["reason"], 9), c["slug"]))
    return candidates


def scan_and_enqueue(store: Any, wiki_root: Path) -> dict:
    """Run :func:`scan_candidates` and upsert the approval queue. No LLM.

    Idempotent: new candidates become ``pending``; an existing ``pending`` row
    refreshes its reason; a ``done``/``skipped`` row re-opens to ``pending``
    only when the source went ``stale`` (spec §14.2 — skipped stays skipped
    until the source changes). ``approved`` rows are never disturbed.

    Returns a summary with queue totals and the current actionable rows.
    """
    candidates = scan_candidates(store, wiki_root)
    conn = store._conn
    enqueued = 0
    for c in candidates:
        row = conn.execute(
            "SELECT status FROM wiki_queue WHERE slug=?", (c["slug"],)
        ).fetchone()
        status = row["status"] if row else None
        if row is None:
            conn.execute(
                "INSERT INTO wiki_queue (slug, title, scope, reason, est_calls, status) "
                "VALUES (?, ?, ?, ?, ?, 'pending')",
                (c["slug"], c["title"], c["scope"], c["reason"], c["est_calls"]),
            )
            enqueued += 1
        elif status == "pending":
            conn.execute(
                "UPDATE wiki_queue SET title=?, scope=?, reason=? WHERE slug=?",
                (c["title"], c["scope"], c["reason"], c["slug"]),
            )
        elif status in ("done", "skipped") and c["reason"] == "stale":
            conn.execute(
                "UPDATE wiki_queue SET reason='stale', status='pending', decided_at=NULL "
                "WHERE slug=?",
                (c["slug"],),
            )
            enqueued += 1
        # approved rows: leave untouched
    conn.commit()
    return _queue_summary(store, scanned=len(candidates))


def _queue_summary(store: Any, scanned: int | None = None) -> dict:
    """Return queue counts + the actionable (pending/approved) rows."""
    conn = store._conn
    counts = {"pending": 0, "approved": 0, "done": 0, "skipped": 0}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM wiki_queue GROUP BY status"
    ):
        counts[row["status"]] = row["n"]
    total_est = conn.execute(
        "SELECT COALESCE(SUM(est_calls), 0) AS s FROM wiki_queue "
        "WHERE status IN ('pending','approved')"
    ).fetchone()["s"]
    rows = conn.execute(
        "SELECT slug, title, scope, reason, est_calls, status FROM wiki_queue "
        "WHERE status IN ('pending','approved') ORDER BY status, slug"
    ).fetchall()
    out = {
        **counts,
        "total_est_calls": int(total_est or 0),
        "queue": [dict(r) for r in rows],
    }
    if scanned is not None:
        out["scanned"] = scanned
    return out


_INDEX_LINE_RE = re.compile(r"^- \[\[([^\]]+)\]\] — (.+)$")


def _index_entries(wiki_root: Path) -> list[tuple[str, str]]:
    """Parse ``index.md`` into ``[(slug, title)]`` (public pages only)."""
    path = wiki_root / "index.md"
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _INDEX_LINE_RE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def query(
    store: Any, wiki_root: Path, query_text: str, *,
    top_k: int = 5, authorized_scopes: list[str] | None = None,
    _file_back: bool = False,
) -> dict:
    """Hybrid wiki query: vector ranking unioned with index.md lexical hits.

    Vector search runs over the ``wiki`` namespace (plus ``wiki-private`` when
    ``authorized_scopes`` is non-empty); the curated ``index.md`` (public only)
    backfills lexical title matches. Returns ranked page bodies with
    ``source_refs`` provenance. Private pages are excluded for unauthorized
    callers in both paths.

    ``_file_back`` is a reserved seam (deferred query-file-back) — raises
    ``NotImplementedError``.
    """
    if _file_back:
        return file_answer(store, wiki_root, query_text,
                           top_k=top_k, authorized_scopes=authorized_scopes)

    wiki_root = Path(wiki_root)
    authorized_scopes = authorized_scopes or []
    namespaces = ["wiki"] + (["wiki-private"] if authorized_scopes else [])

    ranked: list[dict] = []
    seen: set[str] = set()

    try:
        hits = store.search_vectors(query_text, namespaces=namespaces,
                                    top_k=max(top_k * 2, top_k))
    except Exception as exc:  # noqa: BLE001
        _log.warning("wiki query: vector search failed: %s", exc)
        hits = []

    for h in hits:
        meta = h.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:  # noqa: BLE001
                meta = {}
        slug = meta.get("slug")
        rel = meta.get("rel_path")
        if not slug or slug in seen or not rel:
            continue
        page = _load_page(wiki_root / rel)
        if page is None:
            continue
        if page.scope == "private" and not authorized_scopes:
            continue
        seen.add(slug)
        ranked.append({
            "slug": page.slug, "title": page.title, "type": page.type,
            "scope": page.scope, "score": round(float(h.get("score", 0.0)), 4),
            "body": page.body, "source_refs": page.source_refs,
        })
        if len(ranked) >= top_k:
            break

    if len(ranked) < top_k:
        terms = [t for t in re.split(r"\W+", query_text.lower()) if len(t) > 2]
        for slug, title in _index_entries(wiki_root):
            if slug in seen:
                continue
            if terms and any(t in title.lower() for t in terms):
                page = _find_page_by_slug(store, wiki_root, slug)
                if page is None or page.scope == "private":
                    continue
                seen.add(slug)
                ranked.append({
                    "slug": page.slug, "title": page.title, "type": page.type,
                    "scope": page.scope, "score": 0.0,
                    "body": page.body, "source_refs": page.source_refs,
                })
                if len(ranked) >= top_k:
                    break

    return {"query": query_text, "results": ranked[:top_k]}


def file_answer(
    store: Any, wiki_root: Path, query_text: str, *,
    top_k: int = 5,
    authorized_scopes: list[str] | None = None,
    tier: str = "tier_smart",
    model: str | None = None,
) -> dict:
    """Synthesize a cited answer from top-k wiki hits read from disk.

    Runs ``query()`` to get ranked page slugs, reads their FULL markdown
    bodies from disk, then calls the local LLM via the
    ``wiki_file_answer`` prompt. Cites source page slugs/paths in the
    answer.

    Fails gracefully when no LLM is available: returns the raw top hits
    with a ``no_llm`` note in the answer field so the caller always gets
    a usable result.

    Returns a dict with keys:
        query (str), results (list), answer (str), sources (list[str])
    """
    from . import embeddings as _emb
    from .llm.prompts import load_prompt

    wiki_root = Path(wiki_root)
    raw = query(store, wiki_root, query_text,
                top_k=top_k, authorized_scopes=authorized_scopes)
    results = raw["results"]

    if not results:
        return {
            "query": query_text,
            "results": [],
            "answer": f"No wiki pages found for query: {query_text!r}",
            "sources": [],
        }

    # Build pages block from full disk bodies (results already have body).
    pages_block_parts: list[str] = []
    sources: list[str] = []
    for r in results:
        # body is already the full page body (loaded by query() via _load_page).
        slug = r["slug"]
        sources.append(slug)
        pages_block_parts.append(
            f"### [[{slug}]] — {r['title']} "
            f"(type={r['type']}, scope={r['scope']})\n{r['body']}"
        )
    pages_block = "\n\n".join(pages_block_parts)

    # Try LLM synthesis.
    try:
        prompt_tmpl = load_prompt("wiki_file_answer")
        prompt = prompt_tmpl.format(
            query=query_text,
            top_k=top_k,
            pages_block=pages_block[:16000],
        )
        raw_answer = _emb.get_provider().complete(
            prompt,
            tier=tier,
            model=model,
            max_tokens=2048,
            temperature=0.1,
            timeout=120.0,
            op="wiki_file_answer",
        )
        if raw_answer:
            import re as _re
            raw_answer = _re.sub(
                r"<think>.*?</think>", "", raw_answer, flags=_re.DOTALL
            ).strip()
        answer = raw_answer or "(LLM returned empty response)"
    except Exception as exc:  # noqa: BLE001
        _log.warning("wiki file_answer: LLM call failed: %s", exc)
        bodies_summary = "\n\n".join(
            f"[[{r['slug']}]]: {r['body'][:400]}" for r in results
        )
        answer = (
            f"(LLM unavailable — raw top hits below)\n\n{bodies_summary}"
        )

    return {
        "query": query_text,
        "results": results,
        "answer": answer,
        "sources": sources,
    }


def write_source_page(
    store: Any, wiki_root: Path, *, source_id: str, title: str, body: str,
    url: str = "", scope: str = "public", project: str = "_global",
    today: str | None = None,
) -> str | None:
    """Mechanically write/refresh a ``source`` page (no LLM). Returns the slug.

    Source connectors (discussions, issues) use this to land raw content into
    the wiki as a ``source`` page. The scan→approve→ingest loop distills it
    later — this keeps connectors cheap (no token spend) and consistent with
    migration. Idempotent via ``source_hash``; returns None when unchanged.
    """
    wiki_root = Path(wiki_root)
    today = today or date.today().isoformat()
    slug = "source-" + _slugify(source_id)
    existing = _find_page_by_slug(store, wiki_root, slug)
    source_hash = _hash_text(body)
    if existing is not None and existing.source_hash == source_hash:
        return None  # unchanged — skip the write

    page = WikiPage(
        id=existing.id if existing else _stable_page_id(Path(scope) / slug),
        slug=slug, title=title or slug, type="source",
        projects=(existing.projects if existing and existing.projects
                  else [project or "_global"]),
        scope=scope, body=body,
        source_refs=[url] if url else [source_id],
        created=existing.created if (existing and existing.created) else today,
        updated=today, source_hash=source_hash,
    )
    path = page_path(wiki_root, page)
    if path.exists():
        _backup_page(path)
    _atomic_write_page(path, render_page(page))
    try:
        _index_pages(store, wiki_root, [page])
        _write_public_index(wiki_root)
        _write_private_indexes(wiki_root)
    except Exception as exc:  # noqa: BLE001
        _log.warning("write_source_page: index update failed for %s: %s", slug, exc)
    return slug


def queue_decision(store: Any, slug: str, decision: str) -> dict:
    """Approve or skip a queued candidate. ``decision`` in {'approve','skip'}.

    Approval is the gate: only ``approved`` rows may be ingested non-dry. A
    skipped row is excluded from future scans until its source goes stale.
    """
    if decision not in ("approve", "skip"):
        return {"status": "error", "reason": f"unknown decision {decision!r}"}
    new = "approved" if decision == "approve" else "skipped"
    cur = store._conn.execute(
        "UPDATE wiki_queue SET status=?, decided_at=datetime('now') "
        "WHERE slug=? AND status IN ('pending','approved')",
        (new, slug),
    )
    store._conn.commit()
    if cur.rowcount == 0:
        return {"status": "noop", "reason": f"{slug} not in pending/approved"}
    return {"status": "ok", "slug": slug, "decision": new}


def ingest_queued(
    store: Any, wiki_root: Path, slug: str, *,
    authorized_scopes: list[str] | None = None, dry_run: bool = True,
    today: str | None = None, tier: str = "tier_smart", model: str | None = None,
) -> dict:
    """Distill one queued source page. Live writes require an ``approved`` row.

    Reconstructs the source text from the page's underlying file
    (``source_refs[0]``) when available, else the page body, then calls
    :func:`ingest_source` updating the same source slug in place. Marks the
    queue row ``done`` on success.
    """
    from . import config as _cfg
    if model is None:
        model = _cfg.get("wiki_ingest_model") or None
    wiki_root = Path(wiki_root)
    page = _find_page_by_slug(store, wiki_root, slug)
    if page is None:
        return {"status": "error", "reason": f"source page {slug!r} not found"}

    row = store._conn.execute(
        "SELECT status FROM wiki_queue WHERE slug=?", (slug,)
    ).fetchone()
    status = row["status"] if row else None
    if not dry_run and status != "approved":
        return {"status": "denied",
                "reason": f"{slug} not approved (status={status})"}

    source_text = page.body
    source_ref = str(page.source_refs[0]) if page.source_refs else ""
    if source_ref:
        op = Path(source_ref)
        if op.is_file():
            try:
                source_text = op.read_text(encoding="utf-8")
            except OSError:
                pass
    target_project = page.projects[0] if page.projects else "_global"

    result = ingest_source(
        store, wiki_root, source_kind="memory",
        source_id=source_ref or slug, source_title=page.title,
        source_text=source_text, source_slug=slug,
        target_scope=page.scope, target_project=target_project,
        authorized_scopes=authorized_scopes, dry_run=dry_run,
        today=today, tier=tier, model=model,
    )

    if dry_run:
        store._conn.execute(
            "UPDATE wiki_queue SET diff_preview=? WHERE slug=?",
            (json.dumps(result.get("pages", [])), slug),
        )
        store._conn.commit()
    elif result.get("status") in ("ok", "skipped"):
        store._conn.execute(
            "UPDATE wiki_queue SET status='done', decided_at=datetime('now') "
            "WHERE slug=?", (slug,),
        )
        store._conn.commit()
    return result


def ingest_approved(
    store: Any, wiki_root: Path, *, limit: int = 10,
    authorized_scopes: list[str] | None = None, dry_run: bool = False,
    today: str | None = None, tier: str = "tier_smart", model: str | None = None,
) -> dict:
    """Batch-ingest up to ``limit`` approved queue rows. ``limit`` is the cost cap."""
    from . import config as _cfg
    if model is None:
        model = _cfg.get("wiki_ingest_model") or None
    rows = store._conn.execute(
        "SELECT slug FROM wiki_queue WHERE status='approved' ORDER BY slug LIMIT ?",
        (max(0, limit),),
    ).fetchall()
    results = [
        ingest_queued(store, wiki_root, r["slug"],
                      authorized_scopes=authorized_scopes, dry_run=dry_run,
                      today=today, tier=tier, model=model)
        for r in rows
    ]
    ok = sum(1 for r in results if r.get("status") in ("ok", "skipped"))
    return {"processed": len(results), "ok": ok, "limit": limit,
            "results": results}


def ingest_source(
    store: Any,
    wiki_root: Path,
    *,
    source_kind: str,
    source_id: str,
    source_title: str = "",
    source_text: str = "",
    source_slug: str = "",
    url: str = "",
    target_scope: str = "public",
    target_project: str = "_global",
    authorized_scopes: list[str] | None = None,
    dry_run: bool = True,
    today: str | None = None,
    tier: str = "tier_smart",
    model: str | None = None,
) -> dict:
    """Distill one source into wiki pages: candidate discovery → LLM → write.

    Single normalized entrypoint for all sources (memory, discussion, issue).
    ``dry_run=True`` (default) writes nothing and returns proposed diffs.

    Access model (§3.7): a private target (``target_scope='private'``) requires
    ``target_project`` to be in ``authorized_scopes``; otherwise the call is
    denied. Unauthorized private *page_updates* proposed by the LLM are silently
    dropped rather than written.

    Idempotence: the source page records a ``source_hash``; an unchanged source
    skips the LLM call entirely.

    Returns a dict with ``status`` in {dry_run, ok, skipped, denied, llm_failed}.
    """
    from . import embeddings as _emb
    from . import config as _cfg
    if model is None:
        model = _cfg.get("wiki_ingest_model") or None

    wiki_root = Path(wiki_root)
    today = today or date.today().isoformat()
    target_project = target_project or "_global"
    authorized_scopes = authorized_scopes or []

    # --- access gating for the primary target ---
    if target_scope == "private" and target_project not in authorized_scopes:
        return {
            "status": "denied",
            "reason": f"private scope {target_project!r} not authorized",
            "pages": [],
        }

    # --- idempotence: skip unchanged source ---
    source_hash = _hash_text(source_text)
    source_slug = source_slug or (
        "source-" + _slugify(source_id or source_title or "untitled"))
    src_existing = _find_page_by_slug(store, wiki_root, source_slug)
    if src_existing is not None and src_existing.source_hash == source_hash:
        return {"status": "skipped", "reason": "unchanged source_hash",
                "source": source_id, "pages": []}

    # --- candidate discovery (deterministic, no LLM) ---
    namespaces = ["wiki"]
    if target_scope == "private" and target_project in authorized_scopes:
        namespaces.append("wiki-private")
    candidate_text, candidate_slugs = _build_candidate_context(
        store, wiki_root, source_text, namespaces)

    # --- LLM distillation ---
    result = _emb.wiki_ingest(
        source_kind=source_kind, source_title=source_title,
        source_text=source_text, candidate_pages=candidate_text,
        target_scope=target_scope, target_project=target_project,
        tier=tier, model=model,
    )
    if result.get("_fallback"):
        return {"status": "llm_failed", "source": source_id, "pages": []}

    refs = [url] if url else [source_id]

    # --- assemble pages to write ---
    pages_to_write: list[WikiPage] = []
    diffs: list[dict] = []

    sp = dict(result.get("source_page") or {})
    sp["slug"] = source_slug  # deterministic from source_id → guarantees idempotence
    sp["type"] = "source"
    sp["scope"] = target_scope
    src_page = _page_from_update(sp, src_existing, target_scope, target_project,
                                 today, source_refs=refs, source_hash=source_hash)
    pages_to_write.append(src_page)
    diffs.append({"slug": src_page.slug, "type": "source", "scope": src_page.scope,
                  "action": "update" if src_existing else "create",
                  "title": src_page.title, "chars": len(src_page.body)})

    for upd in result.get("page_updates", []):
        slug = upd.get("slug")
        if not slug:
            continue
        existing = _find_page_by_slug(store, wiki_root, slug)
        scope = upd.get("scope") or (existing.scope if existing else target_scope)
        if scope == "private":
            scope_name = (existing.projects[0] if existing and existing.projects
                          else target_project)
            if scope_name not in authorized_scopes:
                _log.info("wiki ingest: dropping unauthorized private page %s", slug)
                continue
        page = _page_from_update(upd, existing, target_scope, target_project,
                                 today, source_refs=refs)
        pages_to_write.append(page)
        diffs.append({"slug": page.slug, "type": page.type, "scope": page.scope,
                      "action": "update" if existing else "create",
                      "title": page.title, "chars": len(page.body)})

    log_line = (f"## [{today}] ingest | {source_title or source_id} "
                f"({len(pages_to_write)} pages)")

    if dry_run:
        return {
            "status": "dry_run", "source": source_id, "pages": diffs,
            "candidates": candidate_slugs,
            "assumptions": result.get("assumptions", []),
            "log_line": log_line,
        }

    # --- deterministic write phase ---
    written = 0
    for p in pages_to_write:
        path = page_path(wiki_root, p)
        if path.exists():
            _backup_page(path)
        _atomic_write_page(path, render_page(p))
        written += 1
    vec = _index_pages(store, wiki_root, pages_to_write)
    _write_public_index(wiki_root)
    _write_private_indexes(wiki_root)
    _append_log(wiki_root, today, "ingest", source_title or source_id, written)
    _append_inbox(wiki_root, result.get("assumptions", []), today, source_id)

    return {
        "status": "ok", "source": source_id, "pages": diffs,
        "written": written, "vectors": vec,
        "assumptions": result.get("assumptions", []),
    }


def migrate(
    store: Any,
    wiki_root: Path,
    *,
    dry_run: bool = True,
    sources: list[Path] | None = None,
    private_scopes: dict[str, list[str]] | None = None,
    project_roots: list[Path] | None = None,
    today: str | None = None,
) -> dict:
    """Mechanical migration: one source file → one ``source`` wiki page.

    Discovery:
    - Public auto-memory files via ``iter_user_memory_files()`` (already
      excludes ``private/`` subdirs).
    - Private files: files under any ``private/`` subdir in both the
      auto-memory tree and per-project ``.memory/`` trees.
    - Per-project ``.memory/*.md`` files via ``_project_to_memory_dir``.
    - One ``project/<project>.md`` index page per project that has source pages.

    Idempotent — keyed on ``source_refs``.  A source file already present in any
    page's ``source_refs`` frontmatter is skipped.

    Slug collisions: suffixed ``<slug>-<project>`` then ``<slug>-<project>-N``.

    Args:
        store: SkillStore instance (not used for writing; reserved for future
               source-ref scanning from the DB).
        wiki_root: root directory of the wiki vault.
        dry_run: When True, scan and report without writing anything.
        sources: Optional explicit list of source paths (overrides discovery).
        private_scopes: Mapping of project label → list of scope names.
            Defaults to the ``wiki_private_scopes`` config value.
        project_roots: Optional list of repo roots whose literal ``<root>/.memory/``
            trees should also be migrated (decisions.md, patterns.md, etc.).
            These are distinct from the encoded auto-memory tree. None = skip.
        today: ISO date for log.md / index timestamps.  Defaults to today.

    Returns:
        dry_run=True:  manifest dict with counts (writes nothing).
        dry_run=False: result dict with written counts.
    """
    from .memory_index import iter_user_memory_files, _USER_MEMORY_ROOT
    from .master_state import _strip_frontmatter

    wiki_root = Path(wiki_root)
    _today = today or date.today().isoformat()

    if private_scopes is None:
        try:
            from . import config as _cfg
            private_scopes = dict(_cfg.get("wiki_private_scopes") or {})
        except Exception:
            private_scopes = {}

    # Collect already-converted source_refs from existing wiki pages.
    existing_refs = _collect_existing_source_refs(wiki_root)

    # ---- Discovery --------------------------------------------------------

    # Each entry: (source_path, mem_dir, project_label)
    candidate_triples: list[tuple[Path, Path, str]] = []

    if sources is not None:
        # Explicit override — caller supplies paths; use wiki_root as mem_dir placeholder.
        for p in sources:
            candidate_triples.append((p, wiki_root, p.parent.name))
    else:
        auto_memory_root = _USER_MEMORY_ROOT  # ~/.claude/projects/

        # --- Public pass: iter_user_memory_files gives non-private files ---
        public_files = iter_user_memory_files()
        for f in public_files:
            # Derive project_label from the encoded dir two levels up:
            # f = ~/.claude/projects/<enc>/memory/foo.md  → enc = project_label
            try:
                mem_dir = f.parents[0]  # .../memory/
                enc_dir = f.parents[1]  # .../projects/<enc>/
                project_label = enc_dir.name
            except IndexError:
                mem_dir = f.parent
                project_label = "unknown"
            candidate_triples.append((f, mem_dir, project_label))

        # --- Private pass: files under private/ subdirs in auto-memory tree ---
        if auto_memory_root.exists():
            for project_dir in auto_memory_root.iterdir():
                if not project_dir.is_dir():
                    continue
                mem_dir = project_dir / "memory"
                if not mem_dir.is_dir():
                    continue
                private_dir = mem_dir / "private"
                if not private_dir.is_dir():
                    continue
                for f in private_dir.rglob("*.md"):
                    if f.is_file():
                        candidate_triples.append((f, mem_dir, project_dir.name))

        # --- Per-project .memory/ trees (public + private) ---
        # Scan each project's auto-memory dir; also look for .memory/ dirs
        # attached to known project roots by trying _project_to_memory_dir on
        # the most common work roots.
        # We cover this by also reading all *.md files in each project's
        # auto-memory dir that iter_user_memory_files may have missed
        # (e.g. MEMORY.md at root, or files in non-private subdirs).
        seen_paths: set[Path] = {t[0] for t in candidate_triples}

        if auto_memory_root.exists():
            for project_dir in auto_memory_root.iterdir():
                if not project_dir.is_dir():
                    continue
                mem_dir = project_dir / "memory"
                if not mem_dir.is_dir():
                    continue
                for f in mem_dir.rglob("*.md"):
                    if not f.is_file():
                        continue
                    if f in seen_paths:
                        continue
                    seen_paths.add(f)
                    candidate_triples.append((f, mem_dir, project_dir.name))

        # --- Literal per-repo .memory/ trees (decisions.md, patterns.md) ---
        # Distinct from the encoded auto-memory tree above. project_label is the
        # repo dir name so scope resolution (private/ subdirs, keyword/key match)
        # works the same way.
        for root in (project_roots or []):
            mem_dir = Path(root) / ".memory"
            if not mem_dir.is_dir():
                continue
            for f in mem_dir.rglob("*.md"):
                if not f.is_file():
                    continue
                if f in seen_paths:
                    continue
                seen_paths.add(f)
                candidate_triples.append((f, mem_dir, Path(root).name))

    # ---- Build source list ------------------------------------------------

    # Structure: list of dicts with all info needed to create a page.
    # Deferred until after idempotency check.

    # Track slugs used in THIS run (starts with slugs from existing pages).
    used_slugs: set[str] = set()
    for md_path in wiki_root.rglob("*.md"):
        if not md_path.is_file():
            continue
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm, _ = parse_frontmatter(text)
        if fm.get("slug"):
            used_slugs.add(fm["slug"])

    # Per-project source page list (for building project index pages).
    # Keyed by human project name → list of (slug, title).
    project_source_pages: dict[str, list[tuple[str, str]]] = {}

    pages_to_write: list[WikiPage] = []
    skipped_already_converted = 0
    collisions: list[str] = []
    by_project: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    public_count = 0
    private_count = 0

    for source_path, mem_dir, project_label in candidate_triples:
        # Skip non-markdown or excluded filenames.
        if source_path.suffix.lower() != ".md":
            continue
        if source_path.name in _SKIP_FILENAMES:
            continue

        # Idempotency check.
        if str(source_path) in existing_refs:
            skipped_already_converted += 1
            continue

        # Determine scope and project.
        scope, project_name = _resolve_scope_and_project(
            source_path, mem_dir, project_label, private_scopes
        )

        # Determine if this is genuinely cross-cutting (auto-memory files that
        # appear under a directory NOT matching any specific project).
        # Convention: treat files whose project_label doesn't look like a
        # specific project path as [_global].  For now, the encoded dir name
        # IS the project, so we always set it.
        projects_list = [project_name]

        # Read the source file.
        try:
            raw_text = source_path.read_text(encoding="utf-8")
        except OSError:
            _log.warning("migrate: cannot read %s", source_path)
            continue

        body_text = _strip_frontmatter(raw_text)

        # Extract dates from existing frontmatter if present.
        existing_fm, _ = parse_frontmatter(raw_text)
        created_date = (
            str(existing_fm.get("created") or "")
            or _file_iso_date(source_path)
        )
        updated_date = (
            str(existing_fm.get("updated") or "")
            or _file_iso_date(source_path)
        )

        # Derive title: first H1 if present, else humanize stem.
        title = _extract_h1_title(body_text) or _humanize(source_path.stem)

        # Build slug and resolve collisions.
        base_slug = _slugify(source_path.stem)
        slug = _unique_slug(base_slug, project_name, used_slugs)
        if slug != base_slug:
            collisions.append(f"{base_slug} → {slug} (project={project_name})")
        used_slugs.add(slug)

        # Stable id derived from source path.
        page_id = _stable_page_id(source_path)

        page = WikiPage(
            id=page_id,
            slug=slug,
            title=title,
            type="source",
            projects=projects_list,
            scope=scope,
            body=body_text,
            source_refs=[str(source_path)],
            created=created_date,
            updated=updated_date,
        )

        pages_to_write.append(page)

        # Accumulate stats.
        proj_key = project_name
        by_project[proj_key] = by_project.get(proj_key, 0) + 1
        scope_key = scope if scope == "public" else project_name
        by_scope[scope_key] = by_scope.get(scope_key, 0) + 1
        if scope == "public":
            public_count += 1
            project_source_pages.setdefault(project_name, []).append((slug, title))
        else:
            private_count += 1

    if dry_run:
        return {
            "dry_run": True,
            "would_write": len(pages_to_write),
            "public": public_count,
            "private": private_count,
            "by_project": by_project,
            "by_scope": by_scope,
            "collisions": collisions,
            "skipped_already_converted": skipped_already_converted,
        }

    # ---- Non-dry-run: write pages -----------------------------------------

    written = 0
    index_pages_written = 0

    for page in pages_to_write:
        dest = page_path(wiki_root, page)
        try:
            _atomic_write_page(dest, render_page(page))
            written += 1
            _log.debug("migrate: wrote %s", dest)
        except Exception as exc:
            _log.warning("migrate: failed to write %s: %s", dest, exc)

    # --- Project index pages (one per project with public source pages) ---
    for proj_name, source_entries in project_source_pages.items():
        links = "\n".join(f"- [[{s}]] — {t}" for s, t in source_entries)
        body = f"# {proj_name}\n\nSource pages migrated from auto-memory.\n\n{links}\n"
        proj_slug = _slugify(proj_name)
        proj_slug = _unique_slug(proj_slug, proj_name, used_slugs)
        used_slugs.add(proj_slug)
        proj_page = WikiPage(
            id=_stable_page_id(wiki_root / "project" / f"{proj_slug}.md"),
            slug=proj_slug,
            title=proj_name,
            type="project",
            projects=[proj_name],
            scope="public",
            body=body,
            created=_today,
            updated=_today,
        )
        dest = wiki_root / "pages" / "project" / f"{proj_slug}.md"
        try:
            _atomic_write_page(dest, render_page(proj_page))
            index_pages_written += 1
        except Exception as exc:
            _log.warning("migrate: failed to write project page %s: %s", dest, exc)

    # --- Public index.md ---
    _write_public_index(wiki_root)

    # --- Private per-scope index files ---
    _write_private_indexes(wiki_root)

    # --- log.md append ---
    summary = f"{written} source pages + {index_pages_written} project pages"
    _append_log(wiki_root, _today, "migrate", summary, written + index_pages_written)

    return {
        "dry_run": False,
        "written": written,
        "public": public_count,
        "private": private_count,
        "skipped": skipped_already_converted,
        "index_pages": index_pages_written,
    }


def _write_public_index(wiki_root: Path) -> None:
    """(Re)generate <wiki_root>/index.md from public pages only.

    Grouped by type, alphabetized within group.  Private pages are never
    included.  Per-scope private indexes are written separately.
    """
    pages_root = wiki_root / "pages"
    if not pages_root.exists():
        return

    by_type: dict[str, list[tuple[str, str]]] = {}
    for md_path in pages_root.rglob("*.md"):
        if not md_path.is_file():
            continue
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm, _ = parse_frontmatter(text)
        slug = fm.get("slug") or md_path.stem
        title = fm.get("title") or slug
        page_type = fm.get("type") or "entity"
        # Only public pages go in the shared index.
        if (fm.get("scope") or "public") == "private":
            continue
        by_type.setdefault(page_type, []).append((slug, title))

    lines = ["# Wiki Index\n"]
    for ptype in sorted(by_type):
        lines.append(f"\n## {ptype}\n")
        for slug, title in sorted(by_type[ptype], key=lambda x: x[0]):
            lines.append(f"- [[{slug}]] — {title}")
    lines.append("")

    index_path = wiki_root / "index.md"
    try:
        _atomic_write_page(index_path, "\n".join(lines))
    except Exception as exc:
        _log.warning("migrate: failed to write index.md: %s", exc)


def _write_private_indexes(wiki_root: Path) -> None:
    """Write per-scope index files under _private/<scope>/index.md.

    These are separate from the public index.md and never leak private titles
    into the public catalog.  Each index file carries proper wiki frontmatter
    so ``reindex`` treats it as a valid page with a unique slug.
    """
    private_root = wiki_root / "_private"
    if not private_root.exists():
        return

    for scope_dir in private_root.iterdir():
        if not scope_dir.is_dir():
            continue
        scope_name = scope_dir.name
        entries: list[tuple[str, str]] = []
        for md_path in scope_dir.glob("*.md"):
            if md_path.name == "index.md":
                continue
            if not md_path.is_file():
                continue
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, _ = parse_frontmatter(text)
            slug = fm.get("slug") or md_path.stem
            title = fm.get("title") or slug
            entries.append((slug, title))

        if not entries:
            continue

        links_body = "\n".join(f"- [[{s}]] — {t}" for s, t in sorted(entries, key=lambda x: x[0]))
        index_slug = f"private-index-{_slugify(scope_name)}"
        index_page = WikiPage(
            id=_stable_page_id(wiki_root / "_private" / scope_name / "index.md"),
            slug=index_slug,
            title=f"Private Index — {scope_name}",
            type="project",
            projects=[scope_name],
            scope="private",
            body=f"# Private Index — {scope_name}\n\n{links_body}\n",
            created=date.today().isoformat(),
            updated=date.today().isoformat(),
        )
        index_path = scope_dir / "index.md"
        try:
            _atomic_write_page(index_path, render_page(index_page))
        except Exception as exc:
            _log.warning("migrate: failed to write private index %s: %s", index_path, exc)


def _append_log(wiki_root: Path, today: str, operation: str,
                summary: str, count: int) -> None:
    """Append a line to <wiki_root>/log.md.

    Format: ``## [YYYY-MM-DD] <operation> | <summary> (<count> pages)``
    Append-only; the ``## [`` prefix is the parse contract used by wiki.status.
    """
    log_path = wiki_root / "log.md"
    line = f"## [{today}] {operation} | {summary} ({count} pages)\n"
    try:
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        _atomic_write_page(log_path, existing + line)
    except Exception as exc:
        _log.warning("migrate: failed to append to log.md: %s", exc)
