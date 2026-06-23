"""LLM Wiki knowledge layer — core file IO, page model, edge derivation, reindex, status.

The wiki is a directory of interlinked Markdown pages that serve as the
canonical source of truth. The SQLite tables (wiki_pages, wiki_edges) and the
vector namespace (wiki / wiki-private) are fully derived from the pages and
rebuilt by ``reindex``. Never edit the derived tables directly.

Vault layout:
    <wiki_root>/
        pages/<type>/<slug>.md    — public pages, foldered by type
        _private/<project>/<slug>.md — private pages, gated at query time

NB: vault.py is the credential vault (keyring/age). This module is the wiki.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
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

    # Derive scope from directory if not in frontmatter.
    if not fm.get("scope") and "_private" in path.parts:
        scope = "private"

    return WikiPage(
        id=page_id, slug=slug, title=title, type=page_type,
        projects=list(projects), scope=scope, body=body,
        tags=list(tags), aliases=list(aliases),
        source_refs=list(source_refs), created=created, updated=updated,
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
