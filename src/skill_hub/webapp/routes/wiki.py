"""Wiki browse and search routes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

# Slug = kebab tokens, optionally '/'-segmented. Never contains '..' — used to
# reject path-traversal attempts before any filesystem lookup.
_SLUG_RE = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")

# Tags/attributes permitted in rendered page-body HTML (bleach allow-list).
_MD_ALLOWED_TAGS = [
    "p", "br", "hr", "span", "div", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "b", "i", "u", "s", "del", "code", "pre",
    "ul", "ol", "li", "a", "img",
    "table", "thead", "tbody", "tr", "th", "td",
]
_MD_ALLOWED_ATTRS = {
    "a": ["href", "title"],
    "img": ["src", "alt", "title"],
    "code": ["class"],
    "span": ["class"],
    "th": ["align"],
    "td": ["align"],
}


def _wiki_root() -> Path:
    from skill_hub import config as _cfg
    return Path(
        _cfg.get("wiki_root")
        or Path.home() / ".claude" / "mcp-skill-hub" / "wiki"
    )


def _docs_root() -> Path:
    from skill_hub import config as _cfg
    return Path(_cfg.get("wiki_docs_root") or Path.home() / "Documents")


def _safe_doc_path(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``root``; reject traversal and _sensitive paths."""
    root = root.resolve()
    candidate = (root / rel).resolve()
    try:
        rel_parts = candidate.relative_to(root).parts
    except ValueError:
        return None  # escaped the docs root
    if any(part == "_sensitive" for part in rel_parts):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _provider_available() -> bool:
    try:
        from skill_hub.llm import get_provider
        return get_provider() is not None
    except Exception:  # noqa: BLE001
        return False


def _authorized_scopes() -> list[str]:
    """Scopes visible on this local machine — union of configured private scopes."""
    from skill_hub import config as _cfg
    cfg = _cfg.get("wiki_private_scopes") or {}
    scopes: set[str] = set()
    if isinstance(cfg, dict):
        for v in cfg.values():
            if isinstance(v, list):
                scopes.update(v)
    return sorted(scopes)


def _wiki_health(store: Any) -> dict | None:
    """Return wiki vault health + queue summary. Reuses dashboard._db_metrics logic."""
    from skill_hub import wiki as _wiki
    wiki_root = _wiki_root()
    try:
        st = _wiki.status(store, wiki_root)
        summary = _wiki._queue_summary(store)
        return {
            "pages_db": st["pages_db"],
            "pages_disk": st["pages_disk"],
            "drift": st["drift"],
            "edges": st["edges"],
            "dangling": st["dangling_edges"],
            "orphans": st["orphans"],
            "last_log": st.get("last_log") or "",
            "queue_pending": summary["pending"],
            "queue_approved": summary["approved"],
            "queue_done": summary["done"],
            "queue_skipped": summary["skipped"],
        }
    except Exception:  # noqa: BLE001
        return None


def _list_pages(store: Any, authorized: list[str]) -> list[dict]:
    """Return all wiki_pages rows with inbound/outbound edge counts."""
    conn = store._conn
    # Private pages visible only when there are authorized scopes.
    scope_filter = "" if authorized else " WHERE scope != 'private'"
    rows = conn.execute(
        f"""
        SELECT
            wp.slug, wp.title, wp.type, wp.scope, wp.projects,
            wp.updated,
            COUNT(DISTINCT e_out.id) AS outbound,
            COUNT(DISTINCT e_in.id)  AS inbound
        FROM wiki_pages wp
        LEFT JOIN wiki_edges e_out ON e_out.src_slug = wp.slug
        LEFT JOIN wiki_edges e_in  ON e_in.dst_slug  = wp.slug AND e_in.resolved = 1
        {scope_filter}
        GROUP BY wp.slug
        ORDER BY wp.title COLLATE NOCASE
        """
    ).fetchall()
    return [dict(r) for r in rows]


@router.get("/wiki", response_class=HTMLResponse)
def wiki_index(request: Request, q: str = "") -> Any:
    store = request.app.state.store
    templates = request.app.state.templates
    authorized = _authorized_scopes()
    health = _wiki_health(store)

    results = None
    if q:
        from skill_hub import wiki as _wiki
        res = _wiki.query(store, _wiki_root(), q, top_k=10,
                          authorized_scopes=authorized)
        results = res.get("results", [])

    pages = _list_pages(store, authorized)
    return templates.TemplateResponse(
        request,
        "wiki.html",
        {
            "active_tab": "wiki",
            "health": health,
            "pages": pages,
            "q": q,
            "results": results,
            "authorized": authorized,
        },
    )


@router.get("/wiki/ingest", response_class=HTMLResponse)
def wiki_ingest_page(request: Request) -> Any:
    from skill_hub import doc_extract
    templates = request.app.state.templates
    docs_root = _docs_root()
    entries = doc_extract.list_documents(docs_root)
    return templates.TemplateResponse(
        request,
        "wiki_ingest.html",
        {
            "active_tab": "wiki",
            "docs_root": str(docs_root),
            "entries": entries,
            "scope_label": "private / career",
            "provider_available": _provider_available(),
        },
    )


@router.post("/wiki/ingest/write", response_class=HTMLResponse)
async def wiki_ingest_write(request: Request) -> Any:
    from skill_hub import doc_extract, wiki as _wiki
    templates = request.app.state.templates
    store = request.app.state.store
    wiki_root = _wiki_root()
    docs_root = _docs_root()
    form = await request.form()
    rels = form.getlist("rel")
    results: list[dict] = []
    for rel in rels:
        if not isinstance(rel, str):
            continue  # ignore accidental file uploads on the 'rel' field
        fp = _safe_doc_path(docs_root, rel)
        if fp is None:
            results.append({"rel": rel, "status": "excluded"})
            continue
        # Isolate per-file: a markitdown failure on one doc must not abort the batch.
        try:
            doc = doc_extract.extract_text(fp)
        except Exception as exc:  # noqa: BLE001 — fail soft per file
            results.append({"rel": rel, "status": "error", "detail": str(exc)})
            continue
        if doc.error or not doc.markdown:
            results.append({"rel": rel, "status": "error",
                            "detail": doc.error or "empty"})
            continue
        # NOTE: source_id slugs collapse punctuation, so two rel paths that differ
        # only in punctuation would collide on the same source slug (last-write-wins).
        slug = _wiki.write_source_page(
            store, wiki_root, source_id=f"doc:{rel}", title=doc.title,
            body=doc.markdown, url="", scope="private", project="career",
        )
        results.append({"rel": rel,
                        "status": "written" if slug else "unchanged",
                        "slug": slug})
    _wiki.scan_and_enqueue(store, wiki_root)
    summary = _wiki._queue_summary(store)
    return templates.TemplateResponse(
        request, "_wiki_ingest_queue.html",
        {"results": results, "summary": summary, "queue": summary["queue"],
         "provider_available": _provider_available()},
    )


@router.post("/wiki/ingest/preview", response_class=HTMLResponse)
async def wiki_ingest_preview(request: Request) -> Any:
    from skill_hub import doc_extract, pii_gate
    templates = request.app.state.templates
    form = await request.form()
    rels = form.getlist("rel")
    docs_root = _docs_root()
    previews: list[dict] = []
    for rel in rels:
        if not isinstance(rel, str):
            continue  # ignore accidental file uploads on the 'rel' field
        fp = _safe_doc_path(docs_root, rel)
        if fp is None:
            previews.append({"rel": rel, "error": "unavailable or excluded path",
                             "title": "", "excerpt": "", "flags": []})
            continue
        doc = doc_extract.extract_text(fp)
        flags = [{"label": h.pattern, "snippet": h.match}
                 for h in pii_gate.scan(doc.markdown)] if doc.markdown else []
        previews.append({
            "rel": rel,
            "error": doc.error,
            "title": doc.title,
            "excerpt": doc.markdown[:1500],
            "flags": flags,
        })
    return templates.TemplateResponse(
        request, "_wiki_ingest_preview.html", {"previews": previews})


@router.get("/wiki/{slug:path}", response_class=HTMLResponse)
def wiki_page(request: Request, slug: str) -> Any:
    store = request.app.state.store
    templates = request.app.state.templates
    authorized = _authorized_scopes()

    # Reject slugs that could escape the vault (path traversal) before lookup.
    if not _SLUG_RE.fullmatch(slug) or ".." in slug.split("/"):
        return templates.TemplateResponse(
            request,
            "wiki_page.html",
            {"active_tab": "wiki", "page": None, "slug": slug,
             "outbound": [], "authorized": authorized},
            status_code=404,
        )

    from skill_hub import wiki as _wiki
    page = _wiki._find_page_by_slug(store, _wiki_root(), slug)

    if page is None:
        return templates.TemplateResponse(
            request,
            "wiki_page.html",
            {"active_tab": "wiki", "page": None, "slug": slug,
             "outbound": [], "authorized": authorized},
            status_code=404,
        )

    # Private gate: only show private pages when authorized.
    if page.scope == "private" and not authorized:
        return templates.TemplateResponse(
            request,
            "wiki_page.html",
            {"active_tab": "wiki", "page": None, "slug": slug,
             "outbound": [], "authorized": authorized, "private_blocked": True},
            status_code=403,
        )

    # Outbound links (resolved only, pointing to existing pages).
    conn = store._conn
    out_rows = conn.execute(
        "SELECT dst_slug, edge_kind FROM wiki_edges "
        "WHERE src_slug = ? AND resolved = 1 ORDER BY dst_slug",
        (slug,),
    ).fetchall()
    # Fetch titles for linked pages.
    outbound = []
    for r in out_rows:
        title_row = conn.execute(
            "SELECT title FROM wiki_pages WHERE slug = ?", (r["dst_slug"],)
        ).fetchone()
        outbound.append({
            "slug": r["dst_slug"],
            "title": title_row["title"] if title_row else r["dst_slug"],
            "kind": r["edge_kind"],
        })

    # Render markdown body to HTML.
    body_html = _render_md(page.body)

    return templates.TemplateResponse(
        request,
        "wiki_page.html",
        {
            "active_tab": "wiki",
            "page": page,
            "body_html": body_html,
            "outbound": outbound,
            "authorized": authorized,
            "private_blocked": False,
        },
    )


def _render_md(text: str) -> str:
    """Render a page body to HTML for a ``| safe`` template.

    Page bodies are not fully trusted (some are ingested from external
    sources such as GitHub Discussions), so this never emits unsanitized
    HTML. Rich markdown is rendered only when both the ``markdown`` renderer
    and the ``bleach`` sanitizer are installed; otherwise the body is escaped
    into a ``<pre>`` block.
    """
    import html
    try:
        import markdown
    except ImportError:
        return "<pre>" + html.escape(text) + "</pre>"
    rendered = markdown.markdown(
        text, extensions=["fenced_code", "tables", "nl2br"]
    )
    try:
        import bleach
    except ImportError:
        # No sanitizer installed — do not pass raw HTML to a |safe template.
        return "<pre>" + html.escape(text) + "</pre>"
    return bleach.clean(
        rendered, tags=_MD_ALLOWED_TAGS, attributes=_MD_ALLOWED_ATTRS, strip=True
    )
