"""Wiki browse and search routes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


def _wiki_root() -> Path:
    from skill_hub import config as _cfg
    return Path(
        _cfg.get("wiki_root")
        or Path.home() / ".claude" / "mcp-skill-hub" / "wiki"
    )


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


@router.get("/wiki/{slug:path}", response_class=HTMLResponse)
def wiki_page(request: Request, slug: str) -> Any:
    store = request.app.state.store
    templates = request.app.state.templates
    authorized = _authorized_scopes()

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
    """Convert markdown to HTML — uses markdown lib if available, else escaped pre."""
    try:
        import markdown
        return markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "nl2br"],
        )
    except ImportError:
        import html
        return "<pre>" + html.escape(text) + "</pre>"
