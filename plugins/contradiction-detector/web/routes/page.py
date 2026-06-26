"""Web routes for the contradiction-detector plugin."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


def _get_wiki_root() -> Path:
    from skill_hub import config as _cfg
    from pathlib import Path
    return Path(
        _cfg.get("wiki_root")
        or Path.home() / ".claude" / "mcp-skill-hub" / "wiki"
    )


def _list_findings(store, status: str = "pending", limit: int = 100) -> list[dict]:
    conn = store._conn
    rows = conn.execute(
        """
        SELECT id, page_a, page_b, claim_a, claim_b, confidence,
               resolution_status, resolution, detected_at
        FROM plugin_contradiction_findings
        WHERE resolution_status = ?
        ORDER BY confidence DESC, detected_at DESC
        LIMIT ?
        """,
        (status, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_page_titles(store, slugs: list[str]) -> dict[str, str]:
    conn = store._conn
    titles = {}
    for slug in slugs:
        row = conn.execute(
            "SELECT title FROM wiki_pages WHERE slug = ?", (slug,)
        ).fetchone()
        titles[slug] = row["title"] if row else slug
    return titles


def _get_page_body(wiki_root: Path, slug: str) -> str:
    from skill_hub import wiki as _wiki
    store = Request.app.state.store if hasattr(Request, 'app') else None
    if store:
        page = _wiki._find_page_by_slug(store, wiki_root, slug)
        if page:
            return page.body
    return ""


@router.get("/", response_class=HTMLResponse)
def index(request: Request, status: str = "pending"):
    store = request.app.state.store
    templates = request.app.state.templates
    wiki_root = _get_wiki_root()

    findings = _list_findings(store, status)
    all_slugs = set()
    for f in findings:
        all_slugs.add(f["page_a"])
        all_slugs.add(f["page_b"])
    titles = _get_page_titles(store, list(all_slugs))

    from skill_hub import wiki as _wiki
    st = _wiki.status(store, wiki_root)

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "active_tab": "contradiction-detector",
            "findings": findings,
            "titles": titles,
            "status_filter": status,
            "wiki_stats": st,
        },
    )


@router.get("/finding/{finding_id:int}", response_class=HTMLResponse)
def finding_detail(request: Request, finding_id: int):
    store = request.app.state.store
    templates = request.app.state.templates
    wiki_root = _get_wiki_root()

    row = store._conn.execute(
        """
        SELECT id, page_a, page_b, claim_a, claim_b, confidence,
               resolution_status, resolution, resolved_by, resolved_at, detected_at
        FROM plugin_contradiction_findings
        WHERE id = ?
        """,
        (finding_id,),
    ).fetchone()

    if not row:
        return templates.TemplateResponse(
            request,
            "review.html",
            {"active_tab": "contradiction-detector", "error": "Finding not found"},
            status_code=404,
        )

    finding = dict(row)
    titles = _get_page_titles(store, [finding["page_a"], finding["page_b"]])

    from skill_hub import wiki as _wiki
    page_a = _wiki._find_page_by_slug(store, wiki_root, finding["page_a"])
    page_b = _wiki._find_page_by_slug(store, wiki_root, finding["page_b"])

    return templates.TemplateResponse(
        request,
        "detail.html",
        {
            "active_tab": "contradiction-detector",
            "finding": finding,
            "titles": titles,
            "page_a": page_a,
            "page_b": page_b,
        },
    )


from pathlib import Path
