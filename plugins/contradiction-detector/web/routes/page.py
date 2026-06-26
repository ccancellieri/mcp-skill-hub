"""Web routes for the contradiction-detector plugin."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
WIKI_ROOT = Path.home() / ".claude" / "mcp-skill-hub" / "wiki"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_wiki_root() -> Path:
    try:
        from skill_hub import config as _cfg
        return Path(_cfg.get("wiki_root") or WIKI_ROOT)
    except Exception:
        return WIKI_ROOT


def _list_findings(conn, status: str = "pending", limit: int = 100) -> list[dict]:
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


def _get_page_titles(conn, slugs: list[str]) -> dict[str, str]:
    titles = {}
    for slug in slugs:
        row = conn.execute(
            "SELECT title FROM wiki_pages WHERE slug = ?", (slug,)
        ).fetchone()
        titles[slug] = row["title"] if row else slug
    return titles


def _get_page_body(wiki_root: Path, slug: str) -> str:
    page_path = wiki_root / "pages" / f"{slug}.md"
    if page_path.exists():
        return page_path.read_text()
    private_path = wiki_root / "_private" / f"{slug}.md"
    if private_path.exists():
        return private_path.read_text()
    return "Page not found"


def _get_wiki_stats(conn, wiki_root: Path) -> dict:
    pages_db = conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM wiki_edges").fetchone()[0]
    pages_disk = 0
    if wiki_root.exists():
        pages_disk = sum(1 for _ in wiki_root.rglob("*.md"))
    return {"pages_db": pages_db, "edges": edges, "pages_disk": pages_disk}


@router.get("/", response_class=HTMLResponse)
def index(request: Request, status: str = "pending"):
    templates = request.app.state.templates
    wiki_root = _get_wiki_root()

    with _get_conn() as conn:
        findings = _list_findings(conn, status)
        all_slugs = set()
        for f in findings:
            all_slugs.add(f["page_a"])
            all_slugs.add(f["page_b"])
        titles = _get_page_titles(conn, list(all_slugs))
        wiki_stats = _get_wiki_stats(conn, wiki_root)

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "active_tab": "contradiction-detector",
            "findings": findings,
            "titles": titles,
            "status_filter": status,
            "wiki_stats": wiki_stats,
        },
    )


@router.get("/finding/{finding_id:int}", response_class=HTMLResponse)
def finding_detail(request: Request, finding_id: int):
    templates = request.app.state.templates
    wiki_root = _get_wiki_root()

    with _get_conn() as conn:
        row = conn.execute(
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
        titles = _get_page_titles(conn, [finding["page_a"], finding["page_b"]])

    page_a_body = _get_page_body(wiki_root, finding["page_a"])
    page_b_body = _get_page_body(wiki_root, finding["page_b"])

    return templates.TemplateResponse(
        request,
        "detail.html",
        {
            "active_tab": "contradiction-detector",
            "finding": finding,
            "titles": titles,
            "page_a_body": page_a_body,
            "page_b_body": page_b_body,
        },
    )
