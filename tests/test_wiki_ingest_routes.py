# tests/test_wiki_ingest_routes.py
from __future__ import annotations
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.templating import Jinja2Templates

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_hub.store import SkillStore  # noqa: E402
from skill_hub.webapp.routes import wiki as wiki_routes  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    store = SkillStore(db)
    wiki_root = tmp_path / "wiki"
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "cv.md").write_text(
        "# CV\n\nReach me at jane@example.com or +1 415 555 0100.", encoding="utf-8")
    sens = docs_root / "_sensitive"
    sens.mkdir()
    (sens / "secret.md").write_text("ssn 000-00-0000", encoding="utf-8")

    monkeypatch.setattr(wiki_routes, "_wiki_root", lambda: wiki_root)
    monkeypatch.setattr(wiki_routes, "_docs_root", lambda: docs_root)

    tpl_dir = SRC / "skill_hub" / "webapp" / "templates"
    app = FastAPI()
    app.state.store = store
    app.state.templates = Jinja2Templates(directory=str(tpl_dir))
    app.state.core_nav = []
    app.state.plugin_nav = []
    app.include_router(wiki_routes.router)
    return app, TestClient(app), docs_root, wiki_root


def test_ingest_page_lists_docs_excluding_sensitive(env):
    _, client, _, _ = env
    r = client.get("/wiki/ingest")
    assert r.status_code == 200
    assert "cv.md" in r.text
    assert "secret.md" not in r.text
    assert "_sensitive" not in r.text


def test_preview_extracts_and_flags_pii(env):
    _, client, _, _ = env
    r = client.post("/wiki/ingest/preview", data={"rel": "cv.md"})
    assert r.status_code == 200
    assert "jane@example.com" in r.text   # retained, not redacted
    assert "email" in r.text              # pii flag label shown
    assert "phone" in r.text


def test_preview_rejects_sensitive_path(env):
    _, client, _, _ = env
    r = client.post("/wiki/ingest/preview", data={"rel": "_sensitive/secret.md"})
    # Defense in depth: sensitive paths are never extracted.
    assert "ssn" not in r.text
    assert "000-00-0000" not in r.text


def test_write_creates_private_career_page_and_enqueues(env):
    _, client, _, wiki_root = env
    r = client.post("/wiki/ingest/write", data={"rel": "cv.md"})
    assert r.status_code == 200
    pages = [p for p in (wiki_root / "_private" / "career").glob("*.md")
             if p.name != "index.md"]
    assert len(pages) == 1
    assert pages[0].name.startswith("source-doc-")
    body = pages[0].read_text(encoding="utf-8")
    assert "jane@example.com" in body          # PII retained in private scope
    assert "written" in r.text                  # per-file result reported


def test_write_unchanged_is_idempotent(env):
    _, client, _, _ = env
    client.post("/wiki/ingest/write", data={"rel": "cv.md"})
    r = client.post("/wiki/ingest/write", data={"rel": "cv.md"})
    assert "unchanged" in r.text


def test_write_rejects_sensitive(env):
    _, client, _, wiki_root = env
    r = client.post("/wiki/ingest/write", data={"rel": "_sensitive/secret.md"})
    career_dir = wiki_root / "_private" / "career"
    source_pages = [p for p in career_dir.glob("*.md") if p.name != "index.md"] \
        if career_dir.exists() else []
    assert not source_pages
    assert "excluded" in r.text or "unavailable" in r.text


def _seed_one_pending(client):
    client.post("/wiki/ingest/write", data={"rel": "cv.md"})


def test_queue_endpoint_lists_pending(env):
    _, client, _, _ = env
    _seed_one_pending(client)
    r = client.get("/wiki/ingest/queue")
    assert r.status_code == 200
    assert "Distillation queue" in r.text
    assert "pending" in r.text


def test_approve_moves_row_to_approved(env):
    app, client, _, _ = env
    _seed_one_pending(client)
    slug = app.state.store._conn.execute(
        "SELECT slug FROM wiki_queue LIMIT 1").fetchone()["slug"]
    r = client.post("/wiki/ingest/approve", data={"slug": slug})
    assert r.status_code == 200
    status = app.state.store._conn.execute(
        "SELECT status FROM wiki_queue WHERE slug=?", (slug,)).fetchone()["status"]
    assert status == "approved"


def test_write_honors_scope_and_project(env):
    app, client, _, _ = env
    r = client.post("/wiki/ingest/write",
                    data={"rel": "cv.md", "scope": "public", "project": "research"})
    assert r.status_code == 200
    row = app.state.store._conn.execute(
        "SELECT scope, projects FROM wiki_pages WHERE type='source'").fetchone()
    assert row["scope"] == "public"
    assert "research" in row["projects"]


def test_write_unknown_scope_falls_back_to_private(env):
    app, client, _, _ = env
    r = client.post("/wiki/ingest/write",
                    data={"rel": "cv.md", "scope": "bogus", "project": ""})
    assert r.status_code == 200
    row = app.state.store._conn.execute(
        "SELECT scope, projects FROM wiki_pages WHERE type='source'").fetchone()
    assert row["scope"] == "private"          # unknown scope rejected
    assert "career" in row["projects"]        # empty project defaults to career


def test_write_honors_custom_root(env, tmp_path):
    _, client, _, wiki_root = env
    alt = tmp_path / "altdocs"
    alt.mkdir()
    (alt / "note.md").write_text("# Note\n\nfrom alt root", encoding="utf-8")
    r = client.post("/wiki/ingest/write",
                    data={"root": str(alt), "rel": "note.md"})
    assert r.status_code == 200
    assert "written" in r.text
    pages = [p for p in (wiki_root / "_private" / "career").glob("*.md")
             if p.name != "index.md"]
    assert any("from alt root" in p.read_text(encoding="utf-8") for p in pages)


def test_ingest_page_filters_by_query(env):
    _, client, docs_root, _ = env
    (docs_root / "resume.md").write_text("# Resume", encoding="utf-8")
    r = client.get("/wiki/ingest", params={"q": "resume"})
    assert r.status_code == 200
    assert "resume.md" in r.text
    assert "cv.md" not in r.text


def test_ingest_page_paginates(env):
    _, client, docs_root, _ = env
    for n in ("a.md", "b.md", "c.md"):
        (docs_root / n).write_text(f"# {n}", encoding="utf-8")
    # 4 supported docs total (cv.md + a/b/c); size=2 → 2 shown, 2 pages.
    r1 = client.get("/wiki/ingest", params={"size": 2, "page": 1})
    assert r1.status_code == 200
    assert r1.text.count('id="cb-') == 2
    assert "page 1 of 2" in r1.text
    r2 = client.get("/wiki/ingest", params={"size": 2, "page": 2})
    assert r2.text.count('id="cb-') == 2
    assert "page 2 of 2" in r2.text


def test_ingest_page_size_all_shows_everything(env):
    _, client, docs_root, _ = env
    for n in ("a.md", "b.md", "c.md"):
        (docs_root / n).write_text(f"# {n}", encoding="utf-8")
    r = client.get("/wiki/ingest", params={"size": 0})
    assert r.status_code == 200
    assert r.text.count('id="cb-') == 4   # all supported docs, no pagination


def test_distill_dry_run_writes_nothing(env, monkeypatch):
    app, client, _, wiki_root = env
    _seed_one_pending(client)
    slug = app.state.store._conn.execute(
        "SELECT slug FROM wiki_queue LIMIT 1").fetchone()["slug"]
    # stub the LLM distill so no provider is needed
    import skill_hub.wiki as wmod
    monkeypatch.setattr(
        wmod, "ingest_queued",
        lambda *a, **k: {"status": "ok",
                         "pages": [{"slug": "entity-jane", "action": "create"}]})
    r = client.post("/wiki/ingest/distill", data={"slug": slug, "dry_run": "1"})
    assert r.status_code == 200
    assert "entity-jane" in r.text
    # still queued (dry run does not mark done)
    status = app.state.store._conn.execute(
        "SELECT status FROM wiki_queue WHERE slug=?", (slug,)).fetchone()["status"]
    assert status in ("pending", "approved")
