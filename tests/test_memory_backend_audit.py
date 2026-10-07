"""The migration audit must stay read-only and never imply semantic qualification."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "audit_memory_backend.py"
SPEC = importlib.util.spec_from_file_location("audit_memory_backend", SCRIPT)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def _db(path: Path, sources: list[Path]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE vectors (namespace TEXT, metadata TEXT)")
        conn.executemany(
            "INSERT INTO vectors VALUES (?, ?)",
            [("memory:user-project", json.dumps({"path": str(source)})) for source in sources],
        )


def _page(root: Path, name: str, source: Path, *, scope: str = "public",
          source_hash: str = "") -> None:
    from skill_hub.wiki import WikiPage, render_page

    page = WikiPage(id=name, slug=name, title=name, type="source",
                    projects=["_global"], scope=scope, body="Verified summary",
                    source_refs=[str(source)], source_hash=source_hash)
    file = root / "pages" / scope / f"{name}.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(render_page(page), encoding="utf-8")


def test_audit_reports_scope_and_stale_hash_without_qualifying(tmp_path):
    from skill_hub.wiki import _hash_text

    sources = [tmp_path / f"source-{n}.md" for n in range(3)]
    for source in sources:
        source.write_text(source.name, encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    _db(db, sources)
    wiki = tmp_path / "wiki"
    _page(wiki, "current", sources[0], source_hash=_hash_text(sources[0].read_text()))
    _page(wiki, "stale", sources[1], scope="private", source_hash="not-current")
    _page(wiki, "missing-hash", sources[2])

    report = audit_module.audit(db, wiki)

    assert report["raw_source_paths"] == 3
    assert report["referenced_raw_source_paths"] == 3
    assert report["raw_paths_referenced_by_public_pages"] == 2
    assert report["raw_paths_referenced_by_private_pages"] == 1
    assert report["raw_paths_with_current_first_source_hash"] == 1
    assert report["wiki_pages_missing_source_hash"] == 1
    assert report["wiki_pages_with_stale_first_source"] == 1
    assert report["reference_coverage_complete"] is True  # Paths only.
    assert report["safe_to_retire_raw_index"] is False
    assert report["semantic_quality_qualified"] is False
    assert str(tmp_path) not in json.dumps(report)


def test_missing_root_or_uncovered_source_never_passes_coverage(tmp_path, capsys):
    sources = [tmp_path / "covered.md", tmp_path / "uncovered.md"]
    for source in sources:
        source.write_text("source", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    _db(db, sources)
    missing = audit_module.audit(db, tmp_path / "absent-wiki")
    assert missing["wiki_pages_root_available"] is False
    assert missing["reference_coverage_complete"] is False

    wiki = tmp_path / "wiki"
    _page(wiki, "covered", sources[0])
    malformed = wiki / "pages" / "public" / "broken.md"
    malformed.write_text("no frontmatter", encoding="utf-8")
    unreadable = wiki / "pages" / "public" / "invalid-utf8.md"
    unreadable.write_bytes(b"\xff")
    report = audit_module.audit(db, wiki)

    assert report["unreferenced_raw_source_paths"] == 1
    assert report["wiki_pages_malformed"] == 1
    assert report["wiki_pages_unreadable"] == 1
    assert report["reference_coverage_complete"] is False
    assert str(tmp_path) not in capsys.readouterr().err


def test_database_is_opened_read_only_and_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    _db(db, [source])
    original = db.read_bytes()
    real_connect = sqlite3.connect
    calls = []

    def checked_connect(database, *args, **kwargs):
        calls.append((database, kwargs.get("uri")))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(audit_module.sqlite3, "connect", checked_connect)
    report = audit_module.audit(db, tmp_path / "missing-wiki")

    assert calls == [(db.resolve().as_uri() + "?mode=ro", True)]
    assert db.read_bytes() == original
    assert report["safe_to_retire_raw_index"] is False


def test_unknown_declared_scope_cannot_complete_coverage(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    _db(db, [source])
    wiki = tmp_path / "wiki"
    _page(wiki, "unknown-scope", source)
    page = wiki / "pages" / "public" / "unknown-scope.md"
    page.write_text(page.read_text().replace("scope: public\n", ""), encoding="utf-8")

    report = audit_module.audit(db, wiki)

    assert report["referenced_raw_source_paths"] == 1
    assert report["wiki_pages_unknown_scope"] == 1
    assert report["reference_coverage_complete"] is False
