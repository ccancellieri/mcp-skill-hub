from __future__ import annotations
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_hub.doc_extract import (  # noqa: E402
    ExtractedDoc, extract_text, DocEntry, list_documents, SUPPORTED_EXTS,
)


def test_extract_md_passthrough(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# My Title\n\nbody text", encoding="utf-8")
    doc = extract_text(p)
    assert doc.error is None
    assert doc.fmt == "md"
    assert doc.title == "My Title"
    assert "body text" in doc.markdown


def test_extract_txt_uses_filename_title(tmp_path):
    p = tmp_path / "plain_notes.txt"
    p.write_text("just text, no heading", encoding="utf-8")
    doc = extract_text(p)
    assert doc.error is None
    assert doc.fmt == "txt"
    assert doc.title == "Plain Notes"
    assert "just text" in doc.markdown


def test_extract_unsupported_ext_returns_error(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    doc = extract_text(p)
    assert doc.error is not None
    assert doc.markdown == ""


def test_extract_docx_or_skip(tmp_path):
    md = pytest.importorskip("markitdown")  # noqa: F841
    # Build a minimal docx via python-docx if present, else skip.
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Resume", level=1)
    d.add_paragraph("Senior engineer")
    fp = tmp_path / "cv.docx"
    d.save(fp)
    doc = extract_text(fp)
    assert doc.error is None
    assert doc.fmt == "docx"
    assert "Senior engineer" in doc.markdown


def test_extract_docx_without_markitdown(tmp_path, monkeypatch):
    # Simulate markitdown being absent: extraction reports unavailable, no crash.
    import skill_hub.doc_extract as de
    monkeypatch.setattr(de, "_get_markitdown", lambda: None)
    fp = tmp_path / "cv.docx"
    fp.write_bytes(b"PK\x03\x04")  # zip magic; content irrelevant here
    doc = extract_text(fp)
    assert doc.error is not None
    assert "extractor unavailable" in doc.error
    assert doc.markdown == ""


def test_list_documents_excludes_sensitive_and_dotfiles(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("x", encoding="utf-8")
    (tmp_path / "~$lock.docx").write_bytes(b"x")
    sens = tmp_path / "_sensitive"
    sens.mkdir()
    (sens / "secret.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF-1.4")
    (sub / "note.jpg").write_bytes(b"x")

    entries = list_documents(tmp_path)
    rels = {e.rel for e in entries}
    assert "a.md" in rels
    assert "sub/b.pdf" in rels
    assert "sub/note.jpg" in rels and not next(e for e in entries if e.rel == "sub/note.jpg").supported
    assert "a.md" in rels and next(e for e in entries if e.rel == "a.md").supported
    # excluded entirely:
    assert ".hidden.md" not in rels
    assert "~$lock.docx" not in rels
    assert not any(r.startswith("_sensitive") for r in rels)


def test_supported_exts_contains_core_formats():
    for e in (".md", ".txt", ".docx", ".pdf"):
        assert e in SUPPORTED_EXTS
