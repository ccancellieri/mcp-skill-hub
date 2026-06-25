"""Pure document text extraction for wiki ingest. No disk writes, no LLM.

.md/.txt are read directly; richer formats (.docx/.pdf/.pptx/.xlsx/.html) go
through markitdown, which is an OPTIONAL dependency. When markitdown is not
installed, those formats return an ``error`` while .md/.txt keep working.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Extensions markitdown handles (beyond the always-on .md/.txt passthrough).
_MARKITDOWN_EXTS = frozenset({".docx", ".pdf", ".pptx", ".xlsx", ".html", ".htm"})
_PLAIN_EXTS = frozenset({".md", ".markdown", ".txt"})
SUPPORTED_EXTS = _PLAIN_EXTS | _MARKITDOWN_EXTS

_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class ExtractedDoc:
    title: str
    markdown: str
    fmt: str
    error: str | None = None


@dataclass
class DocEntry:
    rel: str
    name: str
    ext: str
    size: int
    supported: bool


def _humanize(stem: str) -> str:
    words = re.split(r"[_\-\s]+", stem.strip())
    return " ".join(w.capitalize() for w in words if w) or stem


def _title_from_markdown(md: str, fallback_stem: str) -> str:
    m = _H1_RE.search(md)
    if m:
        return m.group(1).strip()
    return _humanize(fallback_stem)


def _get_markitdown():
    """Return a MarkItDown instance, or None if the package is unavailable.

    Isolated so tests can monkeypatch it to simulate absence.
    """
    try:
        from markitdown import MarkItDown
    except Exception:  # noqa: BLE001 — optional dep
        return None
    return MarkItDown()


def extract_text(path: Path) -> ExtractedDoc:
    path = Path(path)
    ext = path.suffix.lower()
    fmt = ext.lstrip(".") or "?"

    if ext in _PLAIN_EXTS:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ExtractedDoc("", "", fmt, f"read failed: {exc}")
        fmt = "md" if ext in (".md", ".markdown") else "txt"
        return ExtractedDoc(_title_from_markdown(text, path.stem), text, fmt, None)

    if ext in _MARKITDOWN_EXTS:
        md = _get_markitdown()
        if md is None:
            return ExtractedDoc(
                "", "", fmt,
                "document extractor unavailable: install with "
                "pip install 'skill-hub[docs]'",
            )
        try:
            result = md.convert(str(path))
            text = (getattr(result, "text_content", None) or "").strip()
        except Exception as exc:  # noqa: BLE001 — corrupt/unsupported file
            return ExtractedDoc("", "", fmt, f"extraction failed: {exc}")
        if not text:
            return ExtractedDoc("", "", fmt, "extraction produced no text")
        return ExtractedDoc(_title_from_markdown(text, path.stem), text, fmt, None)

    return ExtractedDoc("", "", fmt, f"unsupported file type: {ext or '(none)'}")


def list_documents(root: Path) -> list[DocEntry]:
    root = Path(root)
    entries: list[DocEntry] = []
    if not root.is_dir():
        return entries
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune _sensitive and dot-directories in place so we never descend.
        dirnames[:] = [
            d for d in dirnames if d != "_sensitive" and not d.startswith(".")
        ]
        for name in filenames:
            if name.startswith(".") or name.startswith("~$"):
                continue
            fp = Path(dirpath) / name
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            ext = fp.suffix.lower()
            rel = fp.relative_to(root).as_posix()
            entries.append(DocEntry(rel, name, ext, size, ext in SUPPORTED_EXTS))
    entries.sort(key=lambda e: e.rel)
    return entries
