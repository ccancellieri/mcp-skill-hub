"""A4 — Plugin memory adapter.

Plugins declare memory roots they read from in plugin.json:

    "memory": {
        "reads":  ["/abs/path/**/*.md", "~/relative/**"],
        "writes": ["~/.claude/mcp-skill-hub/plugin_x/**"]
    }

Each matched ``.md`` file is embedded into the shared vector index under the
namespace ``memory:{plugin_name}`` so ``search_context`` / ``search_vectors``
surface them alongside core skills and tasks.

``writes`` are advisory (for audit / dashboard display); no enforcement here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from .plugin_registry import iter_enabled_plugins

_log = logging.getLogger(__name__)

_MAX_DOC_BYTES = 200_000  # hard cap to avoid embedding huge memory files


def _expand_globs(plugin_path: Path, globs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in globs:
        p = Path(str(pattern)).expanduser()
        if p.is_absolute():
            base, rel = _split_glob(p)
            try:
                out.extend(base.glob(rel))
            except OSError:
                continue
        else:
            try:
                out.extend(plugin_path.glob(str(p)))
            except OSError:
                continue
    # de-dup, keep files only, markdown only
    seen: set[Path] = set()
    clean: list[Path] = []
    for f in out:
        try:
            if not f.is_file() or f.suffix.lower() not in {".md", ".markdown", ".txt"}:
                continue
            if f in seen:
                continue
            seen.add(f)
            clean.append(f)
        except OSError:
            continue
    return clean


def _split_glob(p: Path) -> tuple[Path, str]:
    """Split absolute glob into the static-prefix directory and the remainder."""
    parts = p.parts
    static: list[str] = []
    rest: list[str] = []
    seen_magic = False
    for part in parts:
        if seen_magic or any(ch in part for ch in "*?[]"):
            seen_magic = True
            rest.append(part)
        else:
            static.append(part)
    base = Path(*static) if static else Path("/")
    rel = "/".join(rest) if rest else "*"
    return base, rel


def iter_plugin_memory_reads(plugin: dict[str, Any]) -> list[Path]:
    mem = plugin["manifest"].get("memory") or {}
    reads = mem.get("reads") or []
    return _expand_globs(plugin["path"], reads)


_DEFAULT_CHUNK_CHARS = 4000
_DEFAULT_CHUNK_OVERLAP = 400


def _index_chunk_config(store: Any, namespace: str) -> tuple[int, int]:
    """Return (chunk_size, overlap) from ``vector_index_config`` or defaults."""
    try:
        row = store._conn.execute(
            "SELECT chunk_size, chunk_overlap FROM vector_index_config WHERE name = ?",
            (namespace,),
        ).fetchone()
        if row and row["chunk_size"] and row["chunk_size"] > 0:
            return int(row["chunk_size"]), int(row["chunk_overlap"] or 0)
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_CHUNK_CHARS, _DEFAULT_CHUNK_OVERLAP


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy paragraph-aware chunker. Returns a list of chunk strings.

    Splits on blank lines first; if a single paragraph exceeds ``chunk_size``
    it's hard-sliced with ``overlap`` char carryover. Ensures every chunk has
    content (empty strings filtered).
    """
    if len(text) <= chunk_size:
        return [text]
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        # Single paragraph too big — hard slice.
        if len(p) > chunk_size:
            i = 0
            while i < len(p):
                chunks.append(p[i:i + chunk_size])
                i += max(1, chunk_size - overlap)
        else:
            buf = p
    if buf:
        chunks.append(buf)
    return chunks or [text]


def _embed_file(store: Any, f: Path, namespace: str, plugin_name: str,
                level: str | None = None) -> bool:
    try:
        if f.stat().st_size > _MAX_DOC_BYTES:
            return False
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _log.debug("memory read failed: %s (%s)", f, exc)
        return False
    if not text.strip():
        return False
    chunk_size, overlap = _index_chunk_config(store, namespace)
    chunks = _split_text(text, chunk_size, overlap) if chunk_size > 0 else [text]
    try:
        if len(chunks) == 1:
            store.upsert_vector(
                namespace=namespace, doc_id=str(f), text=chunks[0],
                metadata={"path": str(f), "plugin": plugin_name},
                level=level, source=plugin_name,
            )
        else:
            for i, chunk in enumerate(chunks):
                store.upsert_vector(
                    namespace=namespace,
                    doc_id=f"{f}#chunk-{i:03d}",
                    text=chunk,
                    metadata={"path": str(f), "plugin": plugin_name,
                              "chunk_index": i, "chunk_count": len(chunks)},
                    level=level, source=plugin_name,
                )
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort
        _log.warning("memory upsert_vector failed: %s (%s)", f, exc)
        return False


def index_plugin_memory(store: Any) -> dict[str, int]:
    """Embed each enabled plugin's declared memory files.

    Supports two declaration styles in plugin.json::

        # M2 — per-index routing (preferred):
        "memory": {"indexes": [
          {"name": "career:profile",   "reads": ["refs/**/*.md"], "level": "L3"},
          {"name": "career:narrative", "reads": ["state/drafts/**/*.md"]}
        ]}

        # A4 legacy — flat reads go to memory:<plugin>:
        "memory": {"reads": ["~/notes/**/*.md"]}

    Returns ``{namespace: files_indexed}``. Never raises.
    """
    counts: dict[str, int] = {}
    for plugin in iter_enabled_plugins():
        mem = plugin["manifest"].get("memory") or {}

        # New style — per-index routing.
        for idx in (mem.get("indexes") or []):
            ns = idx.get("name")
            reads = idx.get("reads") or []
            if not ns or not reads:
                continue
            level = idx.get("level")
            files = _expand_globs(plugin["path"], reads)
            indexed = 0
            for f in files:
                if _embed_file(store, f, ns, plugin["name"], level=level):
                    indexed += 1
            if indexed:
                counts[ns] = counts.get(ns, 0) + indexed

        # Legacy flat reads — single memory:<plugin> namespace.
        if mem.get("reads"):
            ns = f"memory:{plugin['name']}"
            files = iter_plugin_memory_reads(plugin)
            indexed = 0
            for f in files:
                if _embed_file(store, f, ns, plugin["name"]):
                    indexed += 1
            if indexed:
                counts[ns] = counts.get(ns, 0) + indexed
    return counts
