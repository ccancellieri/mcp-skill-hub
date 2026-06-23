"""CodeGraph context adapter for search_context enrichment.

Queries the project's CodeGraph index (when present) and returns a bounded
block of relevant symbol/definition context for injection into search_context.

Design constraints:
- Never raises: every public function catches all exceptions and returns a
  safe empty string on any failure.
- Import-guarded: heavy imports (subprocess) are deferred until first use.
- Cached: the index-presence check is memoised per repo root so it is cheap
  on the hot path.
- Bounded: top-k and hard char cap prevent codegraph noise from overwhelming
  the context budget.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Bounds mirroring the teaching-injection discipline in router/preloader.py.
_CODEGRAPH_TOP_K: int = 5
_CODEGRAPH_MAX_CHARS: int = 800

# CLI query timeout — enough for a local SQLite lookup, never a hang.
_QUERY_TIMEOUT_SECS: float = 5.0

# Index directory name (matches orchestrator/engine.py).
_CODEGRAPH_DIR = ".codegraph"
_CODEGRAPH_DB = "codegraph.db"

# Cache: str(root) → (epoch_checked, has_index: bool)
_index_cache: dict[str, tuple[float, bool]] = {}
_CACHE_TTL = 60.0  # seconds between re-checks


# ---------------------------------------------------------------------------
# Index detection (cached)
# ---------------------------------------------------------------------------

def has_codegraph_index(repo_root: Path) -> bool:
    """Return True when *repo_root* has a non-empty CodeGraph index.

    Checks for ``.codegraph/codegraph.db`` (same logic as
    ``orchestrator.engine.probe_codegraph``, but without the freshness
    calculation — presence is all we need here).  The result is cached for
    ``_CACHE_TTL`` seconds so repeated calls on the hot path are cheap.
    """
    try:
        key = str(repo_root)
        now = time.monotonic()
        if key in _index_cache:
            ts, result = _index_cache[key]
            if now - ts < _CACHE_TTL:
                return result

        index_dir = repo_root / _CODEGRAPH_DIR
        db_path = index_dir / _CODEGRAPH_DB
        result = index_dir.is_dir() and db_path.exists() and db_path.stat().st_size > 0
        _index_cache[key] = (now, result)
        return result
    except Exception:
        return False


def _clear_cache() -> None:
    """Test helper — clear the detection cache."""
    _index_cache.clear()


# ---------------------------------------------------------------------------
# CLI query
# ---------------------------------------------------------------------------

def _find_codegraph_bin() -> str | None:
    """Return the path to the codegraph executable, or None if not found."""
    import shutil
    return shutil.which("codegraph")


def _run_query(query: str, repo_root: Path, top_k: int) -> list[dict]:
    """Shell out to ``codegraph query --json`` and return parsed results.

    Returns an empty list on any failure (missing binary, timeout, bad JSON,
    index error).
    """
    import json
    import subprocess

    bin_path = _find_codegraph_bin()
    if bin_path is None:
        logger.debug("codegraph_context: binary not found")
        return []

    try:
        result = subprocess.run(
            [bin_path, "query", "--json", "--limit", str(top_k),
             "--path", str(repo_root), query],
            capture_output=True,
            timeout=_QUERY_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.debug("codegraph_context: query timed out for %s", repo_root)
        return []
    except Exception as exc:
        logger.debug("codegraph_context: query failed: %s", exc)
        return []

    if result.returncode != 0:
        logger.debug(
            "codegraph_context: query exited %d: %s",
            result.returncode,
            result.stderr.decode(errors="replace")[:120],
        )
        return []

    try:
        data = json.loads(result.stdout.decode(errors="replace"))
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.debug("codegraph_context: JSON parse failed: %s", exc)

    return []


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def _format_block(rows: list[dict], max_chars: int) -> str:
    """Turn raw codegraph query rows into a bounded markdown context block.

    Each row shape: ``{"node": {name, kind, qualifiedName, filePath,
    startLine, signature, ...}, "score": float}``.

    Returns an empty string when *rows* is empty.
    """
    if not rows:
        return ""

    lines: list[str] = ["## CodeGraph Symbols\n"]
    for row in rows:
        try:
            node = row.get("node") or {}
            name = node.get("name") or ""
            kind = node.get("kind") or ""
            file_path = node.get("filePath") or ""
            start_line = node.get("startLine")
            sig = (node.get("signature") or "").strip()
            score = row.get("score", 0.0)

            loc = f"{file_path}:{start_line}" if start_line else file_path
            sig_snippet = f" — `{sig[:80]}`" if sig else ""
            lines.append(
                f"- **{name}** ({kind}) {loc} (score={score:.1f}){sig_snippet}"
            )
        except Exception:
            continue

    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars].rstrip() + "…"
    return block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_context_block(query: str, repo_root: Path) -> str:
    """Return a bounded CodeGraph context block for *query* at *repo_root*.

    Returns an empty string when:
    - No index is present at *repo_root*.
    - The ``codegraph`` binary is not installed.
    - The query times out or returns no results.
    - Any other error occurs.

    Never raises.
    """
    try:
        if not has_codegraph_index(repo_root):
            return ""

        rows = _run_query(query, repo_root, top_k=_CODEGRAPH_TOP_K)
        return _format_block(rows, max_chars=_CODEGRAPH_MAX_CHARS)
    except Exception as exc:
        logger.debug("codegraph_context.get_context_block: unexpected error: %s", exc)
        return ""
