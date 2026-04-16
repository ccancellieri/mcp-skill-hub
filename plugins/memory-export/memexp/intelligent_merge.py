"""LLM-driven 3-way merge for conflicting rows and markdown files.

Wraps :func:`skill_hub.llm.get_provider` so the plugin works against Claude,
OpenAI, or local Ollama with a single tier dropdown in the UI. The provider
import is **lazy** so unit tests can monkeypatch ``_get_provider`` without
pulling litellm into the import graph.

A small SQLite-backed cache (`merge_cache.sqlite`) memoises results keyed by
``(target, sha256(local), sha256(incoming), tier)`` so re-imports of the same
snapshot don't re-pay the LLM call cost.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = (
    Path.home() / ".claude" / "mcp-skill-hub" / "state" / "merge_cache.sqlite"
)


# ---------------------------------------------------------------------------
# Cache


def _ensure_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS merge_cache (
            cache_key TEXT PRIMARY KEY,
            target    TEXT NOT NULL,
            tier      TEXT NOT NULL,
            kind      TEXT NOT NULL,   -- 'row' | 'file'
            payload   TEXT NOT NULL,
            created_at REAL NOT NULL
        )"""
    )
    return conn


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _cache_key(target: str, local_hash: str, incoming_hash: str, tier: str) -> str:
    return f"{target}|{local_hash}|{incoming_hash}|{tier}"


def _cache_get(conn: sqlite3.Connection, key: str) -> str | None:
    cur = conn.execute("SELECT payload FROM merge_cache WHERE cache_key = ?", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def _cache_put(
    conn: sqlite3.Connection, key: str, target: str, tier: str, kind: str, payload: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO merge_cache "
        "(cache_key, target, tier, kind, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (key, target, tier, kind, payload, time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# LLM provider — lazy + injectable for tests

_ProviderFactory = Callable[[], Any]
_provider_factory: _ProviderFactory | None = None


def set_provider_factory(factory: _ProviderFactory | None) -> None:
    """Override the default provider factory (used by tests)."""
    global _provider_factory
    _provider_factory = factory


def _get_provider() -> Any:
    if _provider_factory is not None:
        return _provider_factory()
    from skill_hub.llm import get_provider  # local import to keep startup light

    return get_provider()


# ---------------------------------------------------------------------------
# Prompts

_ROW_PROMPT = """\
You are merging two versions of the same database row from a memory store.
Both versions share the same primary key. Produce a single merged JSON
object that:
- preserves the primary key exactly
- keeps every concrete fact present in either version
- prefers the more recent / more specific value when they conflict
- never invents new fields or facts

Table: {table}
Primary key: {pk_json}

Local version (already in the destination DB):
{local_json}

Incoming version (from the snapshot):
{incoming_json}

Respond with ONLY a single JSON object containing the merged row. No prose,
no markdown fences.
"""

_FILE_PROMPT = """\
You are merging two versions of the same memory markdown file. Produce a
single merged document that:
- preserves every unique fact present in either version
- never invents new claims
- when both versions assert different facts about the same item, prefer the
  more specific / more recent wording and keep the alternative as a brief
  parenthetical note
- preserves the original heading structure

File: {rel_path}

Local version:
---LOCAL---
{local_text}
---END---

Incoming version:
---INCOMING---
{incoming_text}
---END---

Respond with ONLY the merged markdown body. No prose, no fences.
"""


def _strip_json_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences; strip them defensively."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the first fence line and the trailing fence.
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API


def merge_row(
    table: str,
    pk: dict,
    local: dict,
    incoming: dict,
    tier: str = "tier_cheap",
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    max_tokens: int = 1024,
) -> dict:
    """Return a single merged row. PK is preserved verbatim from ``pk``."""
    if local == incoming:
        return dict(local)

    local_json = json.dumps(local, sort_keys=True, default=str)
    incoming_json = json.dumps(incoming, sort_keys=True, default=str)
    cache_key = _cache_key(
        f"row:{table}", _hash(local_json), _hash(incoming_json), tier
    )

    conn = _ensure_cache(cache_path)
    try:
        cached = _cache_get(conn, cache_key)
        if cached is not None:
            try:
                merged = json.loads(cached)
                merged.update(pk)
                return merged
            except json.JSONDecodeError:
                pass  # cache poisoned; refetch

        provider = _get_provider()
        prompt = _ROW_PROMPT.format(
            table=table,
            pk_json=json.dumps(pk, sort_keys=True, default=str),
            local_json=local_json,
            incoming_json=incoming_json,
        )
        try:
            text = provider.complete(
                prompt, tier=tier, max_tokens=max_tokens, temperature=0.1
            )
        except Exception as e:  # noqa: BLE001
            _log.warning("row merge LLM call failed (table=%s): %s", table, e)
            return dict(incoming)  # fall back to incoming

        try:
            merged = json.loads(_strip_json_fences(text))
        except json.JSONDecodeError as e:
            _log.warning(
                "row merge LLM returned invalid JSON (table=%s): %s; raw=%r",
                table, e, text[:200],
            )
            return dict(incoming)

        if not isinstance(merged, dict):
            _log.warning("row merge LLM returned non-object: %r", text[:120])
            return dict(incoming)

        merged.update(pk)  # PK is sacred — never let the LLM mutate it
        _cache_put(conn, cache_key, f"row:{table}", tier, "row", json.dumps(merged))
        return merged
    finally:
        conn.close()


def merge_markdown(
    rel_path: str,
    local_text: str,
    incoming_text: str,
    tier: str = "tier_cheap",
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    max_tokens: int = 4096,
) -> str:
    """Return a single merged markdown body."""
    if local_text == incoming_text:
        return local_text

    cache_key = _cache_key(
        f"file:{rel_path}", _hash(local_text), _hash(incoming_text), tier
    )

    conn = _ensure_cache(cache_path)
    try:
        cached = _cache_get(conn, cache_key)
        if cached is not None:
            return cached

        provider = _get_provider()
        prompt = _FILE_PROMPT.format(
            rel_path=rel_path,
            local_text=local_text,
            incoming_text=incoming_text,
        )
        try:
            text = provider.complete(
                prompt, tier=tier, max_tokens=max_tokens, temperature=0.2
            )
        except Exception as e:  # noqa: BLE001
            _log.warning("file merge LLM call failed (path=%s): %s", rel_path, e)
            return incoming_text

        merged = text.strip()
        # If the LLM wrapped the result in fences, strip them.
        if merged.startswith("```"):
            merged = re.sub(r"^```[a-zA-Z0-9]*\n", "", merged)
            merged = re.sub(r"\n```\s*$", "", merged).strip()

        _cache_put(conn, cache_key, f"file:{rel_path}", tier, "file", merged)
        return merged
    finally:
        conn.close()
