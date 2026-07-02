"""Ladder-synthesized context digests (#135).

The per-prompt hook cascade is deterministic-only (no prose strategy), so
long wiki pages, memory files, and skills reach the prompt squeezed but not
summarized. An abstractive pass cannot run inline — a remote round-trip blows
the hook budget — so digests are precomputed OFF the hot path:

- The hook asks :func:`digest_or_squeezed`: a cached digest whose content
  hash still matches is injected instantly; otherwise the squeezed raw text
  is injected and the source is queued (``digest = ''`` row, raw retained in
  ``content``).
- Background passes (the async-enrich worker and the reindex sweep) call
  :func:`refresh_pending`, which builds digests through the escalation
  ladder (``op="context_digest"`` → quota-aware, metered per provider).

Injected snippets carry a ``(digest)`` marker so degradation to raw text
stays visible.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

log = logging.getLogger(__name__)

# Sources shorter than this are injected squeezed-raw — a digest would not
# meaningfully compress them and costs a ladder call.
DIGEST_MIN_CHARS = 700
DIGEST_MAX_TOKENS = 300

_DIGEST_PROMPT = """\
Condense the document below into a dense context digest for an AI coding
assistant. Keep: decisions and their reasons, constraints, exact identifiers
(paths, commands, flags, names), and actionable instructions. Drop:
repetition, pleasantries, generic explanations. Output plain prose/bullets,
no preamble, at most ~200 words.

Document:
{text}
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def lookup(store: Any, key: str, content: str) -> str | None:
    """Cached digest for ``key`` iff the content hash still matches."""
    try:
        row = store._conn.execute(
            "SELECT content_hash, digest FROM context_digests WHERE key = ?",
            (key,),
        ).fetchone()
    except Exception:  # noqa: BLE001 - cache is best-effort
        return None
    if row and row["digest"] and row["content_hash"] == _hash(content):
        return row["digest"]
    return None


def mark_pending(store: Any, key: str, content: str) -> None:
    """Queue ``key`` for a background digest build (idempotent)."""
    h = _hash(content)
    try:
        row = store._conn.execute(
            "SELECT content_hash, digest FROM context_digests WHERE key = ?",
            (key,),
        ).fetchone()
        if row and row["content_hash"] == h:
            return  # already digested or already queued for this content
        store._conn.execute(
            "INSERT OR REPLACE INTO context_digests"
            " (key, content_hash, digest, content, updated_at)"
            " VALUES (?, ?, '', ?, datetime('now'))",
            (key, h, content),
        )
        store._conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("digest: mark_pending failed for %s: %s", key, exc)


def digest_or_squeezed(store: Any, key: str, raw: str, *,
                       context: str = "", site: str = "") -> tuple[str, bool]:
    """Hot-path entry: ``(text, is_digest)`` for one source document.

    Cache hit → the digest, instantly. Miss/stale → the deterministic
    squeezed raw text, with the source queued for background digestion.
    Never raises and never makes an LLM call.
    """
    from . import maybe_compress, squeeze_whitespace

    raw = raw or ""
    squeezed = squeeze_whitespace(
        maybe_compress(raw, context=context, site=site) if site else raw
    )
    if len(raw) < DIGEST_MIN_CHARS:
        return squeezed, False
    cached = lookup(store, key, raw)
    if cached:
        return cached, True
    mark_pending(store, key, raw)
    return squeezed, False


def build_digest(text: str) -> str:
    """One abstractive digest via the escalation ladder. ``""`` on failure."""
    from ..llm.request import request
    out = request(
        "cheap",
        _DIGEST_PROMPT.format(text=text[:16000]),
        op="context_digest",
        timeout=60.0,
        temperature=0.2,
        max_tokens=DIGEST_MAX_TOKENS,
    )
    out = (out or "").strip()
    # A digest that failed to compress is useless — keep raw in that case.
    if not out or len(out) >= len(text):
        return ""
    return out


def refresh_pending(store: Any, *, limit: int = 10) -> int:
    """Digest up to ``limit`` queued sources. Returns the number built.

    Runs OFF the hot path (async-enrich worker, reindex sweep). The update is
    guarded on ``content_hash`` so a source that changed while the digest was
    being built stays pending for the next pass.
    """
    try:
        rows = store._conn.execute(
            "SELECT key, content_hash, content FROM context_digests"
            " WHERE digest = '' AND content != '' LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return 0
    built = 0
    for row in rows:
        d = build_digest(row["content"])
        if not d:
            continue
        try:
            store._conn.execute(
                "UPDATE context_digests"
                " SET digest = ?, content = '', provider = 'ladder',"
                "     updated_at = datetime('now')"
                " WHERE key = ? AND content_hash = ?",
                (d, row["key"], row["content_hash"]),
            )
            store._conn.commit()
            built += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("digest: update failed for %s: %s", row["key"], exc)
    return built
