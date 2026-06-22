"""Deterministic content compression via the optional ``headroom-ai`` dependency.

This is a thin, import-guarded adapter around `headroom-ai`'s ``ContentRouter``.
Skill Hub stays local-first and offline-safe: when ``headroom-ai`` is not installed
(or anything goes wrong) every function here degrades to a safe passthrough, so the
server behaves exactly as it did before.

We deliberately run the router with the ML "Kompress" path **disabled**, so only the
deterministic compressors execute (SmartCrusher for JSON arrays, plus the log / search /
diff / tabular / HTML compressors). That keeps the install light — no torch, no model
downloads — and means prose and code (which would otherwise route to Kompress) simply
pass through untouched and are left for the LLM. This matches our policy: deterministic
compression for structured payloads, the LLM only for genuine prose synthesis.

headroom-ai is Apache-2.0: https://github.com/headroomlabs-ai/headroom
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Strategies that mean "nothing useful happened" — treat as passthrough.
_PASSTHROUGH_STRATEGIES = {"KOMPRESS", "PASSTHROUGH", "NONE"}

# Matches the reversible-compression markers headroom injects, e.g. ``<<ccr:9f3a..>>``.
_CCR_RE = re.compile(r"<<ccr:([0-9a-fA-F]+)")

# Rough chars-per-token proxy (matches headroom's own estimator for prose).
_CHARS_PER_TOKEN = 4

_DEFAULT_MIN_TOKENS = 200

# Cached singletons. ``_router_failed`` latches so we don't re-probe a broken install
# on every call.
_router = None
_router_failed = False


@dataclass
class CompressedPayload:
    """Result of a compression attempt. ``compressed`` is always safe to use:
    it is the original content verbatim when nothing was (or could be) compressed."""

    compressed: str
    content_type: str  # strategy name, e.g. "SMART_CRUSHER", "LOG", or "PASSTHROUGH"
    ratio: float  # bytes_after / bytes_before; 1.0 means unchanged
    bytes_before: int
    bytes_after: int
    lossy: bool
    ccr_keys: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.bytes_after < self.bytes_before

    @property
    def saved_bytes(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


def is_available() -> bool:
    """True when the optional ``headroom-ai`` dependency can be imported."""
    try:
        import headroom  # noqa: F401

        return True
    except Exception:
        return False


def _get_router():
    """Return a cached deterministic ``ContentRouter``, or None if unavailable."""
    global _router, _router_failed
    if _router is not None:
        return _router
    if _router_failed:
        return None
    try:
        from headroom.transforms.content_router import (
            ContentRouter,
            ContentRouterConfig,
        )

        cfg = ContentRouterConfig()
        # No ML path: never attempt to load the Kompress / ModernBERT model.
        cfg.enable_kompress = False
        # Leave source code to the LLM / code-graph tools rather than AST-mangling it.
        cfg.enable_code_aware = False
        # Keep deterministic compressors on (these are True by default, set explicitly
        # so behaviour is stable if upstream defaults change).
        for attr in (
            "enable_smart_crusher",
            "enable_search_compressor",
            "enable_log_compressor",
            "enable_tabular_compressor",
            "enable_html_extractor",
        ):
            if hasattr(cfg, attr):
                setattr(cfg, attr, True)
        # Never sacrifice error text / tracebacks — the model needs them verbatim.
        if hasattr(cfg, "protect_error_outputs"):
            cfg.protect_error_outputs = True
        # Keep reversible-compression markers so retrieve_compressed() can rehydrate.
        if hasattr(cfg, "ccr_enabled"):
            cfg.ccr_enabled = True
        _router = ContentRouter(cfg)
        return _router
    except Exception as e:  # pragma: no cover - exercised only without headroom
        logger.debug("compression unavailable (router init failed): %s", e)
        _router_failed = True
        return None


def _extract_ccr_keys(text: str) -> list[str]:
    # dict.fromkeys preserves order and de-duplicates.
    return list(dict.fromkeys(_CCR_RE.findall(text)))


def compress_payload(
    content: str,
    *,
    context: str = "",
    min_tokens: int | None = None,
) -> CompressedPayload:
    """Compress structured ``content`` deterministically. Always returns a usable
    payload — falls back to the original text on any miss, error, or small input.

    Args:
        content: The text to compress (tool output, logs, JSON, search results, ...).
        context: Optional query/context for relevance-aware compression (which items
            to keep). Pass the user's query when available.
        min_tokens: Skip compression below this approximate token count. Defaults to
            ``_DEFAULT_MIN_TOKENS`` (small payloads aren't worth the work).
    """
    text = content if isinstance(content, str) else str(content)
    bytes_before = len(text)
    passthrough = CompressedPayload(
        compressed=text,
        content_type="PASSTHROUGH",
        ratio=1.0,
        bytes_before=bytes_before,
        bytes_after=bytes_before,
        lossy=False,
    )

    if not text.strip():
        return passthrough

    threshold = _DEFAULT_MIN_TOKENS if min_tokens is None else int(min_tokens)
    if bytes_before < threshold * _CHARS_PER_TOKEN:
        return passthrough

    router = _get_router()
    if router is None:
        return passthrough

    try:
        result = router.compress(text, context=context or "")
    except Exception as e:
        logger.debug("compress failed, returning original: %s", e)
        return passthrough

    compressed = getattr(result, "compressed", None) or text
    strategy = getattr(result, "strategy_used", None)
    strat_name = getattr(strategy, "name", None) or str(strategy or "PASSTHROUGH")
    strat_name = strat_name.upper()
    bytes_after = len(compressed)

    # Nothing useful happened (prose/code routed to disabled Kompress, or it grew).
    if strat_name in _PASSTHROUGH_STRATEGIES or bytes_after >= bytes_before:
        return passthrough

    ccr_keys = _extract_ccr_keys(compressed)
    ratio = bytes_after / bytes_before if bytes_before else 1.0
    return CompressedPayload(
        compressed=compressed,
        content_type=strat_name,
        ratio=ratio,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lossy=bool(ccr_keys),
        ccr_keys=ccr_keys,
    )


def maybe_compress(content: str, *, context: str = "") -> str:
    """Config-gated convenience used at wiring sites: returns compressed text when the
    ``compression_enabled`` flag is on and compression is effective, otherwise the
    original content verbatim. Reads ``compression_min_tokens`` and
    ``compression_context_aware`` from config. Never raises."""
    try:
        from .. import config as _cfg

        if not _cfg.get("compression_enabled"):
            return content
        ctx = context if _cfg.get("compression_context_aware") else ""
        min_tokens = int(_cfg.get("compression_min_tokens") or _DEFAULT_MIN_TOKENS)
    except Exception:
        return content
    try:
        return compress_payload(content, context=ctx, min_tokens=min_tokens).compressed
    except Exception:  # pragma: no cover - defensive; compress_payload already guards
        return content


def retrieve_original(hash_key: str) -> str | None:
    """Return the original content stashed behind a ``<<ccr:HASH>>`` marker, or None."""
    try:
        from headroom.cache.compression_store import get_compression_store

        entry = get_compression_store().retrieve(hash_key)
    except Exception as e:
        logger.debug("ccr retrieve failed: %s", e)
        return None
    if entry is None:
        return None
    original = getattr(entry, "original_content", None)
    if isinstance(original, str):
        return original
    return entry if isinstance(entry, str) else None
