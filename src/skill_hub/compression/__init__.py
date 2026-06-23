"""Content compression via the optional ``headroom-ai`` dependency.

This is a thin, import-guarded adapter around `headroom-ai`'s ``ContentRouter``.
Skill Hub stays local-first and offline-safe: when ``headroom-ai`` is not installed
(or anything goes wrong) every function here degrades to a safe passthrough, so the
server behaves exactly as it did before.

**Deterministic-first cascade.** Every payload is first run through the *lossless*
deterministic compressors (SmartCrusher for JSON arrays, plus the log / search / diff /
tabular / HTML compressors). These are cheap, offline, and reversible. Only when the
deterministic pass yields nothing *and* the caller opted into lossy compression do we
fall back to the heavier ML paths:

* **Kompress** — a ModernBERT token compressor for prose. LOSSY and *irreversible*
  (it deletes low-salience tokens; there is no rehydration for prose). Gated by the
  ``compression_ml_enabled`` config flag (default ON after eval — avg ratio 0.60,
  avg embedding-fidelity 0.87; see scripts/compression_eval.py) and the
  ``compression_full`` extra (``headroom-ai[ml]``). Auto-no-ops without the extra.
* **code-aware** — tree-sitter AST compression for source code. Gated by
  ``compression_code_aware_enabled`` (default OFF — the eval showed headroom routes
  code to Kompress first, so this path never fires on real tool output).

The single-shot encoder pass is fast and light (no autoregressive generation, no
Ollama). ``kompress_prose()`` exposes it directly for sites that want a light
extractive digest *instead of* a heavy local-LLM summarize (e.g. the searxng web
connector), independent of the ``compression_ml_enabled`` flag.

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

# Cached routers keyed by ``(ml, code)`` capability tuple. ``_router_failed``
# latches so we don't re-probe a broken install on every call.
_routers: dict[tuple[bool, bool], object] = {}
_router_failed = False

# Cached direct Kompress compressor (prose). headroom's ContentRouter sends prose
# to a no-op TEXT strategy, so we call the Kompress model directly with an explicit
# target ratio for the lossy prose path.
_kompress = None
_kompress_failed = False

# Strategy names that mean "lossy ML compression happened" (irreversible).
_LOSSY_STRATEGIES = {"KOMPRESS", "CODE_AWARE"}

_DEFAULT_ML_TARGET_RATIO = 0.6


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


def _build_router(ml: bool, code: bool):
    """Construct a ``ContentRouter`` for the given capability tuple, or None."""
    from headroom.transforms.content_router import (
        ContentRouter,
        ContentRouterConfig,
    )

    cfg = ContentRouterConfig()
    # ML / lossy paths — only when explicitly requested.
    cfg.enable_kompress = bool(ml)
    cfg.enable_code_aware = bool(code)
    # Route detected source code to the tree-sitter compressor when code-aware
    # is on (otherwise headroom would send code to Kompress instead).
    if hasattr(cfg, "prefer_code_aware_for_code"):
        cfg.prefer_code_aware_for_code = bool(code)
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
    # Keep reversible-compression markers so retrieve_compressed() can rehydrate
    # the (lossless) deterministic strategies.
    if hasattr(cfg, "ccr_enabled"):
        cfg.ccr_enabled = True
    return ContentRouter(cfg)


def _get_router(ml: bool = False, code: bool = False):
    """Return a cached ``ContentRouter`` for the capability tuple, or None.

    ``ml``/``code`` select whether the lossy Kompress / code-aware paths are
    enabled. The deterministic-only router is ``(False, False)``.
    """
    global _router_failed
    key = (bool(ml), bool(code))
    cached = _routers.get(key)
    if cached is not None:
        return cached
    if _router_failed:
        return None
    try:
        router = _build_router(*key)
    except Exception as e:  # pragma: no cover - exercised only without headroom
        logger.debug("compression unavailable (router init failed): %s", e)
        _router_failed = True
        return None
    _routers[key] = router
    return router


def _get_kompress():
    """Return a cached ``KompressCompressor`` (ModernBERT), or None if unavailable.

    Loading downloads the model from HuggingFace on first use (needs the
    ``compression_full`` extra). Latched so a broken/offline install isn't
    re-probed on every call.
    """
    global _kompress, _kompress_failed
    if _kompress is not None:
        return _kompress
    if _kompress_failed:
        return None
    try:
        from headroom.transforms.kompress_compressor import KompressCompressor

        _kompress = KompressCompressor()
    except Exception as e:  # pragma: no cover - exercised only without [ml] extra
        logger.debug("kompress unavailable (load failed): %s", e)
        _kompress_failed = True
        return None
    return _kompress


def _kompress_direct(text: str, context: str) -> "CompressedPayload | None":
    """Lossy prose compression via the Kompress model. Returns None on miss/error."""
    kc = _get_kompress()
    if kc is None:
        return None
    try:
        from .. import config as _cfg

        target = float(_cfg.get("compression_ml_target_ratio") or _DEFAULT_ML_TARGET_RATIO)
    except Exception:  # noqa: BLE001
        target = _DEFAULT_ML_TARGET_RATIO
    try:
        out = kc.compress(text, context=context or "", target_ratio=target)
    except Exception as e:  # noqa: BLE001
        logger.debug("kompress_direct failed: %s", e)
        return None
    compressed = getattr(out, "compressed", None) or text
    bytes_before, bytes_after = len(text), len(compressed)
    if bytes_after >= bytes_before:
        return None
    return CompressedPayload(
        compressed=compressed,
        content_type="KOMPRESS",
        ratio=bytes_after / bytes_before if bytes_before else 1.0,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lossy=True,  # Kompress deletes tokens; not reversible.
    )


def _extract_ccr_keys(text: str) -> list[str]:
    # dict.fromkeys preserves order and de-duplicates.
    return list(dict.fromkeys(_CCR_RE.findall(text)))


def _run_router(router, text: str, context: str) -> CompressedPayload | None:
    """Run one router over ``text``; return a winning payload, or None on miss."""
    bytes_before = len(text)
    try:
        result = router.compress(text, context=context or "")
    except Exception as e:  # noqa: BLE001
        logger.debug("compress failed, returning original: %s", e)
        return None
    compressed = getattr(result, "compressed", None) or text
    strategy = getattr(result, "strategy_used", None)
    strat_name = getattr(strategy, "name", None) or str(strategy or "PASSTHROUGH")
    strat_name = strat_name.upper()
    bytes_after = len(compressed)
    # Nothing useful happened (passthrough strategy, or output grew).
    if strat_name in _PASSTHROUGH_STRATEGIES or bytes_after >= bytes_before:
        return None
    ccr_keys = _extract_ccr_keys(compressed)
    return CompressedPayload(
        compressed=compressed,
        content_type=strat_name,
        ratio=bytes_after / bytes_before if bytes_before else 1.0,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        # Lossy iff an ML strategy ran. CCR markers make a result *reversible*
        # (via retrieve_compressed), so a deterministic CCR result is NOT lossy.
        lossy=strat_name in _LOSSY_STRATEGIES,
        ccr_keys=ccr_keys,
    )


def compress_payload(
    content: str,
    *,
    context: str = "",
    min_tokens: int | None = None,
    allow_lossy: bool = False,
) -> CompressedPayload:
    """Compress ``content`` with a deterministic-first cascade. Always returns a
    usable payload — falls back to the original text on any miss, error, or small
    input.

    Args:
        content: The text to compress (tool output, logs, JSON, search results, ...).
        context: Optional query/context for relevance-aware compression (which items
            to keep). Pass the user's query when available.
        min_tokens: Skip compression below this approximate token count. Defaults to
            ``_DEFAULT_MIN_TOKENS`` (small payloads aren't worth the work).
        allow_lossy: When True (and the ``compression_ml_enabled`` /
            ``compression_code_aware_enabled`` flags are on), fall back to the lossy
            Kompress / code-aware paths if the deterministic pass found nothing.
            Leave False for content fed to a local LLM that cannot rehydrate.
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

    # 1. Lossless deterministic pass (always).
    router = _get_router(False, False)
    if router is None:
        return passthrough
    won = _run_router(router, text, context)
    if won is not None:
        return won

    # 2. Lossy fallback — only when opted in and a flag is on. Try the tree-sitter
    #    code-aware path first (it passes through non-code), then the Kompress prose
    #    model. Both are lossy; reached only when the deterministic pass found nothing.
    if allow_lossy:
        ml, code = _lossy_flags()
        if code:
            code_router = _get_router(ml=False, code=True)
            if code_router is not None:
                won = _run_router(code_router, text, context)
                if won is not None:
                    return won
        if ml:
            won = _kompress_direct(text, context)
            if won is not None:
                return won

    return passthrough


def _lossy_flags() -> tuple[bool, bool]:
    """Read the (ml, code) lossy-compression flags from config; (False, False) on error."""
    try:
        from .. import config as _cfg

        return (
            bool(_cfg.get("compression_ml_enabled")),
            bool(_cfg.get("compression_code_aware_enabled")),
        )
    except Exception:  # noqa: BLE001
        return (False, False)


def _emit_compression_event(payload: CompressedPayload, site: str) -> None:
    """Best-effort telemetry: record a ``compression`` event. Never raises."""
    try:
        from ..store import get_store

        get_store().append_event(
            session_id="",
            kind="compression",
            payload={
                "site": site or "?",
                "strategy": payload.content_type,
                "bytes_before": payload.bytes_before,
                "bytes_after": payload.bytes_after,
                "ratio": round(payload.ratio, 4),
                "lossy": payload.lossy,
            },
            tool_name=site or None,
        )
    except Exception as e:  # noqa: BLE001 - telemetry must never break a tool call
        logger.debug("compression telemetry emit failed (non-fatal): %s", e)


def maybe_compress(
    content: str,
    *,
    context: str = "",
    site: str = "",
    allow_lossy: bool = True,
) -> str:
    """Config-gated convenience used at wiring sites: returns compressed text when the
    ``compression_enabled`` flag is on and compression is effective, otherwise the
    original content verbatim. Records a ``compression`` telemetry event on every
    attempt that reaches the compressor. Reads ``compression_min_tokens`` and
    ``compression_context_aware`` from config. Never raises.

    Args:
        site: short label for telemetry (e.g. ``"searxng"``, ``"search_context"``).
        allow_lossy: pass False for content fed into a local LLM (it cannot
            rehydrate lossy text); True for agent-facing tool output.
    """
    try:
        from .. import config as _cfg

        if not _cfg.get("compression_enabled"):
            return content
        ctx = context if _cfg.get("compression_context_aware") else ""
        min_tokens = int(_cfg.get("compression_min_tokens") or _DEFAULT_MIN_TOKENS)
    except Exception:
        return content
    try:
        payload = compress_payload(
            content, context=ctx, min_tokens=min_tokens, allow_lossy=allow_lossy
        )
    except Exception:  # pragma: no cover - defensive; compress_payload already guards
        return content
    # Only emit telemetry for attempts that actually reached the compressor
    # (i.e. were large enough to try) — skip tiny no-op passthroughs.
    if payload.bytes_before >= min_tokens * _CHARS_PER_TOKEN:
        _emit_compression_event(payload, site)
    return payload.compressed


def kompress_prose(text: str, *, context: str = "", site: str = "kompress") -> str:
    """Light, LLM-free extractive prose compression via Kompress (ModernBERT).

    Unlike :func:`maybe_compress`, this calls the Kompress model directly and is
    **independent of the ``compression_ml_enabled`` flag** — use it where a site
    deliberately wants a fast extractive digest in place of a heavy local-LLM
    summarize (e.g. the searxng web connector). It is LOSSY (deletes tokens) but
    grounded: output is a subset of the input, so it cannot hallucinate.

    Returns the compressed text, or the original verbatim on any miss/error or when
    the ``compression_full`` extra is not installed. Records a telemetry event on a
    successful compression. Never raises.
    """
    if not (text and text.strip()):
        return text
    try:
        payload = _kompress_direct(text, context or "")
    except Exception:  # pragma: no cover - _kompress_direct already guards
        return text
    if payload is None:
        return text
    _emit_compression_event(payload, site)
    return payload.compressed


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
