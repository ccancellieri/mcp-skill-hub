"""Deterministic content compression, plus an optional-dependency prose helper.

**Deterministic-first, deterministic-only.** ``compress_payload()`` runs a
dependency-free JSON-minify + duplicate-line-collapse pass over structured/log/
grep output that floods context (see ``_builtin_deterministic``). This is the
only compression strategy the cascade runs (#119: the ML/code-aware advanced
paths, gated on the optional ``headroom-ai`` package, were retired — they never
ran in practice because that package isn't installed).

``kompress_prose()`` is a separate, opt-in helper: a ModernBERT extractive prose
digest via the optional ``headroom-ai[ml]`` dependency, used directly by sites
that want it instead of a heavy local-LLM summarize (e.g. the searxng web
connector). It auto-no-ops (returns the input verbatim) when the dependency
isn't installed. ``retrieve_original()`` is the read-side counterpart of the
``<<ccr:HASH>>`` markers ``webfetch.py`` stashes via headroom's compression
store.

headroom-ai is Apache-2.0: https://github.com/headroomlabs-ai/headroom
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Rough chars-per-token proxy (matches headroom's own estimator for prose).
_CHARS_PER_TOKEN = 4

_DEFAULT_MIN_TOKENS = 200

# Cached direct Kompress compressor (prose), used only by kompress_prose().
_kompress = None
_kompress_failed = False

_DEFAULT_ML_TARGET_RATIO = 0.6


@dataclass
class CompressedPayload:
    """Result of a compression attempt. ``compressed`` is always safe to use:
    it is the original content verbatim when nothing was (or could be) compressed."""

    compressed: str
    content_type: str  # strategy name, e.g. "JSON_MIN", "DEDUP", or "PASSTHROUGH"
    ratio: float  # bytes_after / bytes_before; 1.0 means unchanged
    bytes_before: int
    bytes_after: int
    lossy: bool

    @property
    def changed(self) -> bool:
        return self.bytes_after < self.bytes_before

    @property
    def saved_bytes(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


def _get_kompress():
    """Return a cached ``KompressCompressor`` (ModernBERT), or None if unavailable.

    Loading downloads the model from HuggingFace on first use (needs the
    optional ``headroom-ai[ml]`` dependency). Latched so a broken/offline
    install isn't re-probed on every call.
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
        out = kc.compress(text, context=context or "", target_ratio=_DEFAULT_ML_TARGET_RATIO)
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


def compress_payload(
    content: str,
    *,
    context: str = "",
    min_tokens: int | None = None,
    allow_lossy: bool = False,
) -> CompressedPayload:
    """Compress ``content`` deterministically. Always returns a usable payload —
    falls back to the original text on any miss, error, or small input.

    Args:
        content: The text to compress (tool output, logs, JSON, search results, ...).
        context: Unused by the current (deterministic-only) cascade; kept for
            call-site compatibility.
        min_tokens: Skip compression below this approximate token count. Defaults to
            ``_DEFAULT_MIN_TOKENS`` (small payloads aren't worth the work).
        allow_lossy: Unused by the current (deterministic-only) cascade; kept for
            call-site compatibility (#119).
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

    won = _builtin_deterministic(text)
    if won is not None:
        return won

    return passthrough


def _builtin_deterministic(text: str) -> "CompressedPayload | None":
    """Dependency-free deterministic compression for the structured/log/grep
    output that floods context — the only strategy ``compress_payload`` runs.
    Safe and near-lossless — returns None unless it actually shrinks the payload.

    Two transforms, tried in order:
    * **JSON minify** — strip insignificant whitespace from a pretty-printed JSON
      body (lossless). Catches large ``curl``/API responses.
    * **Run-length line collapse** — fold runs of >=3 identical consecutive lines
      into one with a ``… (xN)`` marker (logs, repeated grep hits, stack spam).
    """
    import json as _json

    bytes_before = len(text)
    best: tuple[str, str] | None = None

    stripped = text.strip()
    if stripped[:1] in "[{":
        try:
            obj = _json.loads(stripped)
            minified = _json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
            if len(minified) < bytes_before:
                best = ("JSON_MIN", minified)
        except Exception:  # noqa: BLE001 - not valid JSON; fall through
            pass

    if best is None:
        lines = text.split("\n")
        out: list[str] = []
        i, n, collapsed = 0, len(lines), False
        while i < n:
            j = i
            while j + 1 < n and lines[j + 1] == lines[i]:
                j += 1
            run = j - i + 1
            if run >= 3:
                out.append(f"{lines[i]}  … (x{run})")
                collapsed = True
            else:
                out.extend(lines[i:j + 1])
            i = j + 1
        if collapsed:
            dedup = "\n".join(out)
            if len(dedup) < bytes_before:
                best = ("DEDUP", dedup)

    if best is None:
        return None
    name, compressed = best
    bytes_after = len(compressed)
    return CompressedPayload(
        compressed=compressed,
        content_type=name,
        ratio=bytes_after / bytes_before if bytes_before else 1.0,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lossy=False,
    )


# Multiplier table: Pressure tier → fraction of configured min_tokens used as the
# effective threshold.  Values < 1.0 lower the bar (compress more eagerly).
# IDLE and LOW leave the configured threshold unchanged (1.0).
_PRESSURE_THRESHOLD_MULTIPLIERS: dict[int, float] = {
    0: 1.0,   # IDLE     — keep configured threshold
    1: 1.0,   # LOW      — keep configured threshold
    2: 0.5,   # MODERATE — halve threshold (compress payloads half the normal size)
    3: 0.25,  # HIGH     — quarter threshold (compress very aggressively)
}


def _effective_min_tokens(configured: int) -> tuple[int, str]:
    """Return (effective_min_tokens, pressure_tier_name).

    When ``compression_headroom_aware`` is False (default), returns
    ``(configured, "DISABLED")`` so behaviour is byte-identical to before.
    Never raises.
    """
    try:
        from .. import config as _cfg
        from ..resource_monitor import snapshot as _snapshot

        if not _cfg.get("compression_headroom_aware"):
            return configured, "DISABLED"
        snap = _snapshot()
        multiplier = _PRESSURE_THRESHOLD_MULTIPLIERS.get(int(snap.pressure), 1.0)
        effective = max(1, int(configured * multiplier))
        return effective, snap.pressure.name
    except Exception:  # noqa: BLE001
        return configured, "DISABLED"


def _emit_compression_event(
    payload: CompressedPayload, site: str, pressure_tier: str = "DISABLED"
) -> None:
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
                "pressure_tier": pressure_tier,
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
        allow_lossy: unused by the current (deterministic-only) cascade; kept
            for call-site compatibility (#119).
    """
    try:
        from .. import config as _cfg

        if not _cfg.get("compression_enabled"):
            return content
        ctx = context if _cfg.get("compression_context_aware") else ""
        configured_min = int(_cfg.get("compression_min_tokens") or _DEFAULT_MIN_TOKENS)
    except Exception:
        return content
    min_tokens, pressure_tier = _effective_min_tokens(configured_min)
    try:
        payload = compress_payload(
            content, context=ctx, min_tokens=min_tokens, allow_lossy=allow_lossy
        )
    except Exception:  # pragma: no cover - defensive; compress_payload already guards
        return content
    # Only emit telemetry for attempts that actually reached the compressor
    # (i.e. were large enough to try) — skip tiny no-op passthroughs.
    if payload.bytes_before >= min_tokens * _CHARS_PER_TOKEN:
        _emit_compression_event(payload, site, pressure_tier)
    return payload.compressed


def kompress_prose(text: str, *, context: str = "", site: str = "kompress") -> str:
    """Light, LLM-free extractive prose compression via Kompress (ModernBERT).

    Calls the Kompress model directly — use it where a site deliberately wants a
    fast extractive digest in place of a heavy local-LLM summarize (e.g. the
    searxng web connector). It is LOSSY (deletes tokens) but grounded: output is
    a subset of the input, so it cannot hallucinate.

    Returns the compressed text, or the original verbatim on any miss/error or when
    the optional ``headroom-ai[ml]`` dependency is not installed. Records a
    telemetry event on a successful compression. Never raises.
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


_WHITESPACE_RE = re.compile(r"\s")
_WS_RUN_RE = re.compile(r"[ \t]{2,}")
_NL_RUN_RE = re.compile(r"\n{3,}")


def squeeze_whitespace(text: str) -> str:
    """Deterministic whitespace normalization for prompt-injected previews:
    collapse runs of spaces/tabs to one space, strip trailing whitespace per
    line, and cap blank-line runs at one empty line. Markdown table alignment
    and indentation are NOT preserved — do not use on code payloads the agent
    must execute; use only where the text is a human/LLM-readable excerpt.
    """
    lines = [_WS_RUN_RE.sub(" ", ln).rstrip() for ln in text.split("\n")]
    return _NL_RUN_RE.sub("\n\n", "\n".join(lines)).strip()


_NORM_RE = re.compile(r"\W+")
_DEDUPE_MIN_CHARS = 40


def dedupe_snippets(parts: list[str]) -> list[str]:
    """Cross-snippet redundancy removal for context injections (#135).

    When one injection assembles several sources (skill + wiki + memory), the
    same load-bearing sentence often appears in more than one of them. Drops
    lines whose normalized form (case/punctuation-insensitive) already
    appeared in an earlier part; short lines (< 40 significant chars) are kept
    unconditionally so headers and list markers survive. Parts emptied by the
    dedupe are removed.
    """
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        kept: list[str] = []
        for line in part.split("\n"):
            norm = _NORM_RE.sub(" ", line).strip().lower()
            if len(norm) >= _DEDUPE_MIN_CHARS:
                if norm in seen:
                    continue
                seen.add(norm)
            kept.append(line)
        cleaned = "\n".join(kept).strip()
        if cleaned:
            out.append(cleaned)
    return out


def truncate_at_word(text: str, limit: int, *, marker: str = " … (truncated)") -> str:
    """Truncate ``text`` to at most ``limit`` characters without cutting a word
    in half. Cuts at the last whitespace at or before ``limit`` and appends
    ``marker``; falls back to a hard cut at ``limit`` when no whitespace is
    found (e.g. one long unbroken token). Returns ``text`` unchanged when it
    already fits.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    matches = list(_WHITESPACE_RE.finditer(window))
    cut = matches[-1].start() if matches else limit
    return text[:cut].rstrip() + marker


_DISPATCH_SIGNAL_RE = re.compile(
    r"\b(sub[- ]?agents?|agents?|fan[- ]?out|dispatch(?:es|ing|ed)?|"
    r"parallel|explore|delegat(?:e|es|ing|ed)|orchestrat(?:e|es|ing|ed))\b",
    re.IGNORECASE,
)

_AGENT_IO_GUIDANCE = (
    "[Skill Hub — agent I/O] When dispatching sub-agents, instruct each to "
    "return compressed findings (conclusions + file:line refs, not raw file "
    "dumps), dedupe across sources, and keep only what changes the next "
    "decision — so their output stays cheap to fold back in."
)


def agent_io_guidance(message: str) -> str | None:
    """One-line guidance for the orchestrator to compress its sub-agents' I/O.

    skill-hub has no hook into sub-agent dispatch prompts, so this surfaces in
    the *main loop's* injected context and only when the prompt signals a
    dispatch (mentions agents / fan-out / parallel / explore / delegate).
    Deterministic (no embedding, no LLM) so it works with the local model down.
    Returns the guidance line, or None when the message shows no dispatch intent.
    """
    if not message or not _DISPATCH_SIGNAL_RE.search(message):
        return None
    return _AGENT_IO_GUIDANCE


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
