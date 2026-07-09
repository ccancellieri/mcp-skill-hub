"""Fetch a URL and return compact, agent-facing text -- the implementation
behind the ``fetch_compressed`` MCP tool.

Policy: plain WebFetch dumps raw HTML/markdown straight into context. This
module fetches the page, strips HTML boilerplate to markdown-ish text, then
compresses it through the *same* deterministic-first cascade searxng's
``_summarize_results`` already uses for web search results (see
``compression.kompress_prose`` / ``compression.maybe_compress``): prose is
compressed lossily toward a fixed target ratio via Kompress, while
fenced code blocks and JSON bodies are compressed lossless-only so their
structure survives intact.

Whenever the returned text differs from what was actually fetched, the raw
fetched content is stashed behind a reversible ``<<ccr:HASH>>`` marker (the
same convention ``retrieve_compressed`` already rehydrates) so a caller can
recover the exact original when a task needs it -- byte-for-byte layout
evaluation, code review, anything the compact digest can't be trusted for.

No new top-level dependency: HTML stripping reuses the optional
``headroom-ai`` HTML extractor already wired into ``compression.compress_payload``
when installed, falling back to a stdlib-only tag-strip otherwise so output
is always boilerplate-free even without the extra.
"""
from __future__ import annotations

import html as _html
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0
_MAX_FETCH_BYTES = 2_000_000  # ~2MB cap on the fetched body
_USER_AGENT = "skill-hub-fetch_compressed/1.0"

_VALID_MODES = ("auto", "raw")

_HTML_TYPES = ("text/html", "application/xhtml+xml")

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

# Stdlib-only fallback strip, used only when headroom is not installed.
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


@dataclass
class FetchError:
    message: str


@dataclass
class FetchedPage:
    text: str
    content_type: str
    truncated: bool


def fetch_url(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> FetchedPage | FetchError:
    """Fetch ``url`` and decode it to text. Never raises -- transport/HTTP
    failures come back as a :class:`FetchError` for the caller to report."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return FetchError(
            f"unsupported URL scheme {parsed.scheme or '(none)'!r} -- "
            f"only http/https URLs are fetched"
        )

    import httpx

    try:
        resp = httpx.get(
            url, timeout=timeout, follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        return FetchError(f"timed out after {timeout:.0f}s fetching {url}")
    except httpx.HTTPStatusError as exc:
        return FetchError(f"HTTP {exc.response.status_code} fetching {url}")
    except httpx.HTTPError as exc:
        return FetchError(f"fetch failed for {url}: {exc}")
    except Exception as exc:  # noqa: BLE001 - unexpected transport failure
        return FetchError(f"fetch failed for {url}: {exc}")

    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    body = resp.content
    truncated = len(body) > _MAX_FETCH_BYTES
    if truncated:
        body = body[:_MAX_FETCH_BYTES]
    text = body.decode(resp.encoding or "utf-8", errors="replace")
    return FetchedPage(text=text, content_type=content_type, truncated=truncated)


def _classify(content_type: str) -> str:
    """"html" / "text" / "binary" from a Content-Type header (missing header
    is treated as text -- lenient, since plenty of servers omit it)."""
    if not content_type:
        return "text"
    if any(content_type.startswith(t) for t in _HTML_TYPES):
        return "html"
    if content_type.startswith("text/") or content_type.endswith(("+json", "+xml")) or (
        content_type in ("application/json", "application/xml")
    ):
        return "text"
    return "binary"


def _fallback_strip_html(raw_html: str) -> str:
    """Dependency-free HTML boilerplate strip, used only when headroom's HTML
    extractor is unavailable. Plain text, not real markdown, but always
    boilerplate-free."""
    text = _SCRIPT_STYLE_RE.sub("", raw_html)
    text = _TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _strip_html(raw_html: str) -> str:
    """Strip HTML boilerplate down to markdown-ish text.

    Routes through the same deterministic HTML-extraction strategy already
    wired into ``compression.compress_payload`` (the cascade searxng/
    search_web use). Falls back to a stdlib tag-strip when headroom is not
    installed, so the result is always boilerplate-free either way.
    """
    from .compression import compress_payload

    try:
        payload = compress_payload(raw_html, min_tokens=0, allow_lossy=False)
    except Exception as e:  # noqa: BLE001 - defensive, cascade already guards
        logger.debug("fetch_compressed: HTML extraction failed, using fallback: %s", e)
        payload = None
    if payload is not None and payload.content_type == "HTML":
        return payload.compressed
    return _fallback_strip_html(raw_html)


def _split_segments(text: str) -> list[tuple[bool, str]]:
    """Split ``text`` into ``(is_code, chunk)`` pairs on fenced ``` code
    blocks. Fenced blocks stay intact for lossless-only compression;
    everything else is prose."""
    segments: list[tuple[bool, str]] = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
        if m.start() > pos:
            segments.append((False, text[pos:m.start()]))
        segments.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        segments.append((False, text[pos:]))
    return segments or [(False, text)]


def _compress_prose(text: str, context: str) -> str:
    """Lossy prose compression toward a fixed target ratio -- the
    same kompress_prose -> maybe_compress fallback searxng's
    ``_summarize_results`` uses for web search results."""
    from .compression import kompress_prose, maybe_compress

    digest = kompress_prose(text, context=context, site="fetch_compressed")
    if digest == text:
        digest = maybe_compress(text, context=context, site="fetch_compressed", allow_lossy=False)
    return digest


def _compress_lossless(text: str, context: str) -> str:
    """Deterministic-only compression (JSON minify, dedup, ...) -- used for
    code blocks and JSON bodies so their structure is never mangled."""
    from .compression import maybe_compress

    return maybe_compress(text, context=context, site="fetch_compressed", allow_lossy=False)


def _compress_document(text: str, *, context: str, is_json: bool) -> str:
    if is_json:
        return _compress_lossless(text, context)
    parts = [
        _compress_lossless(chunk, context) if is_code else _compress_prose(chunk, context)
        for is_code, chunk in _split_segments(text)
    ]
    return "".join(parts)


def _stash_original(original: str, compressed: str, *, url: str) -> str | None:
    """Stash ``original`` behind a fresh ``<<ccr:HASH>>`` marker in headroom's
    shared compression store so ``retrieve_compressed`` can rehydrate it.
    Returns None (no marker) when the store is unavailable -- never raises."""
    try:
        from headroom.cache.compression_store import get_compression_store

        return get_compression_store().store(
            original, compressed,
            tool_name="fetch_compressed",
            query_context=url[:200],
            compression_strategy="FETCH_COMPRESSED",
        )
    except Exception as e:  # noqa: BLE001 - stashing must never break the fetch
        logger.debug("fetch_compressed: stash failed (non-fatal): %s", e)
        return None


def run(url: str, mode: str = "auto") -> str:
    """Fetch ``url`` and return compact text per ``mode``. Implementation
    behind the ``fetch_compressed`` MCP tool -- see its docstring for the
    policy. Never raises: fetch/content-type/config failures come back as a
    plain error string.
    """
    if mode not in _VALID_MODES:
        return f"fetch_compressed error: invalid mode {mode!r} (expected 'auto' or 'raw')"

    fetched = fetch_url(url)
    if isinstance(fetched, FetchError):
        return f"fetch_compressed error: {fetched.message}"

    kind = _classify(fetched.content_type)
    if kind == "binary":
        return (
            f"fetch_compressed error: content-type {fetched.content_type!r} is not "
            f"readable text/HTML/JSON -- fetch_compressed only handles web pages, "
            f"plain text, markdown, and JSON."
        )

    raw_text = fetched.text
    stripped = _strip_html(raw_text) if kind == "html" else raw_text
    if not stripped.strip():
        return f"fetch_compressed: {url} returned no extractable text content."

    is_json = (
        fetched.content_type in ("application/json",)
        or fetched.content_type.endswith("+json")
        or (kind == "text" and stripped.lstrip()[:1] in "{[")
    )

    from . import config as _cfg

    compression_on = mode == "auto" and bool(_cfg.get("compression_enabled"))
    final_text = (
        _compress_document(stripped, context=url, is_json=is_json)
        if compression_on else stripped
    )

    marker = ""
    if final_text != raw_text:
        stash_hash = _stash_original(raw_text, final_text, url=url)
        if stash_hash:
            marker = f"\n\n<<ccr:{stash_hash}>>"

    note = (
        f"\n[truncated: fetch stopped at {_MAX_FETCH_BYTES // 1_000_000}MB]"
        if fetched.truncated else ""
    )
    return f"# {url}{note}\n\n{final_text}{marker}"
