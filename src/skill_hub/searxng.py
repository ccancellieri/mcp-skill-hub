"""SearXNG web search connector — RAG fallback for Stage 4.1.

When skill search returns 0 results above threshold, this module:
1. Auto-detects a reachable SearXNG instance (configured URL → localhost:8080)
2. Queries it for the user's message
3. Summarizes the top results with the local LLM
4. Returns a formatted context string for injection into Claude's systemMessage

Disabled automatically if:
- searxng_enabled is False in config
- No SearXNG instance is reachable
- Ollama is not available (no point summarizing without LLM)
"""

from __future__ import annotations

import re

_searxng_url_cache: str | None = None   # resolved URL, cached once per process
_searxng_url_checked: bool = False       # whether we've done the probe


def _resolve_searxng_url(timeout: float = 5.0) -> str | None:
    """
    Try the configured URL first, then localhost:8080.
    Returns the first URL that responds with HTTP 200, or None.
    Caches the result for the process lifetime.
    """
    global _searxng_url_cache, _searxng_url_checked

    if _searxng_url_checked:
        return _searxng_url_cache

    _searxng_url_checked = True

    import httpx
    from . import config as _cfg

    candidates: list[str] = []
    explicit = str(_cfg.get("searxng_url") or "").strip()
    if explicit:
        candidates.append(explicit.rstrip("/"))
    candidates.append("http://localhost:8989")

    for url in candidates:
        try:
            resp = httpx.get(
                f"{url}/search",
                params={"q": "ping", "format": "json"},
                timeout=timeout,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                _searxng_url_cache = url
                return url
        except Exception:
            continue

    _searxng_url_cache = None
    return None


def _searxng_search(query: str, base_url: str,
                    top_k: int = 3, timeout: float = 5.0) -> list[dict]:
    """
    Query SearXNG JSON API.
    Returns list of {"title": str, "url": str, "snippet": str}.
    """
    import httpx

    resp = httpx.get(
        f"{base_url}/search",
        params={"q": query, "format": "json", "categories": "it,science,general"},
        timeout=timeout,
        follow_redirects=True,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    output: list[dict] = []
    for r in results[:top_k]:
        output.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", r.get("snippet", ""))[:300],
        })
    return output


_SUMMARIZE_PROMPT = """\
Summarize these web search results to help answer the user's question.
Be concise (3-5 sentences). Include the most relevant source URL inline.

Question: {query}

Results:
{results_text}

Summary:"""


def _summarize_results(query: str, results: list[dict]) -> str:
    """
    Condense the fetched results into a context string.

    Default path (``searxng_use_llm_summary`` False): a fast, light, LLM-free
    extractive digest via Kompress (ModernBERT) — no Ollama dependency, grounded
    (cannot hallucinate). Falls back to lossless deterministic compaction, then to
    the raw results, if Kompress is unavailable.

    Legacy path (``searxng_use_llm_summary`` True): an abstractive summary from the
    local LLM (reason_model) — fluent but slow and able to invent facts.

    Returns the digest string, or empty string on total failure.
    """
    if not results:
        return ""

    from . import config as _cfg

    results_text = "\n".join(
        f"{i+1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(results)
    )

    # --- Default: light extractive Kompress, no local LLM --------------------
    if not _cfg.get("searxng_use_llm_summary"):
        from .compression import kompress_prose, maybe_compress

        digest = kompress_prose(results_text, context=query, site="searxng")
        if digest == results_text:
            # Kompress unavailable / no-op — fall back to lossless deterministic
            # compaction (JSON/log/table). Returns the raw text if nothing fires.
            digest = maybe_compress(
                results_text, context=query, site="searxng", allow_lossy=False
            )
        return digest

    # --- Legacy: abstractive summary via the local Ollama LLM ----------------
    from .embeddings import RERANK_MODEL
    from .llm import LLMError, get_provider

    # Deterministic-only here: this text is fed to the local summarize LLM, which
    # cannot rehydrate lossy Kompress output, so never apply the ML paths.
    from .compression import maybe_compress

    results_text = maybe_compress(
        results_text, context=query, site="searxng", allow_lossy=False
    )

    model = str(_cfg.get("reason_model") or RERANK_MODEL)
    resolved = model if "/" in model else f"ollama/{model}"
    prompt = _SUMMARIZE_PROMPT.format(
        query=query[:200],
        results_text=results_text,
    )

    try:
        raw = get_provider().complete(
            prompt, model=resolved,
            max_tokens=220, temperature=0.3,
            timeout=float(_cfg.get("searxng_timeout") or 5) + 20,
        ).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    except LLMError:
        return ""


def searxng_context(query: str) -> str | None:
    """
    Public entry point for Stage 4.1.

    Returns a formatted context string or None if:
    - searxng_enabled is False
    - No SearXNG instance is reachable
    - Ollama is not available
    - Search or summarization fails
    """
    from . import config as _cfg
    from .embeddings import embed_available, EMBED_MODEL

    if not _cfg.is_service_enabled("searxng"):
        return None

    # Skip if embedding is unavailable — no point fetching without summarization
    if not embed_available():
        return None

    probe_timeout = float(_cfg.get("searxng_timeout") or 5)
    search_timeout = float(_cfg.get("searxng_search_timeout") or 15)
    top_k = int(_cfg.get("searxng_top_k") or 3)

    base_url = _resolve_searxng_url(timeout=probe_timeout)
    if not base_url:
        return None

    try:
        results = _searxng_search(query, base_url, top_k=top_k, timeout=search_timeout)
    except Exception:
        return None

    if not results:
        return None

    summary = _summarize_results(query, results)
    sources = " | ".join(r["url"] for r in results if r.get("url"))

    if not summary and not sources:
        return None

    parts = ["[Web context — SearXNG]"]
    if summary:
        parts.append(summary)
    if sources:
        parts.append(f"Sources: {sources}")

    return "\n".join(parts)
