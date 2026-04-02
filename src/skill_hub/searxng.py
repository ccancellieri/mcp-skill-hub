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
    candidates.append("http://localhost:8080")

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
    Use the local LLM (reason_model) to summarize search results.
    Returns the summary string, or empty string on failure.
    """
    if not results:
        return ""

    import httpx as _httpx
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE, RERANK_MODEL

    results_text = "\n".join(
        f"{i+1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(results)
    )

    model = str(_cfg.get("reason_model") or RERANK_MODEL)
    prompt = _SUMMARIZE_PROMPT.format(
        query=query[:200],
        results_text=results_text,
    )

    try:
        resp = _httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 220},
            },
            timeout=float(_cfg.get("searxng_timeout") or 5) + 20,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip think tags if present (DeepSeek R1)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    except Exception:
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
    from .embeddings import ollama_available, EMBED_MODEL

    if not _cfg.get("searxng_enabled"):
        return None

    # Skip if Ollama is unavailable — no point fetching without summarization
    if not ollama_available(EMBED_MODEL):
        return None

    timeout = float(_cfg.get("searxng_timeout") or 5)
    top_k = int(_cfg.get("searxng_top_k") or 3)

    base_url = _resolve_searxng_url(timeout=timeout)
    if not base_url:
        return None

    try:
        results = _searxng_search(query, base_url, top_k=top_k, timeout=timeout)
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
