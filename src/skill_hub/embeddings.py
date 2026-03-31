"""Ollama embedding, re-ranking, and LLM compaction."""

import json
import re

import httpx

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"       # ollama pull nomic-embed-text  (274 MB)
RERANK_MODEL = "deepseek-r1:1.5b"     # ollama pull deepseek-r1:1.5b  (1.1 GB)

_RERANK_PROMPT = """\
You are a relevance judge. Given a user query and a skill description, reply with a
single JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}

Query: {query}
Skill name: {name}
Skill description: {description}
"""


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Return embedding vector from Ollama."""
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": model, "input": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...]], ...}
    embeddings = data.get("embeddings") or data.get("embedding")
    if isinstance(embeddings[0], list):
        return embeddings[0]
    return embeddings


def rerank(query: str, candidates: list[dict],
           model: str = RERANK_MODEL) -> list[dict]:
    """
    Re-rank candidates using a small reasoning model.
    Each candidate dict must have keys: id, name, description, content.
    Returns candidates sorted by re-rank score descending.
    """
    import json

    scored: list[tuple[float, dict]] = []
    for c in candidates:
        prompt = _RERANK_PROMPT.format(
            query=query,
            name=c["name"],
            description=c.get("description", ""),
        )
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            result = json.loads(raw)
            score = float(result.get("score", 0.5))
        except Exception:
            score = 0.5
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


_COMPACT_PROMPT = """\
You are a technical summarizer. Compact this conversation/task into a concise digest.
Output ONLY a JSON object with these fields:
{{
  "title": "<short title, max 10 words>",
  "summary": "<what was discussed/decided, 2-4 sentences>",
  "decisions": ["<key decision 1>", "<key decision 2>"],
  "tools_used": ["<tool or plugin that was relevant>"],
  "open_questions": ["<unresolved item>"],
  "tags": "<comma-separated tags>"
}}

Task/conversation to compact:
{content}
"""

_REWRITE_PROMPT = """\
You are a query optimizer. Given the user's message and conversation context,
rewrite the message as a clear, specific search query that would find the most
relevant skills and tools. Output ONLY the rewritten query, nothing else.

User message: {message}
Context (recent topics): {context}
"""


def compact(content: str, model: str = RERANK_MODEL) -> dict:
    """
    Use a local LLM to compact a conversation/task summary.
    Returns a structured digest dict.
    """
    prompt = _COMPACT_PROMPT.format(content=content[:4000])
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {
        "title": "Untitled",
        "summary": content[:500],
        "decisions": [],
        "tools_used": [],
        "open_questions": [],
        "tags": "",
    }


def rewrite_query(message: str, context: str = "",
                  model: str = RERANK_MODEL) -> str:
    """
    Use a local LLM to rewrite a user message into an optimized search query.
    Saves Claude tokens by doing query understanding locally.
    Falls back to original message on failure.
    """
    prompt = _REWRITE_PROMPT.format(message=message, context=context[:1000])
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw and len(raw) > 5:
            return raw
    except Exception:
        pass
    return message


def ollama_available(model: str = EMBED_MODEL) -> bool:
    """Check whether the required Ollama model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False
