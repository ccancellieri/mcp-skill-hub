"""Ollama embedding + optional deepseek-r1 re-ranking."""

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
            # Strip <think>...</think> blocks emitted by deepseek-r1
            import re
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            result = json.loads(raw)
            score = float(result.get("score", 0.5))
        except Exception:
            score = 0.5
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def ollama_available(model: str = EMBED_MODEL) -> bool:
    """Check whether the required Ollama model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False
