"""Ollama embedding, re-ranking, and LLM compaction."""

import json
import re

import httpx

from . import config as _cfg

# These module-level names are kept for backwards compatibility with imports,
# but always read the live config value.
OLLAMA_BASE = _cfg.get("ollama_base")
EMBED_MODEL = _cfg.get("embed_model")
RERANK_MODEL = _cfg.get("reason_model")

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


_OPTIMIZE_CONTEXT_PROMPT = """\
You are a memory-optimization assistant for an AI coding tool. You are reviewing \
memory files that are loaded into context each session, consuming tokens.

Your job: analyze each memory entry and classify it as one of:
- KEEP — still valuable, well-written
- PRUNE — stale, completed, or no longer relevant (explain why)
- COMPACT — too verbose, can be shortened significantly (provide compacted version)
- MERGE — duplicate or overlapping with another entry (name which one)

For each entry, output a JSON object on its own line:
{{"file": "<filename>", "action": "KEEP|PRUNE|COMPACT|MERGE", "reason": "<why>", \
"compacted": "<shortened content, only if action=COMPACT>"}}

After all entries, output a summary line:
{{"summary": true, "total": <N>, "keep": <N>, "prune": <N>, "compact": <N>, \
"merge": <N>, "est_tokens_saved": <N>}}

Be aggressive about pruning completed projects and compacting verbose entries. \
Each token saved here is saved on EVERY future session.

Memory entries to review:
{content}
"""


def optimize_context(entries: list[dict],
                     model: str = RERANK_MODEL) -> list[dict]:
    """
    Use local LLM to evaluate memory entries and recommend pruning/compaction.

    Args:
        entries: list of {"file": str, "category": str, "tokens": int, "content": str}
        model: reasoning model to use

    Returns:
        list of recommendation dicts from the LLM
    """
    # Format entries for the prompt
    formatted = []
    for e in entries:
        formatted.append(
            f"--- {e['file']} ({e['category']}, ~{e['tokens']} tokens) ---\n"
            f"{e['content'][:2000]}"
        )
    content = "\n\n".join(formatted)

    # Respect compact_max_input_chars but allow more for this use case
    max_chars = int(_cfg.get("compact_max_input_chars")) * 4  # 16k chars
    prompt = _OPTIMIZE_CONTEXT_PROMPT.format(content=content[:max_chars])

    results = []
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300.0,  # longer timeout for many entries
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Extract all JSON objects from the response
        for match in re.finditer(r"\{[^{}]+\}", raw):
            try:
                obj = json.loads(match.group())
                results.append(obj)
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        results.append({
            "error": str(exc),
            "summary": True,
            "total": len(entries),
            "keep": len(entries),
            "prune": 0,
            "compact": 0,
            "merge": 0,
            "est_tokens_saved": 0,
        })

    return results


_CONVERSATION_DIGEST_PROMPT = """\
You are a context-optimization assistant. Summarize this conversation state into a \
compact digest that captures the CURRENT focus, recent decisions, and active intent.

Output ONLY a JSON object:
{{
  "current_focus": "<what the user is working on right now, 1 sentence>",
  "recent_decisions": ["<key decision 1>", "<key decision 2>"],
  "active_plugins": ["<plugin name that seems relevant>"],
  "stale_topics": ["<topic that was discussed earlier but is no longer active>"],
  "suggested_profile": "<profile name if conversation clearly fits one, else null>"
}}

Conversation context:
{content}
"""

_EXHAUSTION_SAVE_PROMPT = """\
You are an AI session-saving assistant. The external AI (Claude) is exhausted/unavailable.
Analyze the conversation state and produce a structured task save so work can resume later.

Output ONLY a JSON object:
{{
  "title": "<descriptive title of what was being worked on, max 12 words>",
  "summary": "<what was accomplished and what remains, 3-5 sentences>",
  "decisions": ["<key decision made during this session>"],
  "next_steps": ["<concrete next action to take when resuming>"],
  "files_modified": ["<file paths that were changed>"],
  "tags": "<comma-separated tags>"
}}

Session state:
{content}
"""


def conversation_digest(messages: list[str],
                        model: str = RERANK_MODEL) -> dict:
    """
    Use local LLM to produce a compact digest of conversation state.
    Used for periodic context compaction and relevance decay tracking.
    """
    content = "\n---\n".join(messages[-10:])  # last 10 messages
    max_chars = int(_cfg.get("compact_max_input_chars")) * 2
    prompt = _CONVERSATION_DIGEST_PROMPT.format(content=content[:max_chars])

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {
        "current_focus": "unknown",
        "recent_decisions": [],
        "active_plugins": [],
        "stale_topics": [],
        "suggested_profile": None,
    }


def exhaustion_save(content: str,
                    model: str = RERANK_MODEL) -> dict:
    """
    Use local LLM to generate a structured task save when Claude is exhausted.
    Returns a digest suitable for store.save_task().
    """
    max_chars = int(_cfg.get("compact_max_input_chars")) * 3  # 12k
    prompt = _EXHAUSTION_SAVE_PROMPT.format(content=content[:max_chars])

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {
        "title": "Session interrupted",
        "summary": content[:500],
        "decisions": [],
        "next_steps": [],
        "files_modified": [],
        "tags": "auto-saved,exhaustion",
    }


def ollama_available(model: str = EMBED_MODEL) -> bool:
    """Check whether the required Ollama model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False
