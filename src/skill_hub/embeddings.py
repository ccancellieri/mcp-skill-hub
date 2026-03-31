"""Ollama embedding, re-ranking, and LLM compaction."""

import json
import re

import httpx

from . import config as _cfg
from .activity_log import log_llm

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

    log_llm("rerank", model=model, candidates=len(candidates))
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
    log_llm("compact", model=model, input_chars=len(content[:4000]))
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
    log_llm("rewrite_query", model=model)
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
    log_llm("optimize_context", model=model, entries=len(entries))
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


_TRIAGE_PROMPT = """\
You are a smart message triage assistant for an AI coding tool (Claude). Your job is to \
decide whether this user message can be handled locally or needs the expensive cloud AI.

You have access to these local /hub-* commands:
- /hub-list-models — show installed Ollama models
- /hub-status — health check, models, DB stats
- /hub-list-tasks — show open/closed tasks
- /hub-list-skills — list indexed skills
- /hub-list-teachings — show teaching rules
- /hub-search-skills <q> — semantic skill search
- /hub-suggest-plugins <q> — find matching plugins
- /hub-configure — view/set config
- /hub-profile — list/switch plugin profiles
- /hub-token-stats — token savings report
- /hub-help — command reference
- /hub-search-context <q> — unified search across skills, tasks, memory
- /hub-optimize-context — analyze memory for pruning
- /hub-save-memory <desc> — save memory entry
- /hub-teach rule -> target — add teaching rule
- /hub-digest — conversation digest

Classify the message into ONE of these actions:

"local_answer" — You can answer this directly. ONLY for: greetings, confirmations ("yes", \
"ok"), questions about what /hub-* commands are available, or trivial clarifications. \
Do NOT answer coding questions or factual questions about the codebase — those need Claude. \
Provide the answer.

"local_action" — This maps to a local /hub-* command. Identify which one. Examples:
  "what models do I have?" → /hub-list-models
  "show my tasks" → /hub-list-tasks
  "what plugins for debugging?" → /hub-suggest-plugins debugging
  "search for MCP skills" → /hub-search-skills MCP server

"enrich_and_forward" — This needs Claude but you can help by:
  - Extracting the core intent (removing filler words)
  - Identifying which files/functions are relevant
  - Suggesting which skills/plugins Claude should use
  Provide a concise hint for Claude.

"pass_through" — Complex coding task, debugging, or creative work that only Claude \
can handle well. No pre-processing needed.

Output ONLY a JSON object:
{{
  "action": "local_answer|local_action|enrich_and_forward|pass_through",
  "answer": "<direct answer if local_answer, else null>",
  "command": "<hub command if local_action, e.g. '/hub-list-tasks', else null>",
  "hint": "<concise hint for Claude if enrich_and_forward, else null>",
  "confidence": <0.0-1.0>,
  "estimated_tokens_saved": <rough estimate of tokens Claude won't need to process>
}}

{context}

User message: {message}
"""


def triage_message(message: str, context: str = "",
                   model: str = RERANK_MODEL) -> dict:
    """
    Use local LLM to triage a user message before sending to Claude.

    Returns a dict with action, optional answer/command/hint, and confidence.
    Falls back to pass_through on any error.
    """
    log_llm("triage", model=model, message_len=len(message))

    ctx_section = ""
    if context:
        ctx_section = f"\nRecent conversation context:\n{context}\n"

    prompt = _TRIAGE_PROMPT.format(
        message=message[:2000],
        context=ctx_section,
    )

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=int(_cfg.get("hook_llm_triage_timeout") or 30),
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate action field
            valid_actions = {"local_answer", "local_action",
                             "enrich_and_forward", "pass_through"}
            if result.get("action") in valid_actions:
                return result
    except Exception:
        pass

    return {
        "action": "pass_through",
        "answer": None,
        "command": None,
        "hint": None,
        "confidence": 0.0,
        "estimated_tokens_saved": 0,
    }


def ollama_available(model: str = EMBED_MODEL) -> bool:
    """Check whether the required Ollama model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False
