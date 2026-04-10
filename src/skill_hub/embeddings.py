"""Ollama embedding, re-ranking, and LLM compaction."""

import json
import re

import httpx

from . import config as _cfg
from .activity_log import get_logger, log_llm, llm_timer

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


def embed(text: str, model: str = EMBED_MODEL, timeout: float = 15.0) -> list[float]:
    """Return embedding vector from Ollama."""
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": model, "input": text},
        timeout=timeout,
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


def smart_memory_write(
    content: str,
    existing_index: str,
    model: str = RERANK_MODEL,
) -> dict:
    """Local LLM generates memory, then self-evaluates quality.

    Returns:
        {
            "result": dict,          # the memory entry (filename, name, etc.)
            "quality": float,        # 0.0-1.0 self-assessed quality score
            "escalate": bool,        # True if Claude should handle this instead
            "reason": str,           # why escalation is needed (or "ok")
        }
    """
    # Try progressively smaller models if primary unavailable
    chosen_model = model
    if not ollama_available(chosen_model):
        for fallback in ("qwen2.5-coder:3b", "deepseek-r1:1.5b"):
            if ollama_available(fallback):
                chosen_model = fallback
                break
        else:
            return {
                "result": {}, "quality": 0.0,
                "escalate": True, "reason": "no_local_model",
            }

    prompt = f"""You are a memory-management assistant for an AI coding tool.

## Task
Generate a memory file from the session context below. Focus on:
- Decisions made and WHY
- User preferences discovered
- Project knowledge not in the code
- Patterns to repeat or avoid

Do NOT save: code patterns (read the code), git history (use git log),
anything already in existing memory files.

## Session context
{content[:4000]}

## Existing memory (avoid duplicates)
{existing_index[:2000]}

## Output
Respond with ONLY this JSON:
{{
  "filename": "<descriptive-kebab-case.md>",
  "name": "<short title>",
  "description": "<one-line description for the memory index>",
  "type": "<user|feedback|project|reference>",
  "content": "<the memory content, 2-6 sentences>",
  "key_entities": ["<list>", "<of>", "<important>", "<names/terms>", "<from the context>"],
  "detail_score": <0.0-1.0 how much important detail you preserved>
}}"""

    try:
        with llm_timer() as _t:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": chosen_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
        log_llm("smart_memory_write", model=chosen_model, duration=_t.duration,
                input_chars=len(content))
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return {
                "result": {}, "quality": 0.0,
                "escalate": True, "reason": "no_json_output",
            }
        entry = json.loads(json_match.group())
    except Exception as exc:
        return {
            "result": {}, "quality": 0.0,
            "escalate": True, "reason": f"llm_error:{str(exc)[:60]}",
        }

    # --- Quality evaluation ---
    mem_content = entry.get("content", "")
    key_entities = entry.get("key_entities", [])
    detail_score = float(entry.get("detail_score", 0.5))

    # Heuristic quality checks
    quality = detail_score

    # Check 1: content not empty or too short
    if len(mem_content) < 30:
        quality *= 0.3

    # Check 2: key entities actually appear in content
    entity_ratio = 1.0
    if key_entities:
        found = sum(1 for e in key_entities if e.lower() in mem_content.lower())
        entity_ratio = found / len(key_entities)
        quality *= (0.5 + 0.5 * entity_ratio)  # scale 0.5-1.0

    # Check 3: content length vs source length ratio
    # Too short = detail loss; too long = not compacted
    source_len = len(content)
    if source_len > 0:
        ratio = len(mem_content) / source_len
        if ratio < 0.01:  # less than 1% — too much lost
            quality *= 0.5
        elif ratio > 0.5:  # more than 50% — not compacted enough
            quality *= 0.8

    # Check 4: is it a duplicate?
    if existing_index and entry.get("filename", "") in existing_index:
        quality *= 0.7

    # Clamp
    quality = max(0.0, min(1.0, quality))

    # Decide escalation threshold
    escalate = quality < 0.4
    reason = "ok"
    if escalate:
        if len(mem_content) < 30:
            reason = "content_too_short"
        elif key_entities and entity_ratio < 0.3:
            reason = "key_entities_missing"
        else:
            reason = f"low_quality:{quality:.2f}"

    # demoted to DEBUG — quality already shown in the STOP "memory saved" line
    get_logger().debug("   smart_memory_quality: quality=%s escalate=%s reason=%s",
                       f"{quality:.2f}", escalate, reason)

    return {
        "result": entry,
        "quality": quality,
        "escalate": escalate,
        "reason": reason,
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

"local_answer" — You can answer this directly. ONLY for messages that are ≤ 120 characters \
AND are one of: bare greetings ("hi", "hello"), single-word confirmations ("yes", "ok", "no"), \
or a direct question about which /hub-* commands exist. NEVER use this for coding questions, \
bug reports, refactoring requests, or anything that references code, files, or errors — those \
need Claude. Provide the answer.

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

"local_agent" — This is a well-defined task that can be handled by a local agent with \
tools (shell commands, skills, file reading, search). Good for: git operations, running \
tests, searching code, reading files, executing skill workflows. NOT for: complex refactoring, \
architecture decisions, multi-file edits, or creative coding that needs Claude's reasoning.

"pass_through" — Complex coding task, debugging, or creative work that only Claude \
can handle well. No pre-processing needed.

Output ONLY a JSON object:
{{
  "action": "local_answer|local_action|local_agent|enrich_and_forward|pass_through",
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

    # Prepend static persona seed to triage prompt (dynamic persona skipped here —
    # store I/O would add 200ms per message in the hot path)
    _seed = str(_cfg.get("local_system_prompt") or "").strip()

    prompt = _TRIAGE_PROMPT.format(
        message=message[:2000],
        context=ctx_section,
    )
    if _seed:
        prompt = f"{_seed}\n\n{prompt}"

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
            valid_actions = {"local_answer", "local_action", "local_agent",
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


_SKILL_LIFECYCLE_PROMPT = """\
You are a context manager for Claude (an AI coding assistant).

## Current conversation context
{context_summary}

## User's new message
{message}

## Currently loaded skills (injected into Claude's context)
{loaded_list}

## Candidate skills (from semantic search, not yet loaded)
{candidate_list}

Decide which skills to KEEP (still useful for this message), ADD (from candidates, \
would help), or DROP (no longer relevant).

Rules:
- Load up to {max_skills} skills total (keep + add). Using fewer is fine if \
candidates are irrelevant, but DO load multiple complementary skills when they \
cover different aspects of the task (e.g. schema design + optimization + best \
practices for a database query).
- Prefer adding skills that cover different angles over loading just the single \
best match.

Also update the context summary: merge the new message's topic in. Keep all prior \
context still relevant. 2–4 sentences max.

Output ONLY this JSON, no markdown, no explanation:
{{"keep": ["skill_id", ...], "add": ["skill_id", ...], "drop": ["skill_id", ...], \
"context_summary": "..."}}"""

_PROMPT_OPT_PROMPT = """\
You are a prompt engineer for Claude (an AI coding assistant).
Rewrite the user message below into a clearer, more structured prompt.

Rules:
- Weave in relevant details from the conversation context (file names, \
technologies, prior decisions) so Claude has full context without re-asking.
- State constraints explicitly (language, framework, coding style).
- When the task has parts, number them and tell Claude the expected output format.
- Preserve the exact intent — never change WHAT is asked, only add clarity.
- Never invent file names, errors, or details not present in the message or context.
- If the message is already specific and complete, return it unchanged.
- Brevity beats padding: add only what genuinely helps.

## Conversation context
{context_summary}

## User message
{message}

Output ONLY the rewritten prompt, no JSON, no explanation:"""


def eval_skill_lifecycle(
    message: str,
    context_summary: str,
    loaded_skills: list[dict],
    candidate_skills: list[dict],
    model: str = RERANK_MODEL,
) -> dict:
    """Decide which skills to keep/add/drop and update the rolling context summary.

    Focused single-task call: structured classification only.
    Runs at temperature=0.0 with a tight token budget (250).

    Returns {"keep": [...], "add": [...], "drop": [...], "context_summary": "..."}
    """
    loaded_list = "\n".join(
        f"  - {s['id']}: {s.get('description', '')[:100]}"
        for s in loaded_skills
    ) or "  (none)"
    candidate_list = "\n".join(
        f"  - {s['id']}: {s.get('description', '')[:100]}"
        for s in candidate_skills
    ) or "  (none)"

    from . import config as _cfg
    max_skills = int(_cfg.get("hook_context_top_k_skills") or 5)

    prompt = _SKILL_LIFECYCLE_PROMPT.format(
        context_summary=context_summary or "(new session — no prior context)",
        message=message[:1200],
        loaded_list=loaded_list,
        candidate_list=candidate_list,
        max_skills=max_skills,
    )

    try:
        with llm_timer() as _t:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 250},
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            import json as _json
            result = _json.loads(json_match.group())
            log_llm("eval_skill_lifecycle", model=model, duration=_t.duration,
                    loaded=len(loaded_skills), candidates=len(candidate_skills))
            return result
        log_llm("eval_skill_lifecycle", model=model, duration=_t.duration,
                loaded=len(loaded_skills), candidates=len(candidate_skills))
    except Exception:
        pass

    return {
        "keep": [s["id"] for s in loaded_skills],
        "add": [],
        "drop": [],
        "context_summary": context_summary,
    }


def optimize_prompt(
    message: str,
    context_summary: str,
    model: str = RERANK_MODEL,
) -> str:
    """Rewrite the user message into a richer, better-structured prompt for Claude.

    Only called when there is meaningful conversation context and the message is
    long enough to benefit from enrichment (≥ 150 chars).
    Runs at temperature=0.2 for more expressive rewrites.

    Returns the optimized prompt string, or the original message on failure.
    """
    prompt = _PROMPT_OPT_PROMPT.format(
        context_summary=context_summary,
        message=message[:1500],
    )

    try:
        with llm_timer() as _t:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 400},
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
        log_llm("optimize_prompt", model=model, duration=_t.duration,
                message_len=len(message))
        # Reject obviously broken outputs
        if result and len(result) > 20 and not result.startswith("{"):
            return result
    except Exception:
        pass

    return message


def dynamic_context_eval(
    message: str,
    context_summary: str,
    loaded_skills: list[dict],
    candidate_skills: list[dict],
    model: str = RERANK_MODEL,
) -> dict:
    """Compatibility wrapper: skill lifecycle + prompt optimization in one call.

    Internally calls eval_skill_lifecycle (fast, structured) and optimize_prompt
    (conditional, warmer temperature) as two separate focused LLM calls.

    Returns {"keep": [...], "add": [...], "drop": [...],
             "context_summary": "...", "optimized_prompt": "..."}
    """
    lifecycle = eval_skill_lifecycle(
        message, context_summary, loaded_skills, candidate_skills, model
    )
    new_summary = lifecycle.get("context_summary", context_summary)

    # Only optimize prompt when there is real conversation context and the
    # message is substantive enough to benefit from enrichment.
    if context_summary and len(message.strip()) >= 150:
        opt = optimize_prompt(message, new_summary, model)
    else:
        opt = message

    return {
        "keep": lifecycle.get("keep", []),
        "add": lifecycle.get("add", []),
        "drop": lifecycle.get("drop", []),
        "context_summary": new_summary,
        "optimized_prompt": opt,
    }


_CACHE_VERIFY_PROMPT = """\
A user asked a coding question and a cached answer exists from a previous session.
Decide if the cached answer is still valid given the current question.

Cached question: {cached_query}
Cached answer (first 400 chars): {cached_response}
Current question: {current_query}

Output ONLY JSON: {{"valid": true/false, "reason": "<one sentence>"}}"""

_DECOMPOSE_PROMPT = """\
The user sent a complex multi-part request to a coding assistant. \
Break it into ordered atomic subtasks so the assistant can focus on one at a time.

Request: {message}

Output ONLY JSON:
{{"parts": ["subtask 1 description", "subtask 2 description", ...], \
"count": <integer>, "context_type": "<refactor|debug|explain|implement|review|other>"}}

Rules:
- Only split if there are genuinely 2+ independent actions (different verbs on different things)
- Do NOT split a single request just because it is long
- Maximum 5 parts"""

_ERROR_EXTRACT_PROMPT = """\
The following text is a coding assistant's response. Extract the primary error or exception \
if one is present, plus a brief fix hint.

Response (first 1500 chars):
{response}

Output ONLY JSON: {{"has_error": true/false, "error_signature": "<ExceptionType: short message>", \
"fix_hint": "<one sentence fix>"}}
If no clear error is present, output: {{"has_error": false}}"""

_AUTO_SKILL_PROMPT = """\
A user has repeatedly asked variations of the same question in a coding tool. \
Generate a local skill JSON that automates it.

Canonical question: {canonical}
Seen {count} times.

A local skill runs shell commands in sequence and formats the output. \
Good skills are for: git operations, running tests, showing project status, listing files.
Bad skills: anything requiring reasoning, code generation, or complex decisions.

Output ONLY a JSON object (no markdown):
{{
  "name": "<kebab-case-name>",
  "description": "<what it does, one sentence>",
  "triggers": ["<phrase 1>", "<phrase 2>", "<phrase 3>"],
  "steps": [
    {{"run": "<shell command>", "as": "<variable_name>"}},
    ...
  ],
  "output": "<template using {{variable_name}} placeholders>"
}}

If the question cannot be automated with shell commands, output: {{"can_automate": false}}"""


def verify_cache_hit(cached_query: str, cached_response: str,
                     current_query: str,
                     model: str = RERANK_MODEL) -> bool:
    """Ask local LLM if a cached answer is still valid for the current query."""
    log_llm("verify_cache_hit", model=model)
    prompt = _CACHE_VERIFY_PROMPT.format(
        cached_query=cached_query[:300],
        cached_response=cached_response[:400],
        current_query=current_query[:300],
    )
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 80}},
            timeout=10.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            import json as _j
            return bool(_j.loads(m.group()).get("valid", False))
    except Exception:
        pass
    return False


def decompose_task(message: str, model: str = RERANK_MODEL) -> dict:
    """Decompose a complex multi-part request into ordered subtasks.

    Returns {"parts": [...], "count": N, "context_type": "..."}
    or {"parts": [message], "count": 1} if no decomposition needed.
    """
    log_llm("decompose_task", model=model, message_len=len(message))
    prompt = _DECOMPOSE_PROMPT.format(message=message[:1500])
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 300}},
            timeout=12.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            import json as _j
            result = _j.loads(m.group())
            if result.get("count", 0) > 1 and result.get("parts"):
                return result
    except Exception:
        pass
    return {"parts": [message], "count": 1, "context_type": "other"}


def extract_error_pattern(response_text: str,
                           model: str = RERANK_MODEL) -> dict | None:
    """Parse a Claude response for error patterns and extract a fix hint.

    Returns {"error_signature": "...", "fix_hint": "..."} or None.
    """
    # Fast pre-check — skip LLM call if no error keywords present
    error_keywords = ("Error", "Exception", "Traceback", "FAILED", "error:",
                       "failed:", "Cannot", "ModuleNotFound", "ImportError",
                       "TypeError", "ValueError", "AttributeError")
    if not any(kw in response_text for kw in error_keywords):
        return None

    log_llm("extract_error", model=model)
    prompt = _ERROR_EXTRACT_PROMPT.format(response=response_text[:1500])
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 150}},
            timeout=10.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            import json as _j
            result = _j.loads(m.group())
            if result.get("has_error") and result.get("error_signature"):
                return result
    except Exception:
        pass
    return None


def generate_auto_skill(canonical: str, count: int,
                         model: str = RERANK_MODEL) -> dict | None:
    """Ask local LLM to generate a local skill JSON for a recurring message pattern."""
    log_llm("generate_auto_skill", model=model, canonical=canonical[:80])
    prompt = _AUTO_SKILL_PROMPT.format(canonical=canonical[:400], count=count)
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 500}},
            timeout=20.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            import json as _j
            result = _j.loads(m.group())
            if result.get("can_automate") is False:
                return None
            if result.get("name") and result.get("steps"):
                return result
    except Exception:
        pass
    return None


def ollama_available(model: str = EMBED_MODEL) -> bool:
    """Check whether the required Ollama model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False
