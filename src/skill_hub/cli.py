"""Direct CLI for skill-hub operations — used by hooks to bypass Claude entirely.

Usage from hooks:
    skill-hub-cli save_task "title" "summary" --tags "mcp,dev"
    skill-hub-cli close_task 3
    skill-hub-cli list_tasks open
    skill-hub-cli search_context "my query"
    skill-hub-cli classify "save this to memory and close"
"""

import json
import sys

from .activity_log import log_hook, log_llm, log_event
from .embeddings import (
    embed, compact, ollama_available, optimize_context,
    EMBED_MODEL, RERANK_MODEL,
    conversation_digest, exhaustion_save,
)
from .store import SkillStore


# Cached embedding of task command examples (computed once per process)
_task_command_vector: list[float] | None = None
_task_command_hash: int | None = None  # tracks config changes


def _get_task_examples() -> list[str]:
    """Load task command examples from config (editable at runtime)."""
    from . import config as _cfg
    return _cfg.get("hook_task_command_examples") or []


def _task_similarity(message: str) -> float:
    """
    Fast embedding-based similarity check (~100ms).
    Compares the message against canonical task command phrases from config.
    Recomputes centroid if the config list changes.
    Returns cosine similarity 0.0–1.0.
    """
    import math

    global _task_command_vector, _task_command_hash

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    try:
        examples = _get_task_examples()
        current_hash = hash(tuple(examples))

        if _task_command_vector is None or _task_command_hash != current_hash:
            centroid_text = " | ".join(examples)
            _task_command_vector = embed(centroid_text)
            _task_command_hash = current_hash

        msg_vec = embed(message)
        return cosine(msg_vec, _task_command_vector)
    except Exception:
        return 0.0


def _classify_intent(message: str) -> dict:
    """
    Classify whether a message is a task command using a two-stage filter:

    Stage 1 — fast embedding similarity (~100ms, no LLM):
      - Messages > 400 chars → "none" immediately (long coding questions)
      - Embed message, compare to canonical task command centroid
      - Below threshold → "none" immediately
      - Above threshold → proceed to Stage 2

    Stage 2 — local LLM classification (~2-5s):
      - Precise classification of the intent
      - Returns structured JSON with extracted title/summary/task_id

    Returns {"intent": "save_task|close_task|list_tasks|search_context|none", ...}
    """
    import httpx
    import re
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE, RERANK_MODEL

    # Stage 1a: length guard — long messages are almost never task commands
    max_len = int(_cfg.get("hook_max_message_length") or 400)
    if len(message) > max_len:
        log_hook("classify_skip", reason="length_guard", length=len(message))
        return {"intent": "none"}

    # Stage 1b: semantic prefilter — skip LLM if message is clearly unrelated
    threshold = float(_cfg.get("hook_semantic_threshold") or 0.35)
    sim = _task_similarity(message)
    if sim < threshold:
        log_hook("classify_skip", reason="low_similarity", sim=f"{sim:.3f}", threshold=threshold)
        return {"intent": "none"}

    log_hook("classify_llm", sim=f"{sim:.3f}", threshold=threshold)
    # Stage 2: LLM classification
    prompt = f"""\
You are a strict command classifier. Classify ONLY explicit task management commands.

IMPORTANT: Most messages are normal coding/work requests — classify them as "none".
Only classify as a task command if the user is EXPLICITLY managing their task list.

Intents (classify as these ONLY if the user is clearly doing task management):
- "save_task": user says to save/park/remember the current discussion (e.g. "save to memory", "park this for later")
- "close_task": user says to close/finish/mark-done a task (e.g. "close task 3", "done with this task")
- "list_tasks": user asks to see their task list (e.g. "show my open tasks", "list tasks")
- "search_context": user asks to find past discussions/work (e.g. "what did we discuss about auth?")
- "none": ANYTHING ELSE — coding questions, bug reports, feature requests, explanations, etc.

Examples of "none" (do NOT classify these as task commands):
- "fix the pagination bug" → none (coding request)
- "add a new endpoint" → none (feature request)
- "explain how this works" → none (question)
- "refactor the query executor" → none (coding request)
- "debug the failing test" → none (coding request)

Reply with ONLY a JSON object:
{{"intent": "<intent>", "title": "<extracted title if any>", "summary": "<extracted summary if any>", "task_id": <number or null>}}

User message: {message}"""

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            log_llm("classify", model=RERANK_MODEL, intent=result.get("intent"))
            return result
    except Exception:
        pass
    return {"intent": "none"}


def _execute_intent(intent: dict, original_message: str) -> str | None:
    """Execute a classified intent. Returns a user-facing message, or None if not handled."""
    store = SkillStore()
    action = intent.get("intent", "none")
    log_hook("execute", intent=action)

    if action == "save_task":
        if not ollama_available(EMBED_MODEL):
            return None
        title = intent.get("title") or "Untitled task"
        summary = intent.get("summary") or original_message
        vector = embed(f"{title}: {summary}")
        tid = store.save_task(title=title, summary=summary, vector=vector)
        store.close()
        return f"Task #{tid} saved (open): \"{title}\""

    elif action == "close_task":
        task_id = intent.get("task_id")
        if task_id:
            task = store.get_task(int(task_id))
            if task and task["status"] == "open":
                content = task["summary"]
                if task["context"]:
                    content += f"\n\n{task['context']}"
                digest = compact(content)
                compact_text = json.dumps(digest, indent=2)
                compact_vector = embed(
                    f"{digest.get('title', '')}: {digest.get('summary', '')}"
                )
                store.close_task(int(task_id), compact_text, compact_vector)
                store.close()
                return (
                    f"Task #{task_id} closed and compacted.\n"
                    f"Summary: {digest.get('summary', 'N/A')}"
                )
        # No task_id — try to close the most recent open task
        tasks = store.list_tasks("open")
        if tasks:
            latest = tasks[0]
            content = latest["summary"]
            digest = compact(content)
            compact_text = json.dumps(digest, indent=2)
            compact_vector = embed(
                f"{digest.get('title', '')}: {digest.get('summary', '')}"
            )
            store.close_task(latest["id"], compact_text, compact_vector)
            store.close()
            return (
                f"Task #{latest['id']} (\"{latest['title']}\") closed and compacted.\n"
                f"Summary: {digest.get('summary', 'N/A')}"
            )
        store.close()
        return "No open tasks to close."

    elif action == "list_tasks":
        tasks = store.list_tasks("open")
        store.close()
        if not tasks:
            return "No open tasks."
        lines = [f"  #{r['id']} {r['title']} — {r['summary'][:80]}..." for r in tasks]
        return f"{len(lines)} open tasks:\n" + "\n".join(lines)

    elif action == "search_context":
        query = intent.get("summary") or original_message
        if not ollama_available(EMBED_MODEL):
            store.close()
            return None
        vector = embed(query)
        tasks = store.search_tasks(vector, top_k=3, status="all")
        store.close()
        if not tasks:
            return "No matching past work found."
        lines = [
            f"  #{t['id']} [{t['status']}] {t['title']} (sim={t['similarity']:.2f})"
            for t in tasks
        ]
        return f"Related tasks:\n" + "\n".join(lines)

    store.close()
    return None


_TOKEN_ESTIMATES: dict[str, int] = {
    "save_task": 500,
    "close_task": 800,
    "list_tasks": 300,
    "search_context": 400,
}


def _build_context_injection(message: str, msg_vector: list[float]) -> str | None:
    """
    Strategy #1 (RAG) + #5 (Auto-Memory): Build a systemMessage with relevant
    context from skills, tasks, teachings, and memory files.

    Returns a context string or None if nothing relevant found.
    """
    from . import config as _cfg
    from pathlib import Path
    import re

    max_chars = int(_cfg.get("hook_context_max_chars") or 2000)
    budget = max_chars
    parts: list[str] = []

    try:
        store = SkillStore()

        # Search skills
        skills = store.search(msg_vector, top_k=2, similarity_threshold=0.4)
        for s in skills:
            if budget <= 0:
                break
            desc = s.get("description", "")[:150]
            # Include truncated content for the top hit
            content_preview = ""
            if skills.index(s) == 0 and s.get("content"):
                content_preview = "\n  " + s["content"][:300].replace("\n", "\n  ")
            snippet = f"Skill [{s['id']}]: {desc}{content_preview}"
            parts.append(snippet)
            budget -= len(snippet)

        # Search past tasks
        tasks = store.search_tasks(msg_vector, top_k=2, min_sim=0.4)
        for t in tasks:
            if budget <= 0:
                break
            compact_info = ""
            if t.get("compact"):
                try:
                    digest = json.loads(t["compact"])
                    decisions = ", ".join(digest.get("decisions", [])[:3])
                    if decisions:
                        compact_info = f" | Decisions: {decisions}"
                except Exception:
                    pass
            snippet = (f"Past work #{t['id']} [{t['status']}]: {t['title']} "
                       f"— {t['summary'][:150]}{compact_info}")
            parts.append(snippet)
            budget -= len(snippet)

        # Search teaching rules
        teachings = store.search_teachings(msg_vector, min_sim=0.5)
        for t in teachings[:3]:
            if budget <= 0:
                break
            snippet = f"Suggestion: when \"{t['rule']}\" → use {t['target_id']}"
            parts.append(snippet)
            budget -= len(snippet)

        store.close()

        # Strategy #5: Auto-memory — find relevant memory files
        memory_dir = (Path.home() / ".claude" / "projects" /
                      "-Users-ccancellieri-work-code" / "memory")
        memory_index = memory_dir / "MEMORY.md"
        if memory_index.exists() and budget > 200:
            index_text = memory_index.read_text(encoding="utf-8", errors="replace")
            # Parse entries: - [file.md](file.md) — description
            entries: list[tuple[str, str, str]] = []
            for match in re.finditer(
                r"-\s*\[([^\]]+)\]\(([^)]+)\)\s*—\s*(.+)$",
                index_text, re.MULTILINE
            ):
                name, filepath, description = match.groups()
                entries.append((filepath.strip(), name.strip(), description.strip()))

            if entries:
                # Embed each description and find the most relevant
                scored: list[tuple[float, str, str, str]] = []
                for filepath, name, desc in entries:
                    try:
                        desc_vec = embed(desc[:200])
                        sim = sum(a * b for a, b in zip(msg_vector, desc_vec))
                        import math
                        na = math.sqrt(sum(x * x for x in msg_vector))
                        nb = math.sqrt(sum(x * x for x in desc_vec))
                        sim = sim / (na * nb) if na and nb else 0.0
                        if sim > 0.4:
                            scored.append((sim, filepath, name, desc))
                    except Exception:
                        continue

                scored.sort(key=lambda x: x[0], reverse=True)

                for sim, filepath, name, desc in scored[:2]:
                    if budget <= 0:
                        break
                    mem_file = memory_dir / filepath
                    if mem_file.exists():
                        content = mem_file.read_text(
                            encoding="utf-8", errors="replace"
                        )[:500]
                        snippet = f"Memory [{name}] (sim={sim:.2f}): {content}"
                        parts.append(snippet)
                        budget -= len(snippet)

    except Exception:
        pass

    if not parts:
        return None

    log_hook("context_inject", parts=len(parts), chars=sum(len(p) for p in parts))
    header = ("[Skill Hub — auto-injected context relevant to this message. "
              "Use this as background knowledge, do not repeat it verbatim.]\n\n")
    return header + "\n\n".join(parts)


def _precompact_hint(message: str) -> str | None:
    """
    Strategy #4: Pre-compact long messages.
    If the message is very long, use local LLM to extract key points
    so Claude can focus on what matters.
    """
    from . import config as _cfg

    threshold = int(_cfg.get("hook_precompact_threshold") or 1500)
    if len(message) <= threshold:
        return None

    log_hook("precompact", length=len(message), threshold=threshold)
    try:
        digest = compact(message)
        summary = digest.get("summary", "")
        if summary:
            return (f"[Skill Hub — pre-compacted summary of the long input below]\n"
                    f"Key points: {summary}\n"
                    f"Decisions: {', '.join(digest.get('decisions', []))}\n"
                    f"Open questions: {', '.join(digest.get('open_questions', []))}")
    except Exception:
        pass
    return None


# Session-level state for conversation tracking (per-process)
_session_messages: list[str] = []
_session_id: str | None = None


def _get_session_id() -> str:
    """Get or create a session ID for this process."""
    global _session_id
    if _session_id is None:
        import os
        _session_id = f"pid-{os.getpid()}"
    return _session_id


def _conversation_digest_if_due(message: str) -> str | None:
    """
    Strategy #2: Periodic conversation digest.
    Every N messages, use local LLM to produce a compact state summary.
    Also tracks relevance decay for auto-eviction (#1).

    Returns a systemMessage with the digest, or None.
    """
    from . import config as _cfg

    _session_messages.append(message)
    n = int(_cfg.get("digest_every_n_messages") or 5)

    if len(_session_messages) < n or len(_session_messages) % n != 0:
        return None

    if not ollama_available(RERANK_MODEL):
        return None

    log_hook("digest", message_count=len(_session_messages))
    try:
        digest = conversation_digest(_session_messages)
        store = SkillStore()
        session_id = _get_session_id()

        # Save conversation state
        stale = json.dumps(digest.get("stale_topics", []))
        store.save_conversation_state(
            session_id=session_id,
            message_count=len(_session_messages),
            digest=json.dumps(digest),
            stale_topics=stale,
            suggested_profile=digest.get("suggested_profile"),
        )

        parts: list[str] = []

        # Inject current focus as context hint
        focus = digest.get("current_focus", "")
        if focus and focus != "unknown":
            parts.append(f"Current focus: {focus}")

        decisions = digest.get("recent_decisions", [])
        if decisions:
            parts.append(f"Recent decisions: {', '.join(decisions[:3])}")

        # Strategy #1: Auto-eviction — detect stale topics and suggest profile switch
        if _cfg.get("eviction_enabled"):
            stale_topics = digest.get("stale_topics", [])
            suggested = digest.get("suggested_profile")
            if stale_topics:
                parts.append(f"Stale topics (no longer active): {', '.join(stale_topics)}")
            if suggested:
                parts.append(
                    f"Conversation fits '{suggested}' profile. "
                    f"Switch with: /hub-profile{suggested}"
                )

        store.close()

        if parts:
            header = "[Skill Hub — conversation digest (auto-generated every "
            header += f"{n} messages)]\n"
            return header + "\n".join(parts)

    except Exception:
        pass

    return None


def _exhaustion_auto_save(context: str) -> str:
    """
    Strategy #3: Exhaustion fallback.
    When Claude is exhausted/unavailable, use local LLM to auto-save the session.
    Saves a task with LLM-generated summary and updates memory.
    """
    log_event("SAVE", "exhaustion auto-save triggered")
    result_parts: list[str] = ["=== Exhaustion Auto-Save ===\n"]

    try:
        # Generate structured save via local LLM
        digest = exhaustion_save(context)

        title = digest.get("title", "Session interrupted")
        summary = digest.get("summary", context[:500])
        tags = digest.get("tags", "auto-saved,exhaustion")
        next_steps = digest.get("next_steps", [])
        files = digest.get("files_modified", [])

        # Save as task
        store = SkillStore()
        vector = embed(f"{title}: {summary}") if ollama_available(EMBED_MODEL) else []
        full_context = json.dumps({
            "decisions": digest.get("decisions", []),
            "next_steps": next_steps,
            "files_modified": files,
        }, indent=2)
        tid = store.save_task(
            title=title,
            summary=summary,
            vector=vector,
            context=full_context,
            tags=tags,
            session_id=_get_session_id(),
        )
        store.close()

        result_parts.append(f"Task #{tid} saved: \"{title}\"\n")
        result_parts.append(f"Summary: {summary}\n")

        if next_steps:
            result_parts.append("Next steps when resuming:")
            for step in next_steps:
                result_parts.append(f"  - {step}")

        if files:
            result_parts.append(f"\nFiles modified: {', '.join(files)}")

        result_parts.append(f"\nTo resume later: search_context(\"{title}\")")
        result_parts.append(f"Or: /list-tasks")

        # Also update memory files
        mem_result = _update_memory_on_exhaustion(digest)
        if mem_result:
            result_parts.append(f"\n{mem_result}")

    except Exception as exc:
        # Even if LLM fails, do a raw save
        try:
            store = SkillStore()
            vector = embed(context[:200]) if ollama_available(EMBED_MODEL) else []
            tid = store.save_task(
                title="Session interrupted (raw save)",
                summary=context[:1000],
                vector=vector,
                tags="auto-saved,exhaustion,raw",
            )
            store.close()
            result_parts.append(f"Raw save: task #{tid} (LLM unavailable: {exc})")
        except Exception as inner:
            result_parts.append(f"Failed to save: {inner}")

    return "\n".join(result_parts)


def _cmd_exhaustion_save(args_str: str) -> str:
    """Handle /exhaustion-save command — manually trigger exhaustion save."""
    from . import config as _cfg

    if not _cfg.get("exhaustion_fallback"):
        return "Exhaustion fallback is disabled. Enable with: /hub-configure exhaustion_fallback true"

    # Use session messages as context, or args_str if provided
    if args_str.strip():
        context = args_str.strip()
    elif _session_messages:
        context = "\n---\n".join(_session_messages[-15:])
    else:
        return "No session context to save. Provide a description: /exhaustion-save <context>"

    return _exhaustion_auto_save(context)


def _cmd_conversation_digest() -> str:
    """Handle /digest command — force a conversation digest now."""
    if not _session_messages:
        return "No messages in this session yet."

    if not ollama_available(RERANK_MODEL):
        return "Ollama/reason model not available for digest."

    try:
        digest = conversation_digest(_session_messages)
        lines = ["=== Conversation Digest ===\n"]
        lines.append(f"Messages in session: {len(_session_messages)}")
        lines.append(f"Current focus: {digest.get('current_focus', 'unknown')}")

        decisions = digest.get("recent_decisions", [])
        if decisions:
            lines.append(f"\nRecent decisions:")
            for d in decisions:
                lines.append(f"  - {d}")

        active = digest.get("active_plugins", [])
        if active:
            lines.append(f"\nActive plugins: {', '.join(active)}")

        stale = digest.get("stale_topics", [])
        if stale:
            lines.append(f"\nStale topics: {', '.join(stale)}")

        suggested = digest.get("suggested_profile")
        if suggested:
            lines.append(f"\nSuggested profile: {suggested}")
            lines.append(f"  Activate: /hub-profile{suggested}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Digest failed: {exc}"


_SAVE_MEMORY_PROMPT = """\
You are a memory-management assistant for an AI coding tool. Given the session \
context, generate a memory file that captures the most important non-obvious \
information worth remembering for future sessions.

Focus on:
- Decisions that were made and WHY
- User preferences discovered
- Project-specific knowledge not in the code
- Patterns that should be repeated or avoided

Do NOT save:
- Code patterns (derivable from reading the code)
- Git history (use git log)
- Anything already in existing memory files

Output ONLY a JSON object:
{{
  "filename": "<descriptive-kebab-case.md>",
  "name": "<short title>",
  "description": "<one-line description for the memory index>",
  "type": "<user|feedback|project|reference>",
  "content": "<the memory content, 2-6 sentences>"
}}

Session context:
{content}

Existing memory files (avoid duplicates):
{existing}
"""


def _cmd_optimize_context() -> str:
    """Handle /optimize-context — local LLM analyzes memory and recommends pruning."""
    from pathlib import Path
    import re

    if not ollama_available(RERANK_MODEL):
        return "Ollama/reason model not available."

    memory_dir = (Path.home() / ".claude" / "projects" /
                  "-Users-ccancellieri-work-code" / "memory")
    memory_index = memory_dir / "MEMORY.md"

    if not memory_index.exists():
        return "No MEMORY.md found."

    index_text = memory_index.read_text(encoding="utf-8", errors="replace")

    # Parse memory entries
    entries: list[dict] = []
    for match in re.finditer(
        r"-\s*\[([^\]]+)\]\(([^)]+)\)\s*—\s*(.+)$",
        index_text, re.MULTILINE
    ):
        name, filepath, description = match.groups()
        mem_file = memory_dir / filepath.strip()
        if mem_file.exists():
            content = mem_file.read_text(encoding="utf-8", errors="replace")
            tokens_est = len(content) // 4
            # Detect category from section headers
            category = "unknown"
            for line in index_text.split("\n"):
                if line.startswith("## "):
                    category = line[3:].strip()
                if filepath.strip() in line:
                    break
            entries.append({
                "file": filepath.strip(),
                "category": category,
                "tokens": tokens_est,
                "content": content,
            })

    if not entries:
        return "No memory files found to analyze."

    total_tokens = sum(e["tokens"] for e in entries)
    lines = [f"=== Context Optimization ===\n"]
    lines.append(f"Analyzing {len(entries)} memory files (~{total_tokens:,} tokens total)...\n")

    # Call local LLM to analyze
    recommendations = optimize_context(entries)

    prune_count = 0
    compact_count = 0
    merge_count = 0
    tokens_saveable = 0

    for rec in recommendations:
        if rec.get("summary"):
            # Summary line
            lines.append(f"\n--- Summary ---")
            lines.append(f"Total files: {rec.get('total', len(entries))}")
            lines.append(f"Keep: {rec.get('keep', 0)}, Prune: {rec.get('prune', 0)}, "
                        f"Compact: {rec.get('compact', 0)}, Merge: {rec.get('merge', 0)}")
            lines.append(f"Est. tokens saveable: ~{rec.get('est_tokens_saved', 0):,}")
            continue

        action = rec.get("action", "KEEP")
        filename = rec.get("file", "?")
        reason = rec.get("reason", "")

        if action == "PRUNE":
            prune_count += 1
            entry = next((e for e in entries if e["file"] == filename), None)
            if entry:
                tokens_saveable += entry["tokens"]
            lines.append(f"  PRUNE  {filename} — {reason}")
        elif action == "COMPACT":
            compact_count += 1
            compacted = rec.get("compacted", "")
            entry = next((e for e in entries if e["file"] == filename), None)
            if entry and compacted:
                saved = entry["tokens"] - len(compacted) // 4
                tokens_saveable += max(0, saved)
            lines.append(f"  COMPACT {filename} — {reason}")
            if compacted:
                lines.append(f"          → {compacted[:150]}...")
        elif action == "MERGE":
            merge_count += 1
            lines.append(f"  MERGE  {filename} — {reason}")
        else:
            lines.append(f"  KEEP   {filename}")

    if prune_count or compact_count or merge_count:
        lines.append(f"\nActions available:")
        if prune_count:
            lines.append(f"  {prune_count} files to prune (~{tokens_saveable:,} tokens saved per session)")
        if compact_count:
            lines.append(f"  {compact_count} files to compact")
        if merge_count:
            lines.append(f"  {merge_count} files to merge")
        lines.append(f"\nTo apply: review recommendations, then manually edit/delete memory files.")
    else:
        lines.append(f"\nAll memory files look good — no changes recommended.")

    return "\n".join(lines)


def _cmd_save_memory(args_str: str) -> str:
    """Handle /save-memory — local LLM generates a memory entry from session context."""
    from pathlib import Path
    import re

    if not ollama_available(RERANK_MODEL):
        return "Ollama/reason model not available."

    memory_dir = (Path.home() / ".claude" / "projects" /
                  "-Users-ccancellieri-work-code" / "memory")
    memory_index = memory_dir / "MEMORY.md"

    # Gather context
    if args_str.strip():
        context = args_str.strip()
    elif _session_messages:
        context = "\n---\n".join(_session_messages[-10:])
    else:
        return "No session context. Provide a description: /save-memory <what to remember>"

    # Read existing memory index to avoid duplicates
    existing = ""
    if memory_index.exists():
        existing = memory_index.read_text(encoding="utf-8", errors="replace")

    # Ask local LLM to generate memory entry
    import httpx
    from .embeddings import OLLAMA_BASE

    prompt = _SAVE_MEMORY_PROMPT.format(
        content=context[:4000],
        existing=existing[:2000],
    )

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return "LLM did not produce valid JSON. Try with a more specific description."
        entry = json.loads(json_match.group())
    except Exception as exc:
        return f"LLM failed: {exc}"

    filename = entry.get("filename", "auto_memory.md")
    name = entry.get("name", "Auto-generated memory")
    description = entry.get("description", "")
    mem_type = entry.get("type", "project")
    content = entry.get("content", "")

    if not content:
        return "LLM generated empty content. Try with more context."

    # Write memory file
    mem_file = memory_dir / filename
    mem_content = f"""---
name: {name}
description: {description}
type: {mem_type}
---

{content}
"""
    mem_file.write_text(mem_content, encoding="utf-8")

    # Append to MEMORY.md index
    index_line = f"- [{filename}]({filename}) — {description}"
    if memory_index.exists():
        current = memory_index.read_text(encoding="utf-8", errors="replace")
        if filename not in current:
            with memory_index.open("a", encoding="utf-8") as f:
                f.write(f"\n{index_line}\n")

    return (f"Memory saved: {mem_file}\n"
            f"  Name: {name}\n"
            f"  Type: {mem_type}\n"
            f"  Description: {description}\n"
            f"  Content: {content[:200]}...")


def _update_memory_on_exhaustion(digest: dict) -> str | None:
    """
    When exhaustion-save fires, also write a memory entry pointing to the saved task.
    Returns status message or None on failure.
    """
    from pathlib import Path

    memory_dir = (Path.home() / ".claude" / "projects" /
                  "-Users-ccancellieri-work-code" / "memory")
    memory_index = memory_dir / "MEMORY.md"

    if not memory_dir.exists():
        return None

    title = digest.get("title", "Session interrupted")
    summary = digest.get("summary", "")
    tags = digest.get("tags", "")
    next_steps = digest.get("next_steps", [])

    # Generate a filename from the title
    import re
    safe_name = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    filename = f"project_{safe_name}.md"

    mem_content = f"""---
name: {title}
description: {summary[:100]}
type: project
---

{summary}

**Next steps:** {'; '.join(next_steps) if next_steps else 'Resume from saved task.'}
**Tags:** {tags}
**Saved by:** exhaustion-fallback (local LLM)
"""

    mem_file = memory_dir / filename
    mem_file.write_text(mem_content, encoding="utf-8")

    # Append to MEMORY.md
    index_line = f"- [{filename}]({filename}) — {summary[:80]}"
    if memory_index.exists():
        current = memory_index.read_text(encoding="utf-8", errors="replace")
        if filename not in current:
            with memory_index.open("a", encoding="utf-8") as f:
                f.write(f"\n{index_line}\n")

    return f"Memory updated: {filename}"


def hook_classify_and_execute(message: str) -> dict:
    """
    Main entry point for the UserPromptSubmit hook.

    Pipeline:
    1. Slash commands → instant local handling (0ms)
    2. Task commands → semantic prefilter + LLM classify → local execution
    3. Normal messages → context injection + pre-compact → allow to Claude
       with enriched systemMessage

    Returns:
      - {"decision": "block", "message": "..."} if handled locally
      - {"decision": "allow"} if Claude should handle it
      - {"decision": "allow", "systemMessage": "..."} if injecting context
    """
    from . import config as _cfg

    # Fast path: known slash commands — no LLM, no embedding, instant
    slash_result = _handle_slash_command(message)
    if slash_result is not None:
        if _cfg.get("token_profiling") and slash_result.get("decision") == "block":
            try:
                store = SkillStore()
                store.log_interception(
                    command_type="slash:" + message.strip().split()[0],
                    message_preview=message,
                    estimated_tokens=300,
                )
                store.close()
            except Exception:
                pass
        return slash_result

    # Semantic path: embedding prefilter → LLM classify
    intent = _classify_intent(message)
    action = intent.get("intent", "none")

    if action != "none":
        result = _execute_intent(intent, message)
        if result:
            if _cfg.get("token_profiling"):
                try:
                    store = SkillStore()
                    store.log_interception(
                        command_type=action,
                        message_preview=message,
                        estimated_tokens=_TOKEN_ESTIMATES.get(action, 400),
                    )
                    store.close()
                except Exception:
                    pass
            return {"decision": "block", "message": result}

    # ── Context enrichment for messages going to Claude ──
    if not _cfg.get("hook_context_injection"):
        return {"decision": "allow"}

    system_parts: list[str] = []

    # Strategy #4: Pre-compact long input
    precompact = _precompact_hint(message)
    if precompact:
        system_parts.append(precompact)

    # Strategy #1 + #5: RAG context injection + memory-aware context
    context: str | None = None
    try:
        msg_vector = embed(message[:500])  # truncate for embedding
        context = _build_context_injection(message, msg_vector)
        if context:
            system_parts.append(context)
    except Exception:
        pass

    # Strategy #2: Periodic conversation digest + relevance decay
    digest_msg = _conversation_digest_if_due(message)
    if digest_msg:
        system_parts.append(digest_msg)

    if system_parts:
        combined = "\n\n".join(system_parts)
        # Log context injection stats
        if _cfg.get("token_profiling"):
            try:
                store = SkillStore()
                store.log_context_injection(
                    message_preview=message,
                    skills=1 if context and "Skill [" in context else 0,
                    tasks=1 if context and "Past work #" in context else 0,
                    teachings=1 if context and "Suggestion:" in context else 0,
                    memory=1 if context and "Memory [" in context else 0,
                    precompacted=bool(precompact),
                    chars=len(combined),
                )
                store.close()
            except Exception:
                pass
        return {"decision": "allow", "systemMessage": combined}

    return {"decision": "allow"}


def _cmd_status() -> str:
    """Full health report — runs locally, no Claude tokens."""
    import httpx
    from . import config as _cfg

    cfg = _cfg.load_config()
    ollama_base = cfg.get("ollama_base", "http://localhost:11434")
    embed_model = cfg.get("embed_model", "nomic-embed-text")
    reason_model = cfg.get("reason_model", "deepseek-r1:1.5b")

    lines = ["=== Skill Hub Status ===\n"]
    lines.append("MCP server:      (check with: skill-hub-cli status)")

    # Ollama
    try:
        resp = httpx.get(f"{ollama_base}/api/tags", timeout=5.0)
        available = [m["name"] for m in resp.json().get("models", [])]
        lines.append(f"Ollama:          ✓ reachable at {ollama_base}")
        embed_ok = any(embed_model in m for m in available)
        lines.append(f"Embed model:     {'✓' if embed_ok else '✗'} {embed_model}"
                     + ("" if embed_ok else f"  ← ollama pull {embed_model}"))
        reason_ok = any(reason_model in m for m in available)
        lines.append(f"Reason model:    {'✓' if reason_ok else '✗'} {reason_model}"
                     + ("" if reason_ok else f"  ← ollama pull {reason_model}"))
        if available:
            lines.append(f"All models:      {', '.join(available)}")
    except Exception as exc:
        lines.append(f"Ollama:          ✗ NOT reachable ({exc})")

    # Hook
    from pathlib import Path
    settings_path = Path.home() / ".claude" / "settings.json"
    hook_configured = False
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            for group in settings.get("hooks", {}).get("UserPromptSubmit", []):
                for h in group.get("hooks", []):
                    if "intercept-task-commands" in h.get("command", ""):
                        hook_configured = True
        except Exception:
            pass
    hook_enabled = cfg.get("hook_enabled", True)
    lines.append(f"Hook:            {'✓ configured' if hook_configured else '✗ NOT configured'}"
                 + (f" {'and enabled' if hook_enabled else '(disabled)'}" if hook_configured else ""))

    # Profiling
    lines.append(f"Token profiling: {'✓ on' if cfg.get('token_profiling', True) else '○ off'}")

    # DB
    try:
        store = SkillStore()
        skills = store.list_skills()
        tasks_all = store.list_tasks("all")
        open_t = sum(1 for r in tasks_all if r["status"] == "open")
        totals = store.get_interception_totals()
        saved = totals["total_tokens_saved"] or 0 if totals else 0
        intercepted = totals["total_interceptions"] or 0 if totals else 0
        lines.append(f"\nDatabase:")
        lines.append(f"  Skills indexed:    {len(skills)}")
        lines.append(f"  Tasks:             {len(tasks_all)} ({open_t} open)")
        lines.append(f"  Intercepted cmds:  {intercepted} (~{saved:,} tokens saved)")
        store.close()
    except Exception as exc:
        lines.append(f"\nDatabase: error — {exc}")

    # Config
    lines.append(f"\nConfig ({_cfg.CONFIG_PATH}):")
    lines.append(f"  embed_model={embed_model}, reason_model={reason_model}")
    lines.append(f"  threshold={cfg.get('hook_semantic_threshold')}, "
                 f"max_len={cfg.get('hook_max_message_length')}, "
                 f"examples={len(cfg.get('hook_task_command_examples', []))}")

    return "\n".join(lines)


def _cmd_token_stats() -> str:
    """Token savings report — runs locally."""
    store = SkillStore()
    totals = store.get_interception_totals()
    ctx_stats = store.get_context_injection_stats()
    store.close()

    lines: list[str] = []

    # Interception stats
    if totals and totals["total_interceptions"]:
        total_saved = totals["total_tokens_saved"] or 0
        total_count = totals["total_interceptions"] or 0
        lines.append("=== Token Savings (blocked commands) ===\n")
        lines.append(f"Total intercepted: {total_count}")
        lines.append(f"Tokens saved:      ~{total_saved:,}\n")

        stats = store.get_interception_stats()
        for row in stats:
            avg = (row["total_tokens_saved"] or 0) // max(row["intercept_count"], 1)
            lines.append(f"  {row['command_type']:<20} {row['intercept_count']:>4}x  ~{row['total_tokens_saved'] or 0:,} tokens")
    else:
        lines.append("No interceptions recorded yet.\n")

    # Context injection stats
    if ctx_stats and ctx_stats.get("total", 0) > 0:
        total = ctx_stats["total"]
        lines.append(f"\n=== Context Injection (enriched messages) ===\n")
        lines.append(f"Total messages enriched: {total}")
        lines.append(f"Avg context injected:   ~{int(ctx_stats.get('avg_chars', 0)) // 4} tokens "
                      f"({int(ctx_stats.get('avg_chars', 0))} chars)\n")
        lines.append("Context sources used:")

        def _pct(n: int) -> str:
            return f"{n / total * 100:.0f}%" if total else "0%"

        w_skills = ctx_stats.get("with_skills", 0) or 0
        w_tasks = ctx_stats.get("with_tasks", 0) or 0
        w_teach = ctx_stats.get("with_teachings", 0) or 0
        w_mem = ctx_stats.get("with_memory", 0) or 0
        w_pre = ctx_stats.get("precompacted", 0) or 0

        lines.append(f"  Skills matched:    {w_skills:>4}x  ({_pct(w_skills)} of messages)")
        lines.append(f"  Tasks matched:     {w_tasks:>4}x  ({_pct(w_tasks)} of messages)")
        lines.append(f"  Teachings matched: {w_teach:>4}x  ({_pct(w_teach)} of messages)")
        lines.append(f"  Memory loaded:     {w_mem:>4}x  ({_pct(w_mem)} of messages)")
        lines.append(f"  Pre-compacted:     {w_pre:>4}x  ({_pct(w_pre)} of messages)")
        lines.append(f"\n  Total context:     ~{(ctx_stats.get('total_chars', 0) or 0) // 4:,} tokens injected")
    else:
        lines.append("\nNo context injections recorded yet.")

    return "\n".join(lines)


def _cmd_help() -> str:
    """Quick reference — runs locally."""
    return """=== Skill Hub Commands ===

All commands start with /hub-* and are intercepted locally (0 Claude tokens).

Status & Config:
  /hub-status              Health check, models, DB stats
  /hub-help                This reference
  /hub-token-stats         Token savings report
  /hub-configure [K] [V]   View/set config
  /hub-list-models         Installed Ollama models
  /hub-pull-model <name>   Download an Ollama model

Skills & Plugins:
  /hub-search-skills Q     Semantic skill search
  /hub-search-context Q    Unified search: skills + tasks + teachings
  /hub-suggest-plugins Q   Suggest plugins for current task
  /hub-list-skills         All indexed skills
  /hub-index-skills        Rebuild skill index
  /hub-index-plugins       Rebuild plugin index
  /hub-toggle-plugin N on  Enable/disable a plugin (via Claude)

Tasks:
  /hub-list-tasks [status] Open/closed/all tasks
  /hub-save-task [title]   Save current work (needs LLM)
  /hub-close-task [N]      Close + compact task (needs LLM)
  /hub-update-task N       Update task (via Claude)
  /hub-reopen-task N       Reopen a closed task

Learning:
  /hub-teach rule -> target  Add teaching rule
  /hub-list-teachings        Show all teaching rules
  /hub-forget-teaching N     Remove a teaching rule

Profiles:
  /hub-profile               List session profiles
  /hub-profile <name>        Activate a profile
  /hub-profile save <name>   Save current state as profile
  /hub-profile delete <name> Remove a profile
  /hub-profile auto <task>   LLM recommends best profile

Context & Memory:
  /hub-digest              Force conversation digest now
  /hub-optimize-context    LLM analyzes memory, recommends pruning
  /hub-save-memory [desc]  LLM generates memory entry from session
  /hub-exhaustion-save     Auto-save session when Claude is exhausted"""


def _cmd_list_models() -> str:
    """List Ollama models — runs locally."""
    import httpx
    from . import config as _cfg

    cfg = _cfg.load_config()
    ollama_base = cfg.get("ollama_base", "http://localhost:11434")
    current_embed = cfg.get("embed_model", "nomic-embed-text")
    current_reason = cfg.get("reason_model", "deepseek-r1:1.5b")

    lines = [f"=== Ollama Models ===\n"]
    lines.append(f"Active embed_model:  {current_embed}")
    lines.append(f"Active reason_model: {current_reason}\n")

    try:
        resp = httpx.get(f"{ollama_base}/api/tags", timeout=5.0)
        installed = resp.json().get("models", [])
        if installed:
            lines.append("Installed:")
            for m in installed:
                name = m["name"]
                size_gb = m.get("size", 0) / 1_073_741_824
                role = ""
                if current_embed in name:
                    role = "  ← embed_model"
                elif current_reason in name:
                    role = "  ← reason_model"
                lines.append(f"  {name:<35} {size_gb:.1f} GB{role}")
        else:
            lines.append("No models installed.")
    except Exception as exc:
        lines.append(f"Ollama not reachable: {exc}")

    return "\n".join(lines)


def _cmd_configure(args: list[str]) -> str:
    """View/set config — runs locally."""
    from . import config as _cfg

    if not args:
        current = _cfg.load_config()
        lines = [f"=== Config ({_cfg.CONFIG_PATH}) ===\n"]
        for k, v in sorted(current.items()):
            val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            if len(val) > 80:
                val = val[:77] + "..."
            lines.append(f"  {k}: {val}")
        return "\n".join(lines)

    if len(args) == 1:
        val = _cfg.get(args[0])
        return f"{args[0]} = {val}"

    key, value = args[0], args[1]
    parsed: str | int | float | bool = value
    if value.lower() in ("true", "false"):
        parsed = value.lower() == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                pass

    current = _cfg.load_config()
    current[key] = parsed
    _cfg.save_config(current)
    return f"Updated: {key} = {parsed}"


def _cmd_list_teachings() -> str:
    """List teaching rules — runs locally."""
    store = SkillStore()
    rows = store.list_teachings()
    store.close()
    if not rows:
        return "No teaching rules. Use teach(rule, suggest) to add one."
    lines = [f"=== Teaching Rules ===\n"]
    for r in rows:
        lines.append(f"  #{r['id']} when \"{r['rule']}\" → suggest {r['target_id']}")
    return "\n".join(lines)


def _cmd_list_skills() -> str:
    """List indexed skills — runs locally."""
    store = SkillStore()
    rows = store.list_skills()
    store.close()
    if not rows:
        return "No skills indexed. Run index_skills() first."
    lines = [f"{len(rows)} skills:\n"]
    for r in rows:
        lines.append(f"  {r['id']}: {r['description'] or '(no description)'}"[:120])
    return "\n".join(lines)


def _cmd_profile(args_str: str) -> str:
    """
    Session profiles — switch plugin sets per work context.

    /profile              — list available profiles
    /profile <name>       — activate a profile (toggle plugins, user must restart)
    /profile save <name>  — save current plugin state as a new profile
    /profile delete <name>— remove a profile
    /profile auto <task>  — LLM suggests the best profile for a task description
    """
    from . import config as _cfg
    from pathlib import Path

    SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
    args = args_str.strip().split(None, 1)
    subcmd = args[0].lower() if args else ""
    rest = args[1].strip() if len(args) > 1 else ""

    cfg = _cfg.load_config()
    profiles: dict = cfg.get("profiles", {})

    # /profile — list profiles
    if not subcmd:
        if not profiles:
            return "No profiles defined. Use /hub-profile save <name> to create one."
        lines = ["=== Session Profiles ===\n"]
        for name, prof in sorted(profiles.items()):
            desc = prof.get("description", "")
            plugins = prof.get("plugins", [])
            if plugins == "__all__":
                count = "all"
            else:
                count = str(len(plugins))
            lines.append(f"  {name:<15} {count:>3} plugins — {desc}")
        lines.append(f"\nUsage: /hub-profile <name> to activate (requires restart)")
        lines.append(f"       /hub-profile save <name> to save current state")
        lines.append(f"       /hub-profile auto <task description>")
        return "\n".join(lines)

    # /profile save <name> [description]
    if subcmd == "save":
        parts = rest.split(None, 1)
        name = parts[0] if parts else ""
        desc = parts[1] if len(parts) > 1 else ""
        if not name:
            return "Usage: /hub-profile save <name> [description]"

        if not SETTINGS_PATH.exists():
            return "Settings file not found."
        settings = json.loads(SETTINGS_PATH.read_text())
        enabled_plugins = settings.get("enabledPlugins", {})

        # Capture currently enabled plugins (short names)
        active = [k.split("@")[0] for k, v in enabled_plugins.items() if v]

        profiles[name] = {
            "description": desc or f"Saved from current session ({len(active)} plugins)",
            "plugins": active,
        }
        cfg["profiles"] = profiles
        _cfg.save_config(cfg)

        return (f"Profile '{name}' saved with {len(active)} plugins:\n"
                f"  {', '.join(active)}")

    # /profile delete <name>
    if subcmd == "delete":
        if not rest:
            return "Usage: /hub-profile delete <name>"
        if rest not in profiles:
            return f"Profile '{rest}' not found."
        del profiles[rest]
        cfg["profiles"] = profiles
        _cfg.save_config(cfg)
        return f"Profile '{rest}' deleted."

    # /profile auto <task description>
    if subcmd == "auto":
        if not rest:
            return "Usage: /hub-profile auto <describe your task>"
        # Use embedding similarity to find the best profile
        try:
            task_vec = embed(rest)
        except Exception:
            return "Ollama not available for auto-profile."

        import math

        best_name = ""
        best_sim = 0.0
        for name, prof in profiles.items():
            desc = prof.get("description", "")
            plugins = prof.get("plugins", [])
            text = f"{name}: {desc}. Plugins: {', '.join(plugins) if isinstance(plugins, list) else 'all'}"
            try:
                prof_vec = embed(text)
                dot = sum(a * b for a, b in zip(task_vec, prof_vec))
                na = math.sqrt(sum(x * x for x in task_vec))
                nb = math.sqrt(sum(x * x for x in prof_vec))
                sim = dot / (na * nb) if na and nb else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            except Exception:
                continue

        if best_name:
            prof = profiles[best_name]
            plugins = prof.get("plugins", [])
            count = "all" if plugins == "__all__" else str(len(plugins))
            return (f"Recommended profile: {best_name} (sim={best_sim:.2f})\n"
                    f"  {prof.get('description', '')}\n"
                    f"  {count} plugins: {', '.join(plugins) if isinstance(plugins, list) else 'all enabled'}\n\n"
                    f"To activate: /hub-profile {best_name}")
        return "No matching profile found. Create one with /hub-profile save <name>."

    # /profile <name> — activate
    name = subcmd
    if name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        return f"Profile '{name}' not found. Available: {available}"

    prof = profiles[name]
    target_plugins = prof.get("plugins", [])

    if not SETTINGS_PATH.exists():
        return "Settings file not found."

    settings = json.loads(SETTINGS_PATH.read_text())
    enabled_plugins = settings.get("enabledPlugins", {})

    if target_plugins == "__all__":
        # Enable everything
        changes = []
        for key in enabled_plugins:
            if not enabled_plugins[key]:
                enabled_plugins[key] = True
                changes.append(f"  ✓ enabled {key.split('@')[0]}")
    else:
        # Enable target plugins, disable everything else
        changes = []
        for key in enabled_plugins:
            short = key.split("@")[0]
            should_enable = short in target_plugins
            was_enabled = enabled_plugins[key]
            if should_enable != was_enabled:
                enabled_plugins[key] = should_enable
                symbol = "✓" if should_enable else "✗"
                action = "enabled" if should_enable else "disabled"
                changes.append(f"  {symbol} {action} {short}")

    settings["enabledPlugins"] = enabled_plugins
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))

    lines = [f"=== Profile '{name}' activated ===\n"]
    lines.append(prof.get("description", ""))
    if changes:
        lines.append(f"\nChanges ({len(changes)}):")
        lines.extend(changes)
    else:
        lines.append("\nNo changes needed — already matching.")
    lines.append(f"\n⚠  Restart Claude Code for changes to take effect.")

    return "\n".join(lines)


# Map of slash command prefixes → local handlers
_SLASH_COMMANDS: dict[str, str] = {
    "/hub-status": "status",
    "/hub-help": "help",
    "/hub-manual": "help",
    "/hub-token-stats": "token_stats",
    "/hub-session-stats": "token_stats",
    "/hub-list-tasks": "list_tasks",
    "/hub-list-models": "list_models",
    "/hub-list-skills": "list_skills",
    "/hub-list-teachings": "list_teachings",
    "/hub-configure": "configure",
    "/hub-save-task": "save_task",
    "/hub-close-task": "close_task",
    "/hub-search-context": "search_context",
    "/hub-search-skills": "search_skills",
    "/hub-index-skills": "index_skills",
    "/hub-index-plugins": "index_plugins",
    "/hub-profile": "profile",
    "/hub-exhaustion-save": "exhaustion_save",
    "/hub-digest": "digest",
    "/hub-optimize-context": "optimize_context",
    "/hub-save-memory": "save_memory",
    "/hub-suggest-plugins": "suggest_plugins",
    "/hub-teach": "teach",
    "/hub-forget-teaching": "forget_teaching",
    "/hub-toggle-plugin": "toggle_plugin",
    "/hub-pull-model": "pull_model",
    "/hub-update-task": "update_task",
    "/hub-reopen-task": "reopen_task",
}


def _handle_slash_command(message: str) -> dict | None:
    """
    Fast-path for /slash-commands. Returns a hook response dict or None if
    the message is not a known slash command.
    """
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    args_str = parts[1] if len(parts) > 1 else ""

    if cmd not in _SLASH_COMMANDS:
        return None

    action = _SLASH_COMMANDS[cmd]

    try:
        if action == "status":
            return {"decision": "block", "message": _cmd_status()}
        elif action == "help":
            return {"decision": "block", "message": _cmd_help()}
        elif action == "token_stats":
            return {"decision": "block", "message": _cmd_token_stats()}
        elif action == "list_models":
            return {"decision": "block", "message": _cmd_list_models()}
        elif action == "list_skills":
            return {"decision": "block", "message": _cmd_list_skills()}
        elif action == "list_teachings":
            return {"decision": "block", "message": _cmd_list_teachings()}
        elif action == "configure":
            return {"decision": "block", "message": _cmd_configure(args_str.split() if args_str else [])}
        elif action == "list_tasks":
            status = args_str.strip() or "open"
            store = SkillStore()
            tasks = store.list_tasks(status)
            store.close()
            if not tasks:
                return {"decision": "block", "message": f"No {status} tasks."}
            lines = [f"  #{r['id']} [{r['status']}] {r['title']} — {r['summary'][:80]}" for r in tasks]
            return {"decision": "block", "message": f"{len(lines)} {status} tasks:\n" + "\n".join(lines)}
        elif action == "save_task":
            # Needs LLM to extract title — fall through to classify
            return None
        elif action == "close_task":
            return None  # needs LLM
        elif action == "search_context":
            if not args_str.strip():
                return {"decision": "block", "message": "Usage: /hub-search-context <query>"}
            return None  # needs embedding + LLM
        elif action == "search_skills":
            if not args_str.strip():
                return {"decision": "block", "message": "Usage: /hub-search-skills <query>"}
            try:
                vec = embed(args_str.strip()[:500])
                store = SkillStore()
                results = store.search(vec, top_k=5, similarity_threshold=0.3)
                store.close()
                if not results:
                    return {"decision": "block", "message": "No matching skills."}
                lines = [f"  {r['id']}: {(r.get('description') or '')[:100]} (sim={r.get('similarity', 0):.2f})"
                         for r in results]
                return {"decision": "block", "message": f"{len(lines)} skills:\n" + "\n".join(lines)}
            except Exception as exc:
                return {"decision": "block", "message": f"Search failed: {exc}"}
        elif action == "suggest_plugins":
            if not args_str.strip():
                return {"decision": "block", "message": "Usage: /hub-suggest-plugins <query>"}
            try:
                vec = embed(args_str.strip()[:500])
                store = SkillStore()
                results = store.suggest_plugins(vec)
                store.close()
                if not results:
                    return {"decision": "block", "message": "No matching plugins."}
                lines = [f"  {r['short_name']}: {(r.get('description') or '')[:80]} "
                         f"(score={r.get('total_score', 0):.2f})" for r in results[:5]]
                return {"decision": "block", "message": f"Suggested plugins:\n" + "\n".join(lines)}
            except Exception as exc:
                return {"decision": "block", "message": f"Suggest failed: {exc}"}
        elif action == "teach":
            if not args_str.strip():
                return {"decision": "block",
                        "message": "Usage: /hub-teach <rule> -> <plugin_or_skill>\n"
                                   "Example: /hub-teach when debugging CSS -> chrome-devtools-mcp"}
            parts_t = args_str.split("->", 1)
            if len(parts_t) < 2:
                return {"decision": "block", "message": "Use '->' to separate rule from target.\n"
                        "Example: /hub-teach when I give a URL -> chrome-devtools-mcp"}
            rule = parts_t[0].strip()
            target = parts_t[1].strip()
            try:
                vec = embed(rule)
                store = SkillStore()
                tid = store.add_teaching(rule, vec, "suggest", "plugin", target)
                store.close()
                return {"decision": "block", "message": f"Teaching #{tid}: when \"{rule}\" -> suggest {target}"}
            except Exception as exc:
                return {"decision": "block", "message": f"Teach failed: {exc}"}
        elif action == "forget_teaching":
            if not args_str.strip():
                return {"decision": "block", "message": "Usage: /hub-forget-teaching <id>"}
            try:
                store = SkillStore()
                ok = store.remove_teaching(int(args_str.strip()))
                store.close()
                return {"decision": "block",
                        "message": f"Teaching removed." if ok else "Teaching not found."}
            except Exception as exc:
                return {"decision": "block", "message": f"Error: {exc}"}
        elif action == "toggle_plugin":
            # Complex — needs settings.json manipulation, let Claude handle
            return None
        elif action == "pull_model":
            if not args_str.strip():
                return {"decision": "block", "message": "Usage: /hub-pull-model <model_name>"}
            import subprocess
            model = args_str.strip()
            if "/" in model or ";" in model or "|" in model or "&" in model:
                return {"decision": "block", "message": f"Invalid model name: {model}"}
            try:
                result = subprocess.run(
                    ["ollama", "pull", model], capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    return {"decision": "block",
                            "message": f"Pulled {model}.\n"
                                       f"To activate: /hub-configure reason_model {model}"}
                return {"decision": "block", "message": f"Pull failed: {result.stderr.strip()}"}
            except Exception as exc:
                return {"decision": "block", "message": f"Pull failed: {exc}"}
        elif action == "update_task":
            return None  # needs LLM
        elif action == "reopen_task":
            if not args_str.strip():
                return {"decision": "block", "message": "Usage: /hub-reopen-task <task_id>"}
            try:
                store = SkillStore()
                ok = store.reopen_task(int(args_str.strip()))
                store.close()
                return {"decision": "block",
                        "message": f"Task reopened." if ok else "Task not found."}
            except Exception as exc:
                return {"decision": "block", "message": f"Error: {exc}"}
        elif action == "index_skills":
            from .indexer import index_all
            store = SkillStore()
            count, errors = index_all(store)
            store.close()
            msg = f"Indexed {count} skills."
            if errors:
                msg += f"\nErrors: " + "; ".join(errors[:5])
            return {"decision": "block", "message": msg}
        elif action == "index_plugins":
            # Plugin indexing is heavier — let Claude handle via MCP
            return None
        elif action == "profile":
            return {"decision": "block", "message": _cmd_profile(args_str)}
        elif action == "exhaustion_save":
            return {"decision": "block", "message": _cmd_exhaustion_save(args_str)}
        elif action == "digest":
            return {"decision": "block", "message": _cmd_conversation_digest()}
        elif action == "optimize_context":
            return {"decision": "block", "message": _cmd_optimize_context()}
        elif action == "save_memory":
            return {"decision": "block", "message": _cmd_save_memory(args_str)}
    except Exception as exc:
        return {"decision": "block", "message": f"Error: {exc}"}

    return None


def main() -> None:
    """CLI entry point for direct invocation."""
    if len(sys.argv) < 2:
        print("Usage: skill-hub-cli <command> [args...]")
        print("Commands: classify, status, help, token_stats, list_models,")
        print("          list_skills, list_teachings, configure, profile,")
        print("          save_task, close_task, list_tasks, search_context,")
        print("          exhaustion_save, digest, optimize_context, save_memory")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "classify":
        message = " ".join(args) if args else sys.stdin.read()
        result = hook_classify_and_execute(message.strip())
        print(json.dumps(result))

    elif cmd == "status":
        print(_cmd_status())

    elif cmd == "help":
        print(_cmd_help())

    elif cmd == "token_stats":
        print(_cmd_token_stats())

    elif cmd == "list_models":
        print(_cmd_list_models())

    elif cmd == "list_skills":
        print(_cmd_list_skills())

    elif cmd == "list_teachings":
        print(_cmd_list_teachings())

    elif cmd == "configure":
        print(_cmd_configure(args))

    elif cmd == "save_task":
        title = args[0] if args else "Untitled"
        summary = args[1] if len(args) > 1 else ""
        tags = args[2] if len(args) > 2 else ""
        store = SkillStore()
        vector = embed(f"{title}: {summary}") if ollama_available() else []
        tid = store.save_task(title=title, summary=summary, vector=vector, tags=tags)
        store.close()
        print(f"Task #{tid} saved: \"{title}\"")

    elif cmd == "close_task":
        task_id = int(args[0]) if args else 0
        if not task_id:
            print("Usage: skill-hub-cli close_task <task_id>")
            sys.exit(1)
        result = _execute_intent({"intent": "close_task", "task_id": task_id}, "")
        print(result or "Failed.")

    elif cmd == "list_tasks":
        status = args[0] if args else "open"
        store = SkillStore()
        tasks = store.list_tasks(status)
        store.close()
        if not tasks:
            print(f"No {status} tasks.")
        else:
            for r in tasks:
                print(f"  #{r['id']} [{r['status']}] {r['title']} — {r['summary'][:80]}")

    elif cmd == "search_context":
        query = " ".join(args)
        result = _execute_intent({"intent": "search_context", "summary": query}, query)
        print(result or "No results.")

    elif cmd == "profile":
        print(_cmd_profile(" ".join(args)))

    elif cmd == "exhaustion_save":
        print(_cmd_exhaustion_save(" ".join(args)))

    elif cmd == "digest":
        print(_cmd_conversation_digest())

    elif cmd == "optimize_context":
        print(_cmd_optimize_context())

    elif cmd == "save_memory":
        print(_cmd_save_memory(" ".join(args)))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
