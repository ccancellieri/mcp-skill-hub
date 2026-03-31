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

from .embeddings import embed, compact, ollama_available, EMBED_MODEL, RERANK_MODEL
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
        return {"intent": "none"}

    # Stage 1b: semantic prefilter — skip LLM if message is clearly unrelated
    threshold = float(_cfg.get("hook_semantic_threshold") or 0.35)
    sim = _task_similarity(message)
    if sim < threshold:
        return {"intent": "none"}

    # Stage 2: LLM classification
    prompt = f"""\
You are a command classifier. Given a user message, determine if it is a task/memory command.

Possible intents:
- "save_task": user wants to save current work/discussion for later (e.g. "save to memory", "park this", "save task", "remember this")
- "close_task": user wants to close/finish a task (e.g. "close task", "done with this", "mark as done", "save and close")
- "list_tasks": user wants to see open tasks (e.g. "what was I working on?", "show tasks", "open tasks")
- "search_context": user wants to find past work (e.g. "what did we discuss about X?", "find my previous work on Y")
- "none": not a task command, just a normal message

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
            return json.loads(json_match.group())
    except Exception:
        pass
    return {"intent": "none"}


def _execute_intent(intent: dict, original_message: str) -> str | None:
    """Execute a classified intent. Returns a user-facing message, or None if not handled."""
    store = SkillStore()
    action = intent.get("intent", "none")

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


def hook_classify_and_execute(message: str) -> dict:
    """
    Main entry point for the UserPromptSubmit hook.
    Returns a hook response dict:
      - {"decision": "block", "message": "..."} if handled locally
      - {"decision": "allow"} if Claude should handle it
    """
    from . import config as _cfg

    intent = _classify_intent(message)
    action = intent.get("intent", "none")

    if action == "none":
        return {"decision": "allow"}

    result = _execute_intent(intent, message)
    if result:
        # Log interception for token profiling (if profiling enabled)
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
        return {
            "decision": "block",
            "message": result,
        }

    return {"decision": "allow"}


def main() -> None:
    """CLI entry point for direct invocation."""
    if len(sys.argv) < 2:
        print("Usage: skill-hub-cli <command> [args...]")
        print("Commands: classify, save_task, close_task, list_tasks, search_context")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "classify":
        message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read()
        result = hook_classify_and_execute(message.strip())
        print(json.dumps(result))

    elif cmd == "save_task":
        title = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        summary = sys.argv[3] if len(sys.argv) > 3 else ""
        tags = sys.argv[4] if len(sys.argv) > 4 else ""
        store = SkillStore()
        vector = embed(f"{title}: {summary}") if ollama_available() else []
        tid = store.save_task(title=title, summary=summary, vector=vector, tags=tags)
        store.close()
        print(f"Task #{tid} saved: \"{title}\"")

    elif cmd == "close_task":
        task_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if not task_id:
            print("Usage: skill-hub-cli close_task <task_id>")
            sys.exit(1)
        result = _execute_intent({"intent": "close_task", "task_id": task_id}, "")
        print(result or "Failed.")

    elif cmd == "list_tasks":
        status = sys.argv[2] if len(sys.argv) > 2 else "open"
        result = _execute_intent({"intent": "list_tasks"}, "")
        print(result or "No tasks.")

    elif cmd == "search_context":
        query = " ".join(sys.argv[2:])
        result = _execute_intent({"intent": "search_context", "summary": query}, query)
        print(result or "No results.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
