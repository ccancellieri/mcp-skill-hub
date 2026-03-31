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

    Fast-path: /slash-commands are routed directly to local handlers
    without any LLM call (~0ms).
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
    if not totals or not totals["total_interceptions"]:
        store.close()
        return "No interceptions recorded yet."

    stats = store.get_interception_stats()
    total_saved = totals["total_tokens_saved"] or 0
    total_count = totals["total_interceptions"] or 0
    store.close()

    lines = [
        f"=== Token Savings ===\n",
        f"Total intercepted: {total_count}",
        f"Tokens saved:      ~{total_saved:,}\n",
    ]
    for row in stats:
        avg = (row["total_tokens_saved"] or 0) // max(row["intercept_count"], 1)
        lines.append(f"  {row['command_type']:<20} {row['intercept_count']:>4}x  ~{row['total_tokens_saved'] or 0:,} tokens")
    return "\n".join(lines)


def _cmd_help() -> str:
    """Quick reference — runs locally."""
    return """=== Skill Hub Commands ===

Slash commands (intercepted locally, 0 Claude tokens):
  /hub-status          Health check, models, DB stats
  /hub-help            This reference
  /token-stats         Token savings report
  /list-tasks [status] Open/closed/all tasks
  /save-task [title]   Save current work (needs LLM)
  /close-task [N]      Close + compact task (needs LLM)
  /search-context Q    Unified search (needs embed model)
  /list-skills         All indexed skills
  /list-teachings      Teaching rules
  /list-models         Installed Ollama models
  /configure [K] [V]   View/set config

These run via local LLM or SQLite — Claude never sees them.

MCP tools (called BY Claude when needed):
  search_skills(query)     suggest_plugins(query)
  teach(rule, suggest)     record_feedback(skill_id, helpful)
  toggle_plugin(name, on)  pull_model(model)
  index_skills()           index_plugins()"""


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


# Map of slash command prefixes → local handlers
_SLASH_COMMANDS: dict[str, str] = {
    "/hub-status": "status",
    "/hub-help": "help",
    "/hub-manual": "help",
    "/token-stats": "token_stats",
    "/list-tasks": "list_tasks",
    "/list-models": "list_models",
    "/list-skills": "list_skills",
    "/list-teachings": "list_teachings",
    "/configure": "configure",
    "/session-stats": "token_stats",
    "/save-task": "save_task",
    "/close-task": "close_task",
    "/search-context": "search_context",
    "/index-skills": "index_skills",
    "/index-plugins": "index_plugins",
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
        elif action in ("save_task", "close_task", "search_context"):
            # These need LLM — fall through to classify
            return None
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
    except Exception as exc:
        return {"decision": "block", "message": f"Error: {exc}"}

    return None


def main() -> None:
    """CLI entry point for direct invocation."""
    if len(sys.argv) < 2:
        print("Usage: skill-hub-cli <command> [args...]")
        print("Commands: classify, status, help, token_stats, list_models,")
        print("          list_skills, list_teachings, configure,")
        print("          save_task, close_task, list_tasks, search_context")
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

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
