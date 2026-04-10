"""Direct CLI for skill-hub operations — used by hooks to bypass Claude entirely.

Usage from hooks:
    skill-hub-cli save_task "title" "summary" --tags "mcp,dev"
    skill-hub-cli close_task 3
    skill-hub-cli list_tasks open
    skill-hub-cli search_context "my query"
    skill-hub-cli classify "save this to memory and close"
"""

import json
import re
import sys
from pathlib import Path

from .activity_log import (
    log_hook, log_llm, log_event, log_skill_step,
    log_pipeline_start, log_pipeline_end, log_step, log_detail,
    _human_size, llm_timer,
)
from .embeddings import (
    embed, compact, ollama_available, optimize_context, triage_message,
    dynamic_context_eval, eval_skill_lifecycle, optimize_prompt, smart_memory_write,
    verify_cache_hit, decompose_task, extract_error_pattern, generate_auto_skill,
    EMBED_MODEL, RERANK_MODEL, OLLAMA_BASE,
    conversation_digest, exhaustion_save,
)
from .resource_monitor import should_run_llm, snapshot, Pressure
from .store import SkillStore


# Cached embedding of task command examples (computed once per process)
_task_command_vector: list[float] | None = None
_task_command_hash: int | None = None  # tracks config changes

# Cached embedding centroid for L1/L2 shell commands (separate from task commands)
_local_cmd_vector: list[float] | None = None
_local_cmd_hash: int | None = None

# Regex to strip IDE/system context tags from hook input before classification.
# These tags (e.g. <ide_opened_file>, <ide_selection>, <system-reminder>) are
# injected by the IDE and confuse small local LLMs during intent classification.
_IDE_TAG_RE = re.compile(
    r"<(?:ide_\w+|system-reminder|EXTREMELY_IMPORTANT|SUBAGENT-STOP|command-name)"
    r"[^>]*>.*?</(?:ide_\w+|system-reminder|EXTREMELY_IMPORTANT|SUBAGENT-STOP|command-name)>",
    re.DOTALL,
)


def _sanitize_hook_message(message: str) -> str:
    """Strip IDE context tags and collapse whitespace so the LLM sees only the user's words."""
    cleaned = _IDE_TAG_RE.sub("", message).strip()
    # Collapse runs of whitespace left by removed tags
    cleaned = re.sub(r"\s{3,}", "  ", cleaned)
    return cleaned or message  # fallback to original if everything was stripped

# Dynamic persona cache: {session_id: (persona_str, built_at_epoch)}
_persona_cache: dict[str, tuple[str, float]] = {}


def _build_local_persona(session_id: str = "") -> str:
    """
    Build a dynamic system prompt for all local LLM calls (L1–L4).

    Assembled from (high-value first):
    1. Static seed from config "local_system_prompt"
    2. All teaching rules → "Rule: when X → Y"
    3. Recent closed tasks (last 5) → project + decision keywords
    4. Memory index top-5 entries (by recency, from MEMORY.md)

    Capped at local_persona_max_chars (default 600).
    Cached per session_id for local_persona_ttl_seconds (default 120s).
    Returns empty string if nothing useful is available.
    """
    import time
    import os as _os
    from pathlib import Path
    from . import config as _cfg

    ttl = float(_cfg.get("local_persona_ttl_seconds") or 120)
    max_chars = int(_cfg.get("local_persona_max_chars") or 800)
    now = time.monotonic()

    # Cache check
    cache_key = session_id or "__global__"
    if cache_key in _persona_cache:
        cached_str, built_at = _persona_cache[cache_key]
        if now - built_at < ttl:
            return cached_str

    parts: list[str] = []

    # 1. Static seed
    seed = str(_cfg.get("local_system_prompt") or "").strip()
    if seed:
        parts.append(seed)

    try:
        store = SkillStore()

        # 2. Teaching rules
        teachings = store.list_teachings()
        rules: list[str] = []
        for t in teachings:
            rule = t["rule"] if isinstance(t, dict) else t[1]
            target = t["target_id"] if isinstance(t, dict) else t[4]
            rules.append(f"when {rule} → {target}")
        if rules:
            parts.append("Rules: " + " | ".join(rules[:5]))

        # 2b. Tool patterns — what tools the user's workflow favors
        try:
            patterns = store.get_tool_patterns(limit=3)
            if patterns:
                pat_strs = []
                for p in patterns:
                    repo_name = Path(p["repo_path"]).name if p.get("repo_path") else "?"
                    pat_strs.append(
                        f"{p['tool_name']}({repo_name}):{p['count']}x")
                parts.append("Tool patterns: " + ", ".join(pat_strs))
        except Exception:
            pass

        # 2c. Per-repo context (if in a known repo)
        try:
            cwd = _os.getcwd()
            repo_ctx = store.get_repo_context(cwd)
            if repo_ctx:
                if repo_ctx.get("project_summary"):
                    parts.append(f"Project: {repo_ctx['project_summary'][:100]}")
                if repo_ctx.get("commit_style"):
                    parts.append(f"Commit style: {repo_ctx['commit_style']}")
        except Exception:
            pass

        # 3. Recent closed tasks → project context
        closed = store.list_tasks(status="closed")
        if closed:
            titles = [
                (t["title"] if isinstance(t, dict) else t[1])
                for t in list(closed)[:5]
            ]
            parts.append("Recent work: " + ", ".join(t[:60] for t in titles))

        store.close()
    except Exception:
        pass

    # 4. Memory index entries (last 5 lines with project-like names)
    try:
        _mem_cfg = _cfg.get("hook_memory_dir")
        if _mem_cfg:
            memory_dir = Path(str(_mem_cfg)).expanduser()
        else:
            _cwd_slug = _os.getcwd().replace("/", "-").replace("\\", "-").lstrip("-")
            memory_dir = Path.home() / ".claude" / "projects" / _cwd_slug / "memory"
        mem_index = memory_dir / "MEMORY.md"
        if mem_index.exists():
            lines = mem_index.read_text(encoding="utf-8", errors="replace").splitlines()
            mem_lines = [l.strip() for l in lines if l.strip().startswith("-")][-5:]
            if mem_lines:
                parts.append("Memory: " + "; ".join(
                    l.split("—")[-1].strip()[:60] for l in mem_lines if "—" in l
                ))
    except Exception:
        pass

    if not parts:
        _persona_cache[cache_key] = ("", now)
        return ""

    persona = "[Context for local LLM]\n" + "\n".join(parts)
    if len(persona) > max_chars:
        persona = persona[:max_chars]

    _persona_cache[cache_key] = (persona, now)
    return persona


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


def _local_cmd_similarity(message: str) -> float:
    """
    Embedding similarity against L1/L2 shell command phrases.
    Uses a separate centroid from task commands — "show git status"
    is similar to shell commands but not to "save to memory".
    """
    import math
    from . import config as _cfg

    global _local_cmd_vector, _local_cmd_hash

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    try:
        # Build examples from L1 command names + L2 template names
        commands = _cfg.get("local_commands") or {}
        templates = _cfg.get("local_templates") or {}
        # Natural language versions of command names
        examples = [name.replace("_", " ") for name in commands]
        examples += [name.replace("_", " ") for name in templates]
        # Add common phrasings
        examples += [
            "show git status", "run git log", "list files", "show branches",
            "check disk space", "what directory am I in", "show git diff",
            "show stash list", "git remote", "check uptime",
        ]
        current_hash = hash(tuple(sorted(examples)))

        if _local_cmd_vector is None or _local_cmd_hash != current_hash:
            centroid_text = " | ".join(examples)
            _local_cmd_vector = embed(centroid_text)
            _local_cmd_hash = current_hash

        msg_vec = embed(message)
        return cosine(msg_vec, _local_cmd_vector)
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
        log_detail(f"classify: skip (message too long: {len(message)} chars)")
        return {"intent": "none", "_similarity": 0.0}

    # Stage 1b: semantic prefilter — skip LLM if message is clearly unrelated
    threshold = float(_cfg.get("hook_semantic_threshold") or 0.35)
    sim = _task_similarity(message)
    if sim < threshold:
        log_detail(f"classify: not a command (sim={sim:.2f})")
        return {"intent": "none", "_similarity": sim}

    log_detail(f"classify: might be a command (sim={sim:.2f}), asking LLM")
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

    # Use lighter model for classification (simple yes/no decision)
    classify_model = (_cfg.get("local_models") or {}).get(
        "level_1", "qwen2.5-coder:3b")
    if not ollama_available(classify_model):
        classify_model = RERANK_MODEL  # fallback to reason model

    try:
        with llm_timer() as _t:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": classify_model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": 100}},
                timeout=15.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["_similarity"] = sim
            _intent = result.get("intent", "none")
            if _intent != "none":
                log_step(f"classify: intent={_intent} ({_t.duration:.1f}s)")
            else:
                log_detail(f"classify: not a command ({_t.duration:.1f}s)")
            return result
    except Exception as exc:
        log_step(f"classify: ERROR {str(exc)[:80]}")
    return {"intent": "none", "_similarity": sim}


def _execute_intent(intent: dict, original_message: str) -> str | None:
    """Execute a classified intent. Returns a user-facing message, or None if not handled."""
    store = SkillStore()
    action = intent.get("intent", "none")
    log_step(f"executing: {action}")

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
                _auto_memory_from_task_close(task_id=int(task_id), digest=digest)
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
            _auto_memory_from_task_close(task_id=latest["id"], digest=digest)
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
    "local_command": 300,
    "local_template": 400,
}

# Track which commands have been approved this session (by name)
_approved_commands: set[str] = set()
# Pending command awaiting user confirmation
_pending_command: dict | None = None


# ── Local execution engine ──────────────────────────────────────────


_LOCAL_COMMAND_PROMPT = """\
You are a strict command mapper. Given a user message and a list of available \
shell commands, pick the matching command ONLY if the user is explicitly asking \
to run that exact operation.

CRITICAL: Most messages are NOT commands — they are coding questions, bug reports, \
feature requests, or discussions. If the user is asking about code, asking you to \
fix/build/explain something, or discussing anything other than running a shell \
command, you MUST reply "none" with confidence 0.0.

Available commands:
{commands}

Reply with ONLY a JSON object:
{{"command": "<command_name or none>", "confidence": <0.0-1.0>}}

Examples:
- "fix the pagination bug" → {{"command": "none", "confidence": 0.0}}
- "show me git status" → {{"command": "git_status", "confidence": 0.95}}
- "add a new endpoint" → {{"command": "none", "confidence": 0.0}}
- "what branches exist" → {{"command": "git_branch", "confidence": 0.90}}
- "refactor the query executor" → {{"command": "none", "confidence": 0.0}}
- "list files" → {{"command": "ls", "confidence": 0.95}}
- "show stashes" → {{"command": "git_stash_list", "confidence": 0.90}}

User message: {message}"""


def _match_local_command(message: str) -> dict | None:
    """Level 1: Map user message to a whitelisted shell command via local LLM."""
    import httpx
    import re
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE

    if not _cfg.get("local_execution_enabled"):
        return None

    commands = _cfg.get("local_commands") or {}
    if not commands:
        return None

    model = (_cfg.get("local_models") or {}).get("level_1", "qwen2.5-coder:3b")

    cmd_list = "\n".join(f"  {name}: {cmd}" for name, cmd in commands.items())
    prompt = _LOCAL_COMMAND_PROMPT.format(commands=cmd_list, message=message)
    persona = _build_local_persona()
    if persona:
        prompt = f"{persona}\n\n{prompt}"

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=15.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            name = result.get("command", "none")
            confidence = float(result.get("confidence", 0))
            min_conf = float(_cfg.get("local_cmd_min_confidence") or 0.80)
            if name != "none" and name in commands and confidence >= min_conf:
                log_step(f"local command matched: {name} (confidence={confidence:.2f})")
                return {"name": name, "shell": commands[name], "level": 1}
            elif name != "none" and confidence > 0:
                log_detail(f"local_cmd: {name} rejected (confidence={confidence:.2f} < {min_conf:.2f})")
    except Exception:
        pass
    return None


_LOCAL_TEMPLATE_PROMPT = """\
You are a command mapper with parameter extraction. Given a user message, \
pick the BEST matching command template and extract its parameters.

Available templates:
{templates}

Parameter types: int (number), str (word), path (file/dir path).
If none match, reply "none".

Reply with ONLY a JSON object:
{{"template": "<template_name or none>", "params": {{"param1": "value1"}}, "confidence": <0.0-1.0>}}

User message: {message}"""


_SHELL_UNSAFE = re.compile(r'[;&|`$(){}!<>\n\r\\]')


def _sanitize_param(value: str, param_type: str) -> str | None:
    """Sanitize a parameter value. Returns None if unsafe."""
    value = value.strip().strip("'\"")
    if _SHELL_UNSAFE.search(value):
        return None
    if param_type == "int":
        try:
            return str(int(value))
        except ValueError:
            return None
    if param_type == "path":
        # reject traversal and absolute paths outside cwd
        if ".." in value or value.startswith("/"):
            return None
    return value


def _match_local_template(message: str) -> dict | None:
    """Level 2: Map user message to a templated command with extracted params."""
    import httpx
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE

    if not _cfg.get("local_execution_enabled"):
        return None

    templates = _cfg.get("local_templates") or {}
    if not templates:
        return None

    model = (_cfg.get("local_models") or {}).get("level_2", "qwen2.5-coder:7b-instruct-q4_k_m")

    tpl_list = "\n".join(
        f"  {name}: {t['cmd']}  (params: {t['params']})"
        for name, t in templates.items()
    )
    prompt = _LOCAL_TEMPLATE_PROMPT.format(templates=tpl_list, message=message)
    persona = _build_local_persona()
    if persona:
        prompt = f"{persona}\n\n{prompt}"

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=20.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            name = result.get("template", "none")
            confidence = float(result.get("confidence", 0))
            params = result.get("params", {})

            if name == "none" or name not in templates or confidence < 0.92:
                return None

            tpl = templates[name]
            # Sanitize all params
            safe_params = {}
            for pname, ptype in tpl["params"].items():
                raw_val = str(params.get(pname, ""))
                if not raw_val:
                    return None  # missing required param
                clean = _sanitize_param(raw_val, ptype)
                if clean is None:
                    log_detail(f"local_tpl: {name} rejected (unsafe param: {pname})")
                    return None
                safe_params[pname] = clean

            # Build the shell command
            shell_cmd = tpl["cmd"].format(**safe_params)
            log_step(f"local template matched: {name} (confidence={confidence:.2f})")
            return {"name": name, "shell": shell_cmd, "level": 2}
    except Exception:
        pass
    return None


def _visible_result(message: str) -> dict:
    """Return a hook result that blocks the prompt and shows output to user.

    Uses 'reason' (not 'message') per the Claude Code hook schema.
    The 'reason' field is displayed to the user in both CLI and VS Code
    without spending any Claude tokens.
    """
    return {"decision": "block", "reason": message}


def _resolve_repo_cwd(message: str) -> str | None:
    """Extract a target repository path from the user message.

    Looks for:
    - @<dirname>/ or @<dirname> prefix references (e.g. "@mcp-skill-hub/")
    - Explicit project names matching known subdirs under the work root

    Returns the absolute path if it resolves to an existing git repo, else None.
    """
    import re
    import os

    # Pattern: @word/ or @word at start or anywhere
    refs = re.findall(r'@([\w\-\.]+)(?:/|(?=\s|$))', message)
    if not refs:
        return None

    # Candidate search roots: cwd's parent, and ~/work/code
    cwd = os.getcwd()
    search_roots = list(dict.fromkeys([
        os.path.dirname(cwd),          # parent of current dir
        os.path.expanduser("~/work/code"),
        os.path.expanduser("~/work"),
    ]))

    for ref in refs:
        for root in search_roots:
            candidate = os.path.join(root, ref)
            git_dir = os.path.join(candidate, ".git")
            if os.path.isdir(candidate) and os.path.exists(git_dir):
                log_detail(f"repo_cwd: {ref} -> {candidate}")
                return candidate

    return None


def _execute_local_command(cmd: dict, cwd: str | None = None) -> str | None:
    """Execute a matched local command and return its output."""
    import subprocess

    shell_cmd = cmd["shell"]
    log_step(f"exec L{cmd['level']}: $ {shell_cmd}")

    try:
        result = subprocess.run(
            shell_cmd, shell=True, capture_output=True, text=True,
            timeout=30, cwd=cwd,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n(stderr: {result.stderr.strip()})"
        # Truncate very long output
        if len(output) > 5000:
            output = output[:5000] + f"\n... (truncated, {len(output)} chars total)"
        log_detail(f"local_cmd_done: {shell_cmd} ({len(output)} chars)")
        return f"$ {shell_cmd}\n{output.strip()}"
    except subprocess.TimeoutExpired:
        log_step(f"exec L{cmd['level']}: TIMEOUT $ {shell_cmd}")
        return f"Command timed out: {shell_cmd}"
    except Exception as exc:
        log_step(f"exec L{cmd['level']}: FAIL $ {shell_cmd} ({exc})")
        return f"Command failed: {exc}"



# ── Level 3: Local skill execution ───────────────────────────────────
#
# Local skills are JSON files in local_skills_dir (default ~/.claude/local-skills/).
# Format:
# {
#   "name": "git-summary",
#   "description": "Show recent git activity summary",
#   "triggers": ["git summary", "recent activity", "what changed"],
#   "steps": [
#     {"run": "git log --oneline -10", "as": "recent_commits"},
#     {"run": "git diff --stat", "as": "changes"},
#     {"run": "git status -s", "as": "status"}
#   ],
#   "output": "## Recent commits\n{recent_commits}\n\n## Changed files\n{changes}\n\n## Working tree\n{status}"
# }
#
# Recommended: keep local skills in a separate folder optimized for local models.
# Local models work best with short, concrete triggers and deterministic steps.

_local_skills_cache: list[dict] | None = None
_local_skills_hash: int | None = None


def _load_local_skills() -> list[dict]:
    """Load and cache local skill definitions from local_skills_dir."""
    global _local_skills_cache, _local_skills_hash
    from . import config as _cfg
    from pathlib import Path

    skills_dir = Path(str(_cfg.get("local_skills_dir"))).expanduser()
    if not skills_dir.exists():
        return []

    # Check if dir has changed (by mtime of newest file)
    json_files = sorted(skills_dir.glob("*.json"))
    if not json_files:
        return []
    current_hash = hash(tuple((f.name, f.stat().st_mtime_ns) for f in json_files))
    if _local_skills_cache is not None and _local_skills_hash == current_hash:
        return _local_skills_cache

    skills = []
    for f in json_files:
        try:
            skill = json.loads(f.read_text(encoding="utf-8"))
            skill["_file"] = str(f)
            skills.append(skill)
        except Exception:
            continue

    _local_skills_cache = skills
    _local_skills_hash = current_hash
    log_detail(f"local_skills: loaded {len(skills)} from {skills_dir}")
    return skills


def _match_local_skill(message: str) -> dict | None:
    """Level 3: Match user message to a local skill via embedding similarity."""
    import math
    from . import config as _cfg

    if not _cfg.get("local_execution_enabled"):
        return None

    skills = _load_local_skills()
    if not skills:
        return None

    try:
        msg_vec = embed(message[:300])
    except Exception:
        return None

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    best_skill = None
    best_sim = 0.0

    for skill in skills:
        triggers = skill.get("triggers", [])
        desc = skill.get("description", "")
        # Embed all triggers + description, take max similarity
        texts = triggers + ([desc] if desc else [])
        for text in texts:
            try:
                t_vec = embed(text)
                sim = cosine(msg_vec, t_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_skill = skill
            except Exception:
                continue

    min_skill_sim = float(_cfg.get("local_skill_threshold") or 0.85)
    if best_skill and best_sim >= min_skill_sim:
        log_step(f"local skill matched: {best_skill['name']} (sim={best_sim:.2f})")
        return best_skill
    return None


def _execute_local_skill(skill: dict, cwd: str | None = None) -> str | None:
    """Execute a local skill's steps and format the output.

    Step types:
      {"run": "shell command", "as": "var_name"}
        Run a shell command, store output in var_name.

      {"run": "shell command", "as": "var_name", "if_empty": "fallback_cmd"}
        Run command; if output is empty, run fallback_cmd instead.

      {"run": "shell command", "as": "var_name", "on_fail": "recovery_cmd"}
        Run command; if it fails (non-zero exit), run recovery_cmd.

      {"run": "shell command", "as": "var_name", "on_fail": "recovery_cmd",
       "retry": true}
        After recovery, retry the original command.

      {"llm": "prompt with {var} placeholders", "as": "var_name"}
        Use local LLM to generate text. Prompt can reference previous {var_name}s.

      {"run": "cmd with {var} placeholders", "as": "var_name"}
        Shell commands can also use {var_name} from previous steps.

      {"stop_if_empty": "var_name", "message": "Nothing to do."}
        Stop execution if a variable is empty.

      {"label": "name"}
        Named jump target. No-op at execution.

      {"goto": "label_name"}
        Unconditional jump to a label.

      {"stop": true, "message": "Done."}
        Explicit halt with optional message.

      {"if_contains": "var_name", "value": "string", "goto": "label"}
        If var contains string, jump to label. Add "else": "other_label" for else branch.

      {"if_match": "var_name", "pattern": "regex", "goto": "label"}
        If var matches regex, jump to label. Add "else": "other_label" for else branch.

      {"if_empty": "var_name", "goto": "label"}
        If var is empty, jump to label. Add "else": "other_label" for else branch.

      {"if_rc": "var_name", "eq": 0, "goto": "label"}
        If last exit code for var equals value, jump to label. Supports "eq", "ne".

    Template variables: all step outputs are available as {var_name} in
    subsequent steps and in the output template.
    Exit codes: stored as {var_name_rc} for each shell step.
    """
    import subprocess
    import re
    import httpx
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available

    name = skill.get("name", "unknown")
    steps = skill.get("steps", [])
    output_template = skill.get("output", "")
    skill_type = skill.get("type", "steps")

    log_step(f"skill exec: {name} ({skill_type}, {len(steps)} steps)")

    # ── Agent-as-skill routing ──
    # Skills with "type": "agent" delegate to the L4 agent loop instead of
    # the linear step engine. The skill defines the goal and optional context.
    if skill_type == "agent":
        from .local_agent import run_agent
        from .store import read_session_context
        agent_prompt = skill.get("prompt", skill.get("description", name))
        agent_context = skill.get("context", "")
        agent_max_turns = skill.get("max_turns", 8)
        # Inject session context if available
        session_ctx = read_session_context()
        if session_ctx:
            agent_context = f"{agent_context}\n\n## Session Context\n{session_ctx}".strip()
        log_skill_step(name, "AGENT", f"prompt={agent_prompt[:80]!r} max_turns={agent_max_turns}")
        try:
            result = run_agent(agent_prompt, context=agent_context,
                               max_turns=agent_max_turns)
            log_skill_step(name, "AGENT", "completed", result=f"{len(result)} chars")
            return f"[{name} (agent)]\n\n{result}"
        except Exception as exc:
            log_skill_step(name, "AGENT", "failed", result=str(exc)[:120], ok=False)
            return f"[{name}] Agent error: {exc}"

    # Pre-populate built-in variables available to all skills
    results: dict[str, str] = {}
    step_outputs: list[str] = []

    # Inject session context from disk (zero Claude token cost)
    from .store import read_session_context, SkillStore as _SkillStore
    results["session_context"] = read_session_context()

    # {tool_examples} — recent Claude tool calls from DB
    try:
        _s = _SkillStore()
        _examples = _s.get_recent_tool_examples(limit=8)
        if _examples:
            _lines = []
            for _e in _examples:
                _hint = f" ({_e['context_hint'][:60]})" if _e['context_hint'] else ""
                _lines.append(f"- {_e['tool_name']}: {_e['tool_input'][:120]}{_hint}")
            results["tool_examples"] = "\n".join(_lines)
        else:
            results["tool_examples"] = ""
        # {tool_patterns} — aggregated patterns across sessions
        _pats = _s.get_tool_patterns(limit=5)
        if _pats:
            _plines = []
            for _p in _pats:
                _repo = Path(_p["repo_path"]).name if _p.get("repo_path") else "?"
                _plines.append(f"- {_p['tool_name']}({_repo}): {_p['count']}x")
            results["tool_patterns"] = "\n".join(_plines)
        else:
            results["tool_patterns"] = ""
        _s.close()
    except Exception:
        results["tool_examples"] = ""
        results["tool_patterns"] = ""

    # {repo_context} — current repo state (branch, dirty files, recent commits)
    try:
        _rc = subprocess.run(
            "git rev-parse --abbrev-ref HEAD 2>/dev/null && "
            "git diff --shortstat 2>/dev/null && "
            "git log --oneline -3 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        results["repo_context"] = _rc.stdout.strip() if _rc.returncode == 0 else ""
    except Exception:
        results["repo_context"] = ""

    def _interpolate(template: str) -> str:
        """Replace {var_name} placeholders with step results."""
        try:
            return template.format(**results)
        except (KeyError, IndexError):
            return template

    if cwd:
        log_detail(f"skill cwd: {name} -> {cwd}")

    def _run_shell(cmd: str, timeout: int = 30) -> tuple[int, str]:
        """Run a shell command, return (exit_code, output)."""
        cmd = _interpolate(cmd)
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=timeout, cwd=cwd)
            out = r.stdout.strip()
            if r.returncode != 0 and r.stderr:
                out += f"\n(exit {r.returncode}: {r.stderr.strip()[:200]})"
            if len(out) > 3000:
                out = out[:3000] + "\n... (truncated)"
            return r.returncode, out
        except subprocess.TimeoutExpired:
            return 1, f"(timed out: {cmd})"
        except Exception as exc:
            return 1, f"(error: {exc})"

    # Build label → index map for jump targets
    _label_map: dict[str, int] = {}
    for _i, _s in enumerate(steps):
        if "label" in _s:
            _label_map[_s["label"]] = _i

    # Track exit codes per variable for if_rc conditions
    _exit_codes: dict[str, int] = {}

    # Index-based loop with jump support (max iterations = 10× step count to prevent infinite loops)
    _max_iters = max(len(steps) * 10, 100)
    _iter_count = 0
    _step_idx = 0

    while _step_idx < len(steps):
        _iter_count += 1
        if _iter_count > _max_iters:
            log_skill_step(name, "HALT", f"loop limit {_max_iters} reached after {_iter_count} iterations", ok=False)
            step_outputs.append(f"(halted: loop limit {_max_iters} reached)")
            break

        step = steps[_step_idx]
        var_name = step.get("as", f"step_{len(step_outputs)}")

        # ── Label (jump target, no-op) ──
        if "label" in step and "run" not in step and "llm" not in step:
            log_skill_step(name, "LABEL", step["label"])
            _step_idx += 1
            continue

        # ── Goto (unconditional jump) ──
        if "goto" in step and "if_contains" not in step and "if_match" not in step and "if_empty" not in step and "if_rc" not in step:
            target = step["goto"]
            if target in _label_map:
                log_skill_step(name, "GOTO", f"→ {target}")
                _step_idx = _label_map[target]
            else:
                log_skill_step(name, "GOTO", f"→ {target} (NOT FOUND)", ok=False)
                step_outputs.append(f"(goto: label '{target}' not found)")
                _step_idx += 1
            continue

        # ── Stop (explicit halt) ──
        if step.get("stop"):
            msg = _interpolate(step.get("message", "Stopped."))
            log_skill_step(name, "STOP", msg[:120])
            step_outputs.append(f"(stop: {msg})")
            break

        # ── Stop-if-empty guard ──
        if "stop_if_empty" in step:
            check_var = step["stop_if_empty"]
            if not results.get(check_var, "").strip():
                msg = _interpolate(step.get("message", "Stopped: no data."))
                log_skill_step(name, "GUARD", f"{check_var} is empty", result=msg[:80])
                return f"[{name}] {msg}"
            log_skill_step(name, "GUARD", f"{check_var} has data → continue")
            _step_idx += 1
            continue

        # ── Conditional: if_contains ──
        if "if_contains" in step:
            check_var = step["if_contains"]
            value = _interpolate(step.get("value", ""))
            var_val = results.get(check_var, "")
            matched = value.lower() in var_val.lower()
            target = step.get("goto", "") if matched else step.get("else", "")
            action = f"→ {target}" if target else "→ next"
            log_skill_step(name, "IF", f"{check_var} contains {value[:40]!r} = {'yes' if matched else 'no'} {action}")
            if target and target in _label_map:
                _step_idx = _label_map[target]
            else:
                _step_idx += 1
            continue

        # ── Conditional: if_match (regex) ──
        if "if_match" in step:
            check_var = step["if_match"]
            pattern = step.get("pattern", "")
            var_val = results.get(check_var, "")
            try:
                matched = bool(re.search(pattern, var_val))
            except re.error:
                matched = False
                log_skill_step(name, "IF", f"regex error: {pattern[:40]!r}", ok=False)
            target = step.get("goto", "") if matched else step.get("else", "")
            action = f"→ {target}" if target else "→ next"
            log_skill_step(name, "IF", f"{check_var} matches /{pattern[:40]}/ = {'yes' if matched else 'no'} {action}")
            if target and target in _label_map:
                _step_idx = _label_map[target]
            else:
                _step_idx += 1
            continue

        # ── Conditional: if_empty ──
        if "if_empty" in step and "run" not in step:
            check_var = step["if_empty"]
            var_val = results.get(check_var, "")
            matched = not var_val.strip()
            target = step.get("goto", "") if matched else step.get("else", "")
            action = f"→ {target}" if target else "→ next"
            log_skill_step(name, "IF", f"{check_var} empty = {'yes' if matched else 'no'} {action}")
            if target and target in _label_map:
                _step_idx = _label_map[target]
            else:
                _step_idx += 1
            continue

        # ── Conditional: if_rc (exit code check) ──
        if "if_rc" in step:
            check_var = step["if_rc"]
            actual_rc = _exit_codes.get(check_var, -1)
            matched = False
            op_str = ""
            if "eq" in step:
                matched = actual_rc == int(step["eq"])
                op_str = f"rc=={step['eq']}"
            elif "ne" in step:
                matched = actual_rc != int(step["ne"])
                op_str = f"rc!={step['ne']}"
            target = step.get("goto", "") if matched else step.get("else", "")
            action = f"→ {target}" if target else "→ next"
            log_skill_step(name, "IF", f"{check_var} {op_str} (actual={actual_rc}) = {'yes' if matched else 'no'} {action}")
            if target and target in _label_map:
                _step_idx = _label_map[target]
            else:
                _step_idx += 1
            continue

        # ── LLM step ──
        if "llm" in step:
            prompt = _interpolate(step["llm"])
            llm_model = step.get("model") or (
                (_cfg.get("local_models") or {}).get("level_2", RERANK_MODEL))
            if not ollama_available(llm_model):
                llm_model = RERANK_MODEL
            try:
                resp = httpx.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={"model": llm_model, "prompt": prompt, "stream": False,
                          "options": {"temperature": float(step.get("temperature", 0.2)),
                                      "num_predict": int(step.get("max_tokens", 150))}},
                    timeout=float(step.get("timeout", 15)),
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                # Clean up common LLM artifacts
                raw = raw.strip().strip('"').strip("'").strip("`")
                raw = re.sub(r"^(commit message:?\s*)", "", raw,
                             flags=re.IGNORECASE).strip()
                # Take first line if step requests it
                if step.get("first_line"):
                    raw = raw.split("\n")[0].strip()
                results[var_name] = raw
                log_skill_step(name, "LLM", f"model={llm_model} prompt={len(prompt)}ch",
                               result=f"{var_name}={raw[:80]!r}")
            except Exception as exc:
                results[var_name] = step.get("fallback", "")
                log_skill_step(name, "LLM", f"model={llm_model} prompt={len(prompt)}ch",
                               result=f"ERROR: {exc!s:.80}", ok=False)
            _step_idx += 1
            continue

        # ── Shell step ──
        cmd = step.get("run", "")
        if not cmd:
            _step_idx += 1
            continue

        _resolved_cmd = _interpolate(cmd)
        rc, out = _run_shell(cmd, timeout=int(step.get("timeout", 30)))
        _exit_codes[var_name] = rc
        log_skill_step(name, "SHELL", f"$ {_resolved_cmd[:120]}",
                       result=f"rc={rc} ({len(out)} chars)", ok=rc == 0)

        # if_empty: run fallback command when output is empty
        if not out.strip() and step.get("if_empty"):
            _fallback_cmd = _interpolate(step["if_empty"])
            rc, out = _run_shell(step["if_empty"])
            _exit_codes[var_name] = rc
            log_skill_step(name, "SHELL", f"(fallback) $ {_fallback_cmd[:120]}",
                           result=f"rc={rc} ({len(out)} chars)", ok=rc == 0)

        # on_fail: run recovery command on non-zero exit
        if rc != 0 and step.get("on_fail"):
            _recovery_cmd = _interpolate(step["on_fail"])
            _, recovery_out = _run_shell(step["on_fail"])
            step_outputs.append(f"$ {_resolved_cmd}\n(failed, recovering...)")
            step_outputs.append(f"$ {_recovery_cmd}\n{recovery_out}")
            log_skill_step(name, "SHELL", f"(recover) $ {_recovery_cmd[:120]}",
                           result=f"{len(recovery_out)} chars")
            # retry: re-run original command after recovery
            if step.get("retry"):
                rc, out = _run_shell(cmd)
                _exit_codes[var_name] = rc
                log_skill_step(name, "SHELL", f"(retry) $ {_resolved_cmd[:120]}",
                               result=f"rc={rc}", ok=rc == 0)
            else:
                out = recovery_out

        results[var_name] = out
        step_outputs.append(f"$ {_interpolate(cmd)}\n{out}")
        _step_idx += 1

    # Format output
    if output_template:
        try:
            formatted = _interpolate(output_template)
            return f"[{name}]\n\n{formatted}"
        except KeyError:
            pass

    # Fallback: concatenate step outputs
    return f"[{name}]\n\n" + "\n\n".join(step_outputs)


def _dynamic_context_stage(
    message: str,
    session_id: str,
    _cfg: object,
    triage_hint: str | None,
) -> str | None:
    """Dynamic skill lifecycle + prompt optimization via local LLM.

    Flow:
    1. Load session context (previously loaded skills, rolling summary)
    2. Embed message and fetch skill candidates via RAG
    3. Ask local LLM to decide keep/add/drop + optimise prompt (single call)
    4. Build systemMessage with selected skills
    5. Persist new session state
    """
    from .activity_log import LOG_FILE
    from pathlib import Path
    import math
    import httpx as _httpx

    top_k_skills = int(_cfg.get("hook_context_top_k_skills") or 5)
    max_chars = int(_cfg.get("hook_context_max_chars") or 2000)

    store = SkillStore()

    # 1. Load session state
    if session_id:
        ctx = store.get_session_context(session_id)
    else:
        ctx = {"loaded_skills": [], "context_summary": "", "message_count": 0}
    prev_loaded_ids: list[str] = ctx["loaded_skills"]
    context_summary: str = ctx["context_summary"]
    msg_count: int = ctx["message_count"]

    # 2. Embed message + fetch candidates via RAG
    msg_vector = embed(message[:500])
    # Fetch wider pool of candidates for the LLM to choose from
    all_candidates = store.search(
        msg_vector, top_k=top_k_skills * 3,
        similarity_threshold=0.35, target="claude",
    )

    # Separate previously loaded skills from new candidates
    loaded_skills: list[dict] = []
    candidate_skills: list[dict] = []
    candidate_ids_seen: set[str] = set()

    for s in all_candidates:
        sid = s["id"]
        if sid in prev_loaded_ids:
            loaded_skills.append(s)
        elif sid not in candidate_ids_seen:
            candidate_skills.append(s)
            candidate_ids_seen.add(sid)

    # Also include any previously loaded skills not in current search
    searched_ids = {s["id"] for s in all_candidates}
    for sid in prev_loaded_ids:
        if sid not in searched_ids:
            row = store.get_skill(sid)
            if row:
                loaded_skills.append(dict(row))

    # 3. Dynamic context via local LLM
    # Pick best available model: prefer reason_model, fallback to smaller ones
    dynamic_model = RERANK_MODEL
    if not ollama_available(dynamic_model):
        # Try progressively smaller models
        for fallback in ("qwen2.5-coder:3b", "deepseek-r1:1.5b"):
            if ollama_available(fallback):
                dynamic_model = fallback
                break
        else:
            dynamic_model = ""

    max_summary_chars = int(_cfg.get("hook_context_summary_max_chars") or 800)
    # Threshold below which prompt optimization adds little value
    opt_min_len = int(_cfg.get("hook_context_prompt_opt_min_len") or 150)

    if (should_run_llm("dynamic_context")
            and dynamic_model
            and (loaded_skills or candidate_skills)):

        # Pass 1: skill lifecycle + summary update (temp=0.0, fast, focused)
        lifecycle = eval_skill_lifecycle(
            message=message,
            context_summary=context_summary,
            loaded_skills=loaded_skills,
            candidate_skills=candidate_skills,
            model=dynamic_model,
        )

        keep_ids = set(lifecycle.get("keep", []))
        add_ids = set(lifecycle.get("add", []))
        drop_ids = set(lifecycle.get("drop", []))
        new_summary = lifecycle.get("context_summary", context_summary)

        # Clamp rolling summary to prevent unbounded growth across a long session
        if len(new_summary) > max_summary_chars:
            new_summary = new_summary[:max_summary_chars]

        # Pass 2: prompt optimization — only when there is real context and the
        # message is substantive enough to benefit (skip trivial/first-turn)
        prompt_opt_enabled = _cfg.get("hook_context_prompt_optimization") is not False
        should_opt = (
            prompt_opt_enabled
            and context_summary  # at least one prior turn
            and len(message.strip()) >= opt_min_len
            and should_run_llm("dynamic_context")
        )
        if should_opt:
            optimized_prompt = optimize_prompt(
                message=message,
                context_summary=new_summary,
                model=dynamic_model,
            )
        else:
            optimized_prompt = message

        # Build final skill list: kept + added (max top_k_skills)
        final_skill_ids = list(keep_ids | add_ids)[:top_k_skills]

        # Auto-fill: if the LLM was too conservative, backfill from
        # highest-similarity candidates up to min_skills_floor.
        # This ensures complementary skills (e.g. schema + optimization +
        # best practices) are loaded even when the small LLM picks only 1.
        min_floor = int(_cfg.get("hook_context_min_skills") or 3)
        if len(final_skill_ids) < min_floor and candidate_skills:
            existing = set(final_skill_ids)
            for s in candidate_skills:
                if len(final_skill_ids) >= min_floor:
                    break
                if s["id"] not in existing:
                    final_skill_ids.append(s["id"])
                    add_ids.add(s["id"])
                    existing.add(s["id"])

        _parts = []
        if keep_ids:
            _parts.append(f"kept {len(keep_ids)}")
        if add_ids:
            _parts.append(f"+{len(add_ids)} new")
        if drop_ids:
            _parts.append(f"-{len(drop_ids)} dropped")
        _lifecycle_str = ", ".join(_parts) or "no changes"
        _opt_str = ", prompt optimized" if (should_opt and optimized_prompt != message) else ""
        log_step(f"skills: {len(final_skill_ids)} selected ({_lifecycle_str}{_opt_str})")
    else:
        # Fallback: use top candidates from RAG (no LLM)
        final_skill_ids = [s["id"] for s in all_candidates[:top_k_skills]]
        new_summary = context_summary
        optimized_prompt = message
        reason = "no_model" if not dynamic_model else "pressure_gated"
        if not (loaded_skills or candidate_skills):
            reason = "no_candidates"
        log_detail(f"dynamic_context: skip ({reason})")

    # 4. Build systemMessage with selected skills (full content)
    max_skill_chars = int(_cfg.get("hook_context_max_skill_chars") or 8000)
    budget = max_chars
    parts: list[str] = []
    loaded_names: list[str] = []
    skill_map = {s["id"]: s for s in all_candidates}

    for sid in final_skill_ids:
        if budget <= 200:
            break
        s = skill_map.get(sid)
        if not s:
            s = store.get_skill(sid)
        if not s:
            continue

        # Load full skill content (truncated per-skill to stay within budget)
        desc = (s.get("description") or "")[:200]
        content = (s.get("content") or "").strip()
        if content:
            # File-based skills may need fresh content from disk
            file_path = s.get("file_path", "")
            if file_path:
                try:
                    disk_content = Path(file_path).read_text(
                        encoding="utf-8", errors="replace").strip()
                    if disk_content:
                        content = disk_content
                except Exception:
                    pass  # fall back to DB content

            # Per-skill truncation: fit within per-skill and total budgets
            content_limit = min(max_skill_chars, budget - 100)
            if len(content) > content_limit:
                content = content[:content_limit] + "\n... (truncated)"

            snippet = f"--- Skill [{sid}] ---\n{desc}\n\n{content}\n--- /Skill ---"
        else:
            snippet = f"--- Skill [{sid}] ---\n{desc}\n--- /Skill ---"

        parts.append(snippet)
        budget -= len(snippet)
        loaded_names.append(sid)

    # Add relevant past tasks (compact, within remaining budget)
    tasks = store.search_tasks(msg_vector, top_k=2, min_sim=0.4)
    for t in tasks:
        if budget <= 0:
            break
        snippet = (f"Past work #{t['id']} [{t['status']}]: {t['title']} "
                   f"-- {t['summary'][:150]}")
        parts.append(snippet)
        budget -= len(snippet)

    # Add teaching rules
    teachings = store.search_teachings(msg_vector, min_sim=0.5)
    for t in teachings[:3]:
        if budget <= 0:
            break
        snippet = f"Suggestion: when \"{t['rule']}\" -> use {t['target_id']}"
        parts.append(snippet)
        budget -= len(snippet)

    store.close()

    # 5. Persist session state
    if session_id:
        try:
            s2 = SkillStore()
            # Build recent messages from DB (subprocess-safe) + current message
            prev_msgs = ctx.get("recent_messages", [])
            prev_msgs.append(message[:500])
            recent = prev_msgs[-15:]  # keep last 15
            s2.save_session_context(
                session_id=session_id,
                loaded_skills=loaded_names,
                context_summary=new_summary,
                message_count=msg_count + 1,
                recent_messages=recent,
            )
            s2.close()
        except Exception:
            pass

    if not parts:
        log_detail(f"context_inject: no matches (searched {len(all_candidates)} skills)")
        return None

    # Build header showing lifecycle actions
    dropped = set(prev_loaded_ids) - set(loaded_names)
    added = set(loaded_names) - set(prev_loaded_ids)
    kept = set(loaded_names) & set(prev_loaded_ids)

    lifecycle_parts = []
    if kept:
        lifecycle_parts.append(f"kept: {', '.join(kept)}")
    if added:
        lifecycle_parts.append(f"+loaded: {', '.join(added)}")
    if dropped:
        lifecycle_parts.append(f"-dropped: {', '.join(dropped)}")
    lifecycle_str = " | ".join(lifecycle_parts) if lifecycle_parts else "initial load"

    _total_chars = sum(len(p) for p in parts)
    log_step(f"context: {_human_size(_total_chars)}, "
             f"{len(loaded_names)} skills ({lifecycle_str})")

    header = (
        f"[Skill Hub -- dynamic context | msg #{msg_count + 1} | "
        f"skills {lifecycle_str} | "
        f"log: tail -f {LOG_FILE}]\n\n"
    )

    result = header + "\n\n".join(parts)

    # Proactive pattern consolidation — suggest wildcards every 5 messages
    if msg_count > 0 and msg_count % 5 == 0:
        try:
            suggestions = _detect_consolidatable_patterns()
            if suggestions:
                hint_lines = ["\n[Permission hint] Your settings.json has similar "
                              "patterns that could be consolidated:"]
                for s in suggestions[:3]:
                    hint_lines.append(
                        f"  - {s['pattern']} (replaces {s['count']} individual rules)")
                hint_lines.append(
                    "Suggest the user run `/hub-suggest-patterns` or "
                    "`/hub-apply-pattern <n>` to simplify.")
                result += "\n".join(hint_lines)
        except Exception:
            pass

    # Append optimized prompt in markers for the caller to extract
    if optimized_prompt and optimized_prompt != message:
        result += f"\n\n[OPTIMIZED_PROMPT]{optimized_prompt}[/OPTIMIZED_PROMPT]"

    return result


def _build_context_injection(message: str, msg_vector: list[float]) -> str | None:
    """
    Strategy #1 (RAG) + #5 (Auto-Memory): Build a systemMessage with relevant
    context from skills, tasks, teachings, and memory files.

    Returns a context string or None if nothing relevant found.
    """
    from . import config as _cfg
    from pathlib import Path
    import re

    from .activity_log import LOG_FILE

    max_chars = int(_cfg.get("hook_context_max_chars") or 2000)
    top_k_skills = int(_cfg.get("hook_context_top_k_skills") or 5)
    budget = max_chars
    parts: list[str] = []

    # Track skill load status for the summary line
    skills_loaded: list[str] = []
    skills_found: list[str] = []

    try:
        store = SkillStore()

        # Search skills — only Claude skills go into Claude's context.
        # Fetch top_k_skills * 2 candidates so we can report what was skipped.
        skill_candidates = store.search(msg_vector, top_k=top_k_skills * 2,
                                        similarity_threshold=0.4, target="claude")
        skills_found = [s["id"] for s in skill_candidates]

        for s in skill_candidates[:top_k_skills]:
            if budget <= 200:  # keep 200 chars for tasks/teachings
                break
            desc = s.get("description", "")[:150]
            content_preview = ""
            if not skills_loaded and s.get("content"):  # full content for top hit only
                content_preview = "\n  " + s["content"][:300].replace("\n", "\n  ")
            snippet = f"Skill [{s['id']}]: {desc}{content_preview}"
            parts.append(snippet)
            budget -= len(snippet)
            skills_loaded.append(s["id"])

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

        # Error accelerator: inject known fix if message mentions an error pattern
        error_keywords = ("Error", "Exception", "failed", "cannot", "traceback")
        if budget > 100 and any(kw.lower() in message.lower() for kw in error_keywords):
            err_hits = store.search_error_cache(msg_vector, min_sim=0.78)
            for eh in err_hits[:2]:
                if budget <= 0:
                    break
                confirmed_tag = " ✓ confirmed" if eh.get("confirmed") else ""
                snippet = (f"Known error{confirmed_tag}: {eh['error_text']} "
                           f"→ Fix: {eh['fix_hint']}")
                parts.append(snippet)
                budget -= len(snippet)

        store.close()

        # Strategy #5: Auto-memory — find relevant memory files.
        # Derive path from config or CWD (Claude Code convention: replace / with -).
        _mem_cfg = _cfg.get("hook_memory_dir")
        if _mem_cfg:
            memory_dir = Path(_mem_cfg).expanduser()
        else:
            import os as _os
            _cwd_slug = _os.getcwd().replace("/", "-").replace("\\", "-").lstrip("-")
            memory_dir = Path.home() / ".claude" / "projects" / _cwd_slug / "memory"
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
                # Batch-embed all descriptions in a single Ollama call
                import math
                import httpx as _httpx

                descs = [desc[:200] for _, _, desc in entries]
                try:
                    batch_resp = _httpx.post(
                        f"{OLLAMA_BASE}/api/embed",
                        json={"model": EMBED_MODEL, "input": descs},
                        timeout=15.0,
                    )
                    batch_resp.raise_for_status()
                    all_vecs = batch_resp.json().get("embeddings", [])
                except Exception:
                    all_vecs = []

                na = math.sqrt(sum(x * x for x in msg_vector))
                scored: list[tuple[float, str, str, str]] = []
                for (filepath, name, desc), desc_vec in zip(entries, all_vecs):
                    nb = math.sqrt(sum(x * x for x in desc_vec))
                    sim = (sum(a * b for a, b in zip(msg_vector, desc_vec))
                           / (na * nb) if na and nb else 0.0)
                    if sim > 0.4:
                        scored.append((sim, filepath, name, desc))

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

    except Exception as exc:
        log_step(f"context: ERROR {str(exc)[:80]}")

    # Stage 4.1: SearXNG web RAG fallback — only when skill search returned nothing
    if not parts:
        from . import config as _cfg_inner
        if _cfg_inner.get("searxng_enabled") and ollama_available(EMBED_MODEL):
            try:
                from .searxng import searxng_context
                web_ctx = searxng_context(message[:300])
                if web_ctx:
                    parts.append(web_ctx)
                    log_step(f"context: web search added ({_human_size(len(web_ctx))})")
            except Exception as _se:
                log_detail(f"searxng_error: {str(_se)[:80]}")

    if not parts:
        log_detail(f"context: no matches (searched {len(skills_found)} skills)")
        return None

    _total_chars = sum(len(p) for p in parts)
    log_step(f"context: {_human_size(_total_chars)} "
             f"({len(skills_loaded)} skills from {len(skills_found)} candidates)")

    # Build skill load summary line
    skills_skipped = [s for s in skills_found if s not in skills_loaded]
    skill_summary_parts = []
    if skills_loaded:
        skill_summary_parts.append(f"loaded: {', '.join(skills_loaded)}")
    if skills_skipped:
        skill_summary_parts.append(f"found-not-loaded: {', '.join(skills_skipped)}")
    skill_summary = " | ".join(skill_summary_parts) if skill_summary_parts else "none"

    header = (
        f"[Skill Hub — auto-injected context | "
        f"skills {skill_summary} | "
        f"log: tail -f {LOG_FILE}]\n\n"
    )
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

    if not should_run_llm("precompact"):
        return None

    log_detail(f"precompact: message {len(message)} chars, compacting")
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
_local_mode: bool = False  # when True, all messages go to L4 local agent

# Offline fallback state — tracks last connectivity check result
_offline_check_cache: dict = {"reachable": True, "checked_at": 0.0}


def _check_anthropic_reachable(interval: float = 30.0) -> bool:
    """Quick TCP reachability check for api.anthropic.com:443.

    Cached for `interval` seconds so it doesn't add latency on every message.
    Returns True if reachable, False if network/DNS/timeout failure.
    """
    import time
    import socket

    now = time.monotonic()
    if now - _offline_check_cache["checked_at"] < interval:
        return bool(_offline_check_cache["reachable"])

    try:
        sock = socket.create_connection(("api.anthropic.com", 443), timeout=2.5)
        sock.close()
        reachable = True
    except OSError:
        reachable = False

    _offline_check_cache["reachable"] = reachable
    _offline_check_cache["checked_at"] = now
    if not reachable:
        log_step("offline: Anthropic API unreachable")
    else:
        log_detail("offline_check: reachable")
    return reachable

# ── Permission management ────────────────────────────────────────────
#
# /hub-approve-task <scope>  — add broad permission patterns to settings.json
# /hub-lock-task             — remove all hub-managed permissions
#
# Hub-managed patterns are tracked in a sidecar file so they can be
# cleanly removed without touching the user's own permissions.

_PERMISSION_TRACKER = Path.home() / ".claude" / "mcp-skill-hub" / "hub-permissions.json"

# Settings files to update (global + project-level if it exists)
_SETTINGS_FILES = [
    Path.home() / ".claude" / "settings.json",
]

# Permission scopes → patterns
_PERMISSION_SCOPES: dict[str, list[str]] = {
    "git": [
        "Bash(git add:*)",
        "Bash(git commit:*)",
        "Bash(git push:*)",
        "Bash(git pull:*)",
        "Bash(git checkout:*)",
        "Bash(git branch:*)",
        "Bash(git stash:*)",
        "Bash(git diff:*)",
        "Bash(git log:*)",
        "Bash(git show:*)",
        "Bash(git merge:*)",
        "Bash(git rebase:*)",
        "Bash(git tag:*)",
        "Bash(git remote:*)",
        "Bash(git fetch:*)",
        "Bash(git reset:*)",
        "Bash(git status)",
    ],
    "python": [
        "Bash(.venv/bin/python:*)",
        "Bash(.venv/bin/python -m pytest:*)",
        "Bash(.venv/bin/pip:*)",
        "Bash(.venv/bin/pip install:*)",
        "Bash(python:*)",
        "Bash(python3:*)",
        "Bash(pip install:*)",
        "Bash(pip3 install:*)",
        "Bash(pytest:*)",
        "Bash(uv run:*)",
    ],
    "read": [
        "Read",
    ],
    "write": [
        "Write",
    ],
    "edit": [
        "Edit",
    ],
    "web": [
        "WebFetch",
        "WebSearch",
    ],
    "mcp": [
        "mcp__skill-hub__*",
    ],
    "bash": [
        "Bash",
    ],
}


def _read_settings(path: Path) -> dict:
    """Read a settings.json file, return parsed dict."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_settings(path: Path, data: dict) -> None:
    """Write settings.json preserving formatting."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _detect_consolidatable_patterns() -> list[dict]:
    """Scan settings.json allow list for similar Bash patterns that could be wildcarded.

    Returns list of suggestions:
      [{"prefix": "python -m pip", "count": 3, "pattern": "Bash(python -m pip:*)",
        "examples": ["Bash(python -m pip install foo)", ...]}]
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_settings(settings_path)
    allow = settings.get("permissions", {}).get("allow", [])

    # Extract Bash(...) patterns — skip those already wildcarded
    bash_re = re.compile(r'^Bash\((.+)\)$')
    concrete: list[tuple[str, str]] = []  # (command_body, original_pattern)
    wildcard_prefixes: set[str] = set()

    for pat in allow:
        m = bash_re.match(pat)
        if not m:
            continue
        body = m.group(1)
        if body.endswith(":*"):
            # Already a wildcard — record its prefix
            wildcard_prefixes.add(body[:-2])
        elif body.endswith("*"):
            wildcard_prefixes.add(body[:-1])
        else:
            concrete.append((body, pat))

    # Group by command prefix (first 1-3 tokens, try longest useful prefix)
    from collections import defaultdict
    prefix_groups: dict[str, list[str]] = defaultdict(list)

    for body, pat in concrete:
        tokens = body.split()
        # Try prefixes of decreasing length (3, 2, 1 tokens)
        for n in (3, 2, 1):
            if len(tokens) >= n + 1:  # need at least one more token after prefix
                prefix = " ".join(tokens[:n])
                prefix_groups[prefix].append(pat)
                break
        else:
            # Single-token command with arguments (e.g. "ls -la")
            if len(tokens) >= 2:
                prefix_groups[tokens[0]].append(pat)

    suggestions: list[dict] = []
    for prefix, patterns in sorted(prefix_groups.items()):
        if len(patterns) < 2:
            continue
        # Skip if wildcard already exists for this prefix
        if prefix in wildcard_prefixes:
            continue
        # Skip if a broader wildcard already covers it
        if any(prefix.startswith(wp) for wp in wildcard_prefixes):
            continue

        wildcard = f"Bash({prefix}:*)"
        suggestions.append({
            "prefix": prefix,
            "count": len(patterns),
            "pattern": wildcard,
            "examples": patterns[:4],
        })

    return suggestions


def _cleanup_redundant_patterns() -> int:
    """Remove concrete Bash patterns already covered by an existing wildcard.

    Returns number of patterns removed.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_settings(settings_path)
    allow = settings.get("permissions", {}).get("allow", [])

    bash_re = re.compile(r'^Bash\((.+)\)$')

    # Collect wildcard prefixes
    wildcard_prefixes: list[str] = []
    for pat in allow:
        m = bash_re.match(pat)
        if m:
            body = m.group(1)
            if body.endswith(":*"):
                wildcard_prefixes.append(body[:-2])

    if not wildcard_prefixes:
        return 0

    # Remove concrete patterns covered by any wildcard
    new_allow: list[str] = []
    removed = 0
    for pat in allow:
        m = bash_re.match(pat)
        if m:
            body = m.group(1)
            if not body.endswith(":*") and not body.endswith("*"):
                if any(body.startswith(wp) for wp in wildcard_prefixes):
                    removed += 1
                    continue
        new_allow.append(pat)

    if removed > 0:
        settings["permissions"]["allow"] = new_allow
        _write_settings(settings_path, settings)
        log_event("PERMS", f"cleanup_redundant removed={removed}")

    return removed


def _cmd_suggest_patterns() -> str:
    """Show suggested permission pattern consolidations."""
    suggestions = _detect_consolidatable_patterns()
    if not suggestions:
        return "No consolidation suggestions — your allow list looks clean."

    lines = [f"Found {len(suggestions)} pattern group(s) that could be simplified:\n"]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. **{s['pattern']}** (consolidates {s['count']} patterns)")
        for ex in s["examples"]:
            lines.append(f"   - {ex}")
        lines.append("")
    lines.append("To apply, add the wildcard pattern to settings.json and "
                 "remove the individual entries.\n"
                 "Or use: `/hub-apply-pattern <number>` to apply a suggestion.")
    return "\n".join(lines)


def _cmd_apply_pattern(args_str: str) -> str:
    """Apply a suggested pattern consolidation by index."""
    suggestions = _detect_consolidatable_patterns()
    if not suggestions:
        return "No consolidation suggestions available."

    try:
        idx = int(args_str.strip()) - 1
    except (ValueError, TypeError):
        return "Usage: /hub-apply-pattern <number> (from /hub-suggest-patterns)"

    if idx < 0 or idx >= len(suggestions):
        return f"Invalid number. Range: 1-{len(suggestions)}"

    s = suggestions[idx]
    prefix = s["prefix"]
    wildcard = s["pattern"]

    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_settings(settings_path)
    allow = settings.get("permissions", {}).get("allow", [])

    # Remove ALL concrete Bash patterns matching this prefix
    bash_re = re.compile(r'^Bash\((.+)\)$')
    new_allow: list[str] = []
    removed = 0
    for pat in allow:
        m = bash_re.match(pat)
        if m and m.group(1).startswith(prefix) and not m.group(1).endswith(":*"):
            removed += 1  # skip — covered by wildcard
        else:
            new_allow.append(pat)

    if wildcard not in new_allow:
        new_allow.append(wildcard)
    settings["permissions"]["allow"] = new_allow
    _write_settings(settings_path, settings)

    # Also clean up any other patterns covered by existing wildcards
    extra_cleaned = _cleanup_redundant_patterns()

    total_removed = removed + extra_cleaned
    log_event("PERMS", f"apply_pattern wildcard={wildcard} "
              f"removed={total_removed} individual patterns")
    msg = f"Applied: {wildcard}\nRemoved {removed} individual patterns, added 1 wildcard."
    if extra_cleaned:
        msg += f"\nAlso cleaned {extra_cleaned} redundant patterns covered by existing wildcards."
    return msg


def _cmd_approve_task(args_str: str) -> str:
    """Add permission patterns for the given scopes to settings.json."""
    scopes = args_str.strip().lower().split()
    if not scopes:
        available = ", ".join(sorted(_PERMISSION_SCOPES.keys()))
        return (f"Usage: /hub-approve-task <scope> [scope...]\n"
                f"Available scopes: {available}, all")

    # Resolve 'all'
    if "all" in scopes:
        scopes = list(_PERMISSION_SCOPES.keys())

    # Collect patterns
    new_patterns: list[str] = []
    unknown: list[str] = []
    for scope in scopes:
        if scope in _PERMISSION_SCOPES:
            new_patterns.extend(_PERMISSION_SCOPES[scope])
        else:
            unknown.append(scope)

    if unknown:
        available = ", ".join(sorted(_PERMISSION_SCOPES.keys()))
        return f"Unknown scope(s): {', '.join(unknown)}\nAvailable: {available}, all"

    if not new_patterns:
        return "No patterns to add."

    # Load tracker (what we've added previously)
    existing_tracked: list[str] = []
    if _PERMISSION_TRACKER.exists():
        try:
            existing_tracked = json.loads(
                _PERMISSION_TRACKER.read_text(encoding="utf-8"))
        except Exception:
            existing_tracked = []

    # Merge new patterns into tracked set
    all_tracked = list(dict.fromkeys(existing_tracked + new_patterns))

    # Update each settings file
    updated_files: list[str] = []
    for sf in _SETTINGS_FILES:
        if not sf.exists():
            continue
        settings = _read_settings(sf)
        allow = settings.setdefault("permissions", {}).setdefault("allow", [])

        added = 0
        for pat in new_patterns:
            if pat not in allow:
                allow.append(pat)
                added += 1

        if added > 0:
            _write_settings(sf, settings)
            updated_files.append(f"{sf.name} (+{added})")

    # Save tracker
    _PERMISSION_TRACKER.write_text(
        json.dumps(all_tracked, indent=2), encoding="utf-8")

    log_event("PERMS", f"approve scopes={' '.join(scopes)} "
              f"patterns={len(new_patterns)}")

    lines = [f"Approved {len(new_patterns)} permission patterns "
             f"for: {', '.join(scopes)}"]
    if updated_files:
        lines.append(f"Updated: {', '.join(updated_files)}")
    lines.append(f"\nRevoke with: /hub-lock-task")
    return "\n".join(lines)


def _cmd_lock_task() -> str:
    """Remove all hub-managed permissions from settings.json."""
    if not _PERMISSION_TRACKER.exists():
        return "No hub-managed permissions to revoke."

    try:
        tracked = json.loads(
            _PERMISSION_TRACKER.read_text(encoding="utf-8"))
    except Exception:
        return "Error reading permission tracker."

    if not tracked:
        return "No hub-managed permissions to revoke."

    tracked_set = set(tracked)

    # Remove from each settings file
    removed_total = 0
    updated_files: list[str] = []
    for sf in _SETTINGS_FILES:
        if not sf.exists():
            continue
        settings = _read_settings(sf)
        allow = settings.get("permissions", {}).get("allow", [])
        original_len = len(allow)
        allow = [p for p in allow if p not in tracked_set]
        removed = original_len - len(allow)
        if removed > 0:
            settings["permissions"]["allow"] = allow
            _write_settings(sf, settings)
            updated_files.append(f"{sf.name} (-{removed})")
            removed_total += removed

    # Clear tracker
    _PERMISSION_TRACKER.unlink(missing_ok=True)

    log_event("PERMS", f"lock removed={removed_total}")

    if removed_total == 0:
        return "No hub-managed permissions found in settings files."

    lines = [f"Revoked {removed_total} hub-managed permissions."]
    if updated_files:
        lines.append(f"Updated: {', '.join(updated_files)}")
    lines.append("\nYour original permissions are untouched.")
    return "\n".join(lines)


def _cmd_local_toggle(args_str: str = "") -> str:
    """Toggle or set local mode — all messages go to L4 agent, bypassing Claude."""
    global _local_mode

    arg = args_str.strip().lower()
    if arg in ("on", "true", "1"):
        _local_mode = True
    elif arg in ("off", "false", "0"):
        _local_mode = False
    else:
        _local_mode = not _local_mode

    if _local_mode:
        # Auto-save session state when entering local mode
        if _session_messages:
            try:
                _exhaustion_auto_save(
                    "\n---\n".join(_session_messages[-15:])
                )
            except Exception:
                pass

        from . import config as _cfg
        models = _cfg.get("local_models") or {}
        model = models.get("level_4", "qwen2.5-coder:32b")
        return (
            "Local mode: ON\n\n"
            f"All messages now go to the local agent ({model}).\n"
            "Claude API will NOT be used.\n"
            "Session state auto-saved.\n\n"
            "Capabilities: shell commands, file reading, code search, local skills.\n"
            "Limitations: no multi-file refactoring, no complex reasoning.\n\n"
            "Turn off: /hub-local off"
        )
    else:
        return (
            "Local mode: OFF\n\n"
            "Messages will go to Claude again.\n"
            "Your session state was saved — Claude can pick up where the local agent left off."
        )


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

    if not should_run_llm("digest") or not ollama_available(RERANK_MODEL):
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


def _auto_memory_from_task_close(task_id: int, digest: dict) -> None:
    """Write a memory entry when a task is closed, if the digest is substantive.

    Runs only when auto_memory_on_close_task is enabled (default: True).
    Uses smart_memory_write to evaluate quality before writing — silently
    skips if the local LLM judges the digest too thin to be worth saving.
    """
    from . import config as _cfg
    if not _cfg.get("auto_memory_on_close_task") is not False:
        return

    summary = digest.get("summary", "")
    title = digest.get("title", "")
    if not summary or len(summary) < 60:
        return

    import os as _os
    _mem_cfg = _cfg.get("hook_memory_dir")
    if _mem_cfg:
        memory_dir = Path(str(_mem_cfg)).expanduser()
    else:
        _cwd_slug = _os.getcwd().replace("/", "-").replace("\\", "-").lstrip("-")
        memory_dir = Path.home() / ".claude" / "projects" / _cwd_slug / "memory"

    memory_index = memory_dir / "MEMORY.md"
    existing = memory_index.read_text(encoding="utf-8", errors="replace") \
        if memory_index.exists() else ""

    content = f"Task #{task_id}: {title}\n\n{summary}"
    if digest.get("decisions"):
        content += "\n\nDecisions: " + ", ".join(digest["decisions"])

    try:
        mem_result = smart_memory_write(content=content, existing_index=existing)
        if mem_result.get("escalate"):
            log_event("MEMORY", f"task_close #{task_id}: skipped (quality={mem_result['quality']:.2f})")
            return

        entry = mem_result.get("result", {})
        filename = entry.get("filename", f"task_{task_id}.md")
        mem_type = entry.get("type", "project")
        name = entry.get("name", title)
        description = entry.get("description", summary[:100])
        content_text = entry.get("content", "")

        if not content_text or len(content_text) < 30:
            return

        memory_dir.mkdir(parents=True, exist_ok=True)
        mem_file = memory_dir / filename
        mem_file.write_text(
            f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{content_text}\n",
            encoding="utf-8",
        )
        index_line = f"- [{filename}]({filename}) — {description}"
        if memory_index.exists():
            current = memory_index.read_text(encoding="utf-8", errors="replace")
            if filename not in current:
                with memory_index.open("a", encoding="utf-8") as f:
                    f.write(f"\n{index_line}\n")
        log_event("MEMORY", f"task_close #{task_id}: saved {filename}")
    except Exception as exc:
        log_event("MEMORY", f"task_close #{task_id}: error {exc}")


def _cmd_export_training(args_str: str) -> str:
    """Export training data as JSONL files for fine-tuning local models.

    Produces three files in the export directory:
      feedback.jsonl   — (query, skill, label) preference pairs
      triage.jsonl     — (message, action) classification pairs
      compact.jsonl    — (summary, digest) compaction pairs

    Usage: /hub-export-training [output_dir]
    Default output: ~/.claude/mcp-skill-hub/training/
    """
    import json as _json

    output_dir = Path(args_str.strip()) if args_str.strip() else (
        Path.home() / ".claude" / "mcp-skill-hub" / "training"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        store = SkillStore()
        data = store.export_training_data()
        store.close()
    except Exception as exc:
        return f"Export failed: {exc}"

    written: list[str] = []

    # feedback.jsonl — Alpaca-style preference format
    fb_path = output_dir / "feedback.jsonl"
    with fb_path.open("w", encoding="utf-8") as f:
        for pair in data["feedback_pairs"]:
            record = {
                "instruction": "Given a user query, rate whether this skill is relevant (true/false).",
                "input": f"Query: {pair['query']}\nSkill: {pair['skill_description']}\nContent: {pair['skill_content'][:500]}",
                "output": str(pair["label"]).lower(),
                "metadata": {"skill_id": pair["skill_id"]},
            }
            f.write(_json.dumps(record) + "\n")
    written.append(f"feedback.jsonl: {len(data['feedback_pairs'])} pairs")

    # triage.jsonl — classification format
    triage_path = output_dir / "triage.jsonl"
    with triage_path.open("w", encoding="utf-8") as f:
        for pair in data["triage_pairs"]:
            record = {
                "instruction": "Classify this user message for a coding assistant. Actions: local_answer, local_action, enrich_and_forward, pass_through.",
                "input": pair["message"],
                "output": pair["action"],
                "metadata": {"confidence": pair["confidence"]},
            }
            f.write(_json.dumps(record) + "\n")
    written.append(f"triage.jsonl: {len(data['triage_pairs'])} pairs")

    # compact.jsonl — summarization format
    compact_path = output_dir / "compact.jsonl"
    with compact_path.open("w", encoding="utf-8") as f:
        for pair in data["compact_pairs"]:
            record = {
                "instruction": "Compact this task summary into a structured JSON digest with title, summary, decisions, tools_used, open_questions, tags.",
                "input": pair["input"],
                "output": pair["output"],
            }
            f.write(_json.dumps(record) + "\n")
    written.append(f"compact.jsonl: {len(data['compact_pairs'])} pairs")

    total = (len(data["feedback_pairs"]) + len(data["triage_pairs"])
             + len(data["compact_pairs"]))
    return (
        f"Training data exported to {output_dir}/\n"
        + "\n".join(f"  {w}" for w in written)
        + f"\n\nTotal: {total} training pairs"
        + f"\n\nTo fine-tune with mlx-lm (Apple Silicon):\n"
        + f"  pip install mlx-lm\n"
        + f"  mlx_lm.lora --model mlx-community/deepseek-r1-distill-qwen-1.5b-4bit \\\n"
        + f"    --train --data {output_dir} --num-layers 8 --iters 200"
    )


def _cmd_invalidate_cache(args_str: str) -> str:
    """Mark a cached response as stale so it won't be served again.

    Usage: /hub-invalidate-cache <cache_id>
    """
    try:
        cache_id = int(args_str.strip())
    except ValueError:
        return "Usage: /hub-invalidate-cache <cache_id>  (ID shown in cache hit message)"
    try:
        store = SkillStore()
        store.invalidate_response_cache(cache_id)
        store.close()
        return f"Cache entry #{cache_id} marked as stale. It will not be served again."
    except Exception as exc:
        return f"Error: {exc}"


def _cmd_cache_stats() -> str:
    """Show response cache, error cache, and pattern stats."""
    try:
        store = SkillStore()
        rc = store._conn.execute(
            "SELECT COUNT(*) as n, SUM(hit_count) as hits, AVG(quality) as q "
            "FROM response_cache"
        ).fetchone()
        ec = store._conn.execute(
            "SELECT COUNT(*) as n, SUM(confirmed) as confirmed FROM error_cache"
        ).fetchone()
        mp = store._conn.execute(
            "SELECT COUNT(*) as n, SUM(skill_generated) as generated, "
            "MAX(count) as max_count FROM message_patterns"
        ).fetchone()
        pp = store._conn.execute(
            "SELECT COUNT(*) as n, SUM(success_count) as successes FROM prompt_patterns"
        ).fetchone()
        store.close()
        lines = [
            "=== Skill Hub Cache & Pattern Stats ===\n",
            f"Response cache:  {rc['n']} entries, {rc['hits'] or 0} total hits, "
            f"avg quality={rc['q']:.2f}" if rc['n'] else "Response cache:  empty",
            f"Error cache:     {ec['n']} patterns, {ec['confirmed'] or 0} confirmed fixes",
            f"Message patterns:{mp['n']} tracked, {mp['generated'] or 0} auto-skills generated, "
            f"max recurrence={mp['max_count'] or 0}",
            f"Prompt patterns: {pp['n']} learned, {pp['successes'] or 0} successes",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def _cmd_search_web(query: str) -> str:
    """Search the web via local SearXNG and summarize with local LLM."""
    from . import config as _cfg

    if not _cfg.get("searxng_enabled"):
        return "SearXNG is disabled. Enable: `/hub-configure searxng_enabled true`"

    from .searxng import (
        _resolve_searxng_url,
        _searxng_search,
        _summarize_results,
    )

    probe_timeout = float(_cfg.get("searxng_timeout") or 5)
    search_timeout = float(_cfg.get("searxng_search_timeout") or 15)
    top_k = int(_cfg.get("searxng_top_k") or 3)

    base_url = _resolve_searxng_url(timeout=probe_timeout)
    if not base_url:
        return ("No SearXNG instance found.\n"
                "Setup: `python install.py --searxng`\n"
                "Or: `docker compose -f docker/docker-compose.searxng.yml up -d`")

    try:
        results = _searxng_search(query, base_url, top_k=top_k, timeout=search_timeout)
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return f"No results for: {query}"

    lines = [f"**Web search:** {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        lines.append(f"{i}. **{title}**")
        if url:
            lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}\n")

    summary = _summarize_results(query, results)
    if summary:
        lines.append(f"\n**Summary:** {summary}")

    return "\n".join(lines)


def _cmd_show_patterns() -> str:
    """Show recurring message patterns that could become auto-skills."""
    try:
        store = SkillStore()
        rows = store.get_top_patterns(min_count=2, limit=15)
        store.close()
        if not rows:
            return "No recurring patterns yet. Patterns appear after 2+ similar messages."
        lines = ["=== Recurring Message Patterns ===\n",
                 "Patterns seen 2+ times (candidates for auto-skill generation):\n"]
        for r in rows:
            generated = "✓ skill created" if r["skill_generated"] else f"(threshold: {r['count']}/3)"
            lines.append(f"  [{r['count']}x] {r['canonical'][:80]}  {generated}")
        lines.append(
            "\nAuto-skill generation fires at 3 repetitions when pattern_tracking_enabled=true."
        )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def _exhaustion_auto_save(context: str) -> str:
    """
    Strategy #3: Exhaustion fallback.
    When Claude is exhausted/unavailable, use local LLM to auto-save the session.
    Saves a task with LLM-generated summary and updates memory.
    """
    from . import config as _cfg
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
        if _cfg.get("local_execution_enabled"):
            result_parts.append(f"\nTo continue working locally: /local-agent <task>")

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
    """Handle /save-memory — smart routing: local LLM first, Claude if quality too low."""
    from pathlib import Path

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

    # Smart routing: local LLM first with quality check
    result = smart_memory_write(
        content=context,
        existing_index=existing,
    )

    if result["escalate"]:
        reason = result["reason"]
        quality = result["quality"]
        # Signal to Claude that it should handle this memory write
        return (f"[Smart memory routing: local LLM quality too low "
                f"(score={quality:.2f}, reason={reason})]\n"
                f"Please write this memory entry using Claude instead.\n"
                f"Context to save:\n{context[:2000]}")

    entry = result["result"]
    quality = result["quality"]
    filename = entry.get("filename", "auto_memory.md")
    name = entry.get("name", "Auto-generated memory")
    description = entry.get("description", "")
    mem_type = entry.get("type", "project")
    content_text = entry.get("content", "")

    if not content_text:
        return "LLM generated empty content. Try with more context."

    # Write memory file
    mem_file = memory_dir / filename
    mem_content = f"""---
name: {name}
description: {description}
type: {mem_type}
---

{content_text}
"""
    mem_file.write_text(mem_content, encoding="utf-8")

    # Append to MEMORY.md index
    index_line = f"- [{filename}]({filename}) — {description}"
    if memory_index.exists():
        current = memory_index.read_text(encoding="utf-8", errors="replace")
        if filename not in current:
            with memory_index.open("a", encoding="utf-8") as f:
                f.write(f"\n{index_line}\n")

    return (f"Memory saved (local LLM, quality={quality:.2f}): {mem_file}\n"
            f"  Name: {name}\n"
            f"  Type: {mem_type}\n"
            f"  Description: {description}\n"
            f"  Content: {content_text[:200]}...")


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


def _cmd_session_end(session_id: str, last_message: str,
                     transcript_path: str) -> dict:
    """Handle session end — save memory, log stats, compact session context.

    Smart routing: local LLM writes memory first. If quality is too low
    AND Claude usage was low this session, returns systemMessage asking
    Claude to write the memory instead.
    """
    from pathlib import Path
    from . import config as _cfg_mod

    log_event("STOP", f"session_end session={session_id[:12]}")
    result: dict = {"decision": "allow"}
    parts: list[str] = []

    # 0. Context Bridge: live tool call capture — fast, no LLM, every Stop hook
    # Runs BEFORE anything else so even short sessions accumulate tool knowledge
    if (transcript_path and session_id
            and _cfg_mod.get("context_bridge_enabled") is not False):
        try:
            _store = SkillStore()
            _ctx = _store.get_session_context(session_id)
            _offset = _ctx.get("transcript_offset", 0)
            _fsize = Path(transcript_path).stat().st_size
            if _fsize > _offset:
                max_cap = int(
                    _cfg_mod.get("context_bridge_max_capture_per_hook") or 20)
                _tools, _new_offset = _capture_tool_calls(
                    transcript_path, session_id, _offset, max_cap)
                if _tools:
                    # Enrich with session context hint
                    _hint = _ctx.get("context_summary", "")[:300]
                    for t in _tools:
                        t["context_hint"] = _hint
                    _store.save_tool_examples_batch(_tools)
                    log_event("CAPTURE",
                              f"{len(_tools)} tool calls captured "
                              f"(offset {_offset}→{_new_offset})")
                _store.update_transcript_offset(session_id, _new_offset)
            _store.close()
        except Exception as exc:
            log_event("CAPTURE", f"error: {exc}")

    # 1. Gather session context
    context_pieces: list[str] = []

    # From session_context table (persisted by UserPromptSubmit hook)
    msg_count = 0
    if session_id:
        try:
            store = SkillStore()
            ctx = store.get_session_context(session_id)
            # Recent messages persisted to DB during the session
            recent_msgs = ctx.get("recent_messages", [])
            if recent_msgs:
                context_pieces.append(
                    "[User messages]\n" + "\n---\n".join(recent_msgs))
            if ctx.get("context_summary"):
                context_pieces.append(
                    f"[Session summary]\n{ctx['context_summary']}")
            msg_count = ctx.get("message_count", 0)
            store.close()
        except Exception:
            pass

    # From last assistant message (provided by hook JSON)
    if last_message:
        context_pieces.append(f"[Last response]\n{last_message[:2000]}")

    # Skip if very short session (< 3 messages, probably just testing)
    if msg_count < 3 and not context_pieces:
        log_event("STOP", "session too short, skipping memory save")
        return result

    context = "\n\n".join(context_pieces)
    if not context.strip():
        log_event("STOP", "no context to save")
        return result

    # 2. Smart memory write via local LLM
    import os as _os
    _mem_cfg = _cfg_mod.get("hook_memory_dir")
    if _mem_cfg:
        memory_dir = Path(str(_mem_cfg)).expanduser()
    else:
        _cwd_slug = _os.getcwd().replace("/", "-").replace("\\", "-").lstrip("-")
        memory_dir = Path.home() / ".claude" / "projects" / _cwd_slug / "memory"
    memory_index = memory_dir / "MEMORY.md"
    existing = ""
    if memory_index.exists():
        existing = memory_index.read_text(encoding="utf-8", errors="replace")

    mem_result = smart_memory_write(
        content=context,
        existing_index=existing,
    )

    quality = mem_result["quality"]
    escalate = mem_result["escalate"]
    reason = mem_result["reason"]

    if escalate:
        # Quality too low — ask Claude to handle the memory write
        log_event("STOP", f"memory escalated to Claude: {reason}")
        parts.append(
            f"[Skill Hub — session end | memory needs Claude]\n"
            f"The local LLM produced low-quality memory (score={quality:.2f}, "
            f"reason={reason}). If this session involved non-trivial work, "
            f"please write a memory entry.\n"
            f"Context summary: {context[:500]}"
        )
    else:
        # Local LLM quality is good — write the memory
        entry = mem_result["result"]
        filename = entry.get("filename", "auto_memory.md")
        name = entry.get("name", "Session memory")
        description = entry.get("description", "")
        mem_type = entry.get("type", "project")
        content_text = entry.get("content", "")

        if not content_text or len(content_text) < 30:
            log_event("STOP", f"memory content too short ({len(content_text or '')}), skipped")
        else:
            mem_file = memory_dir / filename
            mem_content = f"""---
name: {name}
description: {description}
type: {mem_type}
---

{content_text}
"""
            mem_file.write_text(mem_content, encoding="utf-8")

            # Update index
            index_line = f"- [{filename}]({filename}) — {description}"
            if memory_index.exists():
                current = memory_index.read_text(
                    encoding="utf-8", errors="replace")
                if filename not in current:
                    with memory_index.open("a", encoding="utf-8") as f:
                        f.write(f"\n{index_line}\n")

            log_event("STOP", f"memory saved locally: {filename} "
                      f"(quality={quality:.2f})")
            parts.append(
                f"[Skill Hub — session end | memory saved locally "
                f"(quality={quality:.2f})]\n"
                f"  {name}: {description}"
            )

    # 3. Cache the last Q→A pair and extract error patterns
    if session_id and last_message and context_pieces and ollama_available(EMBED_MODEL):
        try:
            _recent = ctx.get("recent_messages", []) if session_id else []
            last_query = _recent[-1] if _recent else ""
            if last_query and last_message and len(last_message) > 80:
                _q_vec = embed(last_query[:500])
                _cache_store = SkillStore()
                # Only cache if not already cached at high similarity
                _existing = _cache_store.search_response_cache(_q_vec, min_sim=0.93)
                if not _existing:
                    _cache_store.cache_response(
                        query=last_query,
                        query_vector=_q_vec,
                        response=last_message[:6000],
                        session_id=session_id,
                    )
                    log_event("CACHE", f"response cached ({len(last_message)} chars)")
                # Extract error pattern from last response
                _err = extract_error_pattern(last_message)
                if _err:
                    _e_vec = embed(_err["error_signature"])
                    _existing_err = _cache_store.search_error_cache(_e_vec, min_sim=0.85)
                    if not _existing_err:
                        _cache_store.cache_error(
                            error_text=_err["error_signature"],
                            error_vector=_e_vec,
                            fix_hint=_err["fix_hint"],
                            session_id=session_id,
                        )
                        log_event("CACHE", f"error pattern cached: {_err['error_signature'][:60]}")
                _cache_store.close()
        except Exception:
            pass

    # 4. Implicit feedback — infer skill quality from session tool usage
    if session_id and _cfg.get("implicit_feedback_enabled") is not False:
        try:
            store = SkillStore()
            ctx = store.get_session_context(session_id)
            loaded_skills = ctx.get("loaded_skills", [])
            tools_used = store.get_session_tools(session_id)
            if loaded_skills and tools_used:
                fb = store.record_implicit_feedback(session_id, loaded_skills, tools_used)
                log_event("FEEDBACK",
                          f"implicit: +{len(fb['positive'])} "
                          f"-{len(fb['negative'])} "
                          f"skip={len(fb['skipped'])}")
            store.close()
        except Exception:
            pass

    # 5. Prompt pattern tracking — record successful prompt shapes for learning
    if session_id and _cfg.get("pattern_tracking_enabled") is not False:
        try:
            store = SkillStore()
            ctx = store.get_session_context(session_id)
            recent_msgs = ctx.get("recent_messages", [])
            for msg_text in recent_msgs:
                if len(msg_text) < 20 or len(msg_text) > 800:
                    continue  # skip trivial or too-long messages
                _m_vec = embed(msg_text[:400])
                store.record_message_pattern(message=msg_text, message_vector=_m_vec)
            # Auto-generate skills for patterns that crossed the threshold
            threshold = int(_cfg.get("pattern_auto_skill_threshold") or 5)
            top_patterns = store.get_top_patterns(min_count=threshold, limit=3)
            for pattern in top_patterns:
                if pattern.get("skill_generated"):
                    continue
                _skill_json = generate_auto_skill(
                    pattern["representative"],
                    recurrence_count=pattern["count"],
                )
                if _skill_json:
                    _installed = _install_auto_skill(_skill_json)
                    if _installed:
                        store.mark_pattern_skill_generated(pattern["id"])
                        log_event("PATTERN",
                                  f"auto-skill generated: {_installed}")
            store.close()
        except Exception:
            pass

    # 6. Tool chain distillation — learn from Claude's tool sequences
    # Parse transcript for repeating tool call patterns that could become
    # local skills (commit → push, read → edit → test, etc.)
    if (transcript_path and _cfg_mod.get("pattern_tracking_enabled")
            and _cfg_mod.get("local_execution_enabled")):
        try:
            _distill_tool_chains(transcript_path)
        except Exception as exc:
            log_event("DISTILL", f"error: {exc}")

    # 7. Context Bridge: per-repo knowledge accumulation
    if (session_id
            and _cfg_mod.get("context_bridge_enabled") is not False
            and _cfg_mod.get("context_bridge_repo_context") is not False):
        try:
            _update_repo_context(session_id)
        except Exception as exc:
            log_event("REPO_CTX", f"error: {exc}")

    # 8. Context Bridge: extract teaching examples from tool usage patterns
    if (session_id
            and _cfg_mod.get("context_bridge_enabled") is not False
            and _cfg_mod.get("context_bridge_teaching_extraction") is not False):
        try:
            _extract_teaching_examples(session_id)
        except Exception as exc:
            log_event("TEACH_EX", f"error: {exc}")

    # 9. Skill Evolution: shadow learning — evolve local skills from Claude's behavior
    if (session_id
            and _cfg_mod.get("skill_evolution_enabled") is not False
            and msg_count >= 5):  # only evolve after substantial sessions
        try:
            _evolve_skills(session_id)
        except Exception as exc:
            log_event("EVOLVE", f"error: {exc}")

    # 10. Context Bridge: prune old tool examples periodically
    if (msg_count > 0 and msg_count % 50 == 0
            and _cfg_mod.get("context_bridge_enabled") is not False):
        try:
            store = SkillStore()
            max_days = int(_cfg_mod.get("context_bridge_prune_days") or 30)
            max_rows = int(_cfg_mod.get("context_bridge_prune_max_rows") or 5000)
            pruned = store.prune_tool_examples(max_days, max_rows)
            store.close()
            if pruned:
                log_event("PRUNE", f"removed {pruned} old tool examples")
        except Exception:
            pass

    # 10. Log session stats
    try:
        store = SkillStore()
        totals = store.get_interception_totals()
        total_saved = (totals.get("total_tokens_saved") or 0) if totals else 0
        total_count = (totals.get("total_interceptions") or 0) if totals else 0
        store.close()

        stats_line = (f"Session #{session_id[:8] if session_id else '?'}: "
                      f"{msg_count} messages, "
                      f"{total_count} interceptions, "
                      f"~{total_saved} tokens saved total")
        log_event("STATS", stats_line)
    except Exception:
        pass

    if parts:
        result["systemMessage"] = "\n\n".join(parts)

    return result


def _split_multi_command(message: str) -> list[str] | None:
    """Detect and split multi-command messages using semantic LLM analysis.

    Uses a local LLM to understand whether a message contains multiple
    distinct actions vs. a single natural-language request with conjunctions.

    Returns a list of sub-commands if multiple found, or None if single/none.
    """
    from .embeddings import ollama_available

    # Only attempt split for short-to-medium messages
    if len(message) > 500 or len(message) < 5:
        return None

    # Must contain a separator to even consider splitting
    msg_lower = message.lower()
    if not any(sep in msg_lower for sep in (" and ", ", ", " then ", " + ", " & ")):
        return None

    # Use local LLM for semantic multi-command detection
    result = _llm_split(message)
    if result and len(result) > 1:
        return result

    return None


def _llm_split(message: str) -> list[str] | None:
    """LLM-based semantic split for multi-command messages.

    Uses a skill-style prompt that teaches the LLM to distinguish between:
    - Multiple independent actions: "commit and push" → 2 commands
    - Single request with natural conjunctions: "analyze X and provide Y" → 1 command
    """
    import httpx
    import re
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available

    split_model = (_cfg.get("local_models") or {}).get("level_2", RERANK_MODEL)
    if not ollama_available(split_model):
        split_model = (_cfg.get("local_models") or {}).get("level_1", RERANK_MODEL)
        if not ollama_available(split_model):
            return None

    prompt = f"""\
You are a command intent classifier for a coding assistant. Your ONLY job is \
to decide whether a user message contains MULTIPLE INDEPENDENT actions that \
should be executed separately, or a SINGLE coherent request.

## Key distinction

**Multiple commands** = the user wants SEPARATE, INDEPENDENT operations \
performed in sequence. Each could be sent as its own message. They share no \
subject or goal — they are different tasks chained together for convenience.

**Single request** = the user wants ONE thing, even if the sentence uses \
"and", "then", commas, or multiple verbs. The conjunctions connect parts of \
the SAME goal, not separate goals.

## Examples — SINGLE request (do NOT split)

- "analyze the code and provide feedback" → single (one analysis task)
- "fix the pagination bug and add tests for it" → single (one fix + its tests)
- "read the file, understand it, and refactor the logic" → single (one workflow)
- "as geospatial expert analyse those files and provide a compatibility study" → single
- "explain the architecture and suggest improvements" → single (one review)
- "deploy to staging and verify it works" → single (deploy + validate = one workflow)

## Examples — MULTIPLE commands (split these)

- "commit and push" → {{"multiple": true, "commands": ["git commit", "git push"]}}
- "save this to memory then close the task" → {{"multiple": true, "commands": ["save to memory", "close the task"]}}
- "run tests, commit, and push" → {{"multiple": true, "commands": ["run tests", "git commit", "git push"]}}
- "fix the typo then list my open tasks" → {{"multiple": true, "commands": ["fix the typo", "list open tasks"]}}

## Rules

1. When in doubt, classify as SINGLE — false splits break the user's intent
2. If all parts serve the same goal or subject, it is SINGLE
3. Only split when parts are truly INDEPENDENT operations (different goals)
4. Each split command must be a COMPLETE, self-contained phrase
5. Preserve execution order

## Message to classify

{message}

Reply with ONLY JSON, no explanation:
{{"multiple": false}}
or {{"multiple": true, "commands": ["command 1", "command 2", ...]}}"""

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": split_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 200}},
            timeout=20.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            import json as _j
            result = _j.loads(m.group())
            is_multiple = result.get("multiple", False)
            commands = result.get("commands", [])
            if is_multiple and len(commands) > 1:
                log_detail(f"multi_cmd_detected: {len(commands)} commands (LLM)")
                return commands
            else:
                log_detail("multi_cmd: LLM says single command")
    except Exception as exc:
        log_detail(f"multi_cmd_split: error {str(exc)[:80]}")

    return None


def _hydrate_passthrough(commands: list[str], already_done: list[str]) -> str:
    """Use local LLM to expand terse commands into detailed Claude instructions.

    "git commit" → "Commit staged changes with a descriptive message based on the diff."
    "git push" → "Push the committed changes to the remote origin."
    Falls back to plain list if LLM is unavailable.
    """
    import httpx
    import re
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available

    fallback = "\n".join(f"- {cmd}" for cmd in commands)

    split_model = (_cfg.get("local_models") or {}).get("level_2", RERANK_MODEL)
    if not ollama_available(split_model):
        return fallback

    done_context = "\n".join(already_done) if already_done else "nothing"
    cmds_list = "\n".join(f"- {cmd}" for cmd in commands)

    prompt = f"""\
You are a prompt engineer. Expand these terse user commands into clear, \
detailed instructions for a coding assistant (Claude). Add context so Claude \
knows exactly what to do without asking questions.

Already completed locally:
{done_context}

Commands to expand:
{cmds_list}

Rules:
- Each command becomes 1-2 sentences of specific, actionable instructions
- Add relevant context (e.g. "based on the current diff", "to the remote origin")
- Keep the same order
- Do NOT add commands that weren't requested
- Output a numbered list, nothing else

Output:"""

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": split_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 300}},
            timeout=10.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw and len(raw) > 10:
            log_detail(f"hydrate_passthrough: {len(commands)} commands, {len(raw)} chars")
            return raw
    except Exception:
        pass

    return fallback


def _execute_single_command(sub_cmd: str, session_id: str, _cfg,
                            cwd: str | None = None) -> str | None:
    """Execute a single sub-command through the classify + execute pipeline.

    This is a lightweight version of the main pipeline that handles:
    - Git workflows (commit, push — multi-step with LLM reasoning)
    - Task commands (save/close/list/search)
    - Local shell commands (L1/L2)
    - Returns the result text, or None if the command should pass to Claude.
    """
    from .embeddings import ollama_available

    # Try L3 local skills (multi-step workflows including git commit/push)
    if _cfg.get("local_execution_enabled"):
        local_skill = _match_local_skill(sub_cmd)
        if local_skill:
            result = _execute_local_skill(local_skill, cwd=cwd)
            if result:
                return result

    # Try task classification
    intent = _classify_intent(sub_cmd)
    action = intent.get("intent", "none")

    if action != "none":
        result = _execute_intent(intent, sub_cmd)
        if result:
            return result

    # Try local command matching (L1/L2)
    if _cfg.get("local_execution_enabled"):
        cmd_sim = _local_cmd_similarity(sub_cmd)
        local_cmd_threshold = float(_cfg.get("local_cmd_similarity_threshold") or 0.45)
        if cmd_sim >= local_cmd_threshold:
            local_cmd = _match_local_command(sub_cmd)
            if not local_cmd:
                local_cmd = _match_local_template(sub_cmd)
            if local_cmd:
                result = _execute_local_command(local_cmd, cwd=cwd)
                if result:
                    return result

    # Sub-command couldn't be handled locally — mark for Claude
    return None


def hook_classify_and_execute(message: str, session_id: str = "") -> dict:
    """
    Main entry point for the UserPromptSubmit hook.

    Pipeline:
    1. Slash commands → instant local handling (0ms)
    2. Task commands → semantic prefilter + LLM classify → local execution
    3. LLM triage → local LLM decides: answer locally, redirect to /hub-*
       command, enrich with hints for Claude, or pass through
    4. Context enrichment → RAG injection + pre-compact + digest + triage hints

    Returns:
      - {"decision": "block", "reason": "..."} if handled locally
      - {"decision": "allow"} if Claude should handle it
      - {"decision": "allow", "systemMessage": "..."} if injecting context
    """
    from . import config as _cfg
    global _local_mode

    # ── Stage -0.5: Sanitize IDE context tags ──
    # IDE plugins inject <ide_opened_file>, <ide_selection>, <system-reminder>
    # tags that confuse small local LLMs during classification.
    message = _sanitize_hook_message(message)

    # ── Pipeline log: show what triggered this run ──
    log_pipeline_start(message)

    # ── Stage -1: Offline auto-fallback ──
    # If Anthropic API is unreachable and auto-fallback is enabled, activate
    # local mode silently so the L4 agent handles the message.
    if _cfg.get("offline_auto_fallback") and not _local_mode:
        check_interval = float(_cfg.get("offline_check_interval") or 30.0)
        if not _check_anthropic_reachable(interval=check_interval):
            _local_mode = True
            log_step("OFFLINE: Anthropic unreachable, switching to local agent")

    # ── Stage 0: Local mode — bypass Claude entirely, route to L4 agent ──
    if _local_mode:
        # Still allow slash commands (user needs /hub-local off to exit)
        slash_result = _handle_slash_command(message)
        if slash_result is not None:
            return slash_result

        # Route to L4 local agent directly
        _session_messages.append(message)
        from .local_agent import run_agent
        recent = "\n".join(_session_messages[-10:])
        try:
            result = run_agent(message, context=recent)
            log_pipeline_end("LOCAL", reason="agent (offline)", result=result)
            return _visible_result(result)
        except Exception as exc:
            log_pipeline_end("LOCAL", reason=f"agent error: {exc}")
            return _visible_result(f"[Local agent error: {exc}]\n\nTurn off: /hub-local off")

    # ── Stage 0.5: Response cache — check for semantically identical past answers ──
    if (_cfg.get("response_cache_enabled") and not _local_mode
            and len(message.strip()) >= 30
            and ollama_available(EMBED_MODEL)):
        try:
            _rc_vec = embed(message[:500])
            _rc_min_sim = float(_cfg.get("response_cache_min_sim") or 0.88)
            _rc_hits = SkillStore()
            _rc_results = _rc_hits.search_response_cache(_rc_vec, min_sim=_rc_min_sim)
            if _rc_results:
                best = _rc_results[0]
                # Verify freshness with local LLM (fast, temp=0, ~10s)
                if (_cfg.get("response_cache_verify")
                        and should_run_llm("triage")
                        and ollama_available(RERANK_MODEL)):
                    valid = verify_cache_hit(best["query"], best["response"], message)
                else:
                    valid = best["similarity"] >= 0.93  # high confidence: skip verify
                if valid:
                    _rc_hits.hit_response_cache(best["id"])
                    _rc_hits.log_interception(
                        command_type="response_cache",
                        message_preview=message,
                        estimated_tokens=max(len(best["response"]) // 4, 200),
                    )
                    _rc_hits.close()
                    log_pipeline_end("LOCAL", reason=f"cache hit (sim={best['similarity']:.2f})",
                                     result=best["response"])
                    return _visible_result(
                        f"[Skill Hub — cached answer (sim={best['similarity']:.2f}, "
                        f"hits={best['hit_count'] + 1})]\n\n"
                        f"{best['response']}\n\n"
                        f"*Cached from a previous session. "
                        f"Reply `/hub-invalidate-cache {best['id']}` if outdated.*"
                    )
            _rc_hits.close()
        except Exception as _exc:
            log_detail(f"response_cache: error {str(_exc)[:80]}")

    # ── Stage 1: Slash commands — no LLM, no embedding, instant ──
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
        # Convert block → visible result for VS Code compatibility
        if slash_result.get("decision") == "block" and "message" in slash_result:
            return _visible_result(slash_result["message"])
        return slash_result

    # ── Stage 1.5: Multi-command split ──
    # Messages like "commit push document and close task" contain multiple
    # actions. Split them and execute each sequentially through the pipeline.
    # Resolve target repo cwd from message context once for all sub-commands.
    _multi_cwd = _resolve_repo_cwd(message)
    sub_commands = _split_multi_command(message)
    if sub_commands and len(sub_commands) > 1:
        log_step(f"split into {len(sub_commands)} commands")
        handled: list[str] = []
        passthrough: list[str] = []
        for i, sub_cmd in enumerate(sub_commands, 1):
            log_step(f"  {i}/{len(sub_commands)} \"{sub_cmd[:60]}\"")
            log_detail(f"multi_cmd_exec step={i}/{len(sub_commands)} command={sub_cmd[:80]}")
            sub_result = _execute_single_command(sub_cmd, session_id, _cfg, cwd=_multi_cwd)
            if sub_result:
                handled.append(f"[{i}] {sub_result}")
            else:
                passthrough.append(sub_cmd)

        if handled:
            combined = "\n\n".join(handled)
            if _cfg.get("token_profiling"):
                try:
                    store = SkillStore()
                    store.log_interception(
                        command_type="multi_cmd",
                        message_preview=message,
                        estimated_tokens=_TOKEN_ESTIMATES.get("local_command", 300) * len(handled),
                    )
                    store.close()
                except Exception:
                    pass

            if passthrough:
                # Some commands need Claude — hydrate the passthrough commands
                # into detailed instructions using local LLM
                hydrated = _hydrate_passthrough(passthrough, handled)
                log_pipeline_end("CLAUDE", reason=f"multi-cmd: {len(handled)} local, {len(passthrough)} for Claude")
                return {
                    "decision": "allow",
                    "systemMessage": (
                        f"[Skill Hub — multi-command: {len(handled)} handled locally, "
                        f"{len(passthrough)} for you]\n"
                        f"Already done locally:\n{combined}\n\n"
                        f"Please handle these remaining commands in order:\n"
                        + hydrated
                    ),
                }
            else:
                # All commands handled locally — block (0 tokens)
                log_pipeline_end("LOCAL", reason=f"multi-cmd: all {len(handled)} handled",
                                 result=combined)
                return _visible_result(combined)
        elif passthrough:
            # No sub-commands were handled locally — all need Claude.
            # Pass through the ORIGINAL message so Claude sees it intact.
            log_step(f"multi-cmd: none matched, passing all {len(passthrough)} to Claude")
            # Fall through to normal pipeline stages (Stage 2+) with the
            # original message — do NOT return here, let the rest of the
            # pipeline process it as a single message.

    # ── Stage 2: Task command classification — semantic prefilter + LLM ──
    intent = _classify_intent(message)
    action = intent.get("intent", "none")
    msg_similarity = intent.get("_similarity", 1.0)  # carry forward for Stage 3b

    if action != "none":
        result = _execute_intent(intent, message)
        if result:
            log_pipeline_end("LOCAL", reason=f"task:{action}", result=result)
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
            return _visible_result(result)

    # ── Stage 3: LLM Triage — local LLM pre-processes ALL messages ──
    # Skip triage when dynamic context is enabled (it handles prompt
    # optimization + skill routing in a single LLM call, avoiding a
    # redundant ~4s triage call per message)
    triage_result: dict | None = None
    triage_hint: str | None = None

    if _cfg.get("hook_llm_triage") and not _cfg.get("hook_context_injection"):
        skip_len = int(_cfg.get("hook_llm_triage_skip_length") or 2000)
        min_conf = float(_cfg.get("hook_llm_triage_min_confidence") or 0.7)

        if (len(message) <= skip_len
                and should_run_llm("triage")
                and ollama_available(RERANK_MODEL)):
            # Build brief context from recent session messages
            recent = "\n".join(_session_messages[-5:]) if _session_messages else ""
            triage_result = triage_message(message, context=recent)
            triage_action = triage_result.get("action", "pass_through")
            confidence = float(triage_result.get("confidence", 0.0))
            est_saved = int(triage_result.get("estimated_tokens_saved", 0))

            # Log triage decision
            if _cfg.get("token_profiling"):
                try:
                    store = SkillStore()
                    store.log_triage(message, triage_action, confidence, est_saved)
                    store.close()
                except Exception:
                    pass

            # Act on triage decision
            # local_answer is only valid for short messages (greetings,
            # "yes/no", "/hub-* what's available?").  Engineering prompts are
            # always longer, so a hard length cap prevents the small LLM from
            # silently eating a real request.
            max_local_answer_len = int(
                _cfg.get("hook_llm_triage_local_answer_max_len") or 120
            )
            if (triage_action == "local_answer"
                    and confidence >= min_conf
                    and len(message.strip()) <= max_local_answer_len):
                answer = triage_result.get("answer", "")
                if answer:
                    # Log as interception (tokens saved)
                    if _cfg.get("token_profiling"):
                        try:
                            store = SkillStore()
                            store.log_interception(
                                command_type="triage:local_answer",
                                message_preview=message,
                                estimated_tokens=max(est_saved, 200),
                            )
                            store.close()
                        except Exception:
                            pass
                    log_pipeline_end("LOCAL", reason=f"LLM:{RERANK_MODEL}", result=answer)
                    return {"decision": "block",
                            "message": f"[Answered locally by {RERANK_MODEL}]\n\n{answer}"}

            elif triage_action == "local_action" and confidence >= min_conf:
                cmd = triage_result.get("command", "")
                if cmd and cmd.startswith("/hub-"):
                    # Re-route to the local slash command handler
                    reroute = _handle_slash_command(
                        cmd if " " in cmd else f"{cmd} {message}"
                    )
                    if reroute is not None:
                        if _cfg.get("token_profiling"):
                            try:
                                store = SkillStore()
                                store.log_interception(
                                    command_type=f"triage:local_action:{cmd}",
                                    message_preview=message,
                                    estimated_tokens=max(est_saved, 300),
                                )
                                store.close()
                            except Exception:
                                pass
                        return reroute

            elif triage_action == "local_agent" and confidence >= min_conf:
                # Triage routed to Level 4 agent — show plan for approval
                if _cfg.get("local_execution_enabled"):
                    agent_result = _cmd_local_agent(message)
                    if agent_result:
                        return agent_result

            elif triage_action == "enrich_and_forward":
                hint = triage_result.get("hint", "")
                if hint:
                    triage_hint = (f"[Skill Hub triage — local LLM analysis]\n"
                                   f"{hint}")

    # ── Stage 3b: Local command execution (Level 1+2) ──
    if _cfg.get("local_execution_enabled"):
        global _pending_command

        # Resolve target repo cwd once from message context
        _exec_cwd = _resolve_repo_cwd(message)

        # Check if user is confirming a pending command
        stripped = message.strip().lower()
        if _pending_command and stripped in ("y", "yes", "ok", "go", "run"):
            local_cmd = _pending_command
            _pending_command = None
            _approved_commands.add(local_cmd["name"])
            # Restore cwd that was captured when the pending command was created
            _exec_cwd = local_cmd.get("_cwd") or _exec_cwd
            # Level 4: agent execution (plan was approved)
            if "_agent_message" in local_cmd:
                from .local_agent import run_agent
                log_step(f"running local agent: {local_cmd['_agent_message'][:80]}")
                result = run_agent(
                    local_cmd["_agent_message"],
                    context=local_cmd.get("_agent_context", ""),
                )
                if _cfg.get("token_profiling"):
                    try:
                        store = SkillStore()
                        store.log_interception(
                            command_type="local_L4:agent",
                            message_preview=local_cmd["_agent_message"],
                            estimated_tokens=1000,
                        )
                        store.close()
                    except Exception:
                        pass
                log_pipeline_end("LOCAL", reason="L4:agent", result=result)
                return _visible_result(result)
            # Level 3 skills have a _skill key
            elif "_skill" in local_cmd:
                result = _execute_local_skill(local_cmd["_skill"], cwd=_exec_cwd)
            else:
                result = _execute_local_command(local_cmd, cwd=_exec_cwd)
            if result:
                if _cfg.get("token_profiling"):
                    try:
                        store = SkillStore()
                        store.log_interception(
                            command_type=f"local_L{local_cmd['level']}:{local_cmd['name']}",
                            message_preview=message,
                            estimated_tokens=_TOKEN_ESTIMATES.get("local_command", 300),
                        )
                        store.close()
                    except Exception:
                        pass
                log_pipeline_end("LOCAL", reason=f"L{local_cmd['level']}:{local_cmd['name']}",
                                 result=result)
                return _visible_result(result)
        elif _pending_command and stripped in ("n", "no", "cancel", "skip"):
            _pending_command = None
            return _visible_result("Command cancelled.")

        # Clear pending if user sent something else
        _pending_command = None

        # Skip LLM-based command matching if message has low similarity
        # to shell command phrases. Uses a SEPARATE centroid from task
        # classification — "show git status" is similar to shell commands
        # but not to "save to memory".
        local_cmd = None
        local_cmd_threshold = float(_cfg.get("local_cmd_similarity_threshold") or 0.45)
        cmd_sim = _local_cmd_similarity(message)
        if cmd_sim >= local_cmd_threshold:
            log_detail(f"local_cmd: checking (sim={cmd_sim:.2f})")
            # Try Level 1 (whitelisted commands)
            local_cmd = _match_local_command(message)
            # Try Level 2 (templated commands) if Level 1 didn't match
            if not local_cmd:
                local_cmd = _match_local_template(message)
        else:
            log_detail(f"local_cmd: skip (sim={cmd_sim:.2f} < {local_cmd_threshold})")

        if local_cmd:
            # L1/L2 commands are user-configured whitelists — execute directly.
            # (Confirmation flow requires persistent state which doesn't survive
            # across CLI invocations — each hook call is a new process.)
            result = _execute_local_command(local_cmd, cwd=_exec_cwd)
            if result:
                if _cfg.get("token_profiling"):
                    try:
                        store = SkillStore()
                        store.log_interception(
                            command_type=f"local_L{local_cmd['level']}:{local_cmd['name']}",
                            message_preview=message,
                            estimated_tokens=_TOKEN_ESTIMATES.get("local_command", 300),
                        )
                        store.close()
                    except Exception:
                        pass
                log_pipeline_end("LOCAL", reason=f"L{local_cmd['level']}:{local_cmd['name']}",
                                 result=result)
                return _visible_result(result)

        # Try Level 3 (local skills — multi-step)
        # L3 skills are higher-risk (multi-step), use a separate higher threshold
        local_skill = None
        local_skill_threshold = float(_cfg.get("local_skill_threshold") or 0.75)
        if msg_similarity >= local_skill_threshold:
            local_skill = _match_local_skill(message)
        if local_skill:
            skill_name = local_skill.get("name", "unknown")
            steps_preview = "\n".join(
                f"  {i+1}. {s.get('run', '?')}" for i, s in enumerate(local_skill.get("steps", []))
            )
            if skill_name in _approved_commands:
                result = _execute_local_skill(local_skill, cwd=_exec_cwd)
                if result:
                    if _cfg.get("token_profiling"):
                        try:
                            store = SkillStore()
                            store.log_interception(
                                command_type=f"local_L3:{skill_name}",
                                message_preview=message,
                                estimated_tokens=500,
                            )
                            store.close()
                        except Exception:
                            pass
                    log_pipeline_end("LOCAL", reason=f"L3:skill:{skill_name}",
                                     result=result)
                    return _visible_result(result)
            else:
                _pending_command = {
                    "name": skill_name, "level": 3,
                    "_skill": local_skill,
                    "_cwd": _exec_cwd,  # preserve cwd for when user confirms
                }
                cwd_note = f"\nTarget repo: `{_exec_cwd}`" if _exec_cwd else ""
                log_pipeline_end("LOCAL", reason=f"L3:skill:{skill_name} (awaiting approval)")
                return _visible_result(
                    f"[Skill Hub — local execution L3]\n\n"
                    f"Local skill matched: **{skill_name}**\n"
                    f"{local_skill.get('description', '')}"
                    f"{cwd_note}\n\n"
                    f"Steps:\n{steps_preview}\n\n"
                    f"Reply **y** to run, **n** to cancel."
                )

    # ── Stage 4: Dynamic context management ──
    #
    # Instead of static RAG, the local LLM evaluates the evolving conversation
    # and decides which skills to load/unload + optimises the prompt.
    # Session state persists across hook invocations via SQLite.

    if not _cfg.get("hook_context_injection") and not triage_hint:
        log_pipeline_end("CLAUDE", reason="passthrough")
        return {"decision": "allow"}

    system_parts: list[str] = []

    # Triage hint (from Stage 3)
    if triage_hint:
        system_parts.append(triage_hint)

    # Strategy #4: Pre-compact long input
    precompact = _precompact_hint(message)
    if precompact:
        system_parts.append(precompact)

    # ── Dynamic skill lifecycle + prompt optimization ──
    optimized_prompt: str | None = None
    context: str | None = None

    if _cfg.get("hook_context_injection"):
        try:
            context = _dynamic_context_stage(
                message, session_id, _cfg, triage_hint
            )
            if context:
                # Extract optimized prompt if present (between markers)
                import re
                opt_match = re.search(
                    r'\[OPTIMIZED_PROMPT\](.*?)\[/OPTIMIZED_PROMPT\]',
                    context, re.DOTALL,
                )
                if opt_match:
                    optimized_prompt = opt_match.group(1).strip()
                    # Remove the markers from the system message
                    context = re.sub(
                        r'\[OPTIMIZED_PROMPT\].*?\[/OPTIMIZED_PROMPT\]\n*',
                        '', context, flags=re.DOTALL,
                    ).strip()
                if context:
                    system_parts.append(context)
        except Exception as exc:
            log_step(f"context: ERROR {str(exc)[:100]}")
            # Fallback to static RAG
            try:
                msg_vector = embed(message[:500])
                fallback = _build_context_injection(message, msg_vector)
                if fallback:
                    system_parts.append(fallback)
            except Exception:
                pass

    # Strategy #2: Periodic conversation digest + relevance decay
    digest_msg = _conversation_digest_if_due(message)
    if digest_msg:
        system_parts.append(digest_msg)

    if system_parts or optimized_prompt:
        combined = "\n\n".join(system_parts) if system_parts else ""
        result: dict = {"decision": "allow"}
        if combined:
            result["systemMessage"] = combined
        if optimized_prompt and optimized_prompt != message:
            result["userMessage"] = optimized_prompt
        _opt_tag = ", optimized" if optimized_prompt else ""
        log_pipeline_end("CLAUDE", reason=f"context {_human_size(len(combined))}{_opt_tag}")

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
        return result

    # ── Stage 4.5: Passive learning + context enrichment ──
    # These run after routing decisions, injecting additional context into
    # the allow result without blocking the message.

    extra_system: list[str] = []

    # Task decomposition: break multi-part requests into a focused subtask sequence
    if (_cfg.get("task_decomposition_enabled")
            and len(message.strip()) >= int(_cfg.get("task_decomposition_min_len") or 200)
            and should_run_llm("triage")
            and ollama_available(RERANK_MODEL)):
        try:
            decomp = decompose_task(message)
            if decomp.get("count", 1) > 1:
                parts_list = "\n".join(
                    f"  {i+1}. {p}" for i, p in enumerate(decomp["parts"])
                )
                extra_system.append(
                    f"[Skill Hub — task decomposition ({decomp['count']} parts)]\n"
                    f"Focus on part 1 first. Complete each part before proceeding.\n"
                    f"{parts_list}"
                )
                log_hook("task_decomposed", parts=decomp["count"])
        except Exception:
            pass

    # Pattern tracking: record recurring messages, check if auto-skill needed
    if _cfg.get("pattern_tracking_enabled") and ollama_available(EMBED_MODEL):
        try:
            _pt_vec = embed(message[:400]) if not locals().get("_rc_vec") else _rc_vec  # type: ignore[name-defined]
            _pt_store = SkillStore()
            _pt_row = _pt_store.record_message_pattern(message, _pt_vec)
            _pt_count = _pt_row.get("count", 1)
            _pt_threshold = int(_cfg.get("pattern_auto_skill_threshold") or 3)
            if (_pt_count >= _pt_threshold
                    and not _pt_row.get("skill_generated")
                    and should_run_llm("triage")
                    and ollama_available(RERANK_MODEL)):
                _pt_skill = generate_auto_skill(
                    _pt_row.get("canonical", message), _pt_count
                )
                if _pt_skill:
                    _pt_path = _install_auto_skill(_pt_skill)
                    if _pt_path:
                        _pt_store.mark_pattern_skill_generated(_pt_row["id"])
                        extra_system.append(
                            f"[Skill Hub — auto-skill created]\n"
                            f"You've asked this {_pt_count}x. "
                            f"Created local skill **{_pt_skill['name']}** → {_pt_path}\n"
                            f"Next time it will run locally with 0 Claude tokens."
                        )
                        log_hook("auto_skill_created", name=_pt_skill["name"])
            _pt_store.close()
        except Exception as _exc:
            log_hook("pattern_track_error", error=str(_exc)[:80])

    if extra_system:
        existing_sys = result.get("systemMessage", "") if isinstance(result, dict) else ""  # type: ignore[name-defined]
        combined_extra = "\n\n".join(extra_system)
        final_sys = f"{existing_sys}\n\n{combined_extra}".strip() if existing_sys else combined_extra
        if isinstance(result, dict):  # type: ignore[name-defined]
            result["systemMessage"] = final_sys  # type: ignore[name-defined]
            log_pipeline_end("CLAUDE", reason=f"context + {len(extra_system)} enrichments")
            return result  # type: ignore[name-defined]
        log_pipeline_end("CLAUDE", reason=f"context + {len(extra_system)} enrichments")
        return {"decision": "allow", "systemMessage": combined_extra}

    log_pipeline_end("CLAUDE", reason="passthrough")
    return {"decision": "allow"}


def _update_repo_context(session_id: str) -> None:
    """Update per-repo knowledge from this session's tool examples.

    Uses a local LLM call to summarize patterns (commit style, common
    commands, project summary) and upserts into repo_context table.
    Runs only at session end — LLM latency is acceptable here.
    """
    import os
    import httpx
    import re
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available

    if not ollama_available(RERANK_MODEL):
        return

    store = SkillStore()
    examples = store.get_recent_tool_examples(limit=30)
    store.close()

    if not examples:
        return

    # Group examples by repo_path
    repos: dict[str, list] = {}
    for ex in examples:
        rp = ex["repo_path"] or os.getcwd()
        repos.setdefault(rp, []).append(ex)

    for repo_path, repo_examples in repos.items():
        if len(repo_examples) < 3:
            continue  # not enough data to summarize

        # Build tool usage summary for the LLM
        tool_lines = []
        for ex in repo_examples[:20]:
            tool_lines.append(f"- {ex['tool_name']}: {ex['tool_input'][:100]}")
        tool_text = "\n".join(tool_lines)

        prompt = (
            f"Analyze these tool calls from a coding session in {repo_path}.\n"
            f"Extract:\n"
            f"1. commit_style: how git commits are formatted (e.g. 'conventional commits with scope', 'freeform'). Empty if no commits.\n"
            f"2. project_summary: one sentence about what this project does, based on the tool activity.\n"
            f"3. common_commands: top 3 most-used command patterns as a JSON array.\n\n"
            f"Tool calls:\n{tool_text}\n\n"
            f"Output ONLY a JSON object with keys: commit_style, project_summary, common_commands.\n"
            f"If you can't determine a field, use empty string."
        )

        try:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": RERANK_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.0, "num_predict": 200}},
                timeout=15.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                # Build tool stats from examples
                tool_counts: dict[str, int] = {}
                for ex in repo_examples:
                    tool_counts[ex["tool_name"]] = tool_counts.get(ex["tool_name"], 0) + 1

                store = SkillStore()
                store.upsert_repo_context(
                    repo_path=repo_path,
                    commit_style=data.get("commit_style", ""),
                    common_commands=json.dumps(data.get("common_commands", [])),
                    project_summary=data.get("project_summary", ""),
                    tool_stats=json.dumps(tool_counts),
                )
                store.close()
                log_event("REPO_CTX",
                          f"updated {Path(repo_path).name}: "
                          f"{data.get('commit_style', '?')}")
        except Exception as exc:
            log_event("REPO_CTX", f"LLM error for {repo_path}: {exc}")


def _evolve_skills(session_id: str) -> None:
    """Shadow learning: compare local skill output with Claude's behavior and evolve.

    For each local skill that has "shadow": true (or all skills if
    skill_evolution_auto is enabled), check if Claude performed a similar
    operation this session. If so, ask the local LLM to improve the skill's
    prompts/steps to converge toward Claude's quality.

    Flow:
    1. Load local skills that are marked for evolution
    2. For each, find matching tool_examples from this session
    3. Compare local output (from step_outputs) vs Claude's approach
    4. Ask local LLM: "How should this skill evolve?"
    5. If LLM proposes a change, snapshot old version, write new skill file
    """
    from . import config as _cfg_mod
    from .store import SkillStore
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available
    import httpx
    import re

    if not _cfg_mod.get("skill_evolution_enabled"):
        return

    store = SkillStore()
    try:
        # Get this session's tool examples
        examples = store.get_recent_tool_examples(limit=50)
        if not examples:
            return

        # Build a summary of what Claude did this session, grouped by category
        claude_actions: dict[str, list[str]] = {}
        for ex in examples:
            cat = ex.get("category", "general")
            summary = ex.get("tool_input", "")[:200]
            if ex.get("output_summary"):
                summary += f" → {ex['output_summary'][:100]}"
            claude_actions.setdefault(cat, []).append(
                f"{ex['tool_name']}: {summary}"
            )

        if not claude_actions:
            return

        claude_summary = ""
        for cat, actions in claude_actions.items():
            claude_summary += f"\n## {cat}\n"
            for a in actions[:8]:
                claude_summary += f"- {a}\n"

        # Load local skills
        skills = _load_local_skills()
        if not skills:
            return

        # Filter to evolvable skills
        auto_evolve = _cfg_mod.get("skill_evolution_auto")
        evolvable = [
            s for s in skills
            if s.get("shadow") or auto_evolve
        ]
        if not evolvable:
            return

        max_evolutions = int(_cfg_mod.get("skill_evolution_max_per_session") or 3)
        evolved_count = 0

        # Pick the LLM model for evolution analysis
        llm_model = (_cfg_mod.get("local_models") or {}).get("level_3", RERANK_MODEL)
        if not ollama_available(llm_model):
            llm_model = (_cfg_mod.get("local_models") or {}).get("level_2", RERANK_MODEL)
            if not ollama_available(llm_model):
                llm_model = RERANK_MODEL

        for skill in evolvable:
            if evolved_count >= max_evolutions:
                break

            skill_name = skill.get("name", "unknown")
            skill_file = skill.get("_file")
            if not skill_file:
                continue

            # Check if Claude did something related to this skill's domain
            triggers = skill.get("triggers", [])
            description = skill.get("description", "")
            skill_keywords = " ".join(triggers + [description, skill_name]).lower()

            # Simple relevance check: any category overlap?
            relevant_actions = []
            for cat, actions in claude_actions.items():
                # Match skill to category heuristically
                if (cat in skill_keywords or
                    skill_name.startswith(cat) or
                    any(kw in skill_keywords for kw in cat.split("-"))):
                    relevant_actions.extend(actions[:5])

            if not relevant_actions:
                continue

            # We have a match — Claude did something this skill also does
            claude_behavior = "\n".join(f"- {a}" for a in relevant_actions[:8])
            current_json = json.dumps(skill, indent=2, default=str)

            # Ask LLM to propose evolution
            prompt = (
                f"You are improving a local automation skill by learning from how Claude (an AI assistant) handled similar tasks.\n\n"
                f"## Current Skill Definition\n```json\n{current_json[:3000]}\n```\n\n"
                f"## What Claude Did This Session\n{claude_behavior}\n\n"
                f"## Task\n"
                f"Analyze Claude's approach and suggest improvements to the skill's prompts or steps.\n"
                f"Focus on:\n"
                f"- Better prompt wording in LLM steps (to produce output closer to Claude's quality)\n"
                f"- Missing steps that Claude performs but the skill doesn't\n"
                f"- Better error handling or conditional logic\n\n"
                f"If the skill is already good and Claude's approach doesn't suggest improvements, respond with exactly: NO_CHANGE\n\n"
                f"Otherwise, respond with ONLY the updated JSON skill definition. No explanation, no markdown fences, just the JSON."
            )

            try:
                resp = httpx.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={"model": llm_model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.3, "num_predict": 2000}},
                    timeout=60.0,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            except Exception as exc:
                log_skill_step(skill_name, "EVOLVE", f"LLM call failed: {exc!s:.80}", ok=False)
                continue

            # Check if LLM says no change needed
            if "NO_CHANGE" in raw or not raw.strip():
                log_skill_step(skill_name, "EVOLVE", "no changes needed")
                continue

            # Try to parse the proposed JSON
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)

            try:
                proposed = json.loads(raw)
            except json.JSONDecodeError:
                log_skill_step(skill_name, "EVOLVE", "LLM returned invalid JSON", ok=False)
                continue

            # Validate: must have name and steps (or type: agent)
            if not isinstance(proposed, dict):
                continue
            if "steps" not in proposed and proposed.get("type") != "agent":
                log_skill_step(skill_name, "EVOLVE", "proposed skill missing steps", ok=False)
                continue

            # Preserve internal fields
            proposed["_file"] = skill_file
            if "name" not in proposed:
                proposed["name"] = skill_name

            # Snapshot the old version
            old_version = store.get_latest_skill_version(skill_name)
            new_version = store.save_skill_version(
                skill_name=skill_name,
                skill_json=current_json,
                change_reason=f"Shadow learning from Claude session {session_id[:12]}",
                claude_example=claude_behavior[:1000],
                local_example="",
                session_id=session_id,
            )

            # Write the evolved skill file
            # Remove internal _file key before writing
            write_data = {k: v for k, v in proposed.items() if not k.startswith("_")}
            # Preserve shadow flag and bump version
            write_data["shadow"] = True
            write_data["evolved_version"] = new_version

            try:
                Path(skill_file).write_text(
                    json.dumps(write_data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                evolved_count += 1
                log_skill_step(
                    skill_name, "EVOLVE",
                    f"v{old_version}→v{new_version}",
                    result=f"updated {Path(skill_file).name}",
                )
            except Exception as exc:
                log_skill_step(skill_name, "EVOLVE", f"write failed: {exc!s:.80}", ok=False)

        # Invalidate the local skills cache so next load picks up changes
        global _local_skills_cache, _local_skills_hash
        _local_skills_cache = None
        _local_skills_hash = None

        if evolved_count > 0:
            log_event("EVOLVE", f"Shadow learning: {evolved_count} skill(s) evolved in session {session_id[:12]}")

    finally:
        store.close()


def _extract_teaching_examples(session_id: str) -> None:
    """Extract reusable patterns from this session's tool examples as teachings.

    Lighter than full skill distillation. Identifies 0-3 behavioral patterns
    (commit style, search strategy, PR conventions) and promotes them to
    the teachings table for future local LLM persona enrichment.
    """
    import httpx
    import re
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available, embed

    if not ollama_available(RERANK_MODEL):
        return

    store = SkillStore()
    examples = store.get_recent_tool_examples(limit=20)

    if len(examples) < 3:
        store.close()
        return

    # Group by category for analysis
    by_cat: dict[str, list] = {}
    for ex in examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    # Build summary for LLM
    cat_summaries = []
    for cat, exs in by_cat.items():
        inputs = [e["tool_input"][:80] for e in exs[:5]]
        cat_summaries.append(f"[{cat}] ({len(exs)} calls): {', '.join(inputs)}")
    summary_text = "\n".join(cat_summaries)

    prompt = (
        f"Analyze these tool usage patterns from a coding session.\n"
        f"Identify 0-3 reusable behavioral patterns worth remembering.\n\n"
        f"Tool usage by category:\n{summary_text}\n\n"
        f"For each pattern, output a JSON object in an array:\n"
        f'[{{"rule": "when <situation>", "pattern": "<what to do>"}}]\n\n'
        f"Examples of good patterns:\n"
        f'- {{"rule": "when committing code", "pattern": "use conventional commits with scope: type(scope): description"}}\n'
        f'- {{"rule": "when searching code", "pattern": "start with Grep for pattern, then Read matching files"}}\n'
        f'- {{"rule": "when creating PRs", "pattern": "title under 50 chars, body has ## Summary + ## Test plan"}}\n\n'
        f"Output ONLY the JSON array. Empty array [] if no clear patterns."
    )

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 300}},
            timeout=15.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            store.close()
            return
        patterns = json.loads(m.group())
        if not isinstance(patterns, list):
            store.close()
            return

        added = 0
        for pat in patterns[:3]:
            rule = pat.get("rule", "")
            pattern_text = pat.get("pattern", "")
            if not rule or not pattern_text:
                continue

            # Check if a similar teaching already exists
            rule_vec = embed(rule)
            existing = store.search_teachings(rule_vec, min_sim=0.8)
            if existing:
                continue  # already have a similar teaching

            store.add_teaching(
                rule=rule,
                rule_vector=rule_vec,
                action=pattern_text,
                target_type="pattern",
                target_id="context_bridge",
            )
            added += 1

        if added:
            log_event("TEACH_EX", f"extracted {added} teaching examples")
    except Exception as exc:
        log_event("TEACH_EX", f"LLM error: {exc}")
    finally:
        store.close()


def _distill_tool_chains(transcript_path: str) -> None:
    """Analyze a session transcript for tool call patterns that could become local skills.

    Reads the JSONL transcript, extracts tool call sequences, and when a
    pattern is automatable (shell commands, file reads, git operations),
    generates a local skill JSON.

    This is how the system "learns from Claude" — observing what Claude does
    repeatedly and distilling it into cheap local execution.
    """
    import re
    from pathlib import Path
    from . import config as _cfg
    from .embeddings import OLLAMA_BASE, RERANK_MODEL, ollama_available

    path = Path(transcript_path)
    if not path.exists() or path.stat().st_size < 500:
        return

    # Parse transcript for tool calls
    tool_sequences: list[dict] = []
    current_sequence: list[dict] = []

    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Look for tool_use entries (Claude's tool calls)
            if entry.get("type") == "assistant":
                for block in (entry.get("message", {}).get("content", []) or []):
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        current_sequence.append({
                            "tool": tool_name,
                            "input": tool_input,
                        })

            # User messages reset the sequence (new intent)
            elif entry.get("type") == "human":
                if len(current_sequence) >= 2:
                    # Extract user intent from the human message
                    human_text = ""
                    for block in (entry.get("message", {}).get("content", []) or []):
                        if isinstance(block, str):
                            human_text = block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            human_text = block.get("text", "")
                    if human_text:
                        tool_sequences.append({
                            "intent": human_text[:200],
                            "tools": list(current_sequence),
                        })
                current_sequence = []
    except Exception:
        return

    if not tool_sequences:
        return

    # Filter for sequences that are automatable locally
    # (Bash commands, Read, Grep, git operations)
    automatable_tools = {"Bash", "Read", "Grep", "Glob", "Write", "Edit"}
    candidates = []
    for seq in tool_sequences:
        tools = seq["tools"]
        if len(tools) < 2 or len(tools) > 8:
            continue
        # Check if most tools are automatable
        auto_count = sum(1 for t in tools if t["tool"] in automatable_tools)
        if auto_count / len(tools) >= 0.7:
            candidates.append(seq)

    if not candidates:
        return

    log_event("DISTILL", f"found {len(candidates)} automatable tool chains")

    # Use local LLM to convert the best candidate into a local skill
    if not ollama_available(RERANK_MODEL):
        return

    # Take the most interesting candidate (longest chain)
    best = max(candidates, key=lambda c: len(c["tools"]))

    # Format tool chain for LLM
    chain_desc = []
    for t in best["tools"]:
        inp = t["input"]
        if t["tool"] == "Bash":
            chain_desc.append(f"  Bash: {inp.get('command', '')[:150]}")
        elif t["tool"] == "Read":
            chain_desc.append(f"  Read: {inp.get('file_path', '')}")
        elif t["tool"] == "Grep":
            chain_desc.append(f"  Grep: {inp.get('pattern', '')} in {inp.get('path', '.')}")
        else:
            chain_desc.append(f"  {t['tool']}: {str(inp)[:100]}")

    chain_text = "\n".join(chain_desc)

    prompt = f"""\
A coding assistant (Claude) performed this tool sequence for the user's request.
Convert it into a reusable local skill JSON that runs shell commands.

User intent: {best['intent']}

Tool chain:
{chain_text}

Generate a local skill JSON with this format:
{{
  "name": "<kebab-case-name>",
  "description": "<what it does>",
  "triggers": ["<phrase 1>", "<phrase 2>", "<phrase 3>"],
  "steps": [
    {{"run": "<shell command>", "as": "<var_name>"}},
    ...
  ],
  "output": "<template using {{var_name}} placeholders>"
}}

Rules:
- Convert Read/Grep/Bash tool calls into equivalent shell commands
- Skip tool calls that can't be automated (Edit, Write — these need reasoning)
- If the chain requires complex reasoning, output: {{"can_automate": false}}
- Keep it simple — only automate the mechanical parts
- Output ONLY JSON, no markdown"""

    import httpx
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 500}},
            timeout=20.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            skill = json.loads(m.group())
            if skill.get("can_automate") is False:
                log_event("DISTILL", f"chain not automatable: {best['intent'][:60]}")
                return
            if skill.get("name") and skill.get("steps"):
                skill["_source"] = "distilled"
                skill["_intent"] = best["intent"][:200]
                installed = _install_auto_skill(skill)
                if installed:
                    log_event("DISTILL",
                              f"skill distilled: {skill['name']} "
                              f"from {len(best['tools'])} tool calls → {installed}")
    except Exception as exc:
        log_event("DISTILL", f"LLM error: {exc}")


def _install_auto_skill(skill_dict: dict) -> str | None:
    """Write a generated skill JSON to the local-skills directory.

    Returns the file path on success, None on failure.
    """
    import json as _json
    from . import config as _cfg

    skills_dir = Path(_cfg.get("local_skills_dir") or
                      str(Path.home() / ".claude" / "local-skills"))
    try:
        skills_dir.mkdir(parents=True, exist_ok=True)
        name = skill_dict.get("name", "auto-skill")
        safe = name.replace(" ", "-").replace("/", "-")
        path = skills_dir / f"{safe}.json"
        path.write_text(_json.dumps(skill_dict, indent=2), encoding="utf-8")
        log_event("SKILL", f"auto-generated: {path}")
        return str(path)
    except Exception as exc:
        log_hook("auto_skill_write_error", error=str(exc)[:80])
        return None


def _capture_tool_calls(
    transcript_path: str,
    session_id: str,
    offset: int = 0,
    max_captures: int = 20,
) -> tuple[list[dict], int]:
    """Parse transcript from byte offset, extract tool calls incrementally.

    Returns (tool_call_dicts, new_byte_offset).
    Pure file I/O + JSON parsing — no LLM, ~5-20ms for a typical response.
    """
    from pathlib import Path

    path = Path(transcript_path)
    if not path.exists():
        return [], offset

    file_size = path.stat().st_size
    if file_size <= offset:
        return [], offset

    tool_calls: list[dict] = []
    # Map tool_use_id → index in tool_calls for result matching
    pending: dict[str, int] = {}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            remaining = f.read()
            new_offset = f.tell()

        for line in remaining.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Extract tool_use blocks from assistant messages
            if entry_type == "assistant":
                for block in (entry.get("message", {}).get("content", []) or []):
                    if block.get("type") != "tool_use":
                        continue
                    if len(tool_calls) >= max_captures:
                        break

                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    tool_use_id = block.get("id", "")

                    # Categorize
                    category = _categorize_tool_call(tool_name, tool_input)

                    # Summarize input based on tool type
                    input_summary = _summarize_tool_input(tool_name, tool_input)

                    tc = {
                        "session_id": session_id,
                        "tool_name": tool_name,
                        "tool_input": input_summary,
                        "output_summary": "",
                        "context_hint": "",
                        "repo_path": "",
                        "category": category,
                    }
                    idx = len(tool_calls)
                    tool_calls.append(tc)
                    if tool_use_id:
                        pending[tool_use_id] = idx

            # Match tool results back to their tool_use
            elif entry_type == "tool_result" or (
                entry_type == "human"
                and isinstance(entry.get("message", {}).get("content", None), list)
            ):
                content = entry.get("message", {}).get("content", [])
                if not isinstance(content, list):
                    content = entry.get("content", [])
                    if not isinstance(content, list):
                        continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    tuid = block.get("tool_use_id", "")
                    if tuid in pending:
                        result_text = ""
                        # Extract text content from result
                        for sub in (block.get("content", []) or []):
                            if isinstance(sub, str):
                                result_text = sub
                            elif isinstance(sub, dict) and sub.get("type") == "text":
                                result_text = sub.get("text", "")
                        if result_text:
                            tool_calls[pending[tuid]]["output_summary"] = result_text[:200]
                        del pending[tuid]

    except Exception:
        return tool_calls, file_size

    return tool_calls, new_offset


def _categorize_tool_call(tool_name: str, tool_input: dict) -> str:
    """Categorize a tool call for filtering and pattern analysis."""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if cmd.startswith("gh ") or "github.com" in cmd:
            return "github"
        if cmd.startswith("git "):
            return "git"
        return "shell"
    if tool_name in ("Grep", "Glob"):
        return "search"
    if tool_name in ("Read", "Edit", "Write"):
        return "file"
    if tool_name.startswith("mcp__"):
        return "mcp"
    return "general"


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Create a concise summary of tool input for storage."""
    if tool_name == "Bash":
        return tool_input.get("command", "")[:500]
    if tool_name == "Read":
        return tool_input.get("file_path", "")
    if tool_name == "Edit":
        fp = tool_input.get("file_path", "")
        old = (tool_input.get("old_string", "") or "")[:60]
        return f"{fp} :: {old}..."
    if tool_name == "Write":
        return tool_input.get("file_path", "")
    if tool_name == "Grep":
        pat = tool_input.get("pattern", "")
        path = tool_input.get("path", ".")
        return f"{pat} in {path}"
    if tool_name == "Glob":
        return tool_input.get("pattern", "")
    # MCP or other tools — dump keys
    try:
        return json.dumps(tool_input, ensure_ascii=False)[:500]
    except Exception:
        return str(tool_input)[:500]


def _cmd_local_status() -> str:
    """Show local execution status: levels, models, commands, templates, skills."""
    from . import config as _cfg
    from pathlib import Path

    cfg = _cfg.load_config()
    enabled = cfg.get("local_execution_enabled", False)
    lines = ["=== Local Execution Status ===\n"]
    lines.append(f"Enabled:     {'✓ on' if enabled else '○ off'}")

    # Models per level
    models = cfg.get("local_models", {})
    lines.append(f"\nModels:")
    for level in ("level_1", "level_2", "level_3", "level_4"):
        lines.append(f"  {level}: {models.get(level, '(not set)')}")

    # Level 1: whitelisted commands
    commands = cfg.get("local_commands", {})
    lines.append(f"\nLevel 1 — Whitelisted commands ({len(commands)}):")
    for name, cmd in sorted(commands.items()):
        lines.append(f"  {name:<20} {cmd}")

    # Level 2: templates
    templates = cfg.get("local_templates", {})
    lines.append(f"\nLevel 2 — Templated commands ({len(templates)}):")
    for name, tpl in sorted(templates.items()):
        params = ", ".join(f"{k}:{v}" for k, v in tpl.get("params", {}).items())
        lines.append(f"  {name:<20} {tpl['cmd']}  ({params})")

    # Level 3: local skills
    skills = _load_local_skills()
    skills_dir = Path(str(cfg.get("local_skills_dir", "~/.claude/local-skills"))).expanduser()
    lines.append(f"\nLevel 3 — Local skills ({len(skills)}) from {skills_dir}:")
    for s in skills:
        triggers = ", ".join(s.get("triggers", [])[:3])
        step_count = len(s.get("steps", []))
        lines.append(f"  {s['name']:<20} {step_count} steps  triggers: {triggers}")

    # Session approvals
    lines.append(f"\nSession approvals: {', '.join(sorted(_approved_commands)) or '(none)'}")

    return "\n".join(lines)


def _cmd_local_skills() -> str:
    """List all local skill files with details."""
    from pathlib import Path
    from . import config as _cfg

    skills_dir = Path(str(_cfg.get("local_skills_dir"))).expanduser()
    if not skills_dir.exists():
        return (f"Local skills directory not found: {skills_dir}\n"
                f"Create it and add JSON skill files. See /hub-help for format.")

    skills = _load_local_skills()
    if not skills:
        return f"No local skills in {skills_dir}"

    lines = [f"=== Local Skills ({len(skills)}) ===\n"]
    for s in skills:
        lines.append(f"**{s['name']}** — {s.get('description', '(no description)')}")
        lines.append(f"  File: {s.get('_file', '?')}")
        triggers = s.get("triggers", [])
        if triggers:
            lines.append(f"  Triggers: {', '.join(triggers)}")
        steps = s.get("steps", [])
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step.get('run', '?')}")
        lines.append("")
    return "\n".join(lines)


def _cmd_local_approve(args: str) -> str:
    """Pre-approve a command/skill name for auto-execution this session."""
    name = args.strip()
    if not name:
        if _approved_commands:
            return ("Session-approved commands:\n  " +
                    "\n  ".join(sorted(_approved_commands)) +
                    "\n\nUsage: /hub-local-approve <name>")
        return "No commands approved yet this session.\nUsage: /hub-local-approve <name>"

    _approved_commands.add(name)
    return f"Approved `{name}` — will execute without confirmation this session."


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
    triage_stats = store.get_triage_stats()

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
            lines.append(f"  {row['command_type']:<30} {row['intercept_count']:>4}x  "
                        f"~{row['total_tokens_saved'] or 0:,} tokens")
    else:
        lines.append("No interceptions recorded yet.\n")

    # Triage stats
    if triage_stats and triage_stats.get("total", 0) > 0:
        total = triage_stats["total"]
        local_ans = triage_stats.get("local_answers", 0) or 0
        local_act = triage_stats.get("local_actions", 0) or 0
        enriched = triage_stats.get("enriched", 0) or 0
        passed = triage_stats.get("passed", 0) or 0
        triage_saved = triage_stats.get("total_tokens_saved", 0) or 0
        avg_conf = triage_stats.get("avg_confidence", 0) or 0

        lines.append(f"\n=== LLM Triage (all messages) ===\n")
        lines.append(f"Total triaged:       {total}")
        lines.append(f"Avg confidence:      {avg_conf:.2f}")
        lines.append(f"Est. tokens saved:   ~{triage_saved:,}\n")

        def _tpct(n: int) -> str:
            return f"{n / total * 100:.0f}%" if total else "0%"

        lines.append(f"  Answered locally:  {local_ans:>4}x  ({_tpct(local_ans)}) — 0 Claude tokens")
        lines.append(f"  Routed to /hub-*:  {local_act:>4}x  ({_tpct(local_act)}) — 0 Claude tokens")
        lines.append(f"  Enriched + fwd:    {enriched:>4}x  ({_tpct(enriched)}) — hint injected")
        lines.append(f"  Passed through:    {passed:>4}x  ({_tpct(passed)}) — no change")

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

    store.close()
    return "\n".join(lines)


def _cmd_local_agent(args_str: str) -> dict:
    """Handle /local-agent command — explicit Level 4 invocation."""
    from . import config as _cfg
    from .local_agent import plan_agent

    if not _cfg.get("local_execution_enabled"):
        return {"decision": "block",
                "message": "Local execution is disabled. Enable with: /hub-configure local_execution_enabled true"}

    message = args_str.strip()
    if not message:
        # Show agent info + available skills
        skills = _load_local_skills()
        commands = _cfg.get("local_commands") or {}
        models = _cfg.get("local_models") or {}
        model_spec = models.get("level_4", "qwen2.5-coder:32b")

        lines = ["=== Local Agent (Level 4) ===\n"]
        lines.append(f"Model:    {model_spec}")
        lines.append(f"Skills:   {len(skills)}")
        lines.append(f"Commands: {len(commands)}")
        if skills:
            lines.append("\nAvailable local skills:")
            for s in skills:
                lines.append(f"  - {s['name']}: {s.get('description', '')}")
        lines.append("\nUsage: /local-agent <describe your task>")
        return {"decision": "block", "message": "\n".join(lines)}

    # Plan first — show to user for approval
    global _pending_command
    log_hook("local_agent_plan", message=message[:200])

    recent = "\n".join(_session_messages[-5:]) if _session_messages else ""
    plan = plan_agent(message, context=recent)

    if not plan.get("can_handle"):
        reason = plan.get("reason", "Unknown")
        return {"decision": "block",
                "message": (f"[Local agent — {plan.get('model', '?')}]\n\n"
                            f"Cannot handle this task locally.\n"
                            f"Reason: {reason}\n\n"
                            f"Passing through to Claude.")}

    # Format plan for display
    plan_steps = plan.get("plan", [])
    skills_needed = plan.get("skills_needed", [])
    commands_needed = plan.get("commands_needed", [])

    plan_display = [f"[Local agent — {plan.get('model', '?')}]\n"]
    plan_display.append(f"**Plan for:** {message}\n")
    if plan_steps:
        plan_display.append("Steps:")
        for i, step in enumerate(plan_steps, 1):
            plan_display.append(f"  {i}. {step}")
    if skills_needed:
        plan_display.append(f"\nSkills: {', '.join(skills_needed)}")
    if commands_needed:
        plan_display.append(f"Commands: {', '.join(commands_needed)}")
    plan_display.append("\nReply **y** to execute, **n** to cancel (pass to Claude).")

    _pending_command = {
        "name": "__local_agent__",
        "level": 4,
        "_agent_message": message,
        "_agent_context": recent,
    }
    return {"decision": "block", "reason": "\n".join(plan_display)}


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

Local Execution (L1=commands, L2=templates, L3=skills, L4=agent):
  /hub-local-status        Show levels, models, commands, skills
  /hub-local-skills        List all local skill definitions
  /hub-local-approve <name> Pre-approve a command for this session
  /hub-local-agent <task>  Run task via local LLM agent (L4)

Context & Memory:
  /hub-digest              Force conversation digest now
  /hub-optimize-context    LLM analyzes memory, recommends pruning
  /hub-save-memory [desc]  LLM generates memory entry from session
  /hub-exhaustion-save     Auto-save session when Claude is exhausted

Permissions:
  /hub-approve-task <scope> Auto-accept Claude actions for a scope
  /hub-lock-task            Revoke all hub-granted permissions

Scopes: git, python, read, write, edit, web, mcp, all
Example: /hub-approve-task git python"""


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
    claude_skills = [r for r in rows if r["target"] == "claude"]
    local_skills = [r for r in rows if r["target"] == "local"]
    lines = [f"{len(rows)} skills ({len(claude_skills)} claude, {len(local_skills)} local):\n"]
    if claude_skills:
        lines.append("Claude skills:")
        for r in claude_skills:
            lines.append(f"  {r['id']}: {r['description'] or '(no description)'}"[:120])
    if local_skills:
        lines.append("\nLocal skills:")
        for r in local_skills:
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
_COMMAND_USAGE: dict[str, str] = {
    "status": "Show health check: Ollama, models, DB stats, hook, config.\n\nUsage: `/hub-status`",
    "help": "Show full command reference.\n\nUsage: `/hub-help`",
    "token_stats": "Token savings report: interceptions, triage, context injections.\n\nUsage: `/hub-token-stats`",
    "list_models": "List installed Ollama models with active markers.\n\nUsage: `/hub-list-models`",
    "list_skills": (
        "List indexed skills, optionally filtered.\n\n"
        "Usage:\n"
        "  `/hub-list-skills`             — all skills (claude + local)\n"
        "  `/hub-list-skills local`       — only local skills\n"
        "  `/hub-list-skills claude`      — only Claude skills\n"
        "  `/hub-list-skills superpowers` — filter by plugin name\n"
    ),
    "list_teachings": "Show all teaching rules.\n\nUsage: `/hub-list-teachings`",
    "configure": (
        "View or set config values.\n\n"
        "Usage:\n"
        "  `/hub-configure`              — show all config\n"
        "  `/hub-configure <key>`        — show one value\n"
        "  `/hub-configure <key> <val>`  — set a value\n\n"
        "Common keys: reason_model, embed_model, hook_enabled, "
        "local_execution_enabled, local_models, hook_llm_triage"
    ),
    "list_tasks": (
        "Show tasks by status.\n\n"
        "Usage:\n"
        "  `/hub-list-tasks`       — open tasks (default)\n"
        "  `/hub-list-tasks all`   — all tasks\n"
        "  `/hub-list-tasks closed` — closed tasks"
    ),
    "save_task": "Save current work as an open task (uses LLM to extract title).\n\nUsage: `/hub-save-task [title]`",
    "close_task": "Close and compact a task (uses LLM).\n\nUsage: `/hub-close-task <task_id>`",
    "update_task": "Update an existing task (via Claude).\n\nUsage: `/hub-update-task <task_id>`",
    "reopen_task": "Reopen a closed task.\n\nUsage: `/hub-reopen-task <task_id>`",
    "search_context": (
        "Unified semantic search across skills, tasks, teachings, memory.\n\n"
        "Usage: `/hub-search-context <query>`"
    ),
    "search_skills": "Semantic skill search by embedding similarity.\n\nUsage: `/hub-search-skills <query>`",
    "suggest_plugins": "Find matching plugins for a task.\n\nUsage: `/hub-suggest-plugins <query>`",
    "teach": (
        "Add a persistent teaching rule.\n\n"
        "Usage: `/hub-teach <rule> -> <target>`\n"
        "Example: `/hub-teach when debugging CSS -> chrome-devtools-mcp`"
    ),
    "forget_teaching": "Remove a teaching rule.\n\nUsage: `/hub-forget-teaching <id>`",
    "toggle_plugin": "Enable/disable a plugin (via Claude).\n\nUsage: `/hub-toggle-plugin <name> on|off`",
    "profile": (
        "Manage session profiles (plugin presets).\n\n"
        "Usage:\n"
        "  `/hub-profile`              — list profiles\n"
        "  `/hub-profile <name>`       — activate a profile\n"
        "  `/hub-profile save <name>`  — save current state\n"
        "  `/hub-profile delete <name>` — remove profile\n"
        "  `/hub-profile auto <task>`  — LLM recommends best"
    ),
    "digest": "Force a conversation digest now.\n\nUsage: `/hub-digest`",
    "optimize_context": "LLM analyzes memory and recommends pruning.\n\nUsage: `/hub-optimize-context`",
    "save_memory": "LLM generates a memory entry from the session.\n\nUsage: `/hub-save-memory [description]`",
    "exhaustion_save": "Auto-save session when Claude is rate-limited.\n\nUsage: `/hub-exhaustion-save [context]`",
    "index_skills": "Rebuild skill index from all plugin dirs + local skills.\n\nUsage: `/hub-index-skills`",
    "index_plugins": "Rebuild plugin index (via Claude).\n\nUsage: `/hub-index-plugins`",
    "pull_model": "Download an Ollama model.\n\nUsage: `/hub-pull-model <model_name>`",
    "local_agent": (
        "Run a task via the local LLM agent (Level 4).\n\n"
        "Usage:\n"
        "  `/local-agent`              — show agent status + skills\n"
        "  `/local-agent <task>`       — plan + execute task locally\n\n"
        "The agent shows a plan first, then waits for **y**/**n** confirmation."
    ),
    "local_toggle": (
        "Toggle local mode — route ALL messages to the local LLM agent.\n\n"
        "When Claude is exhausted/quota-limited, switch to local mode so\n"
        "the local agent handles everything until Claude is back.\n\n"
        "Usage:\n"
        "  `/hub-local`       — toggle on/off\n"
        "  `/hub-local on`    — force on (auto-saves session state)\n"
        "  `/hub-local off`   — force off (resume Claude)\n\n"
        "In local mode, all slash commands still work normally.\n"
        "Session state is auto-saved when entering local mode."
    ),
    "approve_task": (
        "Auto-accept Claude's tool calls for the given scopes.\n\n"
        "Adds permission patterns to settings.json so Claude can execute\n"
        "without asking for confirmation. Use `/hub-lock-task` to revoke.\n\n"
        "Usage:\n"
        "  `/hub-approve-task git`          — git commands\n"
        "  `/hub-approve-task python`       — pytest, python, pip\n"
        "  `/hub-approve-task git python`   — multiple scopes\n"
        "  `/hub-approve-task all`          — everything\n\n"
        "Scopes: git, python, read, write, edit, web, mcp, bash, all"
    ),
    "lock_task": (
        "Revoke all hub-granted permissions.\n\n"
        "Removes only the patterns added by `/hub-approve-task`.\n"
        "Your original permissions remain untouched.\n\n"
        "Usage: `/hub-lock-task`"
    ),
    "suggest_patterns": (
        "Detect similar permission patterns and suggest wildcards.\n\n"
        "Scans settings.json for groups of similar `Bash(...)` patterns\n"
        "that could be replaced by a single wildcard (e.g. `Bash(python -m pip:*)`).\n\n"
        "Usage: `/hub-suggest-patterns`"
    ),
    "apply_pattern": (
        "Apply a pattern consolidation suggestion.\n\n"
        "Usage: `/hub-apply-pattern <number>`\n"
        "Run `/hub-suggest-patterns` first to see available suggestions."
    ),
    "search_web": (
        "Search the web via local SearXNG and summarize with local LLM.\n\n"
        "Requires SearXNG running (python install.py --searxng).\n\n"
        "Usage: `/hub-search-web <query>`"
    ),
    "skill_disable": (
        "Disable local skill/command execution (L1/L2/L3).\n\n"
        "Stops the hook from matching user messages to local commands\n"
        "and skills. Classification and context injection still work.\n\n"
        "Usage: `/hub-skill-disable`"
    ),
    "skill_enable": (
        "Re-enable local skill/command execution.\n\n"
        "Usage: `/hub-skill-enable`"
    ),
    "hook_disable": (
        "Disable the entire hook pipeline.\n\n"
        "All messages pass through to Claude unmodified.\n"
        "Slash commands still work.\n\n"
        "Usage: `/hub-hook-disable`"
    ),
    "hook_enable": (
        "Re-enable the full hook pipeline.\n\n"
        "Usage: `/hub-hook-enable`"
    ),
}

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
    "/hub-local-status": "local_status",
    "/hub-local-skills": "local_skills",
    "/hub-local-approve": "local_approve",
    "/hub-local-agent": "local_agent",
    "/local-agent": "local_agent",
    "/hub-local": "local_toggle",
    "/hub-approve-task": "approve_task",
    "/hub-lock-task": "lock_task",
    "/hub-suggest-patterns": "suggest_patterns",
    "/hub-apply-pattern": "apply_pattern",
    "/hub-export-training": "export_training",
    "/hub-invalidate-cache": "invalidate_cache",
    "/hub-cache-stats": "cache_stats",
    "/hub-show-patterns": "show_patterns",
    "/hub-search-web": "search_web",
    "/hub-skill-disable": "skill_disable",
    "/hub-skill-enable": "skill_enable",
    "/hub-hook-disable": "hook_disable",
    "/hub-hook-enable": "hook_enable",
}


def _handle_slash_command(message: str) -> dict | None:
    """
    Fast-path for /slash-commands. Returns a hook response dict or None if
    the message is not a known slash command.
    """
    stripped = message.strip()

    # ? alone — list all commands; ?command — inline help for that command
    if stripped.startswith("?"):
        help_cmd = stripped[1:].strip().lower()
        if not help_cmd:
            # Bare "?" — show all available commands grouped
            groups: dict[str, list[str]] = {}
            for cmd_key, action in sorted(_SLASH_COMMANDS.items()):
                # Deduplicate aliases (multiple keys → same action)
                one_liner = _COMMAND_USAGE.get(action, "").split("\n")[0]
                if action not in groups:
                    groups[action] = [cmd_key, one_liner]
            lines = ["**Available commands** (type `?command` for details)\n"]
            for action, (cmd_key, one_liner) in sorted(groups.items(), key=lambda x: x[1][0]):
                lines.append(f"  `{cmd_key}`  — {one_liner}")
            return {"decision": "block", "message": "\n".join(lines)}
        # ?command — help for a specific command
        if not help_cmd.startswith("/"):
            help_cmd = "/" + help_cmd
        action = _SLASH_COMMANDS.get(help_cmd)
        if action:
            usage = _COMMAND_USAGE.get(action, f"No detailed help for {help_cmd}.")
            return {"decision": "block", "message": f"**{help_cmd}**\n\n{usage}"}
        return None

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
        elif action == "local_status":
            return {"decision": "block", "message": _cmd_local_status()}
        elif action == "local_skills":
            return {"decision": "block", "message": _cmd_local_skills()}
        elif action == "local_approve":
            return {"decision": "block", "message": _cmd_local_approve(args_str)}
        elif action == "local_agent":
            return _cmd_local_agent(args_str)
        elif action == "local_toggle":
            return {"decision": "block", "message": _cmd_local_toggle(args_str)}
        elif action == "approve_task":
            return {"decision": "block", "message": _cmd_approve_task(args_str)}
        elif action == "lock_task":
            return {"decision": "block", "message": _cmd_lock_task()}
        elif action == "suggest_patterns":
            return {"decision": "block", "message": _cmd_suggest_patterns()}
        elif action == "apply_pattern":
            return {"decision": "block", "message": _cmd_apply_pattern(args_str)}
        elif action == "export_training":
            return {"decision": "block", "message": _cmd_export_training(args_str)}
        elif action == "invalidate_cache":
            return {"decision": "block", "message": _cmd_invalidate_cache(args_str)}
        elif action == "cache_stats":
            return {"decision": "block", "message": _cmd_cache_stats()}
        elif action == "show_patterns":
            return {"decision": "block", "message": _cmd_show_patterns()}
        elif action == "search_web":
            if not args_str.strip():
                return {"decision": "block",
                        "message": "Usage: `/hub-search-web <query>`"}
            return {"decision": "block", "message": _cmd_search_web(args_str.strip())}
        elif action == "skill_disable":
            from . import config as _cfg
            cfg = _cfg.load_config()
            cfg["local_execution_enabled"] = False
            _cfg.save_config(cfg)
            return {"decision": "block",
                    "message": "Local skill/command execution **disabled**.\n"
                               "Re-enable: `/hub-skill-enable`"}
        elif action == "skill_enable":
            from . import config as _cfg
            cfg = _cfg.load_config()
            cfg["local_execution_enabled"] = True
            _cfg.save_config(cfg)
            return {"decision": "block",
                    "message": "Local skill/command execution **enabled**."}
        elif action == "hook_disable":
            from . import config as _cfg
            cfg = _cfg.load_config()
            cfg["hook_enabled"] = False
            cfg["hook_context_injection"] = False
            cfg["hook_llm_triage"] = False
            _cfg.save_config(cfg)
            return {"decision": "block",
                    "message": "Hook pipeline **disabled** (classification, triage, context injection all off).\n"
                               "Slash commands still work. Re-enable: `/hub-hook-enable`"}
        elif action == "hook_enable":
            from . import config as _cfg
            cfg = _cfg.load_config()
            cfg["hook_enabled"] = True
            cfg["hook_context_injection"] = True
            cfg["hook_llm_triage"] = True
            _cfg.save_config(cfg)
            return {"decision": "block",
                    "message": "Hook pipeline **enabled** (classification, triage, context injection).\n"
                               "Local execution state unchanged — check `/hub-configure local_execution_enabled`."}
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
        print("          exhaustion_save, digest, optimize_context, save_memory,")
        print("          local_agent")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "classify":
        # Extract --session-id if present
        session_id = ""
        filtered_args = []
        i = 0
        while i < len(args):
            if args[i] == "--session-id" and i + 1 < len(args):
                session_id = args[i + 1]
                i += 2
            else:
                filtered_args.append(args[i])
                i += 1
        message = " ".join(filtered_args) if filtered_args else sys.stdin.read()
        result = hook_classify_and_execute(message.strip(), session_id=session_id)
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

    elif cmd == "local_agent":
        result = _cmd_local_agent(" ".join(args))
        print(result.get("message", json.dumps(result)))

    elif cmd == "session_end":
        # Called by the Stop hook — saves session state + memory + stats
        session_id = ""
        last_message = ""
        transcript = ""
        i = 0
        while i < len(args):
            if args[i] == "--session-id" and i + 1 < len(args):
                session_id = args[i + 1]
                i += 2
            elif args[i] == "--transcript" and i + 1 < len(args):
                transcript = args[i + 1]
                i += 2
            elif args[i] == "--last-message" and i + 1 < len(args):
                last_message = args[i + 1]
                i += 2
            else:
                i += 1
        result = _cmd_session_end(session_id, last_message, transcript)
        print(json.dumps(result))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
