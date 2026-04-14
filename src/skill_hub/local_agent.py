"""Level 4: Local agent — skill-driven LLM agent loop.

Runs locally (or on a remote LLM) when:
- Claude is rate-limited / exhausted
- Triage classifies the task as simple enough for local execution
- User explicitly invokes /local-agent

The agent's capabilities come primarily from local skills (.json files).
It can also run whitelisted shell commands (Level 1/2 config).

Model is configurable per level_4: local Ollama, remote Ollama, or
OpenAI-compatible API.
"""

import json
import subprocess

import httpx

from . import config as _cfg
from .activity_log import log_hook, log_llm


_AGENT_SYSTEM = """\
You are a local coding assistant with access to tools. You handle simple, \
well-defined tasks: git operations, file reading, running tests, searching code, \
and executing predefined skill workflows.

You have these tools:
- run_skill(name): Execute a local skill by name. Returns the combined output.
- shell(command): Run a whitelisted shell command. Only commands from the \
  approved list are allowed.
- read_file(path): Read a file's contents (max 5000 chars).
- search(pattern, path): Search for a pattern in files (grep -rn).
- list_files(pattern): List files matching a glob pattern.
- done(answer): Return your final answer to the user.

RULES:
- Prefer run_skill over shell when a skill matches the task.
- For shell, only use commands from the whitelist provided.
- Keep answers concise and direct.
- If you can't complete the task with available tools, say so.
- Maximum {max_turns} tool calls per task.

Available local skills:
{skills}

Whitelisted shell commands:
{commands}
"""

_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Execute a local skill by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a whitelisted shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a pattern in files (grep -rn)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory to search", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', 'src/**/*.ts')"}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Return the final answer to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Final answer"}
                },
                "required": ["answer"],
            },
        },
    },
]


# ── Skill loading ──────────────────────────────────────────────────

def _load_agent_skills() -> list[dict]:
    """Load local skills from both filesystem cache and DB (target=local).

    Filesystem JSON skills are the primary source (used for execution).
    DB skills (indexed via /hub-index-skills) provide enriched descriptions
    for the LLM prompt. Merges by name, filesystem wins on conflict.
    """
    from .cli import _load_local_skills
    from .store import SkillStore

    # Filesystem skills (executable — have steps, triggers, etc.)
    fs_skills = _load_local_skills()
    seen = {s.get("name") for s in fs_skills}

    # DB skills with target=local (may have richer descriptions from indexing)
    try:
        store = SkillStore()
        db_rows = store.list_skills(target="local")
        store.close()
        for row in db_rows:
            name = row["name"]
            if name not in seen:
                # DB-only skill — include as info for the LLM
                fs_skills.append({
                    "name": name,
                    "description": row["description"] or "",
                    "_db_only": True,  # no executable steps
                })
                seen.add(name)
    except Exception:
        pass

    return fs_skills


# ── Tool executors ──────────────────────────────────────────────────

def _exec_run_skill(name: str, skills: list[dict]) -> str:
    """Execute a local skill by name."""
    from .cli import _execute_local_skill

    for skill in skills:
        if skill.get("name") == name:
            result = _execute_local_skill(skill)
            return result or "(no output)"
    return f"Skill '{name}' not found."


def _exec_shell(command: str) -> str:
    """Run a shell command if it's in the whitelist or is a safe git command."""
    commands = _cfg.get("local_commands") or {}
    templates = _cfg.get("local_templates") or {}

    # Check if the command matches a whitelisted command exactly
    allowed = set(commands.values())
    # Also allow any command that starts with a whitelisted prefix
    prefixes = {cmd.split("{")[0].strip() for tpl in templates.values()
                for cmd in [tpl.get("cmd", "")]}

    if command in allowed or any(command.startswith(p) for p in prefixes if p):
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30,
            )
            out = result.stdout
            if result.returncode != 0 and result.stderr:
                out += f"\n(exit {result.returncode}: {result.stderr.strip()[:500]})"
            if len(out) > 5000:
                out = out[:5000] + "\n... (truncated)"
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "(timed out)"
        except Exception as exc:
            return f"(error: {exc})"

    # Safety: also allow simple safe commands
    safe_prefixes = ("git ", "ls ", "pwd", "cat ", "head ", "tail ",
                     "wc ", "find ", "grep ", "test ", "echo ")
    if any(command.startswith(p) for p in safe_prefixes):
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30,
            )
            out = result.stdout
            if result.returncode != 0 and result.stderr:
                out += f"\n(exit {result.returncode}: {result.stderr.strip()[:500]})"
            if len(out) > 5000:
                out = out[:5000] + "\n... (truncated)"
            return out.strip() or "(no output)"
        except Exception as exc:
            return f"(error: {exc})"

    return f"Command not allowed: {command}. Use only whitelisted commands or safe prefixes (git, ls, cat, grep, etc.)"


def _exec_read_file(path: str) -> str:
    """Read a file (max 5000 chars)."""
    from pathlib import Path
    try:
        content = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated)"
        return content
    except Exception as exc:
        return f"(error: {exc})"


def _exec_search(pattern: str, path: str = ".") -> str:
    """Grep for a pattern."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.ts", "--include=*.js",
             "--include=*.json", "--include=*.md", "--include=*.yaml", "--include=*.yml",
             pattern, path],
            capture_output=True, text=True, timeout=15,
        )
        out = result.stdout
        if len(out) > 3000:
            out = out[:3000] + "\n... (truncated)"
        return out.strip() or "(no matches)"
    except Exception as exc:
        return f"(error: {exc})"


def _exec_list_files(pattern: str) -> str:
    """List files matching a glob pattern."""
    from pathlib import Path
    try:
        files = sorted(Path(".").glob(pattern))[:50]
        if not files:
            return "(no matches)"
        return "\n".join(str(f) for f in files)
    except Exception as exc:
        return f"(error: {exc})"


def _execute_tool_call(name: str, args: dict, skills: list[dict]) -> str:
    """Dispatch a tool call to the right executor."""
    if name == "run_skill":
        return _exec_run_skill(args.get("name", ""), skills)
    elif name == "shell":
        return _exec_shell(args.get("command", ""))
    elif name == "read_file":
        return _exec_read_file(args.get("path", ""))
    elif name == "search":
        return _exec_search(args.get("pattern", ""), args.get("path", "."))
    elif name == "list_files":
        return _exec_list_files(args.get("pattern", ""))
    elif name == "done":
        return args.get("answer", "")
    return f"Unknown tool: {name}"


# ── Model routing ───────────────────────────────────────────────────

def _resolve_model(level: str = "level_4") -> tuple[str, str]:
    """Resolve model for given level. Returns (base_url, model_name)."""
    models = _cfg.get("local_models") or {}
    model_spec = models.get(level, "qwen2.5-coder:32b")

    if isinstance(model_spec, str) and model_spec.startswith("remote:"):
        remote_url = model_spec[len("remote:"):]
        remote_cfg = _cfg.get("remote_llm") or {}
        model_name = remote_cfg.get("model", "")
        if not model_name:
            # If remote URL looks like Ollama, use a default model
            model_name = "qwen2.5-coder:32b"
        return remote_url, model_name

    # Local Ollama
    from .embeddings import OLLAMA_BASE
    return str(OLLAMA_BASE), str(model_spec)


def _chat_completion(base_url: str, model: str, messages: list[dict],
                     tools: list[dict], timeout: int = 60) -> dict:
    """Call the LLM with tool-use support. Supports Ollama and OpenAI-compatible APIs."""
    remote_cfg = _cfg.get("remote_llm") or {}
    api_key = remote_cfg.get("api_key", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Try OpenAI-compatible endpoint first (works for Ollama too)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        # If /v1/ fails (Ollama < 0.5), fall back to /api/chat
        if resp.status_code == 404:
            url = f"{base_url.rstrip('/')}/api/chat"
            resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


# ── Agent loop ──────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a local coding assistant. Given the user's request and available tools, \
create a brief execution plan.

Available local skills:
{skills}

Whitelisted shell commands:
{commands}

Other tools: read_file, search, list_files

Reply with ONLY a JSON object:
{{
  "can_handle": true/false,
  "plan": ["step 1 description", "step 2 description"],
  "skills_needed": ["skill_name"],
  "commands_needed": ["git status"],
  "reason": "why you can or cannot handle this"
}}

User request: {message}"""


def plan_agent(message: str, context: str = "") -> dict:
    """
    Ask the Level 4 LLM to plan how it would handle the request.
    Returns a plan dict for user approval before execution.
    """
    import re

    base_url, model = _resolve_model("level_3")
    skills = _load_agent_skills()
    commands = _cfg.get("local_commands") or {}

    log_llm("agent_plan", model=model, message_len=len(message))

    skills_desc = "\n".join(
        f"  - {s['name']}: {s.get('description', '')}" for s in skills
    ) or "  (none)"
    commands_desc = "\n".join(
        f"  - {cmd}" for cmd in commands.values()
    ) or "  (none)"

    # Inject session context from disk if no explicit context provided
    if not context:
        from .store import read_session_context
        context = read_session_context()

    prompt = _PLAN_PROMPT.format(
        skills=skills_desc, commands=commands_desc, message=message,
    )
    if context:
        prompt = f"Session context:\n{context}\n\n{prompt}"

    try:
        from .cli import _build_local_persona
        _persona = _build_local_persona()
        if _persona:
            prompt = f"{_persona}\n\n{prompt}"
    except Exception:
        pass

    try:
        timeout = int((_cfg.get("remote_llm") or {}).get("timeout", 120))
        from .llm import LLMError, get_provider
        resolved = model if "/" in model else f"ollama/{model}"
        try:
            raw = get_provider().complete(
                prompt, model=resolved, max_tokens=512,
                temperature=0.2, timeout=timeout,
            )
        except LLMError as exc:
            return {"can_handle": False, "reason": str(exc), "model": model}
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
            plan["model"] = model
            return plan
    except Exception as exc:
        return {"can_handle": False, "reason": str(exc), "model": model}

    return {"can_handle": False, "reason": "Failed to generate plan", "model": model}


def run_agent(message: str, context: str = "", max_turns: int = 8) -> str:
    """
    Run the Level 4 local agent on a user message (after plan approval).

    Args:
        message: User's request
        context: Optional conversation context
        max_turns: Maximum tool call iterations

    Returns:
        Agent's final answer as a string
    """
    base_url, model = _resolve_model()
    skills = _load_agent_skills()
    commands = _cfg.get("local_commands") or {}

    log_llm("agent_start", model=model, message_len=len(message))

    # Build system prompt
    skills_desc = "\n".join(
        f"  - {s['name']}: {s.get('description', '')}" for s in skills
    ) or "  (none — add .json files to local_skills_dir)"
    commands_desc = "\n".join(
        f"  - {cmd}" for cmd in commands.values()
    ) or "  (none)"

    base_system = _AGENT_SYSTEM.format(
        max_turns=max_turns,
        skills=skills_desc,
        commands=commands_desc,
    )

    try:
        from .cli import _build_local_persona
        _persona = _build_local_persona()
        system = f"{_persona}\n\n{base_system}" if _persona else base_system
    except Exception:
        system = base_system

    messages = [{"role": "system", "content": system}]

    # Inject session context from disk (zero Claude token cost)
    if not context:
        from .store import read_session_context
        context = read_session_context()
    if context:
        messages.append({"role": "system", "content": f"Recent context:\n{context}"})

    messages.append({"role": "user", "content": message})

    timeout = int((_cfg.get("remote_llm") or {}).get("timeout", 120))

    for turn in range(max_turns):
        response = _chat_completion(base_url, model, messages, _TOOL_DEFS, timeout)

        if "error" in response:
            log_llm("agent_error", model=model, error=response["error"])
            return f"[Local agent error: {response['error']}]"

        # Extract the assistant message
        choices = response.get("choices", [])
        if not choices:
            # Ollama /api/chat format
            msg = response.get("message", {})
        else:
            # OpenAI format
            msg = choices[0].get("message", {})

        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Add assistant message to history
        messages.append(msg)

        if not tool_calls:
            # No tool calls — agent is done, content is the answer
            log_llm("agent_done", model=model, turns=turn + 1)
            return f"[Local agent — {model}]\n\n{content}" if content else "[Local agent: no response]"

        # Execute tool calls
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

            log_hook("agent_tool", tool=tool_name, args=tool_args)

            # Check for "done" tool
            if tool_name == "done":
                answer = tool_args.get("answer", content or "")
                log_llm("agent_done", model=model, turns=turn + 1)
                return f"[Local agent — {model}]\n\n{answer}"

            result = _execute_tool_call(tool_name, tool_args, skills)

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{turn}_{tool_name}"),
                "name": tool_name,
                "content": result,
            })

    log_llm("agent_max_turns", model=model, turns=max_turns)
    last_content = messages[-1].get("content", "") if messages else ""
    return f"[Local agent — reached max {max_turns} turns]\n\n{last_content}"
