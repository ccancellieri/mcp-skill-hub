#!/usr/bin/env python3
"""PreToolUse hook: auto-approve or block tool calls based on per-project allow-list.

Reads `<cwd>/.claude/skill-hub-allow.yml` (falls back to `~/.claude/skill-hub-allow.yml`).
YAML schema:
    safe_bash_prefixes:   # list of str; Bash commands starting with one of these auto-approve
      - "git status"
      - "git diff"
    safe_tools:           # list of tool names always auto-approved (e.g. Read, Grep)
      - Read
      - Grep
    deny_patterns:        # list of regex; matching Bash commands are blocked with reason
      - "rm\\s+-rf\\s+/"
      - "git\\s+push\\s+.*--force"

Logging appends to ~/.claude/mcp-skill-hub/logs/hook-debug.log.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_cache  # local module, stdlib-only  # noqa: E402

LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"
ACTIVE_TASK_MARKER = (
    Path.home() / ".claude" / "mcp-skill-hub" / "state" / "active_task.json"
)


def load_active_task_override() -> dict:
    """Read active task marker. Returns {} if absent/invalid."""
    if not ACTIVE_TASK_MARKER.exists():
        return {}
    try:
        return json.loads(ACTIVE_TASK_MARKER.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def task_safe_prefixes_from_marker(marker: dict) -> list[str]:
    """Extract per-task extra safe_bash_prefixes from the active task marker.

    Prefixes live in the task's DB row under context JSON key
    'task_safe_prefixes'. The marker may also embed them directly.
    """
    # Inline on the marker (fast path; server may choose to denormalize).
    inline = marker.get("task_safe_prefixes")
    if isinstance(inline, list):
        return [str(p) for p in inline if p]
    # Fallback: read task context from the DB.
    task_id = marker.get("task_id")
    if not isinstance(task_id, int):
        return []
    db = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
    if not db.exists():
        return []
    try:
        import sqlite3
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT context FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return []
        # context may be plain text OR JSON; try JSON.
        try:
            ctx = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return []
        prefixes = ctx.get("task_safe_prefixes") if isinstance(ctx, dict) else None
        if isinstance(prefixes, list):
            return [str(p) for p in prefixes if p]
    except Exception:  # noqa: BLE001
        return []
    return []


def log(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] AUTO_APPROVE {msg}\n")
    except OSError:
        pass


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Minimal fallback parser: top-level keys with list-of-strings values.
        data: dict = {}
        key = None
        for raw in path.read_text().splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith((" ", "\t")) and line.endswith(":"):
                key = line[:-1].strip()
                data[key] = []
            elif key is not None and line.lstrip().startswith("-"):
                val = line.lstrip()[1:].strip().strip('"').strip("'")
                if val:
                    data[key].append(val)
        return data
    except (OSError, Exception) as e:
        log(f"yaml_load_error  path={path}  err={e}")
        return {}


def load_allow_list(cwd: Path) -> dict:
    merged = {"safe_bash_prefixes": [], "safe_tools": [], "deny_patterns": []}
    for path in (Path.home() / ".claude" / "skill-hub-allow.yml",
                 cwd / ".claude" / "skill-hub-allow.yml"):
        data = load_yaml(path)
        for k in merged:
            merged[k].extend(data.get(k, []) or [])
    return merged


def extract_bash_command(tool_input: dict) -> str:
    return (tool_input.get("command") or "").strip()


def llm_classify(tool_name: str, command: str, examples: list[dict],
                 base_url: str, model: str, timeout: float) -> tuple[str, float, str]:
    """Call local Ollama to classify a command. Returns (decision, confidence, reason).

    Never denies — the LLM can only upgrade "unknown" to "allow". Static
    deny_patterns remain the only path to "deny".
    """
    example_lines = []
    for ex in examples[:6]:
        example_lines.append(
            f"- ({ex['tool_name']}) `{ex['command'][:120]}` -> {ex['decision']}"
        )
    prompt = (
        "You are a safety classifier for shell commands invoked by Claude "
        "Code. Return STRICT JSON of the form "
        '{"decision":"allow"|"unknown","confidence":0..1,"reason":"..."}. '
        "Rule: reply \"allow\" only when the command is clearly local, "
        "reversible, read-only, or is a test/build command whose effects are "
        "confined to the current repo. Anything that deletes, force-pushes, "
        "touches credentials, pipes curl|bash, or writes to network/paths "
        "outside the repo -> \"unknown\" (let the user decide). Never reply "
        "\"deny\".\n\n"
        "Recently user-approved examples (for calibration):\n"
        + ("\n".join(example_lines) or "- (none yet)")
        + f"\n\nCommand to classify (tool={tool_name}):\n{command}\n"
    )
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    try:
        parsed = json.loads(data.get("response", "{}"))
    except json.JSONDecodeError:
        return "unknown", 0.0, "llm returned non-json"
    decision = parsed.get("decision", "unknown")
    if decision == "deny":
        decision = "unknown"  # hard rule
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return decision, confidence, str(parsed.get("reason", ""))[:200]


def _in_night_window(cfg: dict) -> bool:
    """True if current hour falls within auto_proceed_window (shared semantics)."""
    window = cfg.get("auto_proceed_window")
    if not isinstance(window, dict):
        return False
    try:
        start = int(window.get("start_hour", 23))
        end = int(window.get("end_hour", 7))
    except (TypeError, ValueError):
        return False
    if start == end:
        return False
    h = datetime.now().hour
    if start < end:
        return start <= h < end
    return h >= start or h < end


def decide(tool_name: str, tool_input: dict, allow: dict) -> tuple[str, str]:
    """Return (decision, reason). decision ∈ {"approve", "block", ""}."""
    if tool_name in allow.get("safe_tools", []):
        return "approve", f"tool '{tool_name}' in safe_tools"

    if tool_name == "Bash":
        cmd = extract_bash_command(tool_input)
        if not cmd:
            return "", ""
        for pat in allow.get("deny_patterns", []):
            try:
                if re.search(pat, cmd):
                    return "block", f"matched deny_pattern: {pat}"
            except re.error:
                continue
        for prefix in allow.get("safe_bash_prefixes", []):
            if cmd == prefix or cmd.startswith(prefix + " ") or cmd.startswith(prefix + "\n"):
                return "approve", f"matched safe_bash_prefix: {prefix}"
    return "", ""


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {}) or {}
    cwd = Path(data.get("cwd") or os.getcwd())

    allow = load_allow_list(cwd)

    # Per-task permissive override: if an active task has auto_approve=True,
    # ADD its task_safe_prefixes to the allow-list (never reduces safety).
    marker = load_active_task_override()
    if marker.get("auto_approve") is True:
        extra = task_safe_prefixes_from_marker(marker)
        if extra:
            allow["safe_bash_prefixes"] = list(allow.get("safe_bash_prefixes", [])) + extra
            log(f"task_override  task={marker.get('task_id')}  "
                f"added_prefixes={len(extra)}")
    # auto_approve is False/None -> no-op (fall through to normal chain).

    decision, reason = decide(tool_name, tool_input, allow)

    preview = (extract_bash_command(tool_input) if tool_name == "Bash" else "")[:80]

    # Static deny is terminal. Otherwise try cache → optional LLM.
    cfg = verdict_cache.load_config()

    # Night-mode / auto-accept-all: when enabled AND within the configured
    # window, approve any tool that isn't explicitly denied. Deny-patterns
    # still win (rm -rf /, git push --force, DROP TABLE, ...). Principle:
    # "after 23, follow suggestions and never stop."
    if decision != "block" and _in_night_window(cfg) and cfg.get(
            "auto_approve_night_mode", True):
        decision, reason = "approve", "night-mode auto-accept"
    if decision != "deny" and tool_name == "Bash":
        cmd = extract_bash_command(tool_input)
        if cmd:
            try:
                conn = verdict_cache.connect()
                ttl = int(cfg.get("auto_approve_verdict_ttl_days", 30))
                hit = verdict_cache.lookup(conn, tool_name, cmd, ttl_days=ttl)
                if hit and hit["decision"] == "allow" and not decision:
                    decision, reason = "approve", (
                        f"cache hit ({hit['source']}, hits={hit['hit_count']})"
                    )
                # Vector-similarity classifier: between exact cache and LLM.
                if (not decision and not hit
                        and cfg.get("vector_autoapprove_enabled", True)):
                    try:
                        # Import skill_hub.embeddings lazily; if unavailable,
                        # silently skip and fall through to LLM / prompt.
                        import importlib.util
                        spec = importlib.util.find_spec("skill_hub.embeddings")
                        if spec is not None:
                            from skill_hub.embeddings import embed  # type: ignore
                            vec = embed(cmd)
                            thresh = float(cfg.get(
                                "vector_autoapprove_threshold", 0.88))
                            vhit = verdict_cache.search_by_vector(
                                conn, vec, threshold=thresh,
                                source_filter="user_approved",
                            )
                            if vhit:
                                decision, reason = "approve", (
                                    f"vector sim={vhit['similarity']:.3f} "
                                    f"≥ {thresh} (from {vhit['source']})"
                                )
                                # Cache as vector-source so next exact match
                                # is O(1).
                                verdict_cache.put_with_vector(
                                    conn, tool_name, cmd, "allow", "vector",
                                    vec, confidence=vhit["similarity"],
                                )
                                log(f"VECTOR hit sim={vhit['similarity']:.3f}")
                    except Exception as e:  # noqa: BLE001
                        log(f"VECTOR error={e}")
                if not decision and cfg.get("auto_approve_llm"):
                    try:
                        base = cfg.get("ollama_base_url", "http://localhost:11434")
                        model = cfg.get("auto_approve_model",
                                        cfg.get("local_models", {}).get("level_1",
                                                                         "qwen2.5-coder:3b"))
                        thresh = float(cfg.get("auto_approve_confidence", 0.85))
                        timeout = float(cfg.get("auto_approve_timeout_s", 4.0))
                        examples = verdict_cache.recent_examples(conn, n=6)
                        llm_dec, conf, llm_reason = llm_classify(
                            tool_name, cmd, examples, base, model, timeout
                        )
                        log(f"LLM  dec={llm_dec}  conf={conf:.2f}  "
                            f"reason=\"{llm_reason[:60]}\"")
                        if llm_dec == "allow" and conf >= thresh:
                            decision, reason = "approve", (
                                f"llm allow conf={conf:.2f}: {llm_reason}"
                            )
                            verdict_cache.put(
                                conn, tool_name, cmd, "allow", "llm", conf
                            )
                    except Exception as e:  # noqa: BLE001
                        log(f"LLM  error={e}")
            except Exception as e:  # noqa: BLE001
                log(f"cache  error={e}")

    log(f"{tool_name}  decision={decision or 'pass'}  cmd=\"{preview}\"  reason={reason}")

    if decision == "approve":
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": reason,
            }
        }
        print(json.dumps(out))
    elif decision == "block":
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
