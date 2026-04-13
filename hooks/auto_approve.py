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
import shlex
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# Built-in preset bundles for adaptive windows + task_type overrides.
# Users can reference these by name in config (prefix_bundle / task_type_bundles).
# "all_non_denied" is a sentinel: when active, we approve anything not denied
# (same semantics as the legacy binary night-mode window).
READ_ONLY_BUNDLE = [
    "sed -n", "head", "tail", "grep", "rg", "find", "wc",
    "file", "stat", "ls", "cat", "tree", "which", "echo", "pwd", "cd",
    "git status", "git diff", "git log", "git branch", "git show",
]
BUILD_BUNDLE = [
    "pytest", "uv run pytest", "npm test", "npm run", "make",
    "uv run", "python -m pytest", "ruff", "mypy", "pyright",
]
DEPLOY_BUNDLE = [
    "docker build", "docker compose", "kubectl get", "kubectl describe",
    "gh pr", "gh run",
]

BUILTIN_BUNDLES: dict[str, list[str]] = {
    "read_only": READ_ONLY_BUNDLE,
    "build": BUILD_BUNDLE,
    "deploy": DEPLOY_BUNDLE,
    "research": READ_ONLY_BUNDLE,  # alias
}
ALL_NON_DENIED = "all_non_denied"

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


def _hour_in_window(h: int, start: int, end: int) -> bool:
    """True if hour h is in [start, end) handling overnight wrap (start>end)."""
    if start == end:
        return False
    if start < end:
        return start <= h < end
    return h >= start or h < end


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
    return _hour_in_window(datetime.now().hour, start, end)


def active_adaptive_window(cfg: dict, now_hour: int | None = None) -> dict | None:
    """Return the first matching adaptive window for the current hour, or None.

    Config schema:
        "adaptive_windows": [
            {"name":"evening","start_hour":18,"end_hour":23,"prefix_bundle":"read_only"},
            {"name":"night","start_hour":23,"end_hour":7,"prefix_bundle":"all_non_denied"}
        ]
    """
    windows = cfg.get("adaptive_windows")
    if not isinstance(windows, list):
        return None
    h = datetime.now().hour if now_hour is None else now_hour
    for w in windows:
        if not isinstance(w, dict):
            continue
        try:
            s = int(w.get("start_hour"))
            e = int(w.get("end_hour"))
        except (TypeError, ValueError):
            continue
        if _hour_in_window(h, s, e):
            return w
    return None


def resolve_bundle(name: str, cfg: dict) -> list[str]:
    """Resolve a bundle name to a list of prefixes.

    Config may override/extend via top-level "prefix_bundles" dict (name -> list).
    Returns the sentinel name as-is inside a 1-item list for all_non_denied,
    handled separately by caller.
    """
    if name == ALL_NON_DENIED:
        return []
    user_bundles = cfg.get("prefix_bundles")
    if isinstance(user_bundles, dict) and name in user_bundles:
        raw = user_bundles[name]
        if isinstance(raw, list):
            return [str(x) for x in raw if x]
    return list(BUILTIN_BUNDLES.get(name, []))


def task_type_prefixes(marker: dict, cfg: dict) -> list[str]:
    """Map active_task.task_type -> extra safe prefixes via task_type_bundles.

    Config:
        "task_type_bundles": {"research": "read_only", "build": "build"}
    Values may be a bundle NAME (str) or an inline list of prefixes.
    """
    ttype = marker.get("task_type")
    if not isinstance(ttype, str) or not ttype:
        return []
    mapping = cfg.get("task_type_bundles")
    if not isinstance(mapping, dict):
        return []
    spec = mapping.get(ttype)
    if isinstance(spec, list):
        return [str(x) for x in spec if x]
    if isinstance(spec, str):
        return resolve_bundle(spec, cfg)
    return []


def _split_bash_tokens(cmd: str) -> tuple[list[str], list[str]]:
    """Return (unquoted_tokens, quoted_tokens) using shlex.

    We re-scan the raw string with shlex.shlex(posix=False) to know whether
    each token originated inside quotes. Tokens inside quotes go to the
    "quoted" list and are EXCLUDED from deny-regex scoping.
    """
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    unquoted: list[str] = []
    quoted: list[str] = []
    # shlex.shlex in posix mode loses quoting info. Walk manually instead.
    # Strategy: tokenise with posix=True to get logical tokens, then for each
    # token find its position in the original string and check if it sits
    # inside a quoted span.
    try:
        tokens = list(shlex.split(cmd, posix=True))
    except ValueError:
        # Unbalanced quotes — treat the entire string as unquoted (safe side:
        # deny patterns still scan everything, same as the legacy behavior).
        return [cmd], []
    # Compute quoted character mask over cmd.
    mask = [False] * len(cmd)
    in_q: str | None = None
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if in_q:
            if c == in_q:
                in_q = None
            else:
                mask[i] = True
            i += 1
            continue
        if c in ("'", '"'):
            in_q = c
        i += 1
    # For each token, locate its first occurrence after the scan cursor and
    # check whether its span overlaps a quoted region.
    cursor = 0
    for tok in tokens:
        if not tok:
            continue
        idx = cmd.find(tok, cursor)
        if idx < 0:
            # Couldn't locate (e.g. due to unescaping) — be conservative and
            # treat as unquoted so deny patterns still fire.
            unquoted.append(tok)
            continue
        end = idx + len(tok)
        # If ANY char of the token span is inside a quoted region, classify
        # as quoted. (Mixed cases are rare; commit -m "..." produces a pure
        # quoted-inside token post-shlex.)
        if any(mask[idx:end]):
            quoted.append(tok)
        else:
            unquoted.append(tok)
        cursor = end
    return unquoted, quoted


def _quote_mask(cmd: str) -> list[bool]:
    """Return a per-character mask where True means the char is inside quotes."""
    mask = [False] * len(cmd)
    in_q: str | None = None
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if in_q:
            if c == "\\" and i + 1 < len(cmd):
                # Skip escaped char inside double quotes.
                mask[i] = True
                mask[i + 1] = True
                i += 2
                continue
            if c == in_q:
                in_q = None
            else:
                mask[i] = True
            i += 1
            continue
        if c in ("'", '"'):
            in_q = c
        i += 1
    return mask


# Shell operators that separate compound commands. Order matters: longer first.
_COMPOUND_OPS = ("&&", "||", ";", "|")


def split_compound_segments(cmd: str) -> list[str]:
    """Split a shell command string into segments on `&&`, `||`, `;`, `|`.

    Operators inside single or double quotes are ignored. Returns trimmed
    segments, dropping empties. If the command has no operators (or they are
    all inside quotes), returns [cmd.strip()].
    """
    if not cmd:
        return []
    mask = _quote_mask(cmd)
    segments: list[str] = []
    start = 0
    i = 0
    n = len(cmd)
    while i < n:
        if mask[i]:
            i += 1
            continue
        matched = None
        for op in _COMPOUND_OPS:
            if cmd.startswith(op, i) and not any(mask[i:i + len(op)]):
                # Don't treat `|` as a split when it's actually `||` already
                # consumed above (order handles this since `||` comes first).
                matched = op
                break
        if matched is not None:
            seg = cmd[start:i].strip()
            if seg:
                segments.append(seg)
            i += len(matched)
            start = i
            continue
        i += 1
    tail = cmd[start:].strip()
    if tail:
        segments.append(tail)
    return segments or [cmd.strip()]


def _cd_target_ok(segment: str, cfg: dict) -> bool:
    """Return True if `segment` is `cd <path>` targeting HOME or a workspace dir.

    Accepts `cd`, `cd -`, `cd ~`, `cd ~/...`, `cd $HOME/...`, or absolute paths
    under HOME or any entry in cfg['workspace_dirs'] (list of str).
    """
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        return False
    if not tokens or tokens[0] != "cd":
        return False
    if len(tokens) == 1:
        return True  # bare `cd` -> HOME
    if len(tokens) > 2:
        # `cd path && other` was already split; extra tokens here are flags.
        # Accept `cd -` (previous dir) but nothing else multi-arg.
        return False
    target = tokens[1]
    if target in ("-", "~"):
        return True
    if target.startswith("~/"):
        return True
    # Expand env vars and user.
    expanded = os.path.expanduser(os.path.expandvars(target))
    home = str(Path.home())
    workspace_dirs = [home]
    extra = cfg.get("workspace_dirs") if isinstance(cfg, dict) else None
    if isinstance(extra, list):
        workspace_dirs.extend(str(p) for p in extra if p)
    # Resolve relative paths against CWD — but for safety only accept absolute
    # paths or `~`-relative paths. Relative `cd foo` is allowed only if foo
    # contains no `..` traversal beyond workspace roots, which we can't check
    # without the runtime CWD; accept plain names without `/` conservatively.
    if not expanded.startswith("/"):
        # Relative path without `..` — accept (conservative, still bounded by
        # CWD which is inside a workspace anyway when the hook fires).
        return ".." not in expanded.split("/")
    for root in workspace_dirs:
        root_abs = os.path.abspath(root).rstrip("/") + "/"
        if (expanded + "/").startswith(root_abs) or expanded == root_abs.rstrip("/"):
            return True
    return False


def _segment_matches_prefix(segment: str, prefixes: list[str]) -> str | None:
    """Return the matching prefix for `segment`, or None."""
    for prefix in prefixes:
        if (segment == prefix
                or segment.startswith(prefix + " ")
                or segment.startswith(prefix + "\n")):
            return prefix
    return None


def scoped_deny_haystack(cmd: str) -> str:
    """Return the deny-pattern scan target: unquoted tokens joined by spaces.

    Quoted arguments (commit messages, echo strings, grep patterns) are
    excluded so benign text inside them can't trigger a deny match.
    """
    unquoted, _ = _split_bash_tokens(cmd)
    return " ".join(unquoted)


def decide(tool_name: str, tool_input: dict, allow: dict,
           bundle_name: str | None = None, cfg: dict | None = None
           ) -> tuple[str, str]:
    """Return (decision, reason). decision ∈ {"approve", "block", ""}.

    `bundle_name`, when provided, activates either a named prefix bundle
    (approve if cmd matches any prefix in the bundle) or the
    "all_non_denied" sentinel (approve anything not denied).
    """
    cfg = cfg or {}
    if tool_name in allow.get("safe_tools", []):
        return "approve", f"tool '{tool_name}' in safe_tools"

    if tool_name == "Bash":
        cmd = extract_bash_command(tool_input)
        if not cmd:
            return "", ""
        # Deny scan: only on unquoted tokens. This fixes the bug where
        # `git commit -m "... rm -rf / ..."` got blocked because its message
        # contained the literal deny string.
        scan = scoped_deny_haystack(cmd)
        for pat in allow.get("deny_patterns", []):
            try:
                if re.search(pat, scan):
                    return "block", f"matched deny_pattern: {pat}"
            except re.error:
                continue
        # Adaptive window bundle (if any).
        if bundle_name == ALL_NON_DENIED:
            return "approve", "adaptive window: all_non_denied"
        window_prefixes: list[str] = []
        if bundle_name:
            window_prefixes = resolve_bundle(bundle_name, cfg)
        all_prefixes = list(allow.get("safe_bash_prefixes", [])) + window_prefixes

        # Split compound commands on &&, ||, ;, | (respecting quotes).
        segments = split_compound_segments(cmd)
        if len(segments) == 1:
            prefix = _segment_matches_prefix(segments[0], all_prefixes)
            if prefix is not None:
                return "approve", f"matched safe_bash_prefix: {prefix}"
            # Also honor `cd <safe-path>` as a single segment.
            if _cd_target_ok(segments[0], cfg):
                return "approve", "cd to workspace/home path"
            return "", ""
        # Compound: every segment must match a prefix (or be a safe `cd`).
        matched_prefixes: list[str] = []
        for seg in segments:
            if _cd_target_ok(seg, cfg):
                matched_prefixes.append("cd")
                continue
            p = _segment_matches_prefix(seg, all_prefixes)
            if p is None:
                return "", ""
            matched_prefixes.append(p)
        return "approve", (
            f"all {len(segments)} segments in allow-list "
            f"({', '.join(sorted(set(matched_prefixes)))})"
        )
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

    cfg = verdict_cache.load_config()

    # Per-task permissive override: if an active task has auto_approve=True,
    # ADD its task_safe_prefixes to the allow-list (never reduces safety).
    # task_type (if present on the marker) maps to a preset bundle via
    # cfg["task_type_bundles"] and is also added additively.
    marker = load_active_task_override()
    if marker.get("auto_approve") is True:
        extra = task_safe_prefixes_from_marker(marker)
        tt_extra = task_type_prefixes(marker, cfg)
        combined = extra + tt_extra
        if combined:
            allow["safe_bash_prefixes"] = (
                list(allow.get("safe_bash_prefixes", [])) + combined
            )
            log(
                f"task_override  task={marker.get('task_id')}  "
                f"task_type={marker.get('task_type')!r}  "
                f"added_prefixes={len(combined)} "
                f"(explicit={len(extra)}, task_type={len(tt_extra)})"
            )
    # auto_approve is False/None -> no-op (fall through to normal chain).

    # Resolve adaptive time-tier. New schema takes precedence; legacy
    # auto_approve_night_mode + auto_proceed_window remains as fallback.
    window = active_adaptive_window(cfg)
    bundle_name = None
    if window is not None:
        bundle_name = window.get("prefix_bundle")
        log(f"adaptive_window  name={window.get('name')!r}  bundle={bundle_name!r}")
    elif _in_night_window(cfg) and cfg.get("auto_approve_night_mode", True):
        # Legacy behavior: binary night-mode = all_non_denied.
        bundle_name = ALL_NON_DENIED

    decision, reason = decide(tool_name, tool_input, allow,
                               bundle_name=bundle_name, cfg=cfg)

    # Non-Bash tools (Edit, Write, ...) don't have deny scanning or prefix
    # matching inside decide(). When the active window is "all_non_denied"
    # (legacy night-mode semantics), approve any non-Bash tool outright —
    # Bash is handled inside decide() with its deny scan.
    if (not decision and bundle_name == ALL_NON_DENIED
            and tool_name != "Bash"):
        decision, reason = "approve", "adaptive window: all_non_denied (non-Bash)"

    preview = (extract_bash_command(tool_input) if tool_name == "Bash" else "")[:80]
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

    # Ask-the-user fallback for ambiguous Bash commands when explicitly enabled.
    if (not decision and tool_name == "Bash"
            and cfg.get("ask_user_on_ambiguous", False)):
        cmd_full = extract_bash_command(tool_input)
        if cmd_full:
            try:
                ask_url = cfg.get("dashboard_server_url",
                                  "http://127.0.0.1:8765")
                host = cfg.get("dashboard_server_host", "127.0.0.1")
                port = int(cfg.get("dashboard_server_port", 8765))
                base = f"http://{host}:{port}"
                body = json.dumps({
                    "prompt": f"Allow this command? {cmd_full[:160]}",
                    "command": cmd_full,
                    "tool_name": tool_name,
                }).encode()
                req = urllib.request.Request(
                    f"{base}/questions/ask", data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as r:
                    qdata = json.loads(r.read().decode())
                qid = qdata.get("id")
                if qid:
                    timeout_s = float(cfg.get("ask_user_timeout_s", 3.0))
                    poll_url = f"{base}/questions/{qid}/answer"
                    deadline = datetime.now().timestamp() + timeout_s
                    import time as _time
                    answered = None
                    while datetime.now().timestamp() < deadline:
                        # Poll the list endpoint to see if status flipped.
                        try:
                            with urllib.request.urlopen(
                                    f"{base}/questions/list", timeout=0.8
                            ) as lr:
                                payload = json.loads(lr.read().decode())
                            for q in payload.get("recent", []):
                                if (q.get("id") == qid
                                        and q.get("status") == "answered"):
                                    answered = q
                                    break
                            if answered:
                                break
                        except Exception:  # noqa: BLE001
                            pass
                        _time.sleep(0.25)
                    if answered:
                        if answered.get("decision") == "allow":
                            decision, reason = "approve", "user (ui) allowed"
                        elif answered.get("decision") == "deny":
                            decision, reason = "block", "user (ui) denied"
                        log(f"ASK_USER  qid={qid}  -> {answered.get('decision')}")
                    else:
                        log(f"ASK_USER  qid={qid}  timeout (no answer)")
            except Exception as e:  # noqa: BLE001
                log(f"ASK_USER  error={e}")

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
