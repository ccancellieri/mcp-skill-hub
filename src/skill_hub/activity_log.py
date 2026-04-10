"""Activity logging for Skill Hub — zero Claude token cost visibility.

Two-tier narrative logging:
  INFO  (default) — human-readable pipeline summaries, decisions, outcomes
  DEBUG           — full internal detail for investigation

Logs to:
1. Rotating daily log file at ~/.claude/mcp-skill-hub/logs/activity.log
2. stderr (visible in terminal, invisible to Claude)

Rotation: daily, total size capped at 50 MB (oldest files pruned).
"""

import logging
import os
import sys
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_DEFAULT_LOG_DIR = Path.home() / ".claude" / "mcp-skill-hub" / "logs"
MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB


def _resolve_log_dir() -> Path:
    """Read log_dir from config, fall back to default."""
    try:
        from . import config as _cfg
        custom = _cfg.get("log_dir")
        if custom:
            return Path(str(custom)).expanduser()
    except Exception:
        pass
    return _DEFAULT_LOG_DIR


LOG_DIR = _resolve_log_dir()
LOG_FILE = LOG_DIR / "activity.log"

_logger: logging.Logger | None = None


def _prune_old_logs() -> None:
    """Delete oldest rotated logs if total size exceeds MAX_TOTAL_BYTES."""
    if not LOG_DIR.exists():
        return
    log_files = sorted(LOG_DIR.glob("activity.log*"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in log_files)
    while total > MAX_TOTAL_BYTES and len(log_files) > 1:
        oldest = log_files.pop(0)
        total -= oldest.stat().st_size
        oldest.unlink(missing_ok=True)


def _get_log_level() -> int:
    """Read log_level from config. Accepts 'DEBUG', 'INFO', etc."""
    try:
        from . import config as _cfg
        level = _cfg.get("log_level", "INFO")
        return getattr(logging, str(level).upper(), logging.INFO)
    except Exception:
        return logging.INFO


def get_logger() -> logging.Logger:
    """Return the singleton activity logger, creating it on first call."""
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_logs()

    _logger = logging.getLogger("skill_hub.activity")
    _logger.setLevel(_get_log_level())
    _logger.propagate = False

    _fmt = logging.Formatter(
        "%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handler 1: rotating daily file
    file_handler = TimedRotatingFileHandler(
        str(LOG_FILE),
        when="midnight",
        backupCount=0,  # we handle pruning ourselves
        encoding="utf-8",
        utc=False,
    )
    file_handler.namer = lambda name: name  # keep default .YYYY-MM-DD suffix
    file_handler.rotator = _rotator_with_prune
    file_handler.setFormatter(_fmt)
    file_handler.setLevel(logging.DEBUG)  # file captures everything
    _logger.addHandler(file_handler)

    # Handler 2: stderr (visible in terminal, invisible to Claude)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(
        "\033[2m[skill-hub] %(message)s\033[0m",  # dim gray
    ))
    stderr_handler.setLevel(logging.INFO)  # stderr only shows INFO+
    _logger.addHandler(stderr_handler)

    return _logger


def _rotator_with_prune(source: str, dest: str) -> None:
    """Rotate the log file then prune if total size exceeds cap."""
    if os.path.exists(source):
        os.rename(source, dest)
    _prune_old_logs()


def _human_size(chars: int) -> str:
    """Convert character count to human-readable size."""
    if chars < 1024:
        return f"{chars}ch"
    return f"{chars // 1024}KB"


def _preview(message: str, max_len: int = 60) -> str:
    """Truncate message to a readable preview."""
    preview = message[:max_len].replace("\n", " ").strip()
    if len(message) > max_len:
        preview += "\u2026"
    return preview


# ── Pipeline framing ───────────────────────────────────────────────
#
# Every hook pipeline run is framed by >> (input) and << (output).
# The << line clearly shows WHO handled the message and WHAT happened.
#
# Passthrough to Claude:
#   14:23:45  >> "fix the pagination bug in the STAC endpoint"
#   14:23:49     skills: 3 loaded (25KB)
#   14:23:49  << CLAUDE  3 skills (25KB, optimized)
#
# Handled locally (command):
#   14:25:01  >> "show git status"
#   14:25:01     local command matched: git_status (confidence=0.92)
#   14:25:01     exec L1: $ git status
#   14:25:02  << LOCAL  L1:git_status  "On branch main, nothing to commit…"
#
# Handled locally (skill):
#   14:26:10  >> "commit and push"
#   14:26:10     local skill matched: git-push (sim=0.93)
#   14:26:10     skill exec: git-push (steps, 7 steps)
#   14:26:10  SKILL [git-push] SHELL  $ git add -A  ->  ok (42 chars)  (ok)
#   14:26:11  SKILL [git-push] SHELL  $ git push    ->  ok (18 chars)  (ok)
#   14:26:11  << LOCAL  skill:git-push  "Changes committed and pushed…"
#
# Handled locally (task command):
#   14:27:00  >> "close task 3"
#   14:27:01     executing: close_task
#   14:27:01  << LOCAL  task:close_task  "Task #3 closed and compacted…"
#

def log_pipeline_start(message: str) -> None:
    """Log the beginning of a hook pipeline run with the user's message."""
    get_logger().info('>> "%s"', _preview(message))


def log_pipeline_end(target: str, reason: str = "", result: str = "") -> None:
    """Log the final outcome of a pipeline run.

    Args:
        target: "LOCAL" or "CLAUDE" — who handles the message.
        reason: what matched or what context was added (e.g. "L1:git_status",
                "3 skills (25KB)", "passthrough").
        result: output preview for LOCAL handling (truncated to 80 chars).
    """
    parts = [target.upper()]
    if reason:
        parts.append(reason)
    if result:
        parts.append(f'"{_preview(result, 80)}"')
    get_logger().info("<< %s", "  ".join(parts))


def log_step(message: str) -> None:
    """Log an important intermediate step during pipeline (indented)."""
    get_logger().info("   %s", message)


def log_detail(message: str) -> None:
    """Log a routine/verbose detail — only visible at DEBUG level."""
    get_logger().debug("   %s", message)


# ── LLM call timing ───────────────────────────────────────────────

class llm_timer:
    """Context manager that tracks LLM call duration.

    Usage::

        with llm_timer() as t:
            result = call_ollama(...)
        log_llm("classify", model="qwen2.5:3b", duration=t.duration, intent="none")
    """

    def __init__(self) -> None:
        self.duration: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "llm_timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        self.duration = time.monotonic() - self._start


# ── Convenience helpers ─────────────────────────────────────────────

def log_tool(tool_name: str, **kwargs: object) -> None:
    """Log an MCP tool invocation."""
    params = "  ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    get_logger().info("TOOL  %-20s  %s", tool_name, params)


def log_hook(action: str, **kwargs: object) -> None:
    """Log a hook interception or classification."""
    params = "  ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    get_logger().info("HOOK  %-20s  %s", action, params)


def log_llm(operation: str, model: str = "", duration: float = 0.0,
            **kwargs: object) -> None:
    """Log a local LLM invocation (embed, rerank, compact, classify).

    Args:
        duration: wall-clock seconds (0 = not measured, omitted from output).
    """
    params = "  ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    dur = f"  ({duration:.1f}s)" if duration > 0 else ""
    get_logger().info("LLM   %-20s  model=%s  %s%s", operation, model, params, dur)


def log_event(category: str, message: str) -> None:
    """Log a general event."""
    get_logger().info("%-8s %s", category.upper(), message)


def log_skill_step(skill_name: str, step_type: str, detail: str,
                   result: str = "", ok: bool = True) -> None:
    """Log an L3 skill step with clear, readable formatting.

    Output format:
      SKILL [git-push] SHELL  $ git push origin main  ->  ok (42 chars)
      SKILL [git-push] SHELL  $ git rev-parse --is-inside-work-tree  ->  exit 128 (81 chars)
      SKILL [git-push] LLM    model=qwen2.5:7b prompt=120ch  ->  "feat(iam): extract..."  (ok)
      SKILL [git-push] IF     dirty contains "M" -> goto:stash  (matched)
      SKILL [git-push] GOTO   -> do_push
      SKILL [git-push] STOP   Rebase conflict -- resolve manually
    """
    result_str = f"  ->  {result}" if result else ""
    if step_type == "SHELL":
        # exit status is embedded in result ("ok (N chars)" or "exit N (N chars)") — skip suffix
        get_logger().info("SKILL [%s] %-6s %s%s", skill_name, step_type, detail, result_str)
    else:
        status = "ok" if ok else "FAIL"
        get_logger().info(
            "SKILL [%s] %-6s %s%s  (%s)",
            skill_name, step_type, detail, result_str, status,
        )


def log_banner() -> None:
    """Log startup banner with session stats — called once when MCP server starts."""
    from . import config as _cfg
    from .store import SkillStore

    cfg = _cfg.load_config()
    logger = get_logger()

    store = SkillStore()
    skill_count = store.count_skills()
    task_counts = store.count_tasks()
    teaching_count = store.count_teachings()
    interceptions = store.count_interceptions()
    tokens_saved = store.total_tokens_saved()
    store.close()

    logger.info("=" * 60)
    logger.info("SKILL HUB SESSION START")
    logger.info("-" * 60)
    logger.info("embed=%s  reason=%s",
                cfg.get("embed_model", "?"), cfg.get("reason_model", "?"))
    logger.info("skills=%d  tasks=%d open/%d closed  teachings=%d",
                skill_count,
                task_counts.get("open", 0),
                task_counts.get("closed", 0),
                teaching_count)
    logger.info("hook=%s  context_inject=%s  token_profiling=%s",
                "on" if cfg.get("hook_enabled") else "off",
                "on" if cfg.get("hook_context_injection") else "off",
                "on" if cfg.get("token_profiling") else "off")
    logger.info("interceptions=%d  tokens_saved=~%d",
                interceptions, tokens_saved)
    logger.info("-" * 60)
    logger.info("LOG: tail -100f %s", LOG_FILE)
    logger.info("=" * 60)
