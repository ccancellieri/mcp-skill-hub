"""Activity logging for Skill Hub — zero Claude token cost visibility.

Logs all MCP tool calls, hook interceptions, and LLM invocations to:
1. Rotating daily log file at ~/.claude/mcp-skill-hub/logs/activity.log
2. stderr (visible in terminal, invisible to Claude)

Rotation: daily, total size capped at 50 MB (oldest files pruned).
"""

import logging
import os
import sys
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


def get_logger() -> logging.Logger:
    """Return the singleton activity logger, creating it on first call."""
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_logs()

    _logger = logging.getLogger("skill_hub.activity")
    _logger.setLevel(logging.INFO)
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
    _logger.addHandler(file_handler)

    # Handler 2: stderr (visible in terminal, invisible to Claude)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(
        "\033[2m[skill-hub] %(message)s\033[0m",  # dim gray
    ))
    _logger.addHandler(stderr_handler)

    return _logger


def _rotator_with_prune(source: str, dest: str) -> None:
    """Rotate the log file then prune if total size exceeds cap."""
    if os.path.exists(source):
        os.rename(source, dest)
    _prune_old_logs()


# ── Convenience helpers ─────────────────────────────────────────────

def log_tool(tool_name: str, **kwargs: object) -> None:
    """Log an MCP tool invocation."""
    params = "  ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    get_logger().info("TOOL  %-20s  %s", tool_name, params)


def log_hook(action: str, **kwargs: object) -> None:
    """Log a hook interception or classification."""
    params = "  ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    get_logger().info("HOOK  %-20s  %s", action, params)


def log_llm(operation: str, model: str = "", **kwargs: object) -> None:
    """Log a local LLM invocation (embed, rerank, compact, classify)."""
    params = "  ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    get_logger().info("LLM   %-20s  model=%s  %s", operation, model, params)


def log_event(category: str, message: str) -> None:
    """Log a general event."""
    get_logger().info("%-6s%s", category.upper(), message)


def log_skill_step(skill_name: str, step_type: str, detail: str,
                   result: str = "", ok: bool = True) -> None:
    """Log an L3 skill step with clear, readable formatting.

    Output format:
      SKILL [git-push] SHELL  $ git push origin main  →  ok (42 chars)
      SKILL [git-push] LLM    model=qwen2.5:7b prompt=120ch  →  "feat(iam): extract..."
      SKILL [git-push] IF     dirty contains "M" → goto:stash  (matched)
      SKILL [git-push] GOTO   → do_push
      SKILL [git-push] STOP   Rebase conflict — resolve manually
    """
    status = "ok" if ok else "FAIL"
    result_str = f"  →  {result}" if result else ""
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
