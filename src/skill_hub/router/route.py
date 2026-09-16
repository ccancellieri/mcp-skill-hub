"""Build bounded context without changing the client's model or user prompt."""
from __future__ import annotations

import os
from typing import Any

from .. import config as _cfg


def _compress_stage(output: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper for callers of the deterministic compressor."""
    from ..compression import maybe_compress

    if not cfg.get("router_compress_context_enabled", True):
        return output
    budget = int(cfg.get("router_compress_budget_tokens", 1500))
    for key in ("systemMessage", "userMessage"):
        text = output.get(key)
        if text and len(text) // 4 > budget:
            output[key] = maybe_compress(text, site=f"router.{key}")
    return output


def route(
    prompt: str,
    session_id: str = "",
    cwd: str = "",
    task_id: int | None = None,
) -> dict[str, Any]:
    """Return supplemental evidence in the existing CLI handoff format.

    No model classification, settings writes, prompt paraphrasing or
    provisioning occurs on this path. Missing context never blocks a turn.
    """
    cfg = _cfg.load_config()
    if (os.environ.get("SKILL_HUB_ROUTER_ENABLED") == "0"
            or not cfg.get("router_enabled", True)
            or not cfg.get("hook_enabled", True)
            or not cfg.get("context_enabled", True)):
        return {}
    from ..context_service import build_context

    try:
        result = build_context(
            prompt, cwd=cwd, session_id=session_id, task_id=task_id, cfg=cfg,
        )
    except Exception:
        return {}
    if not result["context"]:
        return {}
    return {"userMessage": result["context"]}
