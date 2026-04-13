"""Tier 3 — Claude Haiku batched classifier.

A single Haiku call amortises the per-request cost by returning four results
simultaneously:
  1. Classification (complexity / ambiguity / scope / domain_hints / confidence)
  2. Settings optimisation hint (one config key worth tweaking)
  3. Compact hint (/compact suggestion based on estimated context depth)
  4. Subtask decomposition (if the prompt contains independent sub-problems)

Each task can be individually disabled via config (router_haiku_* keys).
The call only fires when:
  - router_haiku_enabled = True  OR  SKILL_HUB_ROUTER_HAIKU env var = "1"
  - Tier-2 confidence < router_haiku_threshold (default 0.7)
  - ANTHROPIC_API_KEY is set
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .. import config as _cfg

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(
    prompt: str,
    cfg: dict[str, Any],
    msg_count: int,
    current_config_summary: str,
    cwd: str = "",
) -> str:
    tasks: list[str] = []

    if cfg.get("router_haiku_classify", True):
        tasks.append(
            '"classification": {'
            '"complexity": 0.0-1.0, '
            '"ambiguity": 0.0-1.0, '
            '"scope": "single|multi|cross-repo", '
            '"domain_hints": ["debugging"|"architecture"|"testing"|"frontend"|"database"|"security"|"devops"|"api"], '
            '"confidence": 0.0-1.0}'
        )

    if cfg.get("router_haiku_settings_opt", True):
        tasks.append(
            '"settings_opt": {"key": "<config_key>", "value": <new_value>, "reason": "<one line>"} '
            '— or null if no change is warranted. '
            f'Current relevant config: {current_config_summary}'
        )

    if cfg.get("router_haiku_compact_hint", True):
        tasks.append(
            f'"compact_hint": {{"suggest_compact": bool, "reason": "<one line>"}} '
            f'— session has approximately {msg_count} messages so far'
        )

    if cfg.get("router_haiku_subtask_decomp", True):
        tasks.append(
            '"subtasks": ["<task 1>", ...] — list ONLY if the message contains 2+ '
            'clearly independent sub-problems; empty list otherwise'
        )

    schema = "{\n  " + ",\n  ".join(tasks) + "\n}"

    project_line = ""
    if cwd:
        project_name = os.path.basename(cwd.rstrip("/"))
        project_line = f"Project context: {project_name} ({cwd})\n\n"

    return (
        "You are a coding-assistant meta-router. Analyse the user message and "
        "return ONLY a JSON object with these fields (no prose, no markdown):\n\n"
        f"{schema}\n\n"
        f"{project_line}"
        f"User message:\n{prompt[:2000]}\n\nJSON:"
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HaikuResult:
    complexity: float = 0.5
    ambiguity: float = 0.3
    scope: str = "single"
    domain_hints: list[str] = field(default_factory=list)
    confidence: float = 0.7
    settings_opt: dict[str, Any] = field(default_factory=dict)
    compact_hint: dict[str, Any] = field(default_factory=dict)
    subtasks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def is_enabled(cfg: dict[str, Any]) -> bool:
    """Return True if Tier 3 is available (API key present + config enabled)."""
    env_flag = os.environ.get("SKILL_HUB_ROUTER_HAIKU", "")
    config_flag = cfg.get("router_haiku_enabled", False)
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    return has_key and (env_flag == "1" or config_flag)


def classify(
    prompt: str,
    cfg: dict[str, Any] | None = None,
    msg_count: int = 0,
    cwd: str = "",
) -> HaikuResult | None:
    """Call Haiku with a batched prompt. Returns None on any error."""
    if cfg is None:
        cfg = _cfg.load_config()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    # Build a compact summary of router-relevant config keys for the settings_opt task
    relevant_keys = [
        "hook_semantic_threshold", "hook_context_injection", "hook_llm_triage",
        "task_decomposition_enabled", "router_ollama_model", "reason_model",
    ]
    config_summary = json.dumps({k: cfg.get(k) for k in relevant_keys if k in cfg})

    system_msg = _build_prompt(prompt, cfg, msg_count, config_summary, cwd=cwd)

    try:
        resp = httpx.post(
            _ANTHROPIC_API,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": _HAIKU_MODEL,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": system_msg}],
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        data: dict[str, Any] = json.loads(m.group())
    except Exception:
        return None

    try:
        cl = data.get("classification") or {}
        result = HaikuResult(
            complexity=float(cl.get("complexity", 0.5)),
            ambiguity=float(cl.get("ambiguity", 0.3)),
            scope=str(cl.get("scope", "single")),
            domain_hints=[str(d) for d in cl.get("domain_hints", [])],
            confidence=float(cl.get("confidence", 0.7)),
        )
        so = data.get("settings_opt") or {}
        if isinstance(so, dict) and so.get("key"):
            result.settings_opt = so
        ch = data.get("compact_hint") or {}
        if isinstance(ch, dict):
            result.compact_hint = ch
        st = data.get("subtasks") or []
        if isinstance(st, list):
            result.subtasks = [str(s) for s in st]
        return result
    except (TypeError, ValueError):
        return None
