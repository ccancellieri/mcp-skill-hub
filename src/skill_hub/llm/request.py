"""Common entry point for optional server-side auxiliary model calls.

Tier intent resolves through server configuration. Explicit model selections
remain pinned by the provider adapter. The foreground hook never calls a model;
L3 refers to the native client agent, not an additional server API tier.
"""
from __future__ import annotations

import os
from typing import Callable

from .provider import LLMError, LLMProvider


def hot_path_only() -> bool:
    """True inside the deterministic foreground hook. No model calls are allowed."""
    return os.environ.get("SKILL_HUB_LOCAL_ONLY") == "1"


def request(
    tier_intent: str,
    prompt: str,
    *,
    local_only: bool = False,
    model: str | None = None,
    op: str = "",
    timeout: float = 60.0,
    temperature: float = 0.2,
    max_tokens: int = 512,
    cache: bool = False,
    cache_ttl: str = "",
    complexity: float | None = None,
    domain: str | None = None,
    get_provider_fn: Callable[[], LLMProvider] | None = None,
) -> str:
    """Route an auxiliary completion or return an empty result on LLMError.

    The foreground hook fails closed before resolving any provider. Background
    callers retain configured routing; an explicit model is passed unchanged.
    get_provider_fn is an injectable provider getter for existing integrations.
    """
    tier = tier_intent if tier_intent.startswith("tier_") else f"tier_{tier_intent}"

    if hot_path_only():
        return ""

    if local_only and model:
        from .escalation import ollama_daemon_reachable
        if not ollama_daemon_reachable():
            return ""

    if get_provider_fn is None:
        from . import get_provider as get_provider_fn  # package singleton

    try:
        return get_provider_fn().complete(
            prompt,
            tier=tier,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            cache=cache,
            cache_ttl=cache_ttl,
            op=op,
            complexity=complexity,
            domain=domain,
        )
    except LLMError:
        return ""
