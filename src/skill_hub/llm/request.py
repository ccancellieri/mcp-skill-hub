"""Unified work-layer LLM call API (issue #128).

``request()`` is the ONE entry point consumers should use for single-turn
completions instead of reaching for ``get_provider()`` directly and
hand-rolling hot-path fast-fail checks. ``tier_intent`` is semantic
("cheap" | "mid" | "smart" — the ``tier_`` prefix is optional, so callers
that already carry a config-style ``tier_*`` string can pass it unchanged)
and resolves through ``config.llm_providers`` inside the provider's
``complete()``; the escalation ladder underneath (:mod:`skill_hub.llm.escalation`)
then picks L1 (local Ollama) / L2 (work gateway) / L3 (personal Anthropic) by
availability, quota, and cooldown. Adding a provider is config-only — no call
site needs to change.
"""
from __future__ import annotations

import os
from typing import Callable

from .provider import LLMError, LLMProvider


def hot_path_only() -> bool:
    """True inside a per-prompt UserPromptSubmit hook (``SKILL_HUB_LOCAL_ONLY=1``).

    On the hot path only the fast local backend is acceptable — a remote
    round-trip blows the hook's latency budget, so the caller must fail fast
    and fall back to a deterministic path instead of computing (and
    discarding) an enriched result. The detached async-escalation worker
    clears this env var before running, so it keeps the full ladder.
    """
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
    """Single-turn completion routed through the escalation ladder.

    ``tier_intent`` is semantic ("cheap"/"mid"/"smart"); it is normalized to
    ``tier_<intent>`` and resolved via ``config.llm_providers`` unless
    ``model`` pins a specific model (still ladder-rescued when that model is
    local Ollama and unreachable — see ``LitellmProvider.chat``).

    Pass ``local_only=True`` for latency-critical hot-path callers (the
    per-prompt hooks): when a ``model`` is pinned and the local Ollama daemon
    is unreachable, returns ``""`` immediately instead of escalating to a
    remote provider. The same fast-fail applies whenever
    ``SKILL_HUB_LOCAL_ONLY=1`` is set (see :func:`hot_path_only`), regardless
    of the ``local_only`` argument. Background/detached callers leave
    ``local_only`` False (and usually pass ``model=None``) to keep the full
    ladder.

    ``get_provider_fn`` lets a caller supply its own (typically module-level,
    test-patchable) provider getter instead of the package singleton;
    resolution stays lazy so a fast-failed call never touches it.

    Returns the generated text, or ``""`` on an ``LLMError`` (transport,
    parse, timeout, auth) so callers can keep their tolerant fallbacks.
    """
    tier = tier_intent if tier_intent.startswith("tier_") else f"tier_{tier_intent}"

    if (local_only or hot_path_only()) and model:
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
