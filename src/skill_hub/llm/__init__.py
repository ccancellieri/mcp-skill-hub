"""Pluggable LLM abstraction (S2 — F-LLM).

One call surface for every LLM-dependent module in skill-hub.

Usage::

    from skill_hub.llm import get_provider
    llm = get_provider()
    text = llm.complete("hello", tier="tier_cheap")

The provider is selected by ``config.llm_providers``. Under the hood we use
litellm so the same code works against Ollama, Anthropic, OpenAI, and other
providers with only a config change.
"""
from .provider import LLMProvider, Message, LLMError
from .litellm_adapter import LitellmProvider, get_provider
from .prompts import load_prompt

__all__ = [
    "LLMProvider",
    "LitellmProvider",
    "Message",
    "LLMError",
    "get_provider",
    "load_prompt",
]
