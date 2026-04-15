"""LLMProvider Protocol — unified call surface for chat/complete/embed.

Implementations must be safe to call from both sync and async code; they can
block — the MCP server is threaded. Errors bubble up as ``LLMError`` so
callers can branch on a single exception type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when an LLM call fails (transport, parse, timeout, auth)."""


@dataclass
class Message:
    role: str   # "system", "user", "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        tier: str = "tier_cheap",
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
        stop: list[str] | None = None,
        cache: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Single-turn text completion. Returns generated text.

        ``cache=True`` attaches an ephemeral ``cache_control`` marker to the
        last user content block for Anthropic models; no-op elsewhere.
        """
        ...

    def chat(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        tier: str = "tier_cheap",
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
        cache: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Multi-turn chat. ``messages`` is a list of ``{role, content}`` dicts
        or ``Message`` dataclasses. Returns the assistant's text."""
        ...

    def embed(
        self,
        text: str | list[str],
        *,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> list[float] | list[list[float]]:
        """Embedding vector(s). Returns a single vector if ``text`` is a string,
        else a list of vectors in the same order as the input."""
        ...
