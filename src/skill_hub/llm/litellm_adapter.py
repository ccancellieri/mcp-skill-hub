"""litellm-backed ``LLMProvider`` implementation.

Handles tier → model resolution, Ollama base-URL wiring, and error wrapping.
Singleton per-process: ``get_provider()`` returns a cached instance.
"""
from __future__ import annotations

import logging
from typing import Any

from .. import config as _cfg
from .provider import LLMError, LLMProvider, Message

_log = logging.getLogger(__name__)


class LitellmProvider:
    """Wraps ``litellm.completion`` and ``litellm.embedding``.

    Tier resolution reads ``config.llm_providers``; callers may also pass an
    explicit ``model`` string (takes precedence over ``tier``).
    """

    def __init__(self) -> None:
        import litellm
        self._litellm = litellm
        # Quieter by default — caller logs outcomes.
        try:
            litellm.suppress_debug_info = True
            litellm.drop_params = True  # silently drop unsupported kwargs
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Helpers

    def _resolve_model(self, tier: str, model: str | None) -> str:
        if model:
            return model
        providers = _cfg.get("llm_providers") or {}
        resolved = providers.get(tier) if isinstance(providers, dict) else None
        if not resolved:
            default_tier = str(_cfg.get("llm_default_tier") or "tier_cheap")
            resolved = (providers or {}).get(default_tier)
        if not resolved:
            raise LLMError(f"no model configured for tier={tier!r}")
        return str(resolved)

    def _api_base(self, model: str) -> str | None:
        if model.startswith("ollama/"):
            return str(_cfg.get("ollama_base") or "http://localhost:11434")
        return None

    def _normalize_messages(
        self, messages: list[Message] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, Message):
                out.append({"role": m.role, "content": m.content})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    # ------------------------------------------------------------------
    # Public API

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
        extra: dict[str, Any] | None = None,
    ) -> str:
        return self.chat(
            [Message(role="user", content=prompt)],
            tier=tier, model=model, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
            extra={**(extra or {}), **({"stop": stop} if stop else {})},
        )

    def chat(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        tier: str = "tier_cheap",
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
        extra: dict[str, Any] | None = None,
    ) -> str:
        resolved_model = self._resolve_model(tier, model)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": self._normalize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        api_base = self._api_base(resolved_model)
        if api_base:
            kwargs["api_base"] = api_base
        if extra:
            kwargs.update(extra)
        try:
            resp = self._litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"completion failed ({resolved_model}): {exc}") from exc
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {resolved_model}: {exc}") from exc

    def embed(
        self,
        text: str | list[str],
        *,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> list[float] | list[list[float]]:
        providers = _cfg.get("llm_providers") or {}
        resolved = model or (providers.get("embed") if isinstance(providers, dict) else None)
        if not resolved:
            resolved = f"ollama/{_cfg.get('embed_model') or 'nomic-embed-text'}"
        api_base = self._api_base(resolved)
        kwargs: dict[str, Any] = {
            "model": resolved,
            "input": text if isinstance(text, list) else [text],
            "timeout": timeout,
        }
        if api_base:
            kwargs["api_base"] = api_base
        try:
            resp = self._litellm.embedding(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"embedding failed ({resolved}): {exc}") from exc
        try:
            vectors = [row["embedding"] for row in resp["data"]]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected embed response from {resolved}: {exc}") from exc
        return vectors[0] if isinstance(text, str) else vectors


_INSTANCE: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Return a process-singleton ``LLMProvider``."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LitellmProvider()
    return _INSTANCE
