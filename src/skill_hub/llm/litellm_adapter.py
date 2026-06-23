"""litellm-backed ``LLMProvider`` implementation.

Handles tier → model resolution, Ollama base-URL wiring, and error wrapping.
Singleton per-process: ``get_provider()`` returns a cached instance.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .. import config as _cfg
from .provider import LLMError, LLMProvider, Message

_log = logging.getLogger(__name__)


def _emit_llm_event(
    *,
    op: str,
    model: str,
    tier: str,
    duration_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status: str,
) -> None:
    """Best-effort telemetry: record an ``llm_call`` event. Never raises."""
    try:
        from ..config import get as cfg_get

        if not cfg_get("llm_metering_enabled"):
            return
        from ..store import get_store

        get_store().append_event(
            session_id="",
            kind="llm_call",
            payload={
                "op": op,
                "model": model,
                "tier": tier,
                "duration_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "status": status,
            },
            tool_name=op or None,
        )
    except Exception as e:  # noqa: BLE001 - telemetry must never break a tool call
        _log.debug("llm_call telemetry emit failed (non-fatal): %s", e)


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
            # Import inside function to avoid circular imports.
            from skill_hub.ollama_client import get_ollama_client
            return get_ollama_client().get_api_base(model)
        return None

    def _normalize_messages(
        self, messages: list[Message] | list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, Message):
                out.append({"role": m.role, "content": m.content})
            else:
                # Preserve content-block lists (used for cache_control markers);
                # strings pass through untouched.
                out.append({"role": m["role"], "content": m["content"]})
        return out

    @staticmethod
    def _supports_cache_control(model: str) -> bool:
        return model.startswith("anthropic/") or model.startswith("claude")

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        """Mark the last user message as an ephemeral cache breakpoint.

        No-op for non-Anthropic models. Used by callers that reuse a long
        prefix across calls (e.g., session_memory incremental refresh).
        """
        if not self._supports_cache_control(model) or not messages:
            return messages
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                # Mark the last text block only.
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["cache_control"] = {"type": "ephemeral"}
                        break
            break
        return messages

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
        cache: bool = False,
        extra: dict[str, Any] | None = None,
        op: str = "",
    ) -> str:
        return self.chat(
            [Message(role="user", content=prompt)],
            tier=tier, model=model, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
            cache=cache,
            extra={**(extra or {}), **({"stop": stop} if stop else {})},
            op=op,
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
        cache: bool = False,
        extra: dict[str, Any] | None = None,
        op: str = "",
    ) -> str:
        resolved_model = self._resolve_model(tier, model)
        normalized = self._normalize_messages(messages)
        if cache:
            normalized = self._apply_cache_control(normalized, resolved_model)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": normalized,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        api_base = self._api_base(resolved_model)
        if api_base:
            kwargs["api_base"] = api_base
        if extra:
            kwargs.update(extra)
        _t0 = time.monotonic()
        try:
            resp = self._litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            _duration_ms = int(round((time.monotonic() - _t0) * 1000))
            _emit_llm_event(
                op=op, model=resolved_model, tier=tier,
                duration_ms=_duration_ms,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                status="error",
            )
            raise LLMError(f"completion failed ({resolved_model}): {exc}") from exc
        _duration_ms = int(round((time.monotonic() - _t0) * 1000))
        try:
            _raw_usage = resp.get("usage") if hasattr(resp, "get") else None
            if _raw_usage is None:
                try:
                    _raw_usage = resp["usage"]
                except (KeyError, TypeError):
                    pass
            if _raw_usage is None:
                _raw_usage = getattr(resp, "usage", None)
            if isinstance(_raw_usage, dict):
                _usage: dict[str, Any] = _raw_usage
            elif _raw_usage is not None:
                _usage = {
                    "prompt_tokens": getattr(_raw_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(_raw_usage, "completion_tokens", 0),
                    "total_tokens": getattr(_raw_usage, "total_tokens", 0),
                }
            else:
                _usage = {}
        except Exception:  # noqa: BLE001
            _usage = {}
        _emit_llm_event(
            op=op, model=resolved_model, tier=tier,
            duration_ms=_duration_ms,
            prompt_tokens=int(_usage.get("prompt_tokens") or 0),
            completion_tokens=int(_usage.get("completion_tokens") or 0),
            total_tokens=int(_usage.get("total_tokens") or 0),
            status="ok",
        )
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
        api_base: str | None = None,
    ) -> list[float] | list[list[float]]:
        providers = _cfg.get("llm_providers") or {}
        resolved = model or (providers.get("embed") if isinstance(providers, dict) else None)
        if not resolved:
            resolved = f"ollama/{_cfg.get('embed_model') or 'nomic-embed-text'}"
        resolved_api_base = api_base or self._api_base(resolved)  # explicit override wins
        kwargs: dict[str, Any] = {
            "model": resolved,
            "input": text if isinstance(text, list) else [text],
            "timeout": timeout,
        }
        if resolved_api_base:
            kwargs["api_base"] = resolved_api_base
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
