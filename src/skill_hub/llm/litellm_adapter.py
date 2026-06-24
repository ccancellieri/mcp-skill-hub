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

# Auxiliary-task routing policy: maps a call-site ``op`` label to a default
# ``(complexity, domain)`` so those tasks flow through the escalation ladder and
# prefer a domain-specialised model. Only ops listed here auto-engage the
# ladder; explicit ``complexity``/``domain`` kwargs always override. Unlisted
# ops keep the prior tier-based behaviour (no regression). Domains are matched
# against per-model ``tags`` in the provider registry.
_OP_ROUTING: dict[str, tuple[float, str]] = {
    "conversation_digest": (0.3, "digest"),
    "compact": (0.3, "digest"),
    "compact_master_state": (0.4, "writing"),
    "smart_memory_write": (0.4, "writing"),
    "rerank": (0.2, "fast"),
    "triage": (0.2, "fast"),
    "wiki_file_answer": (0.5, "reasoning"),
    "wiki_ingest": (0.4, "writing"),
}


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
        cache_ttl: str = "",
    ) -> list[dict[str, Any]]:
        """Mark the last user message as an ephemeral cache breakpoint.

        No-op for non-Anthropic models. Used by callers that reuse a long
        prefix across calls (e.g., session_memory incremental refresh).

        ``cache_ttl`` (e.g. ``"1h"``) requests Anthropic's extended cache TTL
        for long-lived prefixes; omitted/empty uses the default ~5-minute
        ephemeral window. Applied only when the operator has opted into the
        extended tier (``llm_cache_extended_ttl``) so an unsupported
        litellm/SDK never errors the call.
        """
        if not self._supports_cache_control(model) or not messages:
            return messages
        cc: dict[str, Any] = {"type": "ephemeral"}
        if cache_ttl and _cfg.get("llm_cache_extended_ttl"):
            cc["ttl"] = cache_ttl
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(cc),
                    }
                ]
            elif isinstance(content, list) and content:
                # Mark the last text block only.
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["cache_control"] = dict(cc)
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
        cache_ttl: str = "",
        complexity: float | None = None,
        domain: str | None = None,
    ) -> str:
        return self.chat(
            [Message(role="user", content=prompt)],
            tier=tier, model=model, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
            cache=cache,
            extra={**(extra or {}), **({"stop": stop} if stop else {})},
            op=op,
            cache_ttl=cache_ttl,
            complexity=complexity,
            domain=domain,
        )

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_tokens: int,
        temperature: float,
        timeout: float,
        cache: bool,
        extra: dict[str, Any] | None,
        op: str,
        cache_ttl: str,
        tier: str,
    ) -> str:
        """Single litellm completion attempt with an already-resolved model.

        ``messages`` is expected already-normalized (both callers normalize
        before dispatch), so this method does not normalize again.
        """
        normalized = messages
        if cache:
            normalized = self._apply_cache_control(normalized, model, cache_ttl)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": normalized,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        if extra:
            kwargs.update(extra)
        _t0 = time.monotonic()
        try:
            resp = self._litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            _duration_ms = int(round((time.monotonic() - _t0) * 1000))
            _emit_llm_event(
                op=op, model=model, tier=tier,
                duration_ms=_duration_ms,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                status="error",
            )
            raise LLMError(f"completion failed ({model}): {exc}") from exc
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
            op=op, model=model, tier=tier,
            duration_ms=_duration_ms,
            prompt_tokens=int(_usage.get("prompt_tokens") or 0),
            completion_tokens=int(_usage.get("completion_tokens") or 0),
            total_tokens=int(_usage.get("total_tokens") or 0),
            status="ok",
        )
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {model}: {exc}") from exc

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
        cache_ttl: str = "",
        complexity: float | None = None,
        domain: str | None = None,
    ) -> str:
        # --- Auxiliary escalation ladder -------------------------------------
        # Engage only when the caller did not pin a model and left tier default.
        # A signal is required: an explicit complexity/domain, or a known ``op``
        # whose routing policy supplies both (see ``_OP_ROUTING``). Explicit
        # kwargs override the policy. Unknown ops with no signal skip the ladder.
        if model is None and tier == "tier_cheap":
            policy = _OP_ROUTING.get(op)
            eff_complexity = complexity
            eff_domain = domain
            if policy is not None:
                if eff_complexity is None:
                    eff_complexity = policy[0]
                if eff_domain is None:
                    eff_domain = policy[1]
            if eff_complexity is not None or eff_domain is not None:
                return self._chat_via_ladder(
                    self._normalize_messages(messages),
                    complexity=eff_complexity if eff_complexity is not None else 0.5,
                    domain=eff_domain, max_tokens=max_tokens,
                    temperature=temperature, timeout=timeout, cache=cache,
                    extra=extra, op=op, cache_ttl=cache_ttl,
                )

        resolved_model = self._resolve_model(tier, model)
        return self._chat_once(
            self._normalize_messages(messages),
            model=resolved_model,
            api_base=self._api_base(resolved_model),
            api_key=None,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            cache=cache,
            extra=extra,
            op=op,
            cache_ttl=cache_ttl,
            tier=tier,
        )

    def _personal_over_cap(self) -> bool:
        cap = _cfg.get("llm_personal_daily_usd_cap")
        if cap is None:
            return False
        try:
            spent = self._personal_spend_today_usd()
        except Exception:  # noqa: BLE001
            return False
        return spent >= float(cap)

    def _personal_spend_today_usd(self) -> float:
        """Sum today's personal-Claude auxiliary spend from llm_call events.

        ``store.get_events`` signature:
        ``get_events(session_id="", since=0.0, kind="", limit=200)``
        Each row's ``ts`` is a float epoch. Filter the current day with
        ``since=<start-of-today epoch>``.
        """
        import json as _json
        import time as _time
        from .. import model_registry as _mr
        from ..store import get_store
        lt = _time.localtime()
        start = _time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        total = 0.0
        try:
            events = get_store().get_events(kind="llm_call", since=start, limit=5000)
        except Exception:  # noqa: BLE001
            return 0.0
        for ev in events:
            # store.get_events returns ``payload`` as a raw JSON string (the
            # store does not decode it on read — see get_compression_stats).
            raw = ev.get("payload")
            if isinstance(raw, str):
                try:
                    payload = _json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
            else:
                payload = raw or {}
            ev_model = str(payload.get("model") or "")
            if "claude" not in ev_model and not ev_model.startswith("anthropic/"):
                continue
            rate = _mr.blended_usd_per_m(ev_model)
            if rate:
                total += rate * (int(payload.get("total_tokens") or 0) / 1_000_000)
        return total

    def _chat_via_ladder(
        self,
        messages: list[dict[str, Any]],
        *,
        complexity: float,
        domain: str | None = None,
        max_tokens: int,
        temperature: float,
        timeout: float,
        cache: bool,
        extra: dict[str, Any] | None,
        op: str,
        cache_ttl: str,
    ) -> str:
        from . import escalation
        exclude: set[str] = set()
        over_cap = self._personal_over_cap()
        last_exc: Exception | None = None
        for _ in range(64):
            sel = escalation.select(complexity, domain=domain, exclude=exclude)
            if sel is None:
                break
            if over_cap and sel.level == "personal":
                exclude.add(sel.model)
                continue
            try:
                return self._chat_once(
                    messages, model=sel.model, api_base=sel.api_base,
                    api_key=sel.api_key, max_tokens=max_tokens,
                    temperature=temperature, timeout=timeout, cache=cache,
                    extra=extra, op=op, cache_ttl=cache_ttl,
                    tier=f"ladder:{sel.level}",
                )
            except LLMError as exc:
                last_exc = exc
                if escalation.looks_like_quota_error(exc):
                    escalation.mark_cooldown(sel.model)
                exclude.add(sel.model)
                continue
        raise LLMError(f"escalation ladder exhausted (complexity={complexity}): {last_exc}")

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
