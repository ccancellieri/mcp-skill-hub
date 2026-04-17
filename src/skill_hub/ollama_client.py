"""Multi-endpoint Ollama client with circuit breaker and per-model routing.

Provides a priority-ordered list of Ollama endpoints (local + remote cloud),
with in-memory circuit breakers and per-model routing overrides.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OllamaEndpoint:
    """A single Ollama endpoint configuration."""
    name: str
    url: str
    priority: int
    enabled: bool
    auth_header: Optional[str] = None  # full header value, e.g. "Bearer sk-..."


@dataclass
class OllamaClientConfig:
    """Configuration for OllamaMultiClient."""
    endpoints: list[OllamaEndpoint]
    # model name → list of endpoint names in priority order
    model_routing: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Circuit breaker state (per endpoint)
# ---------------------------------------------------------------------------


@dataclass
class _CircuitState:
    failures: list[float] = field(default_factory=list)
    circuit_open_until: Optional[float] = None
    # Dynamic priority offset: incremented when circuit opens
    priority_penalty: int = 0


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

_FAILURE_WINDOW_SECONDS = 60
_FAILURE_THRESHOLD = 3
_CIRCUIT_OPEN_SECONDS = 300
_PRIORITY_DEMOTE_STEP = 10


class OllamaMultiClient:
    """Priority-ordered multi-endpoint Ollama client with circuit breakers.

    Circuit breaker (per endpoint):
    - Tracks failure timestamps within a 60-second window.
    - After 3 failures in that window, opens the circuit for 5 minutes.
    - On success, resets the failure counter and closes the circuit.

    Thread-safe: all circuit breaker state mutations are protected by a Lock.
    """

    def __init__(self, config: OllamaClientConfig) -> None:
        self._config = config
        self._state: dict[str, _CircuitState] = {
            ep.name: _CircuitState() for ep in config.endpoints
        }
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Circuit breaker

    def is_healthy(self, endpoint_name: str) -> bool:
        """Return True if the endpoint's circuit breaker is closed (healthy)."""
        with self._lock:
            state = self._state.get(endpoint_name)
            if state is None:
                return False
            if state.circuit_open_until is not None and time.time() < state.circuit_open_until:
                return False
            return True

    def record_failure(self, endpoint_name: str) -> None:
        """Record a failure; open circuit after 3 failures within 60 seconds."""
        with self._lock:
            state = self._state.get(endpoint_name)
            if state is None:
                _log.warning("record_failure: unknown endpoint %r", endpoint_name)
                return

            now = time.time()
            # Prune failures outside the window
            state.failures = [t for t in state.failures if now - t < _FAILURE_WINDOW_SECONDS]
            state.failures.append(now)

            if len(state.failures) >= _FAILURE_THRESHOLD:
                if state.circuit_open_until is None:  # only open once per failure burst
                    state.circuit_open_until = now + _CIRCUIT_OPEN_SECONDS
                    state.priority_penalty += _PRIORITY_DEMOTE_STEP
                    _log.warning(
                        "ollama_client: circuit opened for %r "
                        "(failures=%d, open until %s, priority_penalty=%d)",
                        endpoint_name,
                        len(state.failures),
                        time.strftime("%H:%M:%S", time.localtime(state.circuit_open_until)),
                        state.priority_penalty,
                    )
                else:
                    # Already open — just extend the deadline
                    state.circuit_open_until = now + _CIRCUIT_OPEN_SECONDS

    def record_success(self, endpoint_name: str) -> None:
        """Reset failure count and close the circuit for this endpoint."""
        with self._lock:
            state = self._state.get(endpoint_name)
            if state is None:
                return
            state.failures.clear()
            state.circuit_open_until = None
            # Keep priority_penalty — do not re-promote automatically

    # ------------------------------------------------------------------
    # Endpoint selection

    def _effective_priority(self, ep: OllamaEndpoint) -> int:
        """Effective priority = base priority + circuit penalty (lower = better)."""
        state = self._state.get(ep.name, _CircuitState())
        return ep.priority + state.priority_penalty

    def get_api_base(self, model: str | None = None) -> str | None:
        """Return the URL of the best available endpoint for *model*.

        Picks from per-model routing overrides first; falls back to global
        priority order.  Returns ``None`` if all endpoints are unhealthy or
        disabled.
        """
        endpoints_by_name: dict[str, OllamaEndpoint] = {
            ep.name: ep for ep in self._config.endpoints
        }

        # Snapshot penalties under the lock to avoid data races during sort
        with self._lock:
            penalties = {name: s.priority_penalty for name, s in self._state.items()}

        # Determine candidate order
        if model and model in self._config.model_routing:
            # Per-model list: names in explicit priority order
            names = self._config.model_routing[model]
            candidates = [endpoints_by_name[n] for n in names if n in endpoints_by_name]
        else:
            # Global priority order (ascending priority value)
            candidates = sorted(
                self._config.endpoints,
                key=lambda ep: ep.priority + penalties.get(ep.name, 0),
            )

        for ep in candidates:
            if not ep.enabled:
                continue
            if self.is_healthy(ep.name):
                return ep.url

        _log.debug("ollama_client: no healthy endpoint available (model=%r)", model)
        return None

    def get_auth_headers(self, endpoint_name: str) -> dict[str, str]:
        """Return Authorization headers for the named endpoint (may be empty)."""
        for ep in self._config.endpoints:
            if ep.name == endpoint_name:
                if ep.auth_header:
                    return {"Authorization": ep.auth_header}
                return {}
        return {}

    # ------------------------------------------------------------------
    # Health check (async)

    async def health_check(
        self, endpoint_name: str | None = None
    ) -> dict[str, bool]:
        """Async HTTP health check against ``{url}/api/tags``.

        Updates circuit breaker state based on reachability.

        Args:
            endpoint_name: If given, check only this endpoint.
                           Otherwise, check all enabled endpoints.

        Returns:
            Mapping of ``{endpoint_name: reachable}``.
        """
        import httpx

        targets = [
            ep for ep in self._config.endpoints
            if ep.enabled and (endpoint_name is None or ep.name == endpoint_name)
        ]

        results: dict[str, bool] = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for ep in targets:
                headers = self.get_auth_headers(ep.name)
                try:
                    resp = await client.get(f"{ep.url}/api/tags", headers=headers)
                    reachable = resp.status_code < 400
                except Exception:
                    reachable = False

                results[ep.name] = reachable
                if reachable:
                    self.record_success(ep.name)
                else:
                    self.record_failure(ep.name)

        return results


# ---------------------------------------------------------------------------
# Module-level singleton + factory
# ---------------------------------------------------------------------------

_client: OllamaMultiClient | None = None
_client_lock = threading.Lock()


def _build_config_from_dict(raw: list[dict]) -> list[OllamaEndpoint]:
    """Convert a list of raw dicts (from JSON config) into OllamaEndpoint objects."""
    endpoints = []
    for item in raw:
        endpoints.append(
            OllamaEndpoint(
                name=str(item.get("name", "unknown")),
                url=str(item.get("url", "http://localhost:11434")),
                priority=int(item.get("priority", 1)),
                enabled=bool(item.get("enabled", True)),
                auth_header=item.get("auth_header") or None,
            )
        )
    return endpoints


def get_ollama_client() -> OllamaMultiClient:
    """Return the process-singleton ``OllamaMultiClient``, initializing lazily.

    Reads ``config.ollama_endpoints`` and ``config.ollama_model_routing``.
    Falls back to a single-endpoint config built from ``config.ollama_base``
    when those keys are absent.
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        # Double-checked locking
        if _client is not None:
            return _client

        from . import config as _cfg

        cfg = _cfg.load_config()
        raw_endpoints = cfg.get("ollama_endpoints")
        model_routing: dict[str, list[str]] = cfg.get("ollama_model_routing") or {}

        if raw_endpoints and isinstance(raw_endpoints, list) and len(raw_endpoints) > 0:
            endpoints = _build_config_from_dict(raw_endpoints)
        else:
            # Fallback: build a single-endpoint config from the legacy ollama_base key
            fallback_url = str(cfg.get("ollama_base") or "http://localhost:11434")
            _log.debug(
                "ollama_client: ollama_endpoints not configured; "
                "falling back to single endpoint at %s",
                fallback_url,
            )
            endpoints = [
                OllamaEndpoint(
                    name="local",
                    url=fallback_url,
                    priority=1,
                    enabled=True,
                )
            ]

        client_config = OllamaClientConfig(
            endpoints=endpoints,
            model_routing=model_routing if isinstance(model_routing, dict) else {},
        )
        _client = OllamaMultiClient(client_config)
        return _client


def reset_client() -> None:
    """Reset the singleton so the next call to ``get_ollama_client`` re-reads config.

    Primarily useful in tests.
    """
    global _client
    with _client_lock:
        _client = None
