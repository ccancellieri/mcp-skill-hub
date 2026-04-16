"""Tests for the multi-endpoint Ollama client with circuit breaker logic."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.ollama_client import (
    OllamaClientConfig,
    OllamaEndpoint,
    OllamaMultiClient,
    get_ollama_client,
    reset_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    endpoints: list[OllamaEndpoint] | None = None,
    model_routing: dict[str, list[str]] | None = None,
) -> OllamaMultiClient:
    """Create an OllamaMultiClient with sensible test defaults."""
    if endpoints is None:
        endpoints = [
            OllamaEndpoint(name="local", url="http://localhost:11434", priority=1, enabled=True),
            OllamaEndpoint(name="remote", url="http://remote:11434", priority=2, enabled=True),
        ]
    return OllamaMultiClient(
        OllamaClientConfig(
            endpoints=endpoints,
            model_routing=model_routing or {},
        )
    )


# ---------------------------------------------------------------------------
# Test: get_api_base returns highest-priority healthy endpoint
# ---------------------------------------------------------------------------


def test_get_api_base_returns_highest_priority():
    """get_api_base should return the url of the lowest-priority-int endpoint."""
    client = _make_client()
    assert client.get_api_base() == "http://localhost:11434"


def test_get_api_base_skips_unhealthy_falls_to_next():
    """When the highest-priority endpoint is down, the next healthy one is used."""
    client = _make_client()
    # Force local to have its circuit open
    client._state["local"].circuit_open_until = time.time() + 9999
    assert client.get_api_base() == "http://remote:11434"


def test_get_api_base_returns_none_when_all_down():
    """Returns None when every endpoint has an open circuit."""
    client = _make_client()
    for state in client._state.values():
        state.circuit_open_until = time.time() + 9999
    assert client.get_api_base() is None


def test_get_api_base_skips_disabled():
    """Disabled endpoints are never selected."""
    endpoints = [
        OllamaEndpoint(name="disabled", url="http://disabled:11434", priority=1, enabled=False),
        OllamaEndpoint(name="active", url="http://active:11434", priority=2, enabled=True),
    ]
    client = _make_client(endpoints=endpoints)
    assert client.get_api_base() == "http://active:11434"


# ---------------------------------------------------------------------------
# Test: circuit breaker opens after 3 failures within 60s
# ---------------------------------------------------------------------------


def test_circuit_opens_after_3_failures():
    """Circuit opens after 3 failures in the 60-second window."""
    client = _make_client()
    assert client.is_healthy("local")

    client.record_failure("local")
    client.record_failure("local")
    assert client.is_healthy("local")  # still healthy at 2 failures

    client.record_failure("local")
    assert not client.is_healthy("local")  # now open


def test_circuit_reopens_after_timeout():
    """Circuit is considered closed (healthy) once circuit_open_until has passed."""
    client = _make_client()
    client.record_failure("local")
    client.record_failure("local")
    client.record_failure("local")
    assert not client.is_healthy("local")

    # Manually expire the circuit
    client._state["local"].circuit_open_until = time.time() - 1
    assert client.is_healthy("local")


# ---------------------------------------------------------------------------
# Test: failures older than 60s are ignored
# ---------------------------------------------------------------------------


def test_old_failures_are_pruned():
    """Failures older than 60 seconds do not count toward the threshold."""
    client = _make_client()
    old = time.time() - 61  # outside the window
    client._state["local"].failures = [old, old]  # 2 stale failures

    client.record_failure("local")  # 1 fresh failure → should NOT open circuit
    assert client.is_healthy("local")


def test_mixed_old_and_new_failures():
    """Mix of stale + fresh failures — only fresh ones count."""
    client = _make_client()
    old = time.time() - 61
    # Pre-load 2 stale + 1 fresh failure
    client._state["local"].failures = [old, old]

    client.record_failure("local")  # 1st fresh
    client.record_failure("local")  # 2nd fresh — total fresh = 2, still healthy
    assert client.is_healthy("local")

    client.record_failure("local")  # 3rd fresh — now opens
    assert not client.is_healthy("local")


# ---------------------------------------------------------------------------
# Test: record_success resets state
# ---------------------------------------------------------------------------


def test_record_success_resets_failures():
    """record_success clears failures and closes the circuit."""
    client = _make_client()
    client.record_failure("local")
    client.record_failure("local")
    client.record_failure("local")
    assert not client.is_healthy("local")

    client.record_success("local")
    assert client.is_healthy("local")
    assert client._state["local"].failures == []
    assert client._state["local"].circuit_open_until is None


# ---------------------------------------------------------------------------
# Test: per-model routing overrides global priority
# ---------------------------------------------------------------------------


def test_model_routing_overrides_global_priority():
    """A model-specific routing list is used instead of global priority order."""
    endpoints = [
        OllamaEndpoint(name="local", url="http://local:11434", priority=1, enabled=True),
        OllamaEndpoint(name="cloud", url="http://cloud:11434", priority=2, enabled=True),
    ]
    # Route deepseek to cloud first
    client = _make_client(
        endpoints=endpoints,
        model_routing={"ollama/deepseek-r1:1.5b": ["cloud", "local"]},
    )
    assert client.get_api_base("ollama/deepseek-r1:1.5b") == "http://cloud:11434"


def test_model_routing_falls_through_when_first_is_down():
    """Per-model routing respects circuit breaker on the first endpoint."""
    endpoints = [
        OllamaEndpoint(name="local", url="http://local:11434", priority=1, enabled=True),
        OllamaEndpoint(name="cloud", url="http://cloud:11434", priority=2, enabled=True),
    ]
    client = _make_client(
        endpoints=endpoints,
        model_routing={"ollama/deepseek-r1:1.5b": ["cloud", "local"]},
    )
    # Take cloud down
    client._state["cloud"].circuit_open_until = time.time() + 9999
    assert client.get_api_base("ollama/deepseek-r1:1.5b") == "http://local:11434"


def test_global_priority_used_when_no_model_routing():
    """When model is not in model_routing, global priority order is used."""
    endpoints = [
        OllamaEndpoint(name="local", url="http://local:11434", priority=1, enabled=True),
        OllamaEndpoint(name="cloud", url="http://cloud:11434", priority=2, enabled=True),
    ]
    client = _make_client(
        endpoints=endpoints,
        model_routing={"ollama/special-model": ["cloud"]},
    )
    # Any other model uses global priority
    assert client.get_api_base("ollama/other-model") == "http://local:11434"


# ---------------------------------------------------------------------------
# Test: get_auth_headers
# ---------------------------------------------------------------------------


def test_get_auth_headers_with_auth():
    endpoints = [
        OllamaEndpoint(
            name="cloud",
            url="http://cloud:11434",
            priority=1,
            enabled=True,
            auth_header="Bearer my-token",
        )
    ]
    client = _make_client(endpoints=endpoints)
    assert client.get_auth_headers("cloud") == {"Authorization": "Bearer my-token"}


def test_get_auth_headers_without_auth():
    client = _make_client()
    assert client.get_auth_headers("local") == {}


def test_get_auth_headers_unknown_endpoint():
    client = _make_client()
    assert client.get_auth_headers("nonexistent") == {}


# ---------------------------------------------------------------------------
# Test: singleton factory fallback from ollama_base
# ---------------------------------------------------------------------------


def test_get_ollama_client_fallback_from_ollama_base():
    """When ollama_endpoints is not configured, falls back to ollama_base."""
    reset_client()
    fake_config = {
        "ollama_base": "http://fallback:11434",
        "ollama_endpoints": [],  # empty → triggers fallback
        "ollama_model_routing": {},
    }
    with patch("skill_hub.config.load_config", return_value=fake_config):
        client = get_ollama_client()

    assert len(client._config.endpoints) == 1
    ep = client._config.endpoints[0]
    assert ep.name == "local"
    assert ep.url == "http://fallback:11434"
    assert ep.enabled is True
    reset_client()  # clean up


def test_get_ollama_client_uses_ollama_endpoints():
    """When ollama_endpoints is configured, it is used directly."""
    reset_client()
    fake_config = {
        "ollama_base": "http://localhost:11434",
        "ollama_endpoints": [
            {"name": "primary", "url": "http://primary:11434", "priority": 1, "enabled": True},
            {"name": "backup", "url": "http://backup:11434", "priority": 2, "enabled": True},
        ],
        "ollama_model_routing": {},
    }
    with patch("skill_hub.config.load_config", return_value=fake_config):
        client = get_ollama_client()

    assert len(client._config.endpoints) == 2
    assert client._config.endpoints[0].name == "primary"
    assert client._config.endpoints[1].name == "backup"
    reset_client()  # clean up


# ---------------------------------------------------------------------------
# Test: priority penalty via record_failure demotes priority
# ---------------------------------------------------------------------------


def test_circuit_failure_demotes_priority():
    """Opening the circuit increments the priority penalty by 10."""
    client = _make_client()
    initial_penalty = client._state["local"].priority_penalty
    # Open circuit
    client.record_failure("local")
    client.record_failure("local")
    client.record_failure("local")
    assert client._state["local"].priority_penalty == initial_penalty + 10


def test_effective_priority_includes_penalty():
    """_effective_priority reflects base priority + accumulated penalty."""
    client = _make_client()
    ep = client._config.endpoints[0]  # "local", priority=1
    assert client._effective_priority(ep) == 1

    client._state["local"].priority_penalty = 10
    assert client._effective_priority(ep) == 11
