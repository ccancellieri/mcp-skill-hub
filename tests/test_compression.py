"""Tests for the deterministic compression adapter (``skill_hub.compression``).

The adapter wraps the optional ``headroom-ai`` dependency. Tests that need real
compression are guarded with ``importorskip`` so the base test run (without the
``compression`` extra) skips them; the passthrough/gating tests run unconditionally.
"""
from __future__ import annotations

import json

import pytest

from skill_hub import compression


# --- behaviour that does NOT require headroom installed ---------------------

def test_empty_input_passthrough():
    out = compression.compress_payload("")
    assert out.content_type == "PASSTHROUGH"
    assert out.compressed == ""
    assert out.ratio == 1.0


def test_small_input_passthrough():
    # Below the token threshold -> returned verbatim, no router involved.
    text = "short bit of text"
    out = compression.compress_payload(text, min_tokens=200)
    assert out.compressed == text
    assert out.content_type == "PASSTHROUGH"
    assert not out.changed


def test_maybe_compress_disabled_returns_original(monkeypatch):
    from skill_hub import config

    big = "x" * 10000
    monkeypatch.setattr(config, "get", lambda k: False if k == "compression_enabled" else config._DEFAULTS.get(k))
    assert compression.maybe_compress(big) == big


def test_retrieve_original_missing_hash_is_none():
    # Unknown hash (or headroom absent) must not raise.
    assert compression.retrieve_original("deadbeefdeadbeefdeadbeef") is None


def test_is_available_returns_bool():
    assert isinstance(compression.is_available(), bool)


# --- behaviour that DOES require headroom -----------------------------------

def test_json_array_compresses_and_keeps_errors():
    pytest.importorskip("headroom")
    rows = [
        {"id": i, "level": "INFO", "msg": f"row {i} ok", "ts": 1000 + i}
        for i in range(60)
    ]
    rows[42] = {"id": 42, "level": "ERROR", "msg": "boom: NullPointer at svc.py:42", "ts": 1042}
    payload = compression.compress_payload(json.dumps(rows))
    assert payload.ratio < 1.0
    assert payload.changed
    assert payload.content_type != "PASSTHROUGH"
    # The error row must survive the crush.
    assert "NullPointer" in payload.compressed


def test_prose_passes_through_unchanged():
    pytest.importorskip("headroom")
    # Prose routes to the disabled Kompress path -> safe passthrough.
    prose = ("The quick brown fox jumps over the lazy dog. " * 100)
    payload = compression.compress_payload(prose)
    assert payload.content_type == "PASSTHROUGH"
    assert payload.compressed == prose
    assert payload.ratio == 1.0


def test_malformed_input_never_raises():
    pytest.importorskip("headroom")
    garbage = "{not json [ <<>> \x00 broken" * 50
    payload = compression.compress_payload(garbage)
    # Always returns a usable payload; compressed is at worst the original.
    assert isinstance(payload.compressed, str)
    assert payload.bytes_before == len(garbage)
