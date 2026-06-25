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


# --- dependency-free built-in deterministic fallback (no headroom needed) ----

def test_builtin_json_minify_is_lossless():
    pretty = json.dumps([{"id": i, "name": f"row{i}", "val": i * 3} for i in range(80)], indent=2)
    out = compression._builtin_deterministic(pretty)
    assert out is not None
    assert out.content_type == "JSON_MIN"
    assert out.bytes_after < out.bytes_before
    assert not out.lossy
    # Lossless: re-parsing the minified text yields the same object.
    assert json.loads(out.compressed) == json.loads(pretty)


def test_builtin_collapses_duplicate_log_lines():
    text = "\n".join(["INFO connecting"] * 50 + ["ERROR boom"] * 4)
    out = compression._builtin_deterministic(text)
    assert out is not None
    assert out.content_type == "DEDUP"
    assert out.bytes_after < out.bytes_before
    assert "x50" in out.compressed and "ERROR boom" in out.compressed


def test_builtin_returns_none_on_incompressible_prose():
    # No JSON, no repeated runs → nothing to do.
    assert compression._builtin_deterministic("a unique sentence with no repeats at all") is None


def test_compress_payload_uses_builtin_when_router_absent(monkeypatch):
    # Force the headroom router to be unavailable; the built-in must still shrink JSON.
    monkeypatch.setattr(compression, "_get_router", lambda ml=False, code=False: None)
    pretty = json.dumps([{"k": i, "v": "x" * 5} for i in range(120)], indent=2)
    out = compression.compress_payload(pretty, min_tokens=50)
    assert out.content_type == "JSON_MIN"
    assert out.changed and not out.lossy


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


# --- _compression_report observability when headroom-ai is absent -----------

def _fake_compression_stats() -> dict:
    return {
        "calls": 0, "hits": 0, "saved": 0, "tokens_saved": 0,
        "bytes_before": 0, "bytes_after": 0, "avg_ratio": 1.0,
        "by_strategy": {},
    }


def test_compression_report_flags_missing_headroom(monkeypatch, tmp_path):
    """When compression_ml_enabled=True but headroom-ai is not installed,
    _compression_report() must include an explicit 'no-op' or 'not installed'
    annotation so token_stats() doesn't silently lie about runtime state."""
    from skill_hub import config
    import skill_hub.server as _server

    # Isolate config to a temp file.
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "cfg.json")
    config.set("compression_enabled", True)
    config.set("compression_ml_enabled", True)
    config.set("compression_code_aware_enabled", False)

    # Simulate headroom-ai absent by patching is_available to return False.
    monkeypatch.setattr(compression, "is_available", lambda: False)
    monkeypatch.setattr(_server._store, "get_compression_stats", _fake_compression_stats)

    report = _server._compression_report()

    # The report must flag that headroom-ai is missing, NOT silently show ml=on.
    assert "not installed" in report or "no-op" in report or "missing" in report, (
        f"Expected 'not installed'/'no-op'/'missing' in report when headroom absent, got:\n{report}"
    )
    # The ml flag must still show the config intent (ml/Kompress=on ...).
    assert "ml/Kompress=on" in report


def test_compression_report_no_flag_when_headroom_present(monkeypatch, tmp_path):
    """When headroom-ai IS available, the report must NOT mention 'not installed'."""
    from skill_hub import config
    import skill_hub.server as _server

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "cfg2.json")
    config.set("compression_enabled", True)
    config.set("compression_ml_enabled", True)
    config.set("compression_code_aware_enabled", False)

    monkeypatch.setattr(compression, "is_available", lambda: True)
    monkeypatch.setattr(_server._store, "get_compression_stats", _fake_compression_stats)

    report = _server._compression_report()

    assert "not installed" not in report
    assert "ml/Kompress=on" in report
