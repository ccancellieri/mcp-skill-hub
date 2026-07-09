"""Tests for the deterministic compression adapter (``skill_hub.compression``).

``compress_payload`` is deterministic-only (#119): JSON minify + duplicate-line
collapse, no optional dependency required. ``kompress_prose``/``retrieve_original``
still wrap the optional ``headroom-ai`` dependency and auto-no-op without it.
"""
from __future__ import annotations

import json

from skill_hub import compression


def test_empty_input_passthrough():
    out = compression.compress_payload("")
    assert out.content_type == "PASSTHROUGH"
    assert out.compressed == ""
    assert out.ratio == 1.0


def test_small_input_passthrough():
    # Below the token threshold -> returned verbatim, compressor never reached.
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


def test_compress_payload_uses_builtin_deterministic():
    pretty = json.dumps([{"k": i, "v": "x" * 5} for i in range(120)], indent=2)
    out = compression.compress_payload(pretty, min_tokens=50)
    assert out.content_type == "JSON_MIN"
    assert out.changed and not out.lossy


def test_prose_passes_through_unchanged():
    # No JSON, no repeated lines -> nothing for the deterministic pass to catch.
    prose = ("The quick brown fox jumps over the lazy dog. " * 100)
    payload = compression.compress_payload(prose)
    assert payload.content_type == "PASSTHROUGH"
    assert payload.compressed == prose
    assert payload.ratio == 1.0


def test_malformed_input_never_raises():
    garbage = "{not json [ <<>> \x00 broken" * 50
    payload = compression.compress_payload(garbage)
    # Always returns a usable payload; compressed is at worst the original.
    assert isinstance(payload.compressed, str)
    assert payload.bytes_before == len(garbage)


# --- truncate_at_word ---------------------------------------------------

def test_truncate_at_word_fits_unchanged():
    assert compression.truncate_at_word("short text", 100) == "short text"


def test_truncate_at_word_cuts_at_last_whitespace():
    text = "one two three four five"
    out = compression.truncate_at_word(text, 13)
    assert out == "one two … (truncated)"
    assert "threef" not in out.replace(" ", "")  # no mid-word cut


def test_truncate_at_word_no_whitespace_hard_cuts():
    text = "x" * 50
    out = compression.truncate_at_word(text, 10)
    assert out == "x" * 10 + " … (truncated)"


def test_truncate_at_word_custom_marker():
    out = compression.truncate_at_word("alpha beta gamma", 10, marker="...")
    assert out.endswith("...")
    assert len(out) < len("alpha beta gamma")


# --- _compression_report observability (#119: deterministic-only) -----------

def _fake_compression_stats() -> dict:
    return {
        "calls": 0, "hits": 0, "saved": 0, "tokens_saved": 0,
        "bytes_before": 0, "bytes_after": 0, "avg_ratio": 1.0,
        "by_strategy": {},
    }


def test_compression_report_has_no_dormant_capability_messaging(monkeypatch, tmp_path):
    """_compression_report() no longer explains a missing-extra/ml/code-aware
    capability that doesn't exist -- compression is deterministic-only."""
    from skill_hub import config
    import skill_hub.server as _server

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "cfg.json")
    config.set("compression_enabled", True)
    monkeypatch.setattr(_server._store, "get_compression_stats", _fake_compression_stats)

    report = _server._compression_report()

    assert "master=on" in report
    for stale in ("headroom", "Kompress", "ml/", "code-aware", "not installed", "no-op"):
        assert stale not in report, f"stale dormant-capability text {stale!r} found in report:\n{report}"


# ---------------------------------------------------------------------------
# squeeze_whitespace — deterministic prose normalization for prompt injection
# ---------------------------------------------------------------------------

def test_squeeze_whitespace_collapses_runs_and_blank_lines():
    from skill_hub.compression import squeeze_whitespace
    raw = "col a    col b\t\tcol c   \n\n\n\n- item   one  \n   spaced    out\n"
    out = squeeze_whitespace(raw)
    assert out == "col a col b col c\n\n- item one\n spaced out"


def test_squeeze_whitespace_preserves_single_spacing():
    from skill_hub.compression import squeeze_whitespace
    text = "plain prose line\n\nsecond paragraph"
    assert squeeze_whitespace(text) == text
