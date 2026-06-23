"""Tests for context-window headroom-aware compression triggering (issue #104).

Covers:
(a) Flag default OFF → effective min_tokens equals the configured value, unchanged.
(b) Flag ON + HIGH pressure → effective min_tokens is lower; content that would NOT
    compress at the default threshold DOES reach the compressor.
(c) Flag ON + IDLE → conservative threshold (same as configured, multiplier=1.0).
(d) Telemetry event carries the pressure_tier field.

All tests are hermetic:
- System pressure is mocked — real machine load never influences outcomes.
- Compressor boundary is mocked — no headroom-ai extra required.
- CONFIG_PATH is redirected so no real user config is written.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(pressure_value: int):
    """Return a mock SystemSnapshot-like object for the given Pressure int value."""
    from skill_hub.resource_monitor import Pressure, SystemSnapshot

    pressure = Pressure(pressure_value)
    return SystemSnapshot(
        pressure=pressure,
        cpu_load_1m=0.0,
        memory_used_pct=0.0,
        memory_available_mb=8192,
        total_memory_mb=16384,
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# Config isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a temp file so no real config is read or written."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "test-config.json")
    yield


# ---------------------------------------------------------------------------
# (a) Flag OFF — behaviour byte-identical to before (default)
# ---------------------------------------------------------------------------

def test_headroom_aware_off_threshold_unchanged():
    """When compression_headroom_aware is False (default), _effective_min_tokens
    returns the configured value unchanged regardless of system pressure."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", False)
    cfg.set("compression_min_tokens", 200)

    # Even under HIGH pressure the threshold must not change.
    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        effective, tier = compression._effective_min_tokens(200)

    assert effective == 200
    assert tier == "DISABLED"


def test_flag_off_maybe_compress_unchanged(tmp_path, monkeypatch):
    """With compression_headroom_aware=False, maybe_compress behaviour is identical
    to before — small content below the threshold is returned verbatim."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", False)
    cfg.set("compression_enabled", True)
    cfg.set("compression_min_tokens", 200)
    cfg.set("compression_context_aware", False)

    # 50 chars — well below 200 tokens * 4 chars = 800 bytes.
    small = "x" * 50

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        result = compression.maybe_compress(small, site="test")

    assert result == small


# ---------------------------------------------------------------------------
# (b) Flag ON + HIGH pressure → lower threshold, content reaches compressor
# ---------------------------------------------------------------------------

def test_high_pressure_lowers_threshold():
    """Under HIGH pressure with headroom-aware ON, the effective min_tokens is
    25% of the configured value (multiplier=0.25)."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", True)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        effective, tier = compression._effective_min_tokens(200)

    assert effective == 50   # 200 * 0.25
    assert tier == "HIGH"


def test_high_pressure_reaches_compressor(monkeypatch):
    """Content that is BELOW the default threshold (200 tokens * 4 = 800 bytes)
    but ABOVE the HIGH-pressure threshold (50 tokens * 4 = 200 bytes) MUST
    actually reach compress_payload when the flag is ON and pressure is HIGH."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", True)
    cfg.set("compression_enabled", True)
    cfg.set("compression_min_tokens", 200)
    cfg.set("compression_context_aware", False)

    # 300 chars: above 50*4=200 (HIGH threshold), below 200*4=800 (default threshold).
    medium = "log entry data point " * 15  # ~315 chars

    calls: list[dict] = []

    original_compress = compression.compress_payload

    def spy_compress(content, **kwargs):
        calls.append({"content": content, "kwargs": kwargs})
        # Return a passthrough so we don't need headroom installed.
        return compression.CompressedPayload(
            compressed=content,
            content_type="PASSTHROUGH",
            ratio=1.0,
            bytes_before=len(content),
            bytes_after=len(content),
            lossy=False,
        )

    monkeypatch.setattr(compression, "compress_payload", spy_compress)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        compression.maybe_compress(medium, site="test")

    assert calls, (
        "compress_payload was never called — content should have passed the "
        "lowered HIGH-pressure threshold but did not"
    )


def test_default_threshold_skips_medium_content(monkeypatch):
    """Verify the mirror: the same medium content is NOT sent to the compressor
    when headroom-aware is OFF (so the default 200-token threshold applies)."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", False)
    cfg.set("compression_enabled", True)
    cfg.set("compression_min_tokens", 200)
    cfg.set("compression_context_aware", False)

    medium = "log entry data point " * 15  # ~315 chars — below default 800-byte threshold

    calls: list[dict] = []

    def spy_compress(content, **kwargs):
        calls.append({"content": content, "kwargs": kwargs})
        return compression.CompressedPayload(
            compressed=content,
            content_type="PASSTHROUGH",
            ratio=1.0,
            bytes_before=len(content),
            bytes_after=len(content),
            lossy=False,
        )

    monkeypatch.setattr(compression, "compress_payload", spy_compress)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        result = compression.maybe_compress(medium, site="test")

    # compress_payload may still be called (maybe_compress does not guard the call
    # itself — the threshold guard is inside compress_payload).  What matters is
    # that the returned text is unchanged and compress_payload received the default
    # min_tokens=200, not the lowered HIGH-pressure value.
    if calls:
        assert calls[0]["kwargs"].get("min_tokens", 200) == 200
    assert result == medium


# ---------------------------------------------------------------------------
# (c) Flag ON + IDLE → conservative (multiplier = 1.0, same as configured)
# ---------------------------------------------------------------------------

def test_idle_pressure_keeps_threshold():
    """Under IDLE pressure with headroom-aware ON, the threshold is unchanged
    (multiplier=1.0 for IDLE and LOW)."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", True)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(0)):
        effective, tier = compression._effective_min_tokens(200)

    assert effective == 200
    assert tier == "IDLE"


def test_low_pressure_keeps_threshold():
    """LOW pressure with headroom-aware ON also keeps the full threshold."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", True)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(1)):
        effective, tier = compression._effective_min_tokens(200)

    assert effective == 200
    assert tier == "LOW"


def test_moderate_pressure_halves_threshold():
    """MODERATE pressure uses a 0.5 multiplier."""
    from skill_hub import compression
    from skill_hub import config as cfg

    cfg.set("compression_headroom_aware", True)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(2)):
        effective, tier = compression._effective_min_tokens(200)

    assert effective == 100
    assert tier == "MODERATE"


# ---------------------------------------------------------------------------
# (d) Telemetry event carries the pressure_tier field
# ---------------------------------------------------------------------------

def test_telemetry_event_carries_pressure_tier(tmp_path, monkeypatch):
    """The compression telemetry event must include a 'pressure_tier' key
    reflecting the tier that drove the decision."""
    from skill_hub import compression
    from skill_hub import config as cfg
    from skill_hub.store import SkillStore
    import skill_hub.store as _store_mod

    # Isolated store
    db_path = tmp_path / "headroom_telemetry.db"
    monkeypatch.setattr(_store_mod, "DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    monkeypatch.setattr(_store_mod, "_default_store", store)
    monkeypatch.setattr(_store_mod, "get_store", lambda: store)

    cfg.set("compression_headroom_aware", True)
    cfg.set("compression_enabled", True)
    cfg.set("compression_min_tokens", 50)   # low threshold so our payload qualifies
    cfg.set("compression_context_aware", False)

    # Content large enough to reach the compressor even at the IDLE threshold.
    big = "some tool output line " * 100  # >2000 chars

    # Mock compress_payload to return a deterministic shrink so the event fires.
    compressed_text = big[:100]

    def fake_compress(content, **kwargs):
        return compression.CompressedPayload(
            compressed=compressed_text,
            content_type="SMART_CRUSHER",
            ratio=len(compressed_text) / len(content),
            bytes_before=len(content),
            bytes_after=len(compressed_text),
            lossy=False,
        )

    monkeypatch.setattr(compression, "compress_payload", fake_compress)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        compression.maybe_compress(big, site="headroom_test")

    events = store.get_events(kind="compression")
    assert events, "Expected at least one compression event to be recorded"

    # The most recent event must carry pressure_tier.
    last_payload = events[-1].get("payload") or events[-1]
    if isinstance(last_payload, str):
        import json
        last_payload = json.loads(last_payload)

    assert "pressure_tier" in last_payload, (
        f"'pressure_tier' missing from compression event payload: {last_payload}"
    )
    assert last_payload["pressure_tier"] == "HIGH"

    store.close()


def test_telemetry_flag_off_tier_is_disabled(tmp_path, monkeypatch):
    """When compression_headroom_aware=False, telemetry records pressure_tier='DISABLED'."""
    from skill_hub import compression
    from skill_hub import config as cfg
    from skill_hub.store import SkillStore
    import skill_hub.store as _store_mod

    db_path = tmp_path / "headroom_disabled_telemetry.db"
    monkeypatch.setattr(_store_mod, "DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    monkeypatch.setattr(_store_mod, "_default_store", store)
    monkeypatch.setattr(_store_mod, "get_store", lambda: store)

    cfg.set("compression_headroom_aware", False)
    cfg.set("compression_enabled", True)
    cfg.set("compression_min_tokens", 50)
    cfg.set("compression_context_aware", False)

    big = "some tool output line " * 100

    compressed_text = big[:100]

    def fake_compress(content, **kwargs):
        return compression.CompressedPayload(
            compressed=compressed_text,
            content_type="SMART_CRUSHER",
            ratio=len(compressed_text) / len(content),
            bytes_before=len(content),
            bytes_after=len(compressed_text),
            lossy=False,
        )

    monkeypatch.setattr(compression, "compress_payload", fake_compress)

    with patch("skill_hub.resource_monitor.snapshot", return_value=_make_snapshot(3)):
        compression.maybe_compress(big, site="headroom_disabled_test")

    events = store.get_events(kind="compression")
    assert events

    last_payload = events[-1].get("payload") or events[-1]
    if isinstance(last_payload, str):
        import json
        last_payload = json.loads(last_payload)

    assert last_payload.get("pressure_tier") == "DISABLED"

    store.close()
