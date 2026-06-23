"""Tests for the compression cascade behaviour (issue #56 eval harness companion).

Covers:
(a) JSON array compresses deterministically (content_type != PASSTHROUGH, lossy=False).
(b) Prose with lossy flags OFF returns PASSTHROUGH.
(c) maybe_compress emits a 'compression' event into the store (count delta).
(d) get_compression_stats() returns the documented keys.

All tests are fast and offline-safe — no ML model is loaded. Any assertion that
would require the sentence-transformers model is guarded with skipif.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# ML model availability guard (re-used by skipif markers below)
# ---------------------------------------------------------------------------

def _sentence_transformers_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


_HAS_ST = _sentence_transformers_available()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """A fresh in-memory-backed SkillStore that does not touch the live DB."""
    from skill_hub.store import SkillStore

    db_path = tmp_path / "test_compression_cascade.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture()
def patched_store(isolated_store, monkeypatch):
    """Redirect get_store() to the isolated fixture store.

    get_store() uses the module-global ``_default_store``.  We replace that
    global AND the function itself so both call-sites (direct import and
    lazy import inside compression._emit_compression_event) see our store.
    """
    import skill_hub.store as _store_mod

    monkeypatch.setattr(_store_mod, "_default_store", isolated_store)
    monkeypatch.setattr(_store_mod, "get_store", lambda: isolated_store)
    return isolated_store


# ---------------------------------------------------------------------------
# (a) JSON array compresses deterministically
# ---------------------------------------------------------------------------

def test_json_array_compresses_deterministically():
    """A sufficiently large JSON array must be compressed by a deterministic strategy
    (SmartCrusher or similar), must NOT be lossy, and must actually shrink."""
    pytest.importorskip("headroom")

    from skill_hub.compression import compress_payload

    rows = [
        {"id": i, "level": "INFO", "msg": f"Worker processed batch {i}", "ts": 1700000000 + i * 60}
        for i in range(80)
    ]
    text = json.dumps(rows)

    result = compress_payload(text, allow_lossy=False)

    assert result.content_type != "PASSTHROUGH", (
        f"Expected a deterministic compressor to fire, got PASSTHROUGH "
        f"(bytes_before={result.bytes_before})"
    )
    assert result.lossy is False, "Deterministic strategies must not be lossy"
    assert result.bytes_after < result.bytes_before, "Compressed output must be smaller"
    assert result.ratio < 1.0


# ---------------------------------------------------------------------------
# (b) Prose with lossy flags OFF returns PASSTHROUGH
# ---------------------------------------------------------------------------

def test_prose_passthrough_when_lossy_off():
    """Prose must pass through unchanged when no lossy flags are enabled.

    This works even without headroom installed: the absence of a router causes
    an early passthrough return.  With headroom, the prose path is a no-op for
    the deterministic compressors.
    """
    from skill_hub import config as cfg
    from skill_hub.compression import compress_payload

    # Ensure flags are off (they are off by default, but be explicit).
    original_ml = cfg.get("compression_ml_enabled")
    original_code = cfg.get("compression_code_aware_enabled")
    cfg.set("compression_ml_enabled", False)
    cfg.set("compression_code_aware_enabled", False)

    try:
        prose = (
            "The transformation of software architecture over the past decade has been "
            "profound. Microservices promised autonomy and independent scalability, yet "
            "teams consistently discovered that the operational complexity they introduced "
            "often eclipsed the agility they offered. A well-structured single process with "
            "clear internal boundaries is no less maintainable than a cluster of services. "
        ) * 6  # ~2700 chars — well above the default min_tokens * 4 threshold

        result = compress_payload(prose, allow_lossy=False)
        assert result.content_type == "PASSTHROUGH"
        assert result.compressed == prose
        assert result.ratio == 1.0
        assert result.lossy is False
    finally:
        cfg.set("compression_ml_enabled", original_ml)
        cfg.set("compression_code_aware_enabled", original_code)


# ---------------------------------------------------------------------------
# (c) maybe_compress emits a 'compression' event
# ---------------------------------------------------------------------------

def test_maybe_compress_emits_compression_event(patched_store, monkeypatch):
    """maybe_compress must append a 'compression' event to the store when the
    payload is large enough to reach the compressor (even on passthrough)."""
    pytest.importorskip("headroom")

    from skill_hub import config as cfg

    # Enable the compression master switch and ensure lossy flags are off.
    monkeypatch.setattr(cfg, "get", lambda k, default=None: {
        "compression_enabled": True,
        "compression_context_aware": False,
        "compression_min_tokens": 50,    # lower threshold so our payload qualifies
        "compression_ml_enabled": False,
        "compression_code_aware_enabled": False,
    }.get(k, cfg._DEFAULTS.get(k, default)))

    from skill_hub.compression import maybe_compress

    before_count = len(patched_store.get_events(kind="compression"))

    large_text = ("INFO worker processed batch " * 60)  # >800 chars
    maybe_compress(large_text, site="test_eval", allow_lossy=False)

    after_count = len(patched_store.get_events(kind="compression"))
    assert after_count > before_count, (
        f"Expected at least one new 'compression' event; "
        f"count before={before_count}, after={after_count}"
    )


# ---------------------------------------------------------------------------
# (d) get_compression_stats() returns the documented keys
# ---------------------------------------------------------------------------

_EXPECTED_STATS_KEYS = {
    "calls", "hits", "bytes_before", "bytes_after", "saved",
    "avg_ratio", "tokens_saved", "by_strategy", "by_site",
}


def test_get_compression_stats_returns_documented_keys(isolated_store):
    """get_compression_stats() must always return a dict containing every key
    listed in the docstring, even when there are no events."""
    stats = isolated_store.get_compression_stats()
    assert isinstance(stats, dict)
    missing = _EXPECTED_STATS_KEYS - stats.keys()
    assert not missing, f"get_compression_stats() is missing keys: {missing}"


def test_get_compression_stats_accumulates_events(isolated_store):
    """After appending compression events, stats must reflect them."""
    isolated_store.append_event(
        session_id="",
        kind="compression",
        payload={
            "site": "test",
            "strategy": "SMART_CRUSHER",
            "bytes_before": 1000,
            "bytes_after": 600,
            "ratio": 0.6,
            "lossy": False,
        },
    )
    isolated_store.append_event(
        session_id="",
        kind="compression",
        payload={
            "site": "test",
            "strategy": "LOG",
            "bytes_before": 800,
            "bytes_after": 400,
            "ratio": 0.5,
            "lossy": False,
        },
    )

    stats = isolated_store.get_compression_stats()
    assert stats["calls"] >= 2
    assert stats["hits"] >= 2
    assert stats["bytes_before"] >= 1800
    assert stats["saved"] >= 800
    assert stats["avg_ratio"] < 1.0
    assert stats["tokens_saved"] >= 0
    assert "SMART_CRUSHER" in stats["by_strategy"] or "LOG" in stats["by_strategy"]


# ---------------------------------------------------------------------------
# ML-model-gated fidelity smoke test (skipped when sentence-transformers absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
def test_embedding_fidelity_identical_texts():
    """Sanity: cosine similarity of a text against itself must be ~1.0."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = "The quick brown fox jumps over the lazy dog." * 5
    vecs = model.encode([text, text], normalize_embeddings=True)
    sim = float(np.dot(vecs[0], vecs[1]))
    assert sim > 0.99, f"Self-similarity should be ~1.0, got {sim}"
