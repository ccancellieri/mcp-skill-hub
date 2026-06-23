"""Tests for log_insights.build_digest (Phase 3 digest surface).

Constraints:
- Never imports skill_hub.server.
- Isolates CONFIG_PATH to tmp_path before importing skill_hub.
- Uses a fresh tmp-path DB via isolated_store fixture.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Guard: importing this module must never pull in skill_hub.server.
def test_server_not_imported(assert_server_not_imported):  # noqa: PT019
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    from skill_hub import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
    from skill_hub.store import SkillStore
    db_path = tmp_path / "test_digest.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    monkeypatch.setattr("skill_hub.store._default_store", store)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Return-key contract
# ---------------------------------------------------------------------------

def test_build_digest_keys_empty(isolated_store):
    """build_digest on an empty DB returns all required keys with safe defaults."""
    from skill_hub.log_insights import build_digest
    d = build_digest(hours=24)

    required = {
        "hours", "window", "total", "distinct_sessions",
        "by_kind", "top_failures", "skills",
        "total_injections", "total_feedback", "generated_ts",
    }
    missing = required - d.keys()
    assert not missing, f"build_digest missing keys: {missing}"

    assert d["total"] == 0
    assert d["distinct_sessions"] == 0
    assert d["by_kind"] == {}
    assert d["top_failures"] == []
    assert d["skills"] == []
    assert d["hours"] == 24
    assert d["window"] == "last 24h"
    assert isinstance(d["generated_ts"], float)
    assert d["generated_ts"] > 0


def test_build_digest_custom_hours(isolated_store):
    """hours parameter is reflected in the returned dict."""
    from skill_hub.log_insights import build_digest
    d = build_digest(hours=48)
    assert d["hours"] == 48
    assert d["window"] == "last 48h"


# ---------------------------------------------------------------------------
# Counting events
# ---------------------------------------------------------------------------

def test_build_digest_counts_events(isolated_store):
    """total and by_kind are aggregated from seeded events."""
    t0 = time.time() - 60
    isolated_store.append_event("s1", "tool_invoke", {"q": "a"}, tool_name="t1", ts=t0)
    isolated_store.append_event("s1", "tool_invoke", {"q": "b"}, tool_name="t2", ts=t0 + 1)
    isolated_store.append_event("s2", "tool_result", {"ok": True}, tool_name="t1", ts=t0 + 2)

    from skill_hub.log_insights import build_digest
    d = build_digest(hours=1)

    assert d["total"] == 3
    assert d["distinct_sessions"] == 2
    assert d["by_kind"].get("tool_invoke", 0) == 2
    assert d["by_kind"].get("tool_result", 0) == 1


def test_build_digest_excludes_old_events(isolated_store):
    """Events older than hours window are not counted."""
    t_old = time.time() - 3 * 3600 - 10   # 3h+ ago
    t_new = time.time() - 60               # 1 min ago
    isolated_store.append_event("s1", "tool_invoke", {}, ts=t_old)
    isolated_store.append_event("s2", "tool_invoke", {}, ts=t_new)

    from skill_hub.log_insights import build_digest
    d = build_digest(hours=1)

    assert d["total"] == 1
    assert d["distinct_sessions"] == 1


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_build_digest_top_failures(isolated_store):
    """top_failures contains recurring failure clusters."""
    t0 = time.time() - 60
    payload = json.dumps({"ok": False, "error": "connection refused"})
    for _ in range(3):
        isolated_store.append_event("s", "tool_result", payload, tool_name="search_skills", ts=t0)

    from skill_hub.log_insights import build_digest
    d = build_digest(hours=1)

    assert len(d["top_failures"]) >= 1
    f = d["top_failures"][0]
    assert f["tool"] == "search_skills"
    assert f["count"] == 3


def test_build_digest_top_failures_capped_at_5(isolated_store):
    """top_failures never returns more than 5 entries."""
    t0 = time.time() - 60
    for i in range(10):
        payload = json.dumps({"ok": False, "error": f"unique error {i}"})
        for _ in range(2):
            isolated_store.append_event("s", "tool_result", payload, tool_name=f"tool_{i}", ts=t0)

    from skill_hub.log_insights import build_digest
    d = build_digest(hours=1)

    assert len(d["top_failures"]) <= 5


# ---------------------------------------------------------------------------
# Skill highlights
# ---------------------------------------------------------------------------

def test_build_digest_skill_highlights(isolated_store):
    """skills list reflects injection counts from the DB."""
    for _ in range(3):
        isolated_store.log_skill_injection("skill-alpha", query="q", session_id="s1")
    isolated_store.log_skill_injection("skill-beta", query="q", session_id="s2")

    from skill_hub.log_insights import build_digest
    d = build_digest(hours=1)

    skill_ids = [s["skill_id"] for s in d["skills"]]
    assert "skill-alpha" in skill_ids

    top = d["skills"][0]
    assert top["skill_id"] == "skill-alpha"
    assert top["injections"] == 3
    assert d["total_injections"] == 4


def test_build_digest_no_injection_data(isolated_store):
    """When there are no injections, skills list is empty and totals are zero."""
    from skill_hub.log_insights import build_digest
    d = build_digest(hours=1)
    assert d["skills"] == []
    assert d["total_injections"] == 0
    assert d["total_feedback"] == 0


# ---------------------------------------------------------------------------
# Dashboard: _log_insights_context returns expected shape
# ---------------------------------------------------------------------------

def test_log_insights_context_returns_dict(isolated_store):
    """_log_insights_context returns a dict with 'log_digest' key."""
    from skill_hub.webapp.routes.health import _log_insights_context
    ctx = _log_insights_context()
    assert "log_digest" in ctx
    ld = ctx["log_digest"]
    required = {
        "hours", "window", "total", "distinct_sessions",
        "by_kind", "top_failures", "skills",
        "total_injections", "total_feedback",
    }
    assert required <= ld.keys()


def test_log_insights_context_safe_on_import_error(monkeypatch):
    """_log_insights_context falls back to _LOG_DIGEST_EMPTY when build_digest raises."""
    import skill_hub.log_insights as _li
    monkeypatch.setattr(_li, "build_digest", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    from skill_hub.webapp.routes.health import _log_insights_context
    ctx = _log_insights_context()
    # Should not raise; must return the fallback empty dict
    assert "log_digest" in ctx
    assert ctx["log_digest"]["total"] == 0
