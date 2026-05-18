"""Tests for the no_llm_mode config flag (issue #6).

Covers the acceptance criteria:
  - Flag persists across restarts (round-trip through ``load_config`` / ``save_config``).
  - ``embed_available()`` short-circuits to False without probing any backend.
  - Disabled MCP tools return a clear user-facing error mentioning no_llm_mode.
  - ``status`` output reports "No-LLM mode (N/M tools available)" where N/M
    come from ``capabilities.TOOLS`` (single source of truth — issue #13).
  - The capability matrix flags LLM backends as missing without calling their
    probe functions, so the dashboard verdict is deterministic.
  - The base.html banner middleware sets ``request.state.no_llm_mode`` so the
    sticky banner renders on every page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from skill_hub import capabilities as cap  # noqa: E402
from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub import embeddings as emb  # noqa: E402


def _point_config_at(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", path)
    return path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_no_llm_mode_default_is_false(tmp_path, monkeypatch):
    _point_config_at(monkeypatch, tmp_path)
    assert cfg_mod.load_config()["no_llm_mode"] is False
    assert cfg_mod.get("no_llm_mode") is False


def test_no_llm_mode_persists_across_restarts(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    cfg_mod.set("no_llm_mode", True)

    # On-disk state survives a "restart" (fresh load_config call).
    on_disk = json.loads(path.read_text())
    assert on_disk["no_llm_mode"] is True
    assert cfg_mod.load_config()["no_llm_mode"] is True


# ---------------------------------------------------------------------------
# embed_available short-circuit
# ---------------------------------------------------------------------------

def test_embed_available_returns_false_in_no_llm_mode_without_probing(
    tmp_path, monkeypatch
):
    _point_config_at(monkeypatch, tmp_path)
    cfg_mod.set("no_llm_mode", True)

    # If we *do* probe a backend, this canary will fire and fail the test.
    def _explode():
        raise AssertionError("backend probed despite no_llm_mode=True")

    monkeypatch.setattr(
        "skill_hub.ollama_client.get_ollama_client",
        lambda: _explode(),  # called only if Ollama branch runs
    )

    assert emb.embed_available() is False


def test_embed_unavailable_reason_mentions_no_llm_mode(tmp_path, monkeypatch):
    _point_config_at(monkeypatch, tmp_path)
    cfg_mod.set("no_llm_mode", True)
    msg = emb.embed_unavailable_reason()
    assert "no_llm_mode" in msg.lower()


def test_embed_unavailable_reason_normal_path(tmp_path, monkeypatch):
    _point_config_at(monkeypatch, tmp_path)
    cfg_mod.set("no_llm_mode", False)
    msg = emb.embed_unavailable_reason()
    # When the flag is off the message points at the cascade backends.
    assert "VOYAGE_API_KEY" in msg or "Ollama" in msg


# ---------------------------------------------------------------------------
# capabilities.py respects the flag
# ---------------------------------------------------------------------------

def test_capabilities_matrix_marks_llm_backends_missing(tmp_path, monkeypatch):
    _point_config_at(monkeypatch, tmp_path)
    cfg_mod.set("no_llm_mode", True)

    # Force every backend's *underlying* probe to True so the only thing
    # that can flip them off is the no_llm_mode short-circuit.
    for bid in cap.BACKENDS:
        original = cap.BACKENDS[bid]
        monkeypatch.setitem(
            cap.BACKENDS,
            bid,
            cap.Backend(
                id=original.id,
                label=original.label,
                setup=original.setup,
                check=lambda: True,
            ),
        )

    data = cap.render_matrix()
    by_id = {b["id"]: b for b in data["backends"]}

    # LLM-tied backends report False even though their probe returns True.
    assert by_id[cap.BACKEND_EMBED]["ok"] is False
    assert by_id[cap.BACKEND_OLLAMA]["ok"] is False
    assert by_id[cap.BACKEND_REASON_LLM]["ok"] is False
    assert by_id[cap.BACKEND_VOYAGE]["ok"] is False

    # MCP / DB / Git stay green — they don't need a local LLM.
    assert by_id[cap.BACKEND_MCP]["ok"] is True
    assert by_id[cap.BACKEND_DB]["ok"] is True

    assert data["no_llm_mode"] is True


def test_no_llm_summary_counts_match_capabilities_tools(tmp_path, monkeypatch):
    """Counts must derive from cap.TOOLS so they stay accurate over time."""
    _point_config_at(monkeypatch, tmp_path)
    ns = cap.no_llm_summary()

    assert ns["available"] + ns["disabled"] == ns["total"]
    assert ns["total"] == len(cap.TOOLS)
    assert sorted(ns["available_tools"] + ns["disabled_tools"]) == sorted(
        t.name for t in cap.TOOLS
    )

    # Sanity: search_skills *must* be in the disabled bucket (hard=embed).
    assert "search_skills" in ns["disabled_tools"]
    # And a stdlib-only tool must be available.
    assert "list_teachings" in ns["available_tools"]
    assert "status" in ns["available_tools"]


def test_no_llm_summary_disabled_tools_all_have_llm_hard_dep():
    """Every 'disabled' tool must hard-require an LLM-tier backend."""
    llm_backends = {cap.BACKEND_EMBED, cap.BACKEND_OLLAMA,
                    cap.BACKEND_REASON_LLM, cap.BACKEND_VOYAGE}
    ns = cap.no_llm_summary()
    spec_by_name = {t.name: t for t in cap.TOOLS}
    for name in ns["disabled_tools"]:
        spec = spec_by_name[name]
        assert any(d in llm_backends for d in spec.hard), (
            f"{name} is listed as disabled but has no LLM hard dep"
        )


# ---------------------------------------------------------------------------
# status MCP tool surfaces the flag
# ---------------------------------------------------------------------------

def test_status_tool_reports_no_llm_mode(tmp_path, monkeypatch):
    _point_config_at(monkeypatch, tmp_path)
    cfg_mod.set("no_llm_mode", True)

    # Import lazily — server.py pulls in a lot of subsystems.
    from skill_hub import server as srv

    out = srv.status(section="summary")
    assert "No-LLM mode" in out
    ns = cap.no_llm_summary()
    assert f"{ns['available']}/{ns['total']}" in out


# ---------------------------------------------------------------------------
# Dashboard banner middleware
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
    from skill_hub.services import registry as reg_mod

    reg_mod.set_registry(reg_mod.ServiceRegistry([]))

    class _Pressure:
        def sample(self):
            from skill_hub.services.monitor import ResourceSample
            return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)

        def sustained_seconds(self):
            return 0.0

        def last_sample(self):
            return self.sample()

    reg_mod.set_pressure(_Pressure())

    from skill_hub.webapp.main import create_app
    app = create_app(store=None)
    return TestClient(app)


def test_banner_visible_when_no_llm_mode_on(app_client, tmp_path, monkeypatch):
    cfg_mod.set("no_llm_mode", True)
    r = app_client.get("/status/capabilities")
    assert r.status_code == 200
    assert "no-llm-banner" in r.text
    assert "No-LLM mode is ON" in r.text


def test_banner_hidden_when_no_llm_mode_off(app_client, tmp_path, monkeypatch):
    cfg_mod.set("no_llm_mode", False)
    r = app_client.get("/status/capabilities")
    assert r.status_code == 200
    assert "no-llm-banner" not in r.text
