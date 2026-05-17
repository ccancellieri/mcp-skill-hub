"""Tests for the capability matrix module and /status/capabilities route.

Issue #13: the dashboard view must reflect current backend state and stay
in lock-step with the ``status`` MCP tool. These tests pin both: a frozen
backend probe yields a deterministic verdict per tool, and the page +
JSON endpoint render that verdict consistently.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from skill_hub import capabilities as cap  # noqa: E402
from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub.services import registry as reg_mod  # noqa: E402


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Build a fresh FastAPI app pointed at a tmp config."""
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
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


def _force_backends(monkeypatch, **overrides):
    """Pin every backend probe to a deterministic boolean for tests."""
    defaults = {bid: False for bid in cap.BACKENDS}
    defaults[cap.BACKEND_MCP] = True
    defaults[cap.BACKEND_DB] = True
    defaults.update(overrides)
    for bid, value in defaults.items():
        original = cap.BACKENDS[bid]
        new = cap.Backend(
            id=original.id,
            label=original.label,
            setup=original.setup,
            check=(lambda v=value: v),
        )
        monkeypatch.setitem(cap.BACKENDS, bid, new)


def test_backend_ids_unique_and_covered():
    """Every backend ID referenced by a tool must exist in BACKENDS."""
    declared = set(cap.BACKENDS.keys())
    referenced: set[str] = set()
    for spec in cap.TOOLS:
        referenced.update(spec.hard)
        referenced.update(spec.soft)
    missing = referenced - declared
    assert not missing, f"tools reference unknown backends: {missing}"


def test_matrix_covers_every_mcp_tool():
    """Capability matrix must enumerate every @mcp.tool in server.py.

    Without this guard, a new tool would silently render as 'always works'.
    """
    import re
    server_py = (SRC / "skill_hub" / "server.py").read_text()
    pat = re.compile(r"^@mcp\.tool\(\)\s*\n+def\s+([A-Za-z_][A-Za-z0-9_]*)\(",
                     re.MULTILINE)
    server_tools = set(pat.findall(server_py))
    matrix_tools = {t.name for t in cap.TOOLS}
    assert server_tools, "no @mcp.tool functions found — regex broken?"
    missing = server_tools - matrix_tools
    extra = matrix_tools - server_tools
    assert not missing, f"capabilities matrix missing tools: {sorted(missing)}"
    assert not extra, f"capabilities matrix has unknown tools: {sorted(extra)}"


def test_red_when_hard_dep_missing(monkeypatch):
    _force_backends(monkeypatch)  # everything off except mcp+db
    data = cap.render_matrix()
    verdict = {t["name"]: t["verdict"] for t in data["tools"]}
    # search_skills needs embed → red.
    assert verdict["search_skills"] == "red"
    # search_web needs SearXNG → red.
    assert verdict["search_web"] == "red"


def test_green_when_all_deps_met(monkeypatch):
    _force_backends(
        monkeypatch,
        **{bid: True for bid in cap.BACKENDS},
    )
    data = cap.render_matrix()
    verdict = {t["name"]: t["verdict"] for t in data["tools"]}
    # Everything available → search_skills green.
    assert verdict["search_skills"] == "green"
    # Pure-stdlib tools must always be green.
    assert verdict["list_teachings"] == "green"
    assert verdict["status"] == "green"
    assert verdict["configure"] == "green"


def test_yellow_when_only_soft_dep_missing(monkeypatch):
    # update_marketplace: hard=git, soft=embed. git on, embed off → yellow.
    _force_backends(monkeypatch, git=True, embed=False)
    data = cap.render_matrix()
    verdict = {t["name"]: t["verdict"] for t in data["tools"]}
    assert verdict["update_marketplace"] == "yellow"


def test_summary_counts_sum_to_total(monkeypatch):
    _force_backends(monkeypatch)  # mostly red world
    data = cap.render_matrix()
    s = data["summary"]
    assert s["green"] + s["yellow"] + s["red"] == s["total"]
    assert s["total"] == len(cap.TOOLS)


def test_capabilities_route_renders(app_client, monkeypatch):
    _force_backends(monkeypatch)
    r = app_client.get("/status/capabilities")
    assert r.status_code == 200
    assert "Capabilities" in r.text
    # Status dot HTML must be present so the visual table renders.
    assert "status-dot" in r.text
    assert "dot-red" in r.text
    # A known stdlib tool must appear.
    assert "list_teachings" in r.text


def test_capabilities_json(app_client, monkeypatch):
    _force_backends(monkeypatch)
    r = app_client.get("/api/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert "backends" in data and "tools" in data and "summary" in data
    assert len(data["tools"]) == len(cap.TOOLS)
    assert {b["id"] for b in data["backends"]} == set(cap.BACKENDS.keys())


def test_state_matches_status_tool(monkeypatch):
    """Acceptance criterion 2: state matches `status` MCP tool output exactly.

    Both views must agree on whether Ollama is reachable, whether the
    embed backend is usable, and whether the reasoning model is installed.
    We pin the underlying probes and confirm the matrix flags align.
    """
    _force_backends(
        monkeypatch,
        ollama=False,
        embed=False,
        reason_llm=False,
    )
    data = cap.render_matrix()
    by_id = {b["id"]: b for b in data["backends"]}
    assert by_id[cap.BACKEND_OLLAMA]["ok"] is False
    assert by_id[cap.BACKEND_EMBED]["ok"] is False
    assert by_id[cap.BACKEND_REASON_LLM]["ok"] is False
    # Any tool with hard=embed must now be red (matches the `status`
    # output line "No embedding backend available").
    verdict = {t["name"]: t["verdict"] for t in data["tools"]}
    assert verdict["search_skills"] == "red"
    assert verdict["search_context"] == "red"
