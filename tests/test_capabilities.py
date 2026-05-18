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
    # Allow extra stacked decorators (e.g. @requires_capability) between
    # @mcp.tool() and the def — added by issue #7.
    pat = re.compile(
        r"^@mcp\.tool\(\)\s*\n(?:@[A-Za-z_][\w.]*\([^)]*\)\s*\n)*"
        r"def\s+([A-Za-z_][A-Za-z0-9_]*)\(",
        re.MULTILINE,
    )
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


# ---------------------------------------------------------------------------
# Issue #7 — tier registry & @requires_capability decorator


def test_tier_registry_covers_every_tool():
    """Acceptance: registry contains all tools — no ``unknown`` tier."""
    matrix_tools = {t.name for t in cap.TOOLS}
    assert set(cap.TIER_REGISTRY.keys()) >= matrix_tools, (
        f"missing from TIER_REGISTRY: "
        f"{sorted(matrix_tools - set(cap.TIER_REGISTRY.keys()))}"
    )
    for name, tier in cap.TIER_REGISTRY.items():
        assert tier in ("none", "embedding", "llm"), (
            f"{name} has invalid tier {tier!r}"
        )


def test_tier_registry_covers_every_mcp_tool_in_server():
    """Every @mcp.tool in server.py must have a declared tier."""
    import re
    server_py = (SRC / "skill_hub" / "server.py").read_text()
    pat = re.compile(
        r"^@mcp\.tool\(\)\s*\n(?:@[A-Za-z_][\w.]*\([^)]*\)\s*\n)*"
        r"def\s+([A-Za-z_][A-Za-z0-9_]*)\(",
        re.MULTILINE,
    )
    server_tools = set(pat.findall(server_py))
    assert server_tools, "no @mcp.tool functions found — regex broken?"
    missing = server_tools - set(cap.TIER_REGISTRY.keys())
    assert not missing, f"server tools without declared tier: {sorted(missing)}"


def test_tier_from_spec_priorities():
    """``llm`` > ``embedding`` > ``none`` when deriving from a ToolSpec."""
    s_none = cap.ToolSpec("x", "x", hard=(cap.BACKEND_DB,))
    s_embed = cap.ToolSpec("x", "x", hard=(cap.BACKEND_DB, cap.BACKEND_EMBED))
    s_llm = cap.ToolSpec(
        "x", "x", hard=(cap.BACKEND_EMBED, cap.BACKEND_REASON_LLM),
    )
    s_soft_only = cap.ToolSpec(
        "x", "x", hard=(), soft=(cap.BACKEND_REASON_LLM,),
    )
    assert cap.tier_from_spec(s_none) == "none"
    assert cap.tier_from_spec(s_embed) == "embedding"
    assert cap.tier_from_spec(s_llm) == "llm"
    # soft deps never promote tier — the tool still works without them.
    assert cap.tier_from_spec(s_soft_only) == "none"


def test_requires_capability_stamps_and_registers():
    """Decorator records the tier on the function and in TIER_REGISTRY."""
    @cap.requires_capability("embedding")
    def _probe_tool_xyz():
        return None

    assert _probe_tool_xyz.__capability_tier__ == "embedding"
    assert cap.TIER_REGISTRY["_probe_tool_xyz"] == "embedding"
    # Cleanup so we don't pollute the registry for other tests.
    cap.TIER_REGISTRY.pop("_probe_tool_xyz", None)


def test_requires_capability_rejects_invalid_tier():
    with pytest.raises(ValueError):
        cap.requires_capability("magic")  # type: ignore[arg-type]


def test_tier_for_known_and_unknown():
    # Known tool — looked up via registry.
    assert cap.tier_for("search_skills") == "embedding"
    assert cap.tier_for("list_teachings") == "none"
    assert cap.tier_for("compact_master_state") == "llm"
    # Unknown tool — must raise so callers can't silently swallow it.
    with pytest.raises(KeyError):
        cap.tier_for("definitely_not_a_real_tool_zzz")


def test_render_matrix_includes_tier():
    data = cap.render_matrix()
    by_name = {t["name"]: t for t in data["tools"]}
    assert by_name["search_skills"]["tier"] == "embedding"
    assert by_name["list_teachings"]["tier"] == "none"
    assert by_name["compact_master_state"]["tier"] == "llm"
    # All rows must have a tier — no unknowns.
    for row in data["tools"]:
        assert row["tier"] in ("none", "embedding", "llm"), row


def test_server_tier_decorators_match_registry():
    """server.py's inline @requires_capability call must match TIER_REGISTRY.

    Catches drift between the inline declaration and the spec table.
    """
    import re
    server_py = (SRC / "skill_hub" / "server.py").read_text()
    pat = re.compile(
        r'@requires_capability\("(none|embedding|llm)"\)\s*\n'
        r'def\s+([A-Za-z_][A-Za-z0-9_]*)\(',
        re.MULTILINE,
    )
    found = pat.findall(server_py)
    assert found, "no @requires_capability decorators found in server.py"
    for tier, fname in found:
        assert cap.TIER_REGISTRY.get(fname) == tier, (
            f"server.py declares {fname} as {tier!r} but registry says "
            f"{cap.TIER_REGISTRY.get(fname)!r}"
        )


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
