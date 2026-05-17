"""Unit tests for scripts/import_ruflo_agents.py.

Fixture-based — never touches the user's real ruflo install or
``~/.skill_hub/agents``. Each test builds a fake ruflo plugin cache under
``tmp_path``, runs the importer against it, and asserts on the emitted
YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Make scripts/ importable
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import import_ruflo_agents as _mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_agent(
    cache_root: Path,
    plugin: str,
    version: str,
    name: str,
    description: str,
    model: str,
    body: str,
) -> Path:
    """Create a ruflo-shaped agent .md inside a fake plugin cache."""
    agents_dir = cache_root / plugin / version / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    md = agents_dir / f"{name}.md"
    md.write_text(
        f"""---
name: {name}
description: {description}
model: {model}
---
{body}
""",
        encoding="utf-8",
    )
    return md


@pytest.fixture()
def fake_ruflo_cache(tmp_path: Path) -> Path:
    """Build a fake ruflo cache covering all four in-scope namespaces."""
    root = tmp_path / "ruflo_cache"
    _write_agent(
        root, "ruflo-core", "0.2.1",
        name="coder",
        description="Implementation specialist",
        model="sonnet",
        body="You are a code implementation specialist.",
    )
    _write_agent(
        root, "ruflo-core", "0.2.1",
        name="researcher",
        description="Pathfinder research specialist",
        model="sonnet",
        body="You are a pathfinder researcher.",
    )
    _write_agent(
        root, "ruflo-swarm", "0.2.0",
        name="coordinator",
        description="Swarm coordinator",
        model="sonnet",
        body="You are the swarm coordinator.",
    )
    _write_agent(
        root, "ruflo-autopilot", "0.2.0",
        name="autopilot-coordinator",
        description="Autonomous task completion coordinator",
        model="sonnet",
        body="You are the autopilot coordinator.",
    )
    _write_agent(
        root, "ruflo-federation", "0.2.0",
        name="federation-coordinator",
        description="Federation coordinator",
        model="opus",
        body="You orchestrate federation peers.",
    )
    # An out-of-scope plugin — should be ignored by the importer.
    _write_agent(
        root, "ruflo-goals", "0.2.0",
        name="goal-planner",
        description="Goal planner",
        model="sonnet",
        body="You plan goals.",
    )
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_discover_finds_only_in_scope_plugins(fake_ruflo_cache: Path) -> None:
    agents = _mod.discover_agents(fake_ruflo_cache)
    namespaces = {a.plugin for a in agents}
    assert namespaces == {
        "ruflo-core",
        "ruflo-swarm",
        "ruflo-autopilot",
        "ruflo-federation",
    }
    # ruflo-goals is not in scope and must be ignored.
    assert all(a.plugin != "ruflo-goals" for a in agents)


def test_discover_parses_frontmatter(fake_ruflo_cache: Path) -> None:
    agents = {a.qualified_name: a for a in _mod.discover_agents(fake_ruflo_cache)}
    coder = agents["ruflo-core:coder"]
    assert coder.name == "coder"
    assert coder.description == "Implementation specialist"
    assert coder.model == "sonnet"
    assert "code implementation specialist" in coder.prompt
    fed = agents["ruflo-federation:federation-coordinator"]
    assert fed.model == "opus"


def test_all_known_namespaces_handled(fake_ruflo_cache: Path) -> None:
    """Acceptance criterion: every in-scope ruflo agent namespace is handled."""
    agents = _mod.discover_agents(fake_ruflo_cache)
    qualified = {a.qualified_name for a in agents}
    # The fixture covers at least one agent per required namespace.
    required_prefixes = (
        "ruflo-core:",
        "ruflo-swarm:",
        "ruflo-autopilot:",
        "ruflo-federation:",
    )
    for prefix in required_prefixes:
        assert any(q.startswith(prefix) for q in qualified), (
            f"missing agent for namespace {prefix}"
        )


def test_apply_writes_yaml_files(fake_ruflo_cache: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    count = _mod.run(fake_ruflo_cache, dest, apply=True)
    assert count == 5  # 4 in-scope namespaces, 5 agents total
    yml_files = sorted(p.name for p in dest.glob("*.yml"))
    assert "ruflo-core__coder.yml" in yml_files
    assert "ruflo-swarm__coordinator.yml" in yml_files
    assert "ruflo-autopilot__autopilot-coordinator.yml" in yml_files
    assert "ruflo-federation__federation-coordinator.yml" in yml_files


def test_dry_run_writes_nothing(fake_ruflo_cache: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    count = _mod.run(fake_ruflo_cache, dest, apply=False)
    assert count == 5
    assert not dest.exists() or not any(dest.iterdir())


def test_emitted_yaml_is_valid_and_has_required_fields(
    fake_ruflo_cache: Path, tmp_path: Path
) -> None:
    """The acceptance criterion: verify subagent YAML schema validity."""
    dest = tmp_path / "out"
    _mod.run(fake_ruflo_cache, dest, apply=True)
    for yml_path in dest.glob("*.yml"):
        data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        # Required keys from the ~/.claude/agents/ schema:
        assert isinstance(data, dict)
        for key in ("name", "description", "model", "prompt"):
            assert key in data, f"{yml_path.name} missing {key}"
            assert isinstance(data[key], str)
            assert data[key].strip(), f"{yml_path.name} has empty {key}"
        # Source trace block (importer-added; not in the upstream schema
        # but lets a re-import detect drift):
        assert "source" in data and isinstance(data["source"], dict)
        assert data["source"]["qualified_name"].startswith(
            ("ruflo-core:", "ruflo-swarm:", "ruflo-autopilot:", "ruflo-federation:")
        )


def test_prompt_body_is_preserved_verbatim(
    fake_ruflo_cache: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "out"
    _mod.run(fake_ruflo_cache, dest, apply=True)
    coder_yml = dest / "ruflo-core__coder.yml"
    data = yaml.safe_load(coder_yml.read_text(encoding="utf-8"))
    assert data["prompt"].strip() == "You are a code implementation specialist."


def test_render_handles_special_chars() -> None:
    """Descriptions with em-dashes, quotes, and colons must round-trip."""
    agent = _mod.RufloAgent(
        plugin="ruflo-core",
        name="weird",
        description='Pathfinder — uses "quotes", colons: and backslashes\\',
        model="sonnet",
        prompt="line one\nline two: with colon\n  indented",
        source_path=Path("/dev/null"),
    )
    out = _mod.render_subagent_yaml(agent)
    data = yaml.safe_load(out)
    assert data["description"] == agent.description
    assert data["prompt"].strip() == agent.prompt.strip()


def test_missing_source_dir_is_handled(tmp_path: Path) -> None:
    """Importer must not raise when the ruflo cache doesn't exist."""
    agents = _mod.discover_agents(tmp_path / "does-not-exist")
    assert agents == []


def test_picks_highest_version_when_multiple(tmp_path: Path) -> None:
    """If multiple plugin versions are cached, pick the highest."""
    root = tmp_path / "cache"
    _write_agent(
        root, "ruflo-core", "0.1.0",
        name="coder", description="old", model="sonnet", body="old body",
    )
    _write_agent(
        root, "ruflo-core", "0.2.1",
        name="coder", description="new", model="sonnet", body="new body",
    )
    agents = _mod.discover_agents(root)
    assert len(agents) == 1
    assert agents[0].description == "new"


def test_cli_entry_point(fake_ruflo_cache: Path, tmp_path: Path) -> None:
    """`python scripts/import_ruflo_agents.py --apply` writes files."""
    dest = tmp_path / "out"
    rc = _mod.main(
        ["--source", str(fake_ruflo_cache), "--dest", str(dest), "--apply"]
    )
    assert rc == 0
    assert any(dest.glob("*.yml"))
