"""import_ruflo_agents.py — one-shot importer for ruflo agent personas.

Reads ruflo agent definitions from a Claude Code plugin cache on disk and
emits Claude Code subagent definitions under ``~/.skill_hub/agents/<name>.yml``
matching the ``~/.claude/agents/`` schema (name + description + model +
system prompt body).

This script never imports ruflo or claude-flow as Python — it parses
markdown frontmatter directly. After running it, the imported personas
work in any Claude Code session without ruflo installed.

Scope (issue #24): ``ruflo-core:*``, ``ruflo-swarm:*``,
``ruflo-autopilot:*``, ``ruflo-federation:*``.

Usage:
    python scripts/import_ruflo_agents.py            # dry-run
    python scripts/import_ruflo_agents.py --apply    # write files
    python scripts/import_ruflo_agents.py --source PATH --dest PATH --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Plugins in scope for the importer (per acceptance criteria of issue #24).
# Namespace prefix → plugin directory name under the ruflo plugin cache.
RUFLO_PLUGIN_NAMESPACES = {
    "ruflo-core": "ruflo-core",
    "ruflo-swarm": "ruflo-swarm",
    "ruflo-autopilot": "ruflo-autopilot",
    "ruflo-federation": "ruflo-federation",
}

DEFAULT_SOURCE = Path.home() / ".claude" / "plugins" / "cache" / "ruflo"
DEFAULT_DEST = Path.home() / ".skill_hub" / "agents"


# ---------------------------------------------------------------------------
# Frontmatter parsing — no PyYAML dependency required for reading
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<fm>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class RufloAgent:
    """A parsed ruflo agent definition.

    ``plugin`` is the ruflo plugin slug (e.g. ``ruflo-core``).
    ``qualified_name`` is ``"<plugin>:<name>"`` (e.g. ``ruflo-core:coder``).
    """

    plugin: str
    name: str
    description: str
    model: str
    prompt: str
    source_path: Path

    @property
    def qualified_name(self) -> str:
        return f"{self.plugin}:{self.name}"

    @property
    def output_filename(self) -> str:
        """File-system-safe filename for the emitted ``.yml``.

        Uses the qualified name with the ``:`` replaced by ``__`` so that two
        agents named ``coordinator`` in different plugins don't collide.
        """
        return f"{self.plugin}__{self.name}.yml"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract the YAML frontmatter (as a flat string→string dict) and the body.

    Returns ``({}, text)`` if no frontmatter block is present. The parser is
    intentionally minimal — it handles only ``key: value`` lines, which is
    what every shipped ruflo agent uses. It avoids pulling PyYAML into the
    importer so the script can be audited as zero-dep.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    for line in m.group("fm").splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip()
    return fm, m.group("body")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _resolve_plugin_version_dir(plugin_root: Path) -> Path | None:
    """Pick the highest-versioned subdirectory under a plugin cache entry.

    ruflo plugin cache layout is ``<plugin>/<version>/agents/*.md``. There is
    usually exactly one version directory, but if the user has multiple
    installed we deterministically pick the lexicographically-highest one.
    """
    if not plugin_root.is_dir():
        return None
    versions = sorted(
        (p for p in plugin_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return versions[0] if versions else None


def discover_agents(source: Path) -> list[RufloAgent]:
    """Walk ``source`` and parse every in-scope ruflo agent markdown file."""
    out: list[RufloAgent] = []
    for namespace, plugin_dir in RUFLO_PLUGIN_NAMESPACES.items():
        plugin_root = source / plugin_dir
        version_dir = _resolve_plugin_version_dir(plugin_root)
        if version_dir is None:
            continue
        agents_dir = version_dir / "agents"
        if not agents_dir.is_dir():
            continue
        for md in sorted(agents_dir.glob("*.md")):
            agent = parse_agent_file(md, plugin=namespace)
            if agent is not None:
                out.append(agent)
    return out


def parse_agent_file(path: Path, *, plugin: str) -> RufloAgent | None:
    """Parse a single ruflo agent ``.md`` file. Returns ``None`` on bad input."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm, body = _parse_frontmatter(text)
    name = fm.get("name") or path.stem
    description = fm.get("description", "").strip()
    model = fm.get("model", "sonnet").strip() or "sonnet"
    if not name:
        return None
    return RufloAgent(
        plugin=plugin,
        name=name,
        description=description,
        model=model,
        prompt=body.strip(),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _yaml_escape_block(value: str) -> str:
    """Render a (possibly multi-line) string as a YAML literal block scalar.

    Uses the ``|`` indicator with two-space indentation so the body is
    preserved verbatim — Claude Code's subagent loader treats the system
    prompt as plain text once parsed.
    """
    if not value:
        return '""'
    indented = "\n".join("  " + line for line in value.splitlines())
    return f"|\n{indented}"


def _yaml_escape_inline(value: str) -> str:
    """Quote an inline scalar safely for YAML 1.2 flow output."""
    # Always double-quote and escape backslash + double-quote. This is safe
    # for arbitrary printable text (descriptions can include em-dashes,
    # colons, etc.) without needing the full PyYAML emitter.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_subagent_yaml(agent: RufloAgent) -> str:
    """Emit a Claude Code subagent definition as YAML.

    Schema mirrors the ``~/.claude/agents/`` frontmatter contract:
    ``name`` (string), ``description`` (string), ``model`` (string),
    ``prompt`` (multi-line system prompt). A ``source`` block records the
    ruflo origin so re-imports stay traceable.
    """
    lines = [
        f"name: {_yaml_escape_inline(agent.name)}",
        f"description: {_yaml_escape_inline(agent.description)}",
        f"model: {_yaml_escape_inline(agent.model)}",
        "source:",
        f"  plugin: {_yaml_escape_inline(agent.plugin)}",
        f"  qualified_name: {_yaml_escape_inline(agent.qualified_name)}",
        f"prompt: {_yaml_escape_block(agent.prompt)}",
    ]
    return "\n".join(lines) + "\n"


def write_agent(agent: RufloAgent, dest: Path) -> Path:
    """Render ``agent`` and write it under ``dest/<plugin>__<name>.yml``."""
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / agent.output_filename
    out_path.write_text(render_subagent_yaml(agent), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    source: Path,
    dest: Path,
    *,
    apply: bool,
    stdout=sys.stdout,
) -> int:
    """Discover ruflo agents under ``source`` and emit subagent YAMLs to ``dest``.

    Returns the number of agents processed. Without ``apply``, prints what
    would be written and returns the count without touching disk.
    """
    agents = discover_agents(source)
    if not agents:
        print(f"No ruflo agents found under {source}", file=stdout)
        return 0
    print(
        f"Found {len(agents)} ruflo agents under {source}",
        file=stdout,
    )
    for agent in agents:
        target = dest / agent.output_filename
        if apply:
            write_agent(agent, dest)
            print(f"  wrote {target}  ({agent.qualified_name})", file=stdout)
        else:
            print(
                f"  would write {target}  ({agent.qualified_name})",
                file=stdout,
            )
    if not apply:
        print("\nDry run — pass --apply to write files.", file=stdout)
    return len(agents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Ruflo plugin cache root (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Output directory for emitted subagent YAML files (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write files. Without this flag, the script prints a plan only.",
    )
    args = parser.parse_args(argv)
    run(args.source, args.dest, apply=args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
