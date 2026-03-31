"""Scan ~/.claude/plugins for SKILL.md files, parse metadata, embed, store."""

import re
from pathlib import Path

from . import config as _cfg
from .embeddings import embed, EMBED_MODEL
from .store import Skill, SkillStore

# Plugin directories Claude Code uses
PLUGIN_DIRS: list[Path] = [
    Path.home() / ".claude" / "plugins" / "cache",
    Path.home() / ".claude" / "plugins" / "marketplaces",
    Path.home() / ".claude" / "skills",          # user-local skills
]

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_FIELD_RE    = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)


def _parse_skill_file(path: Path) -> tuple[str, str, str] | None:
    """
    Returns (name, description, content) or None if unparseable.
    Handles both YAML-frontmatter and plain # Heading files.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    name = description = ""
    content = raw

    fm_match = _FRONTMATTER_RE.match(raw)
    if fm_match:
        fm_block = fm_match.group(1)
        fields = dict(_FM_FIELD_RE.findall(fm_block))
        name        = fields.get("name", "").strip()
        description = fields.get("description", "").strip()
        content     = raw[fm_match.end():]
    else:
        # No frontmatter — derive name from first H1 heading
        h1 = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        if h1:
            name = h1.group(1).strip()
        # First non-heading paragraph as description
        body = re.sub(r"^#+.*$", "", raw, flags=re.MULTILINE)
        para = re.search(r"\S.{20,}", body)
        if para:
            description = para.group(0)[:200].strip()

    if not name:
        # Fallback: use directory name
        name = path.parent.name

    return name, description, content


def _skill_id_from_path(path: Path) -> str:
    """
    Derive a stable skill id from the file path.
    e.g. cache/claude-plugins-official/superpowers/.../skills/brainstorm/SKILL.md
      → "superpowers:brainstorm"
    """
    parts = path.parts
    # Find "skills" segment
    try:
        idx = next(i for i in range(len(parts) - 1, -1, -1) if parts[i] == "skills")
        skill_name = parts[idx + 1] if idx + 1 < len(parts) - 1 else path.parent.name
        # Walk backwards to find plugin name (skip version hashes)
        plugin_name = ""
        for p in reversed(parts[:idx]):
            if re.match(r"^[0-9a-f]{8,}$", p) or p in ("cache", "marketplaces",
                                                         "claude-plugins-official",
                                                         "anthropic-agent-skills"):
                continue
            plugin_name = p
            break
        if plugin_name:
            return f"{plugin_name}:{skill_name}"
    except StopIteration:
        pass
    return path.parent.name


def _plugin_from_id(skill_id: str) -> str:
    return skill_id.split(":")[0] if ":" in skill_id else ""


def index_all(store: SkillStore, embed_model: str = EMBED_MODEL,
              use_rerank: bool = False) -> tuple[int, list[str]]:
    """
    Walk all plugin directories, index every SKILL.md found.
    Returns (count_indexed, list_of_errors).
    """
    indexed = 0
    errors: list[str] = []

    def _index_skill_file(skill_file: Path, skill_id: str, plugin: str) -> None:
        nonlocal indexed
        parsed = _parse_skill_file(skill_file)
        if not parsed:
            errors.append(f"parse failed: {skill_file}")
            return
        name, description, content = parsed
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            content=content,
            file_path=str(skill_file),
            plugin=plugin,
        )
        store.upsert_skill(skill)
        embed_text = f"{name}: {description}" if description else name
        try:
            vector = embed(embed_text, model=embed_model)
            store.upsert_embedding(skill_id, embed_model, vector)
            indexed += 1
        except Exception as exc:
            errors.append(f"embed failed for {skill_id}: {exc}")

    # Built-in plugin directories
    for base in PLUGIN_DIRS:
        if not base.exists():
            continue
        for skill_file in base.rglob("SKILL.md"):
            skill_id = _skill_id_from_path(skill_file)
            _index_skill_file(skill_file, skill_id, _plugin_from_id(skill_id))

    # Extra skill/plugin directories from config
    for entry in _cfg.get("extra_skill_dirs") or []:
        if not entry.get("enabled", True):
            continue
        base = Path(entry["path"]).expanduser()
        source = entry.get("source", "extra")
        if not base.exists():
            continue
        for skill_file in base.rglob("SKILL.md"):
            # flat structure: <base>/<skill-name>/SKILL.md → id = source:skill-name
            skill_id = f"{source}:{skill_file.parent.name}"
            _index_skill_file(skill_file, skill_id, source)

    return indexed, errors
