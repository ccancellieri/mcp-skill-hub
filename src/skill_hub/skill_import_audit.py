"""Read-only audit for importing skills from loose local folders.

The audit does not mutate config, copy files, or index anything. It only
classifies candidate files so a later UI/API layer can present an approval
queue before live Skill Hub state changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


SKILL_MD = "skill_md"
LOOSE_MARKDOWN = "loose_markdown"
REFERENCE_MARKDOWN = "reference_markdown"
LOCAL_JSON = "local_json"


@dataclass(frozen=True)
class SkillAuditCandidate:
    path: str
    source_root: str
    format: str
    name: str
    description: str
    llm_targets: list[str]
    harnesses: list[str]
    recommendation: str
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_root": self.source_root,
            "format": self.format,
            "name": self.name,
            "description": self.description,
            "llm_targets": self.llm_targets,
            "harnesses": self.harnesses,
            "recommendation": self.recommendation,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class SkillAuditReport:
    sources: list[str]
    candidates: list[SkillAuditCandidate]
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "candidates": [c.as_dict() for c in self.candidates],
            "errors": self.errors,
        }


@dataclass(frozen=True)
class SkillRepairResult:
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FIELD_RE = re.compile(r"^(\w+)\s*:\s*(.+?)\s*$", re.MULTILINE)
_SYSTEM_PROMPT_RE = re.compile(r"\b(system instruction|you are an? )\b", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def audit_paths(paths: Sequence[str | Path]) -> SkillAuditReport:
    """Scan paths for skill-like files and return a read-only audit report."""
    sources: list[str] = []
    candidates: list[SkillAuditCandidate] = []
    errors: list[str] = []

    for raw_path in paths:
        root = Path(raw_path).expanduser()
        sources.append(str(root))
        if not root.exists():
            errors.append(f"source not found: {root}")
            continue
        for file_path in _candidate_files(root):
            try:
                candidate = _audit_file(file_path, root)
            except Exception as exc:  # noqa: BLE001 - audit should continue
                errors.append(f"{file_path}: {exc}")
                continue
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(key=lambda c: (c.source_root, c.path))
    return SkillAuditReport(sources=sources, candidates=candidates, errors=errors)


def default_source_paths(
    skill_import_sources: Sequence[dict[str, Any]] | None = None,
    extra_skill_dirs: Sequence[dict[str, Any]] | None = None,
) -> list[Path]:
    """Return common user skill locations for a first-pass import audit."""
    paths: list[Path] = []
    sources = skill_import_sources if skill_import_sources is not None else [
        {"path": str(Path.home() / ".agents" / "skills"), "enabled": True},
        {"path": str(Path.home() / ".claude" / "skills"), "enabled": True},
        {"path": str(Path.home() / ".claude" / "local-skills"), "enabled": True},
    ]
    for entry in sources:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        raw_path = entry.get("path")
        if raw_path:
            paths.append(Path(str(raw_path)).expanduser())
    for entry in extra_skill_dirs or []:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        raw_path = entry.get("path")
        if raw_path:
            paths.append(Path(str(raw_path)).expanduser())
    return _dedupe_paths(paths)


def parse_path_list(raw: str) -> list[Path]:
    """Parse a JSON, newline, or comma separated path list."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("paths JSON must be a list")
        return [Path(str(item)).expanduser() for item in data]
    pieces = [piece.strip() for piece in re.split(r"[\n,]", raw) if piece.strip()]
    return [Path(piece).expanduser() for piece in pieces]


def render_markdown(report: SkillAuditReport) -> str:
    """Render a compact human-readable audit report."""
    lines = [
        "# Skill Import Audit",
        "",
        f"Scanned sources: {len(report.sources)}",
        f"Candidates: {len(report.candidates)}",
    ]
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {err}" for err in report.errors)
    if not report.candidates:
        lines.extend(["", "No skill candidates found."])
        return "\n".join(lines)

    lines.extend(["", "## Candidates"])
    for candidate in report.candidates:
        target_text = ", ".join(f"`{target}`" for target in candidate.llm_targets) or "`do_not_import`"
        harness_text = ", ".join(f"`{h}`" for h in candidate.harnesses) or "`none`"
        lines.extend(
            [
                "",
                f"### {candidate.name}",
                "",
                f"- Path: `{candidate.path}`",
                f"- Format: `{candidate.format}`",
                f"- Recommendation: `{candidate.recommendation}`",
                f"- LLM targets: {target_text}",
                f"- Harnesses: {harness_text}",
            ]
        )
        if candidate.description:
            lines.append(f"- Description: {candidate.description}")
        if candidate.issues:
            lines.append("- Issues:")
            lines.extend(f"  - {issue}" for issue in candidate.issues)
    return "\n".join(lines)


def render_json(report: SkillAuditReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)


def repair_importable_skills(report: SkillAuditReport) -> SkillRepairResult:
    """Create generated SKILL.md wrappers for audit candidates needing repair.

    The repair is intentionally non-destructive: source files are not moved or
    rewritten. Loose Markdown and reference Markdown become standalone generated
    skill directories that the existing SKILL.md indexer can import.
    """
    created: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for candidate in report.candidates:
        if candidate.recommendation not in {"normalize", "keep_reference"}:
            continue
        try:
            source_file = Path(candidate.path).expanduser()
            source_root = Path(candidate.source_root).expanduser()
            if not source_file.exists():
                errors.append(f"source not found: {source_file}")
                continue
            dest_dir = _repair_destination(source_root, source_file, candidate)
            dest_file = dest_dir / "SKILL.md"
            if dest_file.exists():
                skipped.append(f"already exists: {dest_file}")
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file.write_text(_render_repaired_skill(candidate, source_file), encoding="utf-8")
            created.append(str(dest_file))
        except Exception as exc:  # noqa: BLE001 - keep repairing other skills
            errors.append(f"{candidate.path}: {exc}")

    return SkillRepairResult(created=created, skipped=skipped, errors=errors)


def _candidate_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if _is_candidate_file(root):
            yield root
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_candidate_file(path):
            continue
        yield path


def _is_candidate_file(path: Path) -> bool:
    if path.name == "SKILL.md":
        return True
    if path.suffix.lower() == ".json":
        return True
    if path.suffix.lower() == ".md":
        return True
    return False


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _audit_file(path: Path, root: Path) -> SkillAuditCandidate | None:
    if path.name == "SKILL.md":
        name, description, body, has_frontmatter_description = _parse_markdown(path)
        return _candidate(
            path, root, SKILL_MD, name, description, body, has_frontmatter_description
        )
    if path.suffix.lower() == ".md":
        name, description, body, has_frontmatter_description = _parse_markdown(path)
        fmt = REFERENCE_MARKDOWN if _is_reference_file(path) else LOOSE_MARKDOWN
        return _candidate(
            path, root, fmt, name, description, body, has_frontmatter_description
        )
    if path.suffix.lower() == ".json":
        parsed = _parse_local_json(path)
        if parsed is None:
            return None
        name, description = parsed
        return SkillAuditCandidate(
            path=str(path),
            source_root=str(root),
            format=LOCAL_JSON,
            name=name,
            description=description,
            llm_targets=["L1"],
            harnesses=["generic_mcp"],
            recommendation="import",
            issues=[],
        )
    return None


def _parse_markdown(path: Path) -> tuple[str, str, str, bool]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    name = ""
    description = ""
    body = raw

    match = _FRONTMATTER_RE.match(raw)
    if match:
        fields = dict(_FIELD_RE.findall(match.group(1)))
        name = _strip_quotes(fields.get("name", "")).strip()
        description = _strip_quotes(fields.get("description", "")).strip()
        body = raw[match.end():]
    has_frontmatter_description = bool(description)

    if not name and _SYSTEM_PROMPT_RE.search(raw):
        name = path.stem

    if not name:
        heading = re.search(r"^#\s+(.+)$", _without_fenced_code(raw), re.MULTILINE)
        name = heading.group(1).strip() if heading else path.stem

    if not description:
        without_headings = re.sub(r"^#+.*$", "", body, flags=re.MULTILINE)
        para = re.search(r"\S.{20,}", without_headings)
        if para:
            description = para.group(0)[:200].strip()

    return name, description, body, has_frontmatter_description


def _repair_destination(
    source_root: Path,
    source_file: Path,
    candidate: SkillAuditCandidate,
) -> Path:
    base = _slugify(candidate.name or source_file.stem)
    if candidate.format == REFERENCE_MARKDOWN:
        try:
            rel = source_file.relative_to(source_root)
        except ValueError:
            rel = source_file.name
        parts = [part for part in Path(rel).parts[:-1] if part != "references"]
        prefix = _slugify("-".join(parts))
        if prefix:
            base = f"{prefix}__{base}"
    return _unique_destination(source_root / base)


def _unique_destination(dest_dir: Path) -> Path:
    if not dest_dir.exists():
        return dest_dir
    if (dest_dir / "SKILL.md").exists():
        return dest_dir
    suffix = 2
    while True:
        candidate = dest_dir.with_name(f"{dest_dir.name}-{suffix}")
        if not candidate.exists() or (candidate / "SKILL.md").exists():
            return candidate
        suffix += 1


def _render_repaired_skill(candidate: SkillAuditCandidate, source_file: Path) -> str:
    raw = source_file.read_text(encoding="utf-8", errors="replace")
    body = _FRONTMATTER_RE.sub("", raw, count=1).strip()
    name = _slugify(candidate.name or source_file.stem)
    if candidate.format == REFERENCE_MARKDOWN:
        parent = source_file.parent.parent.name if source_file.parent.name == "references" else ""
        parent_slug = _slugify(parent)
        if parent_slug:
            name = f"{parent_slug}__{name}"
    description = _single_line_description(candidate.description, name)
    source_note = "_Generated from source Markdown by Skill Hub repair._"
    sections = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
        "---",
        "",
        f"# {candidate.name or name}",
        "",
        source_note,
        "",
    ]
    if body:
        sections.extend([body, ""])
    return "\n".join(sections)


def _single_line_description(description: str, name: str) -> str:
    value = " ".join(description.split())
    if not value:
        value = f"Use when the user asks for {name.replace('-', ' ')}."
    value = value.replace('"', "'")
    return value[:240]


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "imported-skill"


def _parse_local_json(path: Path) -> tuple[str, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not any(key in data for key in ("steps", "triggers", "run")):
        return None
    return str(data.get("name") or path.stem), str(data.get("description") or "")


def _candidate(
    path: Path,
    root: Path,
    fmt: str,
    name: str,
    description: str,
    body: str,
    has_frontmatter_description: bool,
) -> SkillAuditCandidate:
    text = f"{name}\n{description}\n{body}".lower()
    issues: list[str] = []

    if fmt == LOOSE_MARKDOWN:
        issues.append("loose Markdown file is not indexed by the current SKILL.md scanner")
    if fmt == REFERENCE_MARKDOWN:
        issues.append("reference file belongs to a parent skill; do not import as a standalone trigger")
    if not has_frontmatter_description:
        issues.append("missing Skill frontmatter description")
    if _SYSTEM_PROMPT_RE.search(body):
        issues.append("system-prompt shaped content should be rewritten as a skill workflow")
    if "wordpress" in text or "logged-in chrome" in text or "real chrome" in text:
        issues.append("private/browser-sensitive workflow should stay L3-gated")
    if "codex" in text and "claude" in text:
        issues.append("Codex-specific routing should not be injected into Codex itself")

    llm_targets, harnesses = _route_targets(text)
    recommendation = _recommendation(fmt, issues, llm_targets)

    return SkillAuditCandidate(
        path=str(path),
        source_root=str(root),
        format=fmt,
        name=name,
        description=description,
        llm_targets=llm_targets,
        harnesses=harnesses,
        recommendation=recommendation,
        issues=issues,
    )


def _route_targets(text: str) -> tuple[list[str], list[str]]:
    if "wordpress" in text or "logged-in chrome" in text or "real chrome" in text:
        return ["L3"], ["claude_code"]
    if "codex" in text and "claude" in text:
        return ["L3"], ["claude_code"]
    if "fastapi" in text or "performance" in text:
        return ["L2", "L3"], ["claude_code", "codex", "chatgpt", "generic_mcp"]
    if "ogc" in text or "stac" in text or "conformance" in text:
        return ["L2", "L3"], ["claude_code", "codex", "chatgpt", "generic_mcp"]
    return ["L3"], ["claude_code", "codex", "chatgpt", "generic_mcp"]


def _recommendation(fmt: str, issues: list[str], llm_targets: list[str]) -> str:
    if not llm_targets:
        return "reject"
    if fmt == REFERENCE_MARKDOWN:
        return "keep_reference"
    if fmt == LOOSE_MARKDOWN:
        return "normalize"
    if any(issue.startswith("missing Skill frontmatter") for issue in issues):
        return "normalize"
    return "import"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_reference_file(path: Path) -> bool:
    return "references" in path.parts


def _without_fenced_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)
