from __future__ import annotations

import json

from skill_hub.skill_import_audit import audit_paths, default_source_paths, render_markdown


def test_audit_classifies_skill_markdown_for_shared_l3(tmp_path):
    skill_dir = tmp_path / "ogc-api-standards"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: ogc-api-standards
description: Use when working with OGC API or STAC conformance.
---

# OGC API Standards

Fetch the spec, cite requirement IDs, and verify examples.
""",
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])

    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.format == "skill_md"
    assert candidate.name == "ogc-api-standards"
    assert candidate.description == "Use when working with OGC API or STAC conformance."
    assert candidate.llm_targets == ["L2", "L3"]
    assert candidate.harnesses == ["claude_code", "codex", "chatgpt", "generic_mcp"]
    assert candidate.recommendation == "import"
    assert candidate.issues == []


def test_audit_flags_loose_markdown_for_normalization_and_harness_scope(tmp_path):
    loose = tmp_path / "codex-first.md"
    loose.write_text(
        """---
name: codex-first
description: Route implementation work to Codex CLI; Claude specs and reviews.
---

# Codex First

Claude Code sessions only. Codex/other harnesses: skip; never self-delegate.
""",
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])

    candidate = report.candidates[0]
    assert candidate.format == "loose_markdown"
    assert candidate.name == "codex-first"
    assert candidate.llm_targets == ["L3"]
    assert candidate.harnesses == ["claude_code"]
    assert candidate.recommendation == "normalize"
    assert "loose Markdown file is not indexed by the current SKILL.md scanner" in candidate.issues
    assert "Codex-specific routing should not be injected into Codex itself" in candidate.issues


def test_audit_flags_system_prompt_shaped_markdown(tmp_path):
    loose = tmp_path / "fastapi_optimization_skill.md"
    loose.write_text(
        "SYSTEM INSTRUCTION: FastAPI Concurrency & Performance Optimizer\n"
        "You are an elite backend architect specializing in FastAPI.\n",
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])

    candidate = report.candidates[0]
    assert candidate.format == "loose_markdown"
    assert candidate.name == "fastapi_optimization_skill"
    assert candidate.recommendation == "normalize"
    assert "missing Skill frontmatter description" in candidate.issues
    assert "system-prompt shaped content should be rewritten as a skill workflow" in candidate.issues


def test_loose_markdown_name_ignores_code_block_headings(tmp_path):
    loose = tmp_path / "fastapi_optimization_skill.md"
    loose.write_text(
        "SYSTEM INSTRUCTION: FastAPI Concurrency & Performance Optimizer\n"
        "```python\n"
        "# Set default_response_class=ORJSONResponse globally or per-route\n"
        "```\n",
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])

    assert report.candidates[0].name == "fastapi_optimization_skill"


def test_reference_markdown_is_kept_as_reference_not_imported(tmp_path):
    skill_dir = tmp_path / "ogc-api-standards"
    ref_dir = skill_dir / "references"
    ref_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: ogc-api-standards
description: Use when working with OGC API or STAC conformance.
---

# OGC API Standards
""",
        encoding="utf-8",
    )
    (ref_dir / "records-stac.md").write_text(
        "# OGC API Records & STAC\n\nVerify current status before citing.",
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])
    ref = next(c for c in report.candidates if c.path.endswith("records-stac.md"))

    assert ref.format == "reference_markdown"
    assert ref.recommendation == "keep_reference"
    assert ref.llm_targets == ["L2", "L3"]
    assert "reference file belongs to a parent skill; do not import as a standalone trigger" in ref.issues


def test_audit_classifies_local_json_as_l1_only(tmp_path):
    local_skill = tmp_path / "git-summary.json"
    local_skill.write_text(
        json.dumps(
            {
                "name": "git-summary",
                "description": "Summarize local git state.",
                "triggers": ["git summary"],
                "steps": [{"run": "git status"}],
            }
        ),
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])

    candidate = report.candidates[0]
    assert candidate.format == "local_json"
    assert candidate.name == "git-summary"
    assert candidate.llm_targets == ["L1"]
    assert candidate.harnesses == ["generic_mcp"]
    assert candidate.recommendation == "import"


def test_render_markdown_includes_source_summary(tmp_path):
    skill_dir = tmp_path / "publish-to-my-wordpress"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: publish-to-my-wordpress
description: Use when the user asks to draft an article on WordPress.
---

# Publish

Use logged-in Chrome, keep posts as draft, and never auto-publish.
""",
        encoding="utf-8",
    )

    report = audit_paths([tmp_path])
    markdown = render_markdown(report)

    assert "# Skill Import Audit" in markdown
    assert "Scanned sources: 1" in markdown
    assert "publish-to-my-wordpress" in markdown
    assert "`L3`" in markdown
    assert "private/browser-sensitive workflow should stay L3-gated" in markdown


def test_default_source_paths_uses_multiple_configured_sources(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    disabled = tmp_path / "disabled"
    extra = tmp_path / "extra"

    paths = default_source_paths(
        skill_import_sources=[
            {"path": str(first), "source": "first", "enabled": True},
            {"path": str(disabled), "source": "disabled", "enabled": False},
            {"path": str(second), "source": "second"},
        ],
        extra_skill_dirs=[
            {"path": str(extra), "source": "extra", "enabled": True},
            {"path": str(first), "source": "duplicate", "enabled": True},
        ],
    )

    assert [str(p) for p in paths] == [str(first), str(second), str(extra)]
