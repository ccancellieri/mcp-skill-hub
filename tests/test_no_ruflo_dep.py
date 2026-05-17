"""Regression tests for the no-ruflo-runtime-dependency invariant.

Mirrors the two grep gates documented in ``docs/comparison-ruflo.md`` and
enforced by ``.github/workflows/no-ruflo-dep.yml``:

* ``pyproject.toml`` must not depend on ``claude-flow`` or ``ruflo``.
* ``src/`` must not import the ``claude_flow`` package.

Keeping these as Python tests means a contributor catches the regression
locally before CI does.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_does_not_depend_on_ruflo() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    offenders = re.findall(r"(?i)claude-flow|ruflo", content)
    assert not offenders, (
        f"pyproject.toml must not reference claude-flow or ruflo "
        f"(found: {offenders}). See docs/comparison-ruflo.md "
        f"§ 'No-ruflo runtime dependency'."
    )


def test_src_does_not_import_claude_flow() -> None:
    src = REPO_ROOT / "src"
    if not src.is_dir():
        return
    pattern = re.compile(r"^\s*(?:import\s+claude_flow|from\s+claude_flow)\b", re.MULTILINE)
    hits: list[str] = []
    for path in src.rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert not hits, (
        f"src/ must not import claude_flow (offenders: {hits}). "
        f"See docs/comparison-ruflo.md § 'No-ruflo runtime dependency'."
    )
