"""Regression tests for the Managed-Agents design doc (M2 tracking issue #14).

The design doc at ``docs/design/managed-agents-refactor.md`` is the load-bearing
artifact for the M2 milestone. Sub-issues (W1-W5) are only filed once the doc's
decision gates are resolved, so the doc's structure is part of the contract.

These tests guard that contract:

* The five workstreams (W1-W5) remain documented.
* The three blocker open questions (Q1, Q4, Q5) remain marked as decided.
* The reference link to Anthropic's Managed Agents post stays present.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = REPO_ROOT / "docs" / "design" / "managed-agents-refactor.md"


def _doc_text() -> str:
    assert DESIGN_DOC.is_file(), f"missing design doc: {DESIGN_DOC}"
    return DESIGN_DOC.read_text(encoding="utf-8")


def test_design_doc_exists() -> None:
    assert DESIGN_DOC.is_file(), (
        f"docs/design/managed-agents-refactor.md must exist — it is the "
        f"reference artifact for M2 tracking issue #14."
    )


def test_design_doc_lists_all_five_workstreams() -> None:
    text = _doc_text()
    # Each workstream gets its own section header in the design doc.
    missing = [
        marker for marker in ("### W1", "### W2", "### W3", "### W4", "### W5")
        if marker not in text
    ]
    assert not missing, (
        f"design doc must document all five workstreams; missing: {missing}"
    )


def test_design_doc_resolves_blocker_open_questions() -> None:
    """Q1, Q4, Q5 are the blockers for filing W1/W4/W5 sub-issues.

    They must remain marked as decided so the milestone gate stays open.
    """
    text = _doc_text()
    # The Resolutions section spells out the decided status for each.
    resolutions = re.search(
        r"##\s+Resolutions(.+?)(?:\n##\s+|\Z)", text, re.DOTALL
    )
    assert resolutions, "design doc must contain a 'Resolutions' section"
    body = resolutions.group(1)

    for q in ("Q1", "Q4", "Q5"):
        assert f"### {q}" in body, f"Resolutions must cover {q}"
        # Look for an explicit 'decided' marker within ~400 chars of the heading
        # so a future edit can't silently flip a gate back to open.
        idx = body.index(f"### {q}")
        window = body[idx : idx + 1200]
        assert "decided" in window.lower(), (
            f"{q} resolution must remain marked 'decided' "
            f"(unblocks W1/W4/W5 sub-issue filing)"
        )


def test_design_doc_references_managed_agents_post() -> None:
    text = _doc_text()
    assert "anthropic.com/engineering/managed-agents" in text, (
        "design doc must keep the canonical reference link to "
        "Anthropic's Managed Agents engineering post."
    )
