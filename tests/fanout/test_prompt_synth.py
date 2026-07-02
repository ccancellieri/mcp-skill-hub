"""Unit tests for skill_hub.fanout.prompt_synth."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.fanout.prompt_synth import (
    build_standing_preamble,
    draft_prompt,
    gather_repo_context,
)
from skill_hub.fanout.sources import Issue


def _write_repo_fixture(repo: Path) -> None:
    tpl = repo / ".github" / "ISSUE_TEMPLATE"
    tpl.mkdir(parents=True)
    (tpl / "bug.yml").write_text(
        "name: Bug report\n"
        "body:\n  - type: textarea\n    id: reproduction\n    "
        "attributes:\n      label: Steps to reproduce\n"
    )
    (tpl / "feature.yml").write_text("name: Feature request\nbody: []\n")
    (repo / ".github" / "labels.yml").write_text(
        "- name: bug\n  color: d73a4a\n- name: feature\n  color: a2eeef\n"
    )


def test_gather_repo_context_reads_templates(tmp_path: Path):
    _write_repo_fixture(tmp_path)
    ctx = gather_repo_context(tmp_path)
    assert "Steps to reproduce" in ctx["bug_template"]
    assert "Feature request" in ctx["feature_template"]
    assert "bug" in ctx["labels"]
    assert ctx["seed_labels"] == ""  # not present in fixture


def test_gather_repo_context_handles_missing_dir(tmp_path: Path):
    ctx = gather_repo_context(tmp_path / "no-such-repo")
    assert all(v == "" for v in ctx.values())


def test_build_standing_preamble_not_indexed(tmp_path: Path):
    preamble = build_standing_preamble(tmp_path / "no-such-repo")
    assert "STANDING DIRECTIVE:" in preamble
    assert "This repo IS NOT codegraph-indexed" in preamble
    assert "fetch_compressed" in preamble


def test_build_standing_preamble_indexed(tmp_path: Path):
    (tmp_path / ".codegraph").mkdir()
    preamble = build_standing_preamble(tmp_path)
    assert "This repo IS codegraph-indexed" in preamble
    assert "codegraph_search" in preamble
    assert "codegraph_callers" in preamble
    assert "codegraph_callees" in preamble
    assert "codegraph_impact" in preamble
    assert "fetch_compressed" in preamble


def test_draft_prompt_fallback_when_llm_disabled(tmp_path: Path):
    _write_repo_fixture(tmp_path)
    issue = Issue(id="gh:1", title="Broken login", body="500 on /login",
                  labels=["bug"], url="https://example/1", source="gh")
    prompt, quality = draft_prompt(issue, tmp_path, use_llm=False,
                                   store_conn=None)
    assert quality == "fallback"
    assert "Broken login" in prompt
    assert "500 on /login" in prompt
    assert "Acceptance" not in prompt  # fallback template doesn't claim LLM structure
    # Standing tooling directive prepended; repo has no .codegraph/ here.
    assert "STANDING DIRECTIVE:" in prompt
    assert "This repo IS NOT codegraph-indexed" in prompt


def test_draft_prompt_fallback_detects_codegraph_indexed_repo(tmp_path: Path):
    _write_repo_fixture(tmp_path)
    (tmp_path / ".codegraph").mkdir()
    issue = Issue(id="gh:1b", title="Broken login", body="500 on /login", source="gh")
    prompt, quality = draft_prompt(issue, tmp_path, use_llm=False, store_conn=None)
    assert quality == "fallback"
    assert "This repo IS codegraph-indexed" in prompt
    assert "codegraph_search" in prompt


def test_draft_prompt_uses_llm_when_available(tmp_path: Path):
    _write_repo_fixture(tmp_path)
    issue = Issue(id="gh:2", title="Pagination", body="cursor needed", source="gh")

    class _FakeProvider:
        def complete(self, prompt, **kwargs):
            # Long enough to pass the >60-char threshold in draft_prompt
            return (
                "Scope: Add cursor pagination to /items.\n"
                "Acceptance: limit param honored; cursor stable.\n"
                "Files: api/items.py, tests/test_items.py"
            )

    with patch("skill_hub.llm.get_provider", return_value=_FakeProvider()):
        prompt, quality = draft_prompt(issue, tmp_path, use_llm=True,
                                       store_conn=None)
    assert quality == "llm"
    assert "Scope: Add cursor pagination" in prompt
    # The standing directive is prepended by code, independent of whether the
    # LLM itself followed the instruction to include it.
    assert "STANDING DIRECTIVE:" in prompt
    assert "This repo IS NOT codegraph-indexed" in prompt


def test_draft_prompt_cache_roundtrip(tmp_path: Path):
    _write_repo_fixture(tmp_path)
    conn = sqlite3.connect(":memory:")
    issue = Issue(id="gh:3", title="A", body="B", url="https://e/3", source="gh")

    class _FakeProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            return "Cached LLM output that is well over sixty characters long for sure."

    fake = _FakeProvider()
    with patch("skill_hub.llm.get_provider", return_value=fake):
        p1, q1 = draft_prompt(issue, tmp_path, store_conn=conn, use_llm=True)
        p2, q2 = draft_prompt(issue, tmp_path, store_conn=conn, use_llm=True)
    assert q1 == "llm" and q2 == "llm"
    assert p1 == p2
    assert fake.calls == 1  # second call hit the cache
