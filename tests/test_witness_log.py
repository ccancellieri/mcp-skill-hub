"""Tests for the witness-log (M1 / issue #10).

Covers acceptance criteria:
- Append-only enforced (edits raise).
- Filter by repo + since works.
- JSONL parseable by stdlib.
"""
from __future__ import annotations

import json

import pytest

from skill_hub.witness import (
    AppendOnlyError,
    WITNESS_KIND,
    WitnessRecord,
    _iter_records,
    edit_witness,
    format_witness_list,
    list_witness,
    record_witness,
)


@pytest.fixture
def witness_file(tmp_path):
    return tmp_path / "state" / "witness_log.jsonl"


def test_record_witness_appends_jsonl(witness_file):
    rec = record_witness(
        issue="#10",
        pr="#42",
        sha="abc1234",
        repo="ccancellieri/mcp-skill-hub",
        fix_kind="feat",
        fix_summary="add witness log",
        witness_file=witness_file,
        now=1_715_000_000,
    )
    assert isinstance(rec, WitnessRecord)
    assert witness_file.exists()
    lines = witness_file.read_text().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj == {
        "kind": "fix",
        "at": 1_715_000_000,
        "issue": "#10",
        "pr": "#42",
        "sha": "abc1234",
        "repo": "ccancellieri/mcp-skill-hub",
        "fix_kind": "feat",
        "fix_summary": "add witness log",
    }


def test_record_witness_requires_repo(witness_file):
    with pytest.raises(ValueError):
        record_witness(
            issue="#1", pr="", sha="x", repo="",
            witness_file=witness_file,
        )


def test_record_witness_strips_whitespace(witness_file):
    rec = record_witness(
        issue="  #1  ", pr=" #2 ", sha=" abc ",
        repo="  owner/repo  ",
        fix_kind="  fix  ",
        fix_summary="  summary  ",
        witness_file=witness_file,
        now=1,
    )
    assert rec.issue == "#1"
    assert rec.pr == "#2"
    assert rec.sha == "abc"
    assert rec.repo == "owner/repo"
    assert rec.fix_kind == "fix"
    assert rec.fix_summary == "summary"


def test_record_witness_defaults_fix_kind(witness_file):
    rec = record_witness(
        issue="#1", pr="", sha="x", repo="owner/repo",
        fix_kind="",
        witness_file=witness_file,
        now=1,
    )
    assert rec.fix_kind == "fix"


def test_append_only_preserves_history(witness_file):
    record_witness(
        issue="#1", pr="", sha="a", repo="r/r",
        witness_file=witness_file, now=1,
    )
    record_witness(
        issue="#2", pr="", sha="b", repo="r/r",
        witness_file=witness_file, now=2,
    )
    record_witness(
        issue="#3", pr="", sha="c", repo="r/r",
        witness_file=witness_file, now=3,
    )
    lines = witness_file.read_text().splitlines()
    assert len(lines) == 3
    objs = [json.loads(line) for line in lines]
    assert [o["issue"] for o in objs] == ["#1", "#2", "#3"]
    assert [o["at"] for o in objs] == [1, 2, 3]


def test_edit_witness_raises():
    with pytest.raises(AppendOnlyError):
        edit_witness("#1", new_summary="oops")


def test_jsonl_parseable_by_stdlib(witness_file):
    record_witness(
        issue="#1", pr="#2", sha="abc", repo="r/r",
        fix_kind="feat", fix_summary="line one",
        witness_file=witness_file, now=10,
    )
    record_witness(
        issue="#3", pr="#4", sha="def", repo="r/r",
        fix_kind="fix", fix_summary="line two",
        witness_file=witness_file, now=20,
    )
    parsed = [
        json.loads(line)
        for line in witness_file.read_text().splitlines()
        if line.strip()
    ]
    assert len(parsed) == 2
    assert all(p["kind"] == WITNESS_KIND for p in parsed)


def test_list_witness_newest_first(witness_file):
    record_witness(issue="#a", pr="", sha="x", repo="r/r", witness_file=witness_file, now=1)
    record_witness(issue="#b", pr="", sha="x", repo="r/r", witness_file=witness_file, now=3)
    record_witness(issue="#c", pr="", sha="x", repo="r/r", witness_file=witness_file, now=2)
    out = list_witness(witness_file=witness_file)
    assert [r["issue"] for r in out] == ["#b", "#c", "#a"]


def test_list_witness_filter_by_repo(witness_file):
    record_witness(issue="#1", pr="", sha="x", repo="alpha/one", witness_file=witness_file, now=1)
    record_witness(issue="#2", pr="", sha="x", repo="beta/two", witness_file=witness_file, now=2)
    record_witness(issue="#3", pr="", sha="x", repo="alpha/one", witness_file=witness_file, now=3)
    out = list_witness(repo="alpha/one", witness_file=witness_file)
    assert [r["issue"] for r in out] == ["#3", "#1"]
    assert all(r["repo"] == "alpha/one" for r in out)


def test_list_witness_filter_since(witness_file):
    record_witness(issue="#1", pr="", sha="x", repo="r/r", witness_file=witness_file, now=100)
    record_witness(issue="#2", pr="", sha="x", repo="r/r", witness_file=witness_file, now=200)
    record_witness(issue="#3", pr="", sha="x", repo="r/r", witness_file=witness_file, now=300)
    out = list_witness(since=200, witness_file=witness_file)
    assert [r["issue"] for r in out] == ["#3", "#2"]


def test_list_witness_combined_filter(witness_file):
    record_witness(issue="#1", pr="", sha="x", repo="alpha/one", witness_file=witness_file, now=100)
    record_witness(issue="#2", pr="", sha="x", repo="beta/two", witness_file=witness_file, now=200)
    record_witness(issue="#3", pr="", sha="x", repo="alpha/one", witness_file=witness_file, now=300)
    record_witness(issue="#4", pr="", sha="x", repo="alpha/one", witness_file=witness_file, now=50)
    out = list_witness(repo="alpha/one", since=100, witness_file=witness_file)
    assert [r["issue"] for r in out] == ["#3", "#1"]


def test_list_witness_limit(witness_file):
    for i in range(5):
        record_witness(issue=f"#{i}", pr="", sha="x", repo="r/r", witness_file=witness_file, now=i)
    out = list_witness(limit=2, witness_file=witness_file)
    assert len(out) == 2
    assert [r["issue"] for r in out] == ["#4", "#3"]


def test_list_witness_missing_file(tmp_path):
    out = list_witness(witness_file=tmp_path / "missing.jsonl")
    assert out == []


def test_list_witness_skips_non_fix_kinds(witness_file):
    record_witness(issue="#1", pr="", sha="x", repo="r/r", witness_file=witness_file, now=1)
    witness_file.parent.mkdir(parents=True, exist_ok=True)
    with witness_file.open("a") as fh:
        fh.write(json.dumps({"kind": "lint_canary", "at": 2, "selector": "E"}) + "\n")
    out = list_witness(witness_file=witness_file)
    assert len(out) == 1
    assert out[0]["issue"] == "#1"


def test_iter_records_tolerates_garbage(witness_file, tmp_path):
    witness_file.parent.mkdir(parents=True, exist_ok=True)
    witness_file.write_text(
        '{"kind":"fix","at":1,"repo":"r/r","issue":"#1"}\n'
        'not-json\n'
        '\n'
        '"a-string-not-an-object"\n'
        '{"kind":"fix","at":2,"repo":"r/r","issue":"#2"}\n'
    )
    parsed = list(_iter_records(witness_file))
    assert len(parsed) == 2
    assert [p["issue"] for p in parsed] == ["#1", "#2"]


def test_format_witness_list_empty():
    assert "no records" in format_witness_list([])


def test_format_witness_list_renders_rows(witness_file):
    record_witness(
        issue="#1", pr="#2", sha="abc", repo="r/r",
        fix_kind="feat", fix_summary="add it",
        witness_file=witness_file, now=42,
    )
    out = format_witness_list(list_witness(witness_file=witness_file))
    assert "1 record" in out
    assert "r/r" in out
    assert "feat" in out
    assert "#1" in out
    assert "add it" in out


def test_format_witness_list_omits_empty_summary(witness_file):
    record_witness(
        issue="#1", pr="", sha="x", repo="r/r",
        witness_file=witness_file, now=1,
    )
    out = format_witness_list(list_witness(witness_file=witness_file))
    assert " — " not in out
