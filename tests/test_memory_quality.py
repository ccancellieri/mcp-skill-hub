"""Unit tests for memory_quality's source-based scoring and budget rules."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "memory_quality.py"
SPEC = importlib.util.spec_from_file_location("memory_quality", SCRIPT)
assert SPEC and SPEC.loader
memory_quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_quality)


class CharacterTokenizer:
    """Deterministic tokenizer double: one token per character."""

    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def test_chunker_preserves_text_and_overlaps_long_sections():
    text = "A" * 40 + "B" * 40 + "C" * 40
    chunks = memory_quality._chunk_text(text, size=50, overlap=10)

    assert "".join(chunks[0][:40]) == "A" * 40
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert any(chunks[i][-10:] == chunks[i + 1][:10]
               for i in range(len(chunks) - 1))
    assert all(char in "ABC" for chunk in chunks for char in chunk)


def test_budget_hits_obeys_exact_limit_and_retains_provenance():
    tokenizer = CharacterTokenizer()
    hits = [{"id": "source-a", "title": "A", "text": "x" * 20,
             "source_refs": ["docs/a.md"]},
            {"id": "source-b", "title": "B", "text": "y" * 40,
             "source_refs": ["docs/b.md"]}]

    selected, used = memory_quality._budget_hits(hits, tokenizer, budget=65)

    assert used == 65
    assert used <= 65
    assert selected[0]["source_refs"] == ["docs/a.md"]
    assert selected[1]["truncated_for_budget"] is True
    assert all(item["context_tokens"] <= 65 for item in selected)


def test_fact_scoring_requires_source_and_quote_in_selected_context():
    question = {"answerable": True, "required_facts": [{
        "fact": "a", "source_file": "docs/a.md", "source_quote": "exact passage",
    }, {
        "fact": "b", "source_file": "docs/b.md", "source_quote": "missing passage",
    }]}
    selected = [{"source_file": "docs/a.md", "source_refs": ["docs/a.md"],
                 "text": "exact passage", "context_text": "docs/a.md\nexact passage"},
                {"source_file": None, "source_refs": ["docs/b.md"],
                 "text": "paraphrase only", "context_text": "docs/b.md\nparaphrase only"}]

    result = memory_quality._fact_scores(question, selected)

    assert result["source_coverage_recall"] == 0.5
    assert result["facts"][0]["covered"] is True
    assert result["facts"][1]["provenance_only_unknown"] is True
    assert result["abstention_response_assessment"] == "not_evaluated_no_answer_generation"


def test_unanswerable_questions_measure_supporting_evidence_without_noise_penalty():
    question = {"answerable": False, "required_facts": [], "abstention_evidence": [{
        "fact": "unknown", "source_file": "docs/a.md", "source_quote": "availability is unknown",
    }]}
    selected = [{"source_file": "docs/a.md", "source_refs": ["docs/a.md"],
                 "text": "availability is unknown", "context_text": "docs/a.md\navailability is unknown"},
                {"source_file": "docs/other.md", "source_refs": ["docs/other.md"],
                 "text": "related limits", "context_text": "related limits"}]

    result = memory_quality._fact_scores(question, selected)

    assert result["source_coverage_recall"] is None
    assert result["abstention_evidence_source_coverage"] == 1.0
    assert result["abstention_response_assessment"] == "not_evaluated_no_answer_generation"


def test_quote_coverage_cannot_pair_different_hits_or_hidden_provenance():
    question = {"required_facts": [{"fact": "a", "source_file": "docs/a.md",
                                    "source_quote": "exact passage"}]}
    selected = [{"source_file": "docs/a.md", "source_refs": [],
                 "text": "unrelated", "context_text": "docs/a.md\nunrelated"},
                {"source_file": "docs/b.md", "source_refs": [],
                 "text": "exact passage", "context_text": "docs/b.md\nexact passage"}]
    assert not memory_quality._fact_scores(question, selected)["facts"][0]["covered"]
    hidden = [{"source_file": None, "source_refs": ["docs/a.md"],
               "text": "exact passage", "context_text": "exact passage"}]
    fact = memory_quality._fact_scores(question, hidden)["facts"][0]
    assert not fact["source_present"] and not fact["covered"]


def test_machine_paths_are_redacted_from_saved_strings():
    assert memory_quality._sanitize("source at /Users/alice/work/repo/file.md") == (
        "source at <LOCAL_PATH>")


def test_nonempty_work_directory_is_rejected_without_changing_it(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    keep = work / "existing.sqlite3"
    keep.write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="choose a fresh --output-dir"):
        memory_quality._require_empty_workdir(work)

    assert keep.read_bytes() == b"preserve"


def test_empty_or_missing_work_directory_is_allowed(tmp_path):
    memory_quality._require_empty_workdir(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    memory_quality._require_empty_workdir(empty)


def test_relationship_arm_uses_three_seeds_and_caps_unique_results(tmp_path):
    from skill_hub.wiki import WikiPage, page_path, render_page

    root = tmp_path / "wiki"
    root.mkdir()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE wiki_edges (src_slug TEXT, dst_slug TEXT, resolved INTEGER)")
    conn.execute("CREATE TABLE wiki_pages (slug TEXT, rel_path TEXT)")
    pages = {}
    for slug in ("seed-0", "seed-1", "seed-2", "ignored-seed", "neighbor-a", "neighbor-b"):
        page = WikiPage(id=slug, slug=slug, title=slug, type="concept",
                        projects=["_global"], scope="public", body=f"Body {slug}")
        path = page_path(root, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_page(page), encoding="utf-8")
        pages[slug] = path.relative_to(root).as_posix()
        conn.execute("INSERT INTO wiki_pages VALUES (?,?)", (slug, pages[slug]))
    conn.executemany("INSERT INTO wiki_edges VALUES (?,?,1)", [
        ("seed-0", "neighbor-a"), ("seed-1", "neighbor-a"),
        ("seed-2", "neighbor-b"), ("ignored-seed", "ignored-seed"),
    ])
    store = SimpleNamespace(_conn=conn, _benchmark_wiki_root=root)
    seeds = [{"wiki_slug": slug, "id": slug, "score": 1.0, "source_refs": [],
              "title": slug, "text": slug}
             for slug in ("seed-0", "seed-1", "seed-2", "ignored-seed")]

    hits = memory_quality._wiki_one_hop(store, seeds, top_k=4)

    assert len(hits) <= 4
    assert [hit["wiki_slug"] for hit in hits] == [
        "seed-0", "seed-1", "seed-2", "neighbor-a"]
    assert len({hit["wiki_slug"] for hit in hits}) == len(hits)
