"""Frozen bilingual qualification must expose regressions and retrieval failures."""
import importlib.util
from pathlib import Path
import re

import pytest


def bench():
    path = Path(__file__).resolve().parents[1] / "benchmarks/foreground_relevance.py"
    spec = importlib.util.spec_from_file_location("foreground_relevance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_frozen_and_balanced_without_trivial_bilingual_metadata():
    module = bench()
    corpus = module.load_corpus()
    assert len(corpus["cases"]) == 48
    assert len({case["id"] for case in corpus["cases"]}) == 48
    for split in ("calibration", "holdout"):
        for language in ("en", "it"):
            cases = [case for case in corpus["cases"] if case["split"] == split and case["language"] == language]
            assert len(cases) == 12
            assert sum(not c["expected_sources"] for c in cases) == 6
            assert sum(len(c["expected_sources"]) == 1 for c in cases) == 4
            assert sum(len(c["expected_sources"]) > 1 for c in cases) == 2
    assert all(not re.search(r"\b(crea|foglio|diapositive|immagine)\b", row["description"], re.I)
               for row in corpus["skills"])


def test_failed_retrieval_is_not_credited_as_correct_abstention():
    module = bench()
    summary = module.summarize([{
        "expected": [], "selected": [], "category": "followup", "error": "unavailable",
        "elapsed_ms": 1, "original_prompt_preserved": True,
        "shortlist": [], "shortlist_recall": None, "packet_chars": 1, "packet_utf8_bytes": 1,
    }])
    assert summary["correct_abstention_rate"] == 0
    assert summary["failures"] == 1


def report(recall=1, irrelevant=1, abstention=1):
    summary = {"recall": recall, "irrelevant_injections": irrelevant,
               "correct_abstention_rate": abstention, "explicit_identifier_recall": 1,
               "max_ms": 1, "failures": 0, "original_prompt_preserved": True}
    return {"corpus_sha256": "same-corpus", "harness_sha256": "same-harness",
            "acceptance": {"minimum_recall": .9, "maximum_recall_loss": .05,
                           "minimum_abstention": .9, "case_deadline_ms": 2000},
            "summary": {"holdout": {"en": dict(summary), "it": dict(summary)}}}


def test_gate_rejects_language_regressions_even_with_less_noise():
    module = bench()
    baseline = report(irrelevant=5)
    candidate = report(irrelevant=0)
    assert module.compare(candidate, baseline)["passed"]
    candidate["summary"]["holdout"]["it"]["recall"] = .91
    assert not module.compare(candidate, baseline)["passed"]
    assert not module.compare(candidate, baseline)["checks"]["it"]["recall_loss"]


@pytest.mark.parametrize("field,value", [
    ("irrelevant_injections", 5), ("correct_abstention_rate", .5),
    ("explicit_identifier_recall", .5), ("max_ms", 2000),
    ("failures", 1), ("original_prompt_preserved", False),
])
def test_gate_requires_every_fixed_condition(field, value):
    module = bench()
    candidate = report(irrelevant=0)
    candidate["summary"]["holdout"]["en"][field] = value
    assert not module.compare(candidate, report(irrelevant=5))["passed"]


def test_gate_refuses_changed_measurement_harness():
    module = bench()
    candidate = report()
    candidate["harness_sha256"] = "changed"
    with pytest.raises(ValueError, match="identical measurement"):
        module.compare(candidate, report())
