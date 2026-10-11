"""Frozen public foreground qualification; no models, live DB, or config writes.

Capture baseline before edits; inspect calibration only while implementing.
Compare the final candidate once against that immutable baseline report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_compare import validate_corpus
from local_selector import fixture_store
from skill_hub import context_service as context

CORPUS = Path(__file__).with_name("foreground_relevance_cases.json")
CORPUS_SHA256 = "7aa0bf0e77811ec5f8db059244548bec41d9e18d5b1fe8cf432e43256a7cbdb9"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_corpus():
    if sha256(CORPUS) != CORPUS_SHA256:
        raise ValueError("Frozen corpus changed; use a separately preregistered version")
    corpus = json.loads(CORPUS.read_text())
    validate_corpus(corpus)
    return corpus


def summarize(rows):
    positive = [r for r in rows if r["expected"]]
    negative = [r for r in rows if not r["expected"]]
    explicit = [r for r in rows if r["category"] == "explicit"]
    timings = sorted(r["elapsed_ms"] for r in rows)
    selected_count = sum(len(r["selected"]) for r in rows)
    hits = sum(len(set(r["expected"]) & set(r["selected"])) for r in rows)
    return {
        "cases": len(rows),
        "recall": statistics.mean(r["recall"] for r in positive) if positive else None,
        "micro_precision": hits / selected_count if selected_count else None,
        "shortlist_recall": statistics.mean(r["shortlist_recall"] for r in positive) if positive else None,
        "maximum_shortlist_size": max((len(r["shortlist"]) for r in rows), default=0),
        "packet_chars": sum(r["packet_chars"] for r in rows),
        "packet_utf8_bytes": sum(r["packet_utf8_bytes"] for r in rows),
        "irrelevant_injections": sum(len(set(r["selected"]) - set(r["expected"])) for r in rows),
        "correct_abstention_rate": sum(not r["selected"] and not r["error"] for r in negative) / len(negative) if negative else None,
        "explicit_identifier_recall": statistics.mean(r["recall"] for r in explicit) if explicit else None,
        "failures": sum(bool(r["error"]) for r in rows),
        "original_prompt_preserved": all(r["original_prompt_preserved"] for r in rows),
        "p95_ms": timings[max(0, (len(timings) * 95 + 99) // 100 - 1)] if timings else None,
        "max_ms": max(timings) if timings else None,
    }


def run(corpus, context_module=context, split=None):
    rows = []
    # fixture_store would otherwise derive a node ID from the user's configuration.
    with tempfile.TemporaryDirectory(prefix="foreground-relevance-") as directory:
        with patch("skill_hub.store._resolve_node_id", return_value="fixture"):
            store = fixture_store(corpus, Path(directory) / "fixture.db")
        try:
            for case in corpus["cases"]:
                if split and case["split"] != split:
                    continue
                started = time.perf_counter()
                result, error = None, None
                try:
                    # A network attempt is a failing case, never a successful abstention.
                    with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
                        result = context_module.build_context(
                            case["prompt"], cwd="/fixture/repo", store=store,
                            cfg={"context_max_items": 6, "context_max_chars": 6000,
                                 "memory_retrieval_backend": "raw"},
                        )
                    elapsed = (time.perf_counter() - started) * 1000
                    with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
                        candidates, _ = context_module.collect_context_candidates(
                            case["prompt"], cwd="/fixture/repo", store=store,
                            cfg={"context_max_items": 6, "context_max_chars": 6000,
                                 "memory_retrieval_backend": "raw"}, max_skill_items=6,
                        )
                    if any("unavailable" in warning.lower() for warning in result["warnings"]):
                        error = "Context retrieval reported an unavailable source"
                except Exception as exc:
                    error = type(exc).__name__ + ": " + str(exc)
                    elapsed = (time.perf_counter() - started) * 1000
                    candidates = []
                selected = [item["source"] for item in result["items"] if item["kind"] == "skill"] if result else []
                expected = case["expected_sources"]
                shortlist = [item["source"] for item in candidates if item["kind"] == "skill"]
                packet = case["prompt"] + (result["context"] if result else "")
                rows.append({
                    "id": case["id"], "split": case["split"], "language": case["language"],
                    "category": case["category"], "prompt": case["prompt"],
                    "expected": expected, "selected": selected, "error": error,
                    "shortlist": shortlist,
                    "shortlist_recall": len(set(expected) & set(shortlist)) / len(expected) if expected else None,
                    "packet_chars": len(packet), "packet_utf8_bytes": len(packet.encode("utf-8")),
                    "recall": len(set(expected) & set(selected)) / len(expected) if expected else None,
                    "elapsed_ms": elapsed,
                    "original_prompt_preserved": bool(result and result["original_prompt"] == case["prompt"]),
                    "context": result["context"] if result else "",
                })
        finally:
            store.close()
    return {
        "corpus_sha256": CORPUS_SHA256,
        "harness_sha256": sha256(__file__),
        "source_sha256": {"context_service.py": sha256(context_module.__file__),
                          "skill_retrieval.py": sha256(Path(context.__file__).with_name("skill_retrieval.py"))},
        "acceptance": corpus["acceptance"],
        "production_configuration_changed": False, "model_calls": 0,
        "interpretation": "Constructed public source-selection qualification; no whole-task or production-wide claim. Timings cover build_context retrieval, not process startup or an enforced hook deadline. Character/UTF-8 packet sizes are not native token counts. The 100-row FTS candidate ceiling is distinct from the six-item rendered limit. Fixture labels were seen during design, frozen before source edits; no claim of independent sealed evaluation.",
        "summary": {split: {language: summarize([r for r in rows if r["split"] == split and r["language"] == language])
                            for language in ("en", "it")} for split in ("calibration", "holdout")},
        "rows": rows,
    }


def compare(candidate, baseline):
    if candidate["corpus_sha256"] != baseline["corpus_sha256"]:
        raise ValueError("Baseline and candidate must use identical frozen corpora")
    if candidate["harness_sha256"] != baseline["harness_sha256"]:
        raise ValueError("Baseline and candidate must use identical measurement harnesses")
    gates = candidate["acceptance"]
    checks = {}
    for language in ("en", "it"):
        new = candidate["summary"]["holdout"][language]
        old = baseline["summary"]["holdout"][language]
        checks[language] = {
            "minimum_recall": new["recall"] >= gates["minimum_recall"],
            "recall_loss": old["recall"] - new["recall"] <= gates["maximum_recall_loss"],
            "fewer_irrelevant_injections": new["irrelevant_injections"] < old["irrelevant_injections"],
            "abstention": new["correct_abstention_rate"] >= gates["minimum_abstention"],
            "explicit_identifiers": new["explicit_identifier_recall"] == 1.0,
            "deadline": new["max_ms"] < gates["case_deadline_ms"],
            "no_failures": new["failures"] == 0,
            "original_preserved": new["original_prompt_preserved"],
        }
    return {"passed": all(all(values.values()) for values in checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--split", choices=["calibration", "holdout"])
    parser.add_argument("--source-path", type=Path,
                        help="Read-only original context-service snapshot for baseline provenance")
    args = parser.parse_args()
    module = context
    if args.source_path:
        spec = importlib.util.spec_from_file_location("skill_hub._foreground_baseline", args.source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    result = run(load_corpus(), module, args.split)
    if args.baseline:
        result["comparison"] = compare(result, json.loads(args.baseline.read_text()))
        result["baseline_report_sha256"] = sha256(args.baseline)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"corpus_sha256": result["corpus_sha256"],
                      "calibration": result["summary"]["calibration"],
                      "comparison": result.get("comparison")}, indent=2))


if __name__ == "__main__":
    main()
