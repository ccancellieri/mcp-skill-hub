"""Reproducible, offline context-value evaluation.

The corpus is synthetic and public-shaped.  It never reads a user's live Skill Hub
database, project memories, or source repositories.  Text token counts in this
module are estimates; native end-to-end usage is recorded only by the optional
client runner in ``context_value_e2e.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from skill_hub.context_service import build_context
from skill_hub.store import Skill, SkillStore

HARNESS_VERSION = "context-value-v1"
FIXTURE_REVISION = "synthetic-public-shaped-v1"
SCOPES = ("/synthetic/skill-hub", "/synthetic/tellurion")
TOPICS = (
    ("sqlite_wal", "SQLite WAL checkpoint recovery", "retain the last durable checkpoint before reopening"),
    ("hook_deadline", "prompt hook deadline budgeting", "return the original prompt when enrichment exceeds the deadline"),
    ("provider_pin", "provider selection pinning", "preserve an explicit provider selection across capability refresh"),
    ("fts_escape", "FTS query escaping", "quote punctuation before constructing the text-search expression"),
    ("session_scope", "session identity scope", "reject evidence whose verified project root does not match"),
    ("digest_refresh", "context digest refresh", "replace a digest only when its content hash changes"),
    ("native_usage", "native token usage parsing", "count cached input separately without adding it twice"),
    ("approval_policy", "native approval policy", "leave permission decisions to the client unless a deterministic rule exists"),
    ("vector_dimension", "vector dimension migration", "rebuild the index before accepting vectors with a new dimension"),
    ("plugin_manifest", "plugin manifest validation", "reject components outside the declared plugin root"),
    ("tile_cache", "tile cache identity", "include source revision style filter and access scope in the cache key"),
    ("keyset_page", "stable feature pagination", "use a stable identifier as the final keyset tie breaker"),
    ("crs_axis", "CRS axis order handling", "apply the advertised wire CRS axis order at the protocol boundary"),
    ("range_read", "bounded range reads", "cap decoded bytes and validate redirects before remote reads"),
    ("job_retry", "durable ingestion retry", "bind cleanup and retry to an immutable resource incarnation"),
    ("style_revision", "render style revision", "invalidate rendered tiles when the referenced style revision changes"),
    ("stream_cancel", "stream cancellation", "propagate cancellation before scheduling another bounded batch"),
    ("feature_projection", "feature property projection", "preserve stable feature identifiers while selecting requested properties"),
    ("temporal_extent", "temporal extent metadata", "represent an open interval explicitly instead of inventing a timestamp"),
    ("driver_capability", "driver capability refusal", "return an explicit unsupported-operation result instead of an unbounded scan"),
    ("task_pause", "paused task retrieval", "include a paused task only when its exact task identity is supplied"),
    ("memory_supersede", "superseded memory filtering", "exclude records that name a superseding memory entry"),
    ("wiki_scope", "wiki project scope", "match a canonical project label before returning a wiki digest"),
    ("prompt_preserve", "original prompt preservation", "append evidence without rewriting the user's original request"),
    ("cache_prefix", "stable cache prefix", "keep deterministic instructions before request-specific material"),
    ("model_fallback", "local model fallback", "report the unavailable provider before selecting a configured fallback"),
    ("event_order", "event ordering", "sort equal timestamps by the monotonic event sequence"),
    ("schema_expand", "additive schema rollout", "deploy readers before enforcing the newly added field"),
    ("secret_redact", "secret redaction", "remove credential values while retaining the diagnostic field name"),
    ("overload_bound", "bounded overload behavior", "reject excess work before allocating an unbounded request buffer"),
)

PROMPT_FORMS = (
    "Implement {title}; which invariant must the change preserve?",
    "Review a regression in {title}. Cite the required behavior.",
    "Quale vincolo va mantenuto per {title}?",
    "Prepare a focused test for {title} and recover the governing decision.",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def estimate_text_tokens(text: str) -> int:
    """Documented approximation, deliberately not presented as model billing."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def build_cases() -> list[dict[str, Any]]:
    cases = []
    for topic_index, (slug, title, rule) in enumerate(TOPICS):
        scope = SCOPES[topic_index % 2]
        for form_index, form in enumerate(PROMPT_FORMS):
            number = topic_index * len(PROMPT_FORMS) + form_index
            cases.append({
                "id": f"retrieval-{number + 1:03d}-{slug}",
                "split": "calibration" if number < 40 else "holdout",
                "language": "it" if form_index == 2 else "en",
                "scope": scope,
                "prompt": form.format(title=title),
                "topic": slug,
                "expected_markers": [f"FACT_{slug.upper()}", f"SKILL_{slug.upper()}"],
                "forbidden_markers": [f"FOREIGN_{slug.upper()}"],
                "reference_rule": rule,
            })
    return cases


def build_memory_questions() -> list[dict[str, Any]]:
    rows = []
    kinds = ("updated", "conflicting", "unanswerable")
    for index in range(24):
        kind = kinds[index // 8]
        language = "en" if index % 2 == 0 else "it"
        scope = SCOPES[index % 2]
        key = f"qa_{kind}_{index + 1:02d}"
        if kind == "unanswerable":
            prompt = ("Lookup " if language == "en" else "Cerca ") + key + "."
            expected = []
        else:
            prompt = ("What is the current decision for " if language == "en" else
                      "Qual è la decisione corrente per ") + key + "?"
            expected = [f"QA_CURRENT_{index + 1:02d}"]
        rows.append({"id": f"memory-{index + 1:02d}", "category": kind,
                     "language": language, "scope": scope, "prompt": prompt,
                     "topic": key, "expected_markers": expected,
                     "forbidden_markers": [f"QA_FOREIGN_{index + 1:02d}", f"QA_STALE_{index + 1:02d}"]})
    return rows


def seed_store(path: Path, cases: list[dict], questions: list[dict]) -> SkillStore:
    store = SkillStore(db_path=path)
    for index, (slug, title, rule) in enumerate(TOPICS):
        scope = SCOPES[index % 2]
        store.upsert_skill(Skill(
            id=f"fixture:{slug}", name=title,
            description=f"{title}. SKILL_{slug.upper()}: {rule}.",
            content=f"Synthetic benchmark skill for {title}.",
            file_path=f"/synthetic/skills/{slug}/SKILL.md", plugin="context-value-fixture",
        ))
        _insert_memory(store, scope, slug, f"FACT_{slug.upper()}: {rule}.")
        foreign = SCOPES[1] if scope == SCOPES[0] else SCOPES[0]
        _insert_memory(store, foreign, f"foreign_{slug}",
                       f"FOREIGN_{slug.upper()}: unrelated private-project decoy.")
    for index, question in enumerate(questions):
        if question["category"] == "unanswerable":
            continue
        current = f"QA_CURRENT_{index + 1:02d}: approved value revision {index + 3}."
        _insert_memory(store, question["scope"], question["topic"], current)
        if question["category"] == "conflicting":
            _insert_memory(store, question["scope"], f"stale_{question['topic']}",
                           f"QA_STALE_{index + 1:02d}: obsolete value revision {index + 1}.",
                           metadata={"superseded_by": question["topic"]})
        foreign = SCOPES[1] if question["scope"] == SCOPES[0] else SCOPES[0]
        _insert_memory(store, foreign, f"foreign_{question['topic']}",
                       f"QA_FOREIGN_{index + 1:02d}: other-project answer.")
    return store


def _insert_memory(store: SkillStore, project: str, doc_id: str, text: str,
                   metadata: dict | None = None) -> None:
    store._conn.execute(
        "INSERT INTO vectors (namespace, doc_id, vector, norm, metadata, level, source, project) "
        "VALUES ('memory:project', ?, '[]', 0, ?, 'L3', ?, ?)",
        (doc_id, json.dumps(metadata or {}), f"memory:{doc_id}", project),
    )
    store._conn.execute(
        "INSERT INTO context_digests (key, content_hash, digest, content, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (f"memory:{doc_id}", hashlib.sha256(text.encode()).hexdigest(), text, text),
    )
    store._conn.commit()


def _flatten_candidate_text(candidate: dict) -> str:
    return " ".join(str(candidate.get(key, "")) for key in
                    ("title", "text", "content", "digest", "excerpt", "source"))


def _composer_context(prompt: str, scope: str, store: SkillStore) -> dict:
    try:
        from skill_hub.context_composer import compose_context, prepare_composition
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc), "context": "", "items": []}
    draft = prepare_composition(prompt, project_roots=[scope], token_budget=1500,
                                mode="manual", store=store)
    candidates = draft.get("candidates", [])
    # Evaluate the manual composer's deterministic preselection without using
    # gold labels to choose evidence or recording training feedback.
    selected = [str(value) for value in draft.get("selected_ids", [])]
    composed = compose_context(draft["draft_id"], selected_ids=selected,
                               rejected_ids=[], store=store, confirmed=False)
    return {**composed, "status": "complete", "selected_ids": selected,
            "candidate_count": len(candidates),
            "candidate_token_estimate": sum(int(item.get("estimated_tokens", 0)) for item in candidates)}


def _run_condition(condition: str, case: dict, store: SkillStore) -> dict:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    if condition == "A_no_hub":
        result = {"context": "", "items": [], "status": "complete"}
    elif condition == "B_build_context":
        result = build_context(case["prompt"], cwd=case["scope"], store=store,
                               cfg={"context_max_items": 8, "context_max_chars": 6000})
        result["status"] = "complete"
    elif condition == "C_context_composer":
        result = _composer_context(case["prompt"], case["scope"], store)
    else:
        raise ValueError(f"unknown condition: {condition}")
    elapsed_ms = (time.perf_counter() - started) * 1000
    rss_after = process.memory_info().rss
    context = str(result.get("context") or result.get("composed_context") or "")
    context_estimate = estimate_text_tokens(context)
    candidate_estimate = result.get("candidate_token_estimate")
    expected = [marker for marker in case["expected_markers"] if marker in context]
    forbidden = [marker for marker in case["forbidden_markers"] if marker in context]
    all_fixture_markers = [prefix + slug.upper() for slug, _, _ in TOPICS
                           for prefix in ("FACT_", "SKILL_", "FOREIGN_")]
    all_fixture_markers += [prefix + f"{index:02d}" for index in range(1, 25)
                            for prefix in ("QA_CURRENT_", "QA_STALE_", "QA_FOREIGN_")]
    retrieved_markers = sorted(marker for marker in all_fixture_markers if marker in context)
    return {
        "condition": condition, "status": result.get("status", "complete"),
        "reason": result.get("reason"), "context": context,
        "elapsed_ms": elapsed_ms,
        "process_rss_after_bytes": rss_after,
        "process_rss_delta_bytes": rss_after - rss_before,
        "memory_measurement": "harness_process_rss_after_call_not_peak_or_model_memory",
        "context_token_estimate": context_estimate,
        "candidate_token_estimate": candidate_estimate,
        "compression_ratio_estimate": (
            context_estimate / candidate_estimate
            if isinstance(candidate_estimate, int) and candidate_estimate > 0 else None
        ),
        "token_measurement": "utf8_bytes_divided_by_4_estimate",
        "expected_hits": expected, "forbidden_hits": forbidden,
        "retrieved_markers": retrieved_markers,
        "scope_leakage": sorted(set(forbidden)),
        "precision": len(expected) / len(retrieved_markers) if retrieved_markers else (None if not case["expected_markers"] else 0.0),
        "recall": len(expected) / len(case["expected_markers"]) if case["expected_markers"] else None,
        "abstention_correct": (not context) if not case["expected_markers"] else None,
        "constraint_adherent": not forbidden,
        "raw": {key: result[key] for key in
                ("selected_ids", "candidate_count", "warnings", "selector_version") if key in result},
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for condition in ("A_no_hub", "B_build_context", "C_context_composer"):
        selected = [row for row in rows if row["condition"] == condition]
        complete = [row for row in selected if row["status"] == "complete"]
        recalls = [row["recall"] for row in complete if row["recall"] is not None]
        precisions = [row["precision"] for row in complete if row["precision"] is not None]
        abstentions = [row["abstention_correct"] for row in complete if row["abstention_correct"] is not None]
        summary[condition] = {
            "slots": len(selected), "complete": len(complete),
            "unavailable_or_incomplete": len(selected) - len(complete),
            "mean_precision": statistics.fmean(precisions) if precisions else None,
            "mean_recall": statistics.fmean(recalls) if recalls else None,
            "abstention_accuracy": statistics.fmean(abstentions) if abstentions else None,
            "constraint_adherence": statistics.fmean(row["constraint_adherent"] for row in complete) if complete else None,
            "scope_leakage_cases": sum(bool(row["scope_leakage"]) for row in complete),
            "median_context_token_estimate": statistics.median(row["context_token_estimate"] for row in complete) if complete else None,
            "latency_ms_p50": _percentile([row["elapsed_ms"] for row in complete], .50),
            "latency_ms_p95": _percentile([row["elapsed_ms"] for row in complete], .95),
            "process_rss_after_bytes_p50": _percentile(
                [row["process_rss_after_bytes"] for row in complete], .50),
            "process_rss_after_bytes_p95": _percentile(
                [row["process_rss_after_bytes"] for row in complete], .95),
            "process_rss_delta_bytes_p50": _percentile(
                [row["process_rss_delta_bytes"] for row in complete], .50),
            "process_rss_delta_bytes_p95": _percentile(
                [row["process_rss_delta_bytes"] for row in complete], .95),
        }
    return summary


def _percentile(values: list[float | int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run(output_dir: Path) -> dict:
    cases, questions = build_cases(), build_memory_questions()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skill-hub-context-value-") as temp:
        store = seed_store(Path(temp) / "fixture.sqlite", cases, questions)
        try:
            rows = []
            for case in [*cases, *questions]:
                for condition in ("A_no_hub", "B_build_context", "C_context_composer"):
                    rows.append({"case_id": case["id"], "suite": "memory_qa" if "category" in case else "retrieval",
                                 "split": case.get("split"), "language": case["language"],
                                 "category": case.get("category"), **_run_condition(condition, case, store)})
        finally:
            store.close()
    corpus = {"retrieval_cases": cases, "memory_questions": questions}
    manifest = {
        "harness_version": HARNESS_VERSION, "fixture_revision": FIXTURE_REVISION,
        "generated_at": datetime.now(UTC).isoformat(), "python": sys.version,
        "platform": platform.platform(), "corpus_sha256": sha256_json(corpus),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_specs_sha256": hashlib.sha256(
            Path(__file__).with_name("context_value_tasks.py").read_bytes()).hexdigest(),
        "e2e_harness_sha256": hashlib.sha256(
            Path(__file__).with_name("context_value_e2e.py").read_bytes()).hexdigest(),
        "production_composer_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "src" / "skill_hub" / "context_composer.py").read_bytes()
        ).hexdigest(),
        "source_hashes": {
            slug: hashlib.sha256(f"{title}\n{rule}".encode()).hexdigest()
            for slug, title, rule in TOPICS
        },
        "counts": {"retrieval": len(cases), "calibration": sum(c["split"] == "calibration" for c in cases),
                   "holdout": sum(c["split"] == "holdout" for c in cases), "memory_qa": len(questions)},
        "conditions": {"A_no_hub": "no supplemental context", "B_build_context": "current deterministic build_context",
                       "C_context_composer": "production deterministic context composer"},
        "ablations": {name: "separate_not_run" for name in ("learned_ranker", "rag", "openjev")},
        "privacy": "synthetic fixtures only; no live databases, private endpoints, or project source",
        "token_accounting": "offline context values are estimates, never native or billed token counts",
        "performance_accounting": (
            "elapsed_ms is local wall time; RSS is the harness process resident set after each call "
            "and its before/after delta, not peak RSS or model memory"
        ),
    }
    report = {"manifest": manifest, "summary": summarize(rows), "rows": rows}
    (output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "offline-results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/context-value-latest"))
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
