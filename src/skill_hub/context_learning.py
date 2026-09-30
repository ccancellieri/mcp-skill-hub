"""Private, deterministic learning for local context selection.

The learner stores only local composition snapshots in the existing SkillStore
database.  It performs no network, model, telemetry, or background-hook work.
Scores are ranking values from a linear logistic model, not calibrated
probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
from datetime import UTC, datetime
from typing import Any

from .store import SkillStore, get_store

FEATURE_NAMES = (
    "lexical_relevance", "source_kind", "exact_project_match",
    "freshness", "redundancy", "token_length",
)
MIN_COMPOSITIONS = 50
MIN_GROUPS = 5
_LOCK = threading.RLock()
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_KIND_PRIOR = {
    "project_memory": 1.0, "memory": 0.9, "skill": 0.8,
    "task": 0.6, "code": 0.5, "global_skill": 0.4,
}


def _store(value: SkillStore | None) -> SkillStore:
    return value if value is not None else get_store()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    def safe(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(child) for child in item]
        return item

    return json.dumps(
        safe(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _ensure_schema(store: SkillStore) -> None:
    with _LOCK:
        store._conn.executescript("""
        CREATE TABLE IF NOT EXISTS context_learning_compositions (
            composition_id TEXT PRIMARY KEY,
            original_prompt TEXT NOT NULL,
            project_scope TEXT NOT NULL DEFAULT '',
            task_group TEXT NOT NULL,
            mode TEXT NOT NULL,
            confirmed INTEGER NOT NULL,
            task_id TEXT,
            session_id TEXT,
            selector_version TEXT,
            payload_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS context_learning_candidates (
            composition_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            text_snapshot TEXT NOT NULL,
            project_root TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            supplied_features TEXT NOT NULL,
            feature_vector TEXT NOT NULL,
            excerpt TEXT,
            PRIMARY KEY (composition_id, candidate_id)
        );
        CREATE TABLE IF NOT EXISTS context_learning_labels (
            composition_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            label INTEGER NOT NULL CHECK(label IN (0,1)),
            PRIMARY KEY (composition_id, candidate_id)
        );
        CREATE TABLE IF NOT EXISTS context_learning_outcomes (
            composition_id TEXT PRIMARY KEY,
            outcome TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS context_learning_models (
            version TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            evaluation TEXT NOT NULL,
            split TEXT NOT NULL,
            evidence TEXT,
            promoted INTEGER NOT NULL DEFAULT 0,
            trained_at TEXT NOT NULL,
            promoted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS context_learning_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS context_learning_compositions_created
            ON context_learning_compositions(created_at);
        """)
        store._conn.commit()


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _bounded(value: Any, default: float, low: float, high: float) -> float:
    return max(low, min(high, _number(value, default)))


def _tokens(text: Any) -> set[str]:
    return {part.lower() for part in _TOKEN_RE.findall(str(text or ""))}


def _features(prompt: str, candidate: dict[str, Any]) -> list[float]:
    supplied = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    prompt_tokens = _tokens(prompt)
    candidate_tokens = _tokens(f"{candidate.get('title', '')} {candidate.get('text', '')}")
    lexical = _bounded(
        supplied.get("lexical_relevance"),
        len(prompt_tokens & candidate_tokens) / max(1, len(prompt_tokens)),
        0.0,
        1.0,
    )
    length = len(_TOKEN_RE.findall(str(candidate.get("text") or "")))
    return [
        lexical,
        _bounded(
            supplied.get("source_kind"),
            _KIND_PRIOR.get(str(candidate.get("kind", "")), 0.2), 0.0, 1.0,
        ),
        _bounded(supplied.get("exact_project_match"), 0.0, 0.0, 1.0),
        _bounded(supplied.get("freshness"), 0.0, 0.0, 1.0),
        _bounded(supplied.get("redundancy"), 0.0, 0.0, 1.0),
        _bounded(supplied.get("token_length"), math.log1p(length), 0.0, 20.0),
    ]


def _validate_composition(composition: dict[str, Any]) -> tuple[list[dict], set[str], set[str]]:
    required = {"composition_id", "original_prompt", "candidates", "selected_ids", "rejected_ids"}
    missing = sorted(required - composition.keys())
    if missing:
        raise ValueError(f"missing composition fields: {', '.join(missing)}")
    candidates = composition["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")  # noqa: TRY004 - public validation contract
    ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if any(not candidate_id for candidate_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be non-empty and unique")
    selected = {str(value) for value in composition.get("selected_ids") or []}
    rejected = {str(value) for value in composition.get("rejected_ids") or []}
    if selected & rejected:
        raise ValueError("contradictory feedback: candidate is both selected and rejected")
    unknown = (selected | rejected) - set(ids)
    if unknown:
        raise ValueError(f"feedback references unknown candidates: {sorted(unknown)}")
    return candidates, selected, rejected


def record_composition(composition: dict, *, store: SkillStore | None = None) -> dict:
    """Persist a confirmed human selection snapshot and its explicit labels."""
    db = _store(store)
    _ensure_schema(db)
    candidates, selected, rejected = _validate_composition(composition)
    mode = str(composition.get("mode") or "")
    confirmed = composition.get("confirmed") is True
    if not confirmed or mode not in {"training", "mixed"}:
        return {"recorded": False, "composition_id": str(composition["composition_id"]), "labels": 0}
    composition_id = str(composition["composition_id"])
    canonical = _json(composition)
    payload_hash = hashlib.sha256(canonical.encode()).hexdigest()
    explicit_roots = composition.get("project_roots")
    if isinstance(explicit_roots, list):
        roots = sorted({str(root) for root in explicit_roots if isinstance(root, str) and root})
    else:
        # Backward compatibility for compositions recorded before authorized
        # draft scope was included explicitly in the payload.
        roots = sorted({
            str(candidate.get("project_root") or "")
            for candidate in candidates if candidate.get("project_root")
        })
    project_scope = "|".join(roots)
    task_id = composition.get("task_id")
    session_id = composition.get("session_id")
    identity = f"task:{task_id}" if task_id else f"session:{session_id}" if session_id else ""
    task_group = f"{project_scope}\x1f{identity}" if identity else ""
    excerpts = composition.get("excerpts") if isinstance(composition.get("excerpts"), dict) else {}
    with _LOCK:
        existing = db._conn.execute(
            "SELECT payload_hash FROM context_learning_compositions WHERE composition_id=?",
            (composition_id,),
        ).fetchone()
        if existing:
            if existing[0] == payload_hash:
                count = db._conn.execute(
                    "SELECT COUNT(*) FROM context_learning_labels WHERE composition_id=?", (composition_id,)
                ).fetchone()[0]
                return {"recorded": True, "composition_id": composition_id, "labels": count}
            raise ValueError("contradictory feedback for existing composition_id")
        try:
            db._conn.execute("BEGIN IMMEDIATE")
            db._conn.execute(
                "INSERT INTO context_learning_compositions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (composition_id, str(composition["original_prompt"]), project_scope, task_group,
                 mode, int(confirmed), composition.get("task_id"), composition.get("session_id"),
                 composition.get("selector_version"), payload_hash, _now()),
            )
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                vector = _features(str(composition["original_prompt"]), candidate)
                db._conn.execute(
                    "INSERT INTO context_learning_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (composition_id, candidate_id, str(candidate.get("kind") or ""),
                     str(candidate.get("title") or ""), str(candidate.get("source") or ""),
                     str(candidate.get("text") or ""), str(candidate.get("project_root") or ""),
                     str(candidate.get("source_hash") or ""), _json(candidate.get("features") or {}),
                     _json(vector), excerpts.get(candidate_id)),
                )
                label = 1 if candidate_id in selected else 0 if candidate_id in rejected else None
                if label is not None:
                    db._conn.execute(
                        "INSERT INTO context_learning_labels VALUES (?,?,?)",
                        (composition_id, candidate_id, label),
                    )
            db._conn.commit()
        except Exception:
            db._conn.rollback()
            raise
    return {"recorded": True, "composition_id": composition_id, "labels": len(selected | rejected)}


def record_outcome(composition_id: str, outcome: dict, *, store: SkillStore | None = None) -> dict:
    """Record task outcome evidence independently from selection feedback."""
    db = _store(store)
    _ensure_schema(db)
    with _LOCK:
        db._conn.execute(
            "INSERT INTO context_learning_outcomes VALUES (?,?,?) "
            "ON CONFLICT(composition_id) DO UPDATE SET outcome=excluded.outcome, recorded_at=excluded.recorded_at",
            (str(composition_id), _json(outcome), _now()),
        )
        db._conn.commit()
    return {"recorded": True, "composition_id": str(composition_id)}


def _model_rows(db: SkillStore) -> list[Any]:
    return db._conn.execute(
        "SELECT version,model,evaluation,split,evidence,promoted,trained_at,promoted_at "
        "FROM context_learning_models ORDER BY trained_at,version"
    ).fetchall()


def get_learning_status(*, store: SkillStore | None = None) -> dict:
    db = _store(store)
    _ensure_schema(db)
    compositions = db._conn.execute("SELECT COUNT(*) FROM context_learning_compositions").fetchone()[0]
    labels = db._conn.execute("SELECT COUNT(*) FROM context_learning_labels").fetchone()[0]
    eligible_compositions = db._conn.execute(
        "SELECT COUNT(DISTINCT c.composition_id) "
        "FROM context_learning_compositions c "
        "JOIN context_learning_labels l USING(composition_id) "
        "WHERE c.task_group <> ''"
    ).fetchone()[0]
    model_rows = _model_rows(db)
    models = len(model_rows)
    promoted = db._conn.execute(
        "SELECT value FROM context_learning_state WHERE key='promoted_version'"
    ).fetchone()
    state = "active" if promoted else "needs_review" if models else "untrained"
    version_summaries = []
    for row in model_rows:
        model = json.loads(row[1])
        version_summaries.append({
            "version": row[0],
            "trained_at": row[6],
            "evaluation": json.loads(row[2]),
            "active": bool(promoted and promoted[0] == row[0]),
            "dataset_hash": model.get("dataset_hash"),
        })
    return {
        "state": state, "compositions": compositions, "labels": labels,
        "models": models, "eligible_compositions": eligible_compositions,
        "promoted_version": promoted[0] if promoted else None,
        "minimum_compositions": MIN_COMPOSITIONS,
        "version_summaries": version_summaries,
    }


def _train_rows(db: SkillStore) -> list[tuple[str, str, list[float], int, str]]:
    rows = db._conn.execute("""
        SELECT c.composition_id,c.task_group,k.feature_vector,l.label,c.created_at
        FROM context_learning_compositions c
        JOIN context_learning_candidates k USING(composition_id)
        JOIN context_learning_labels l USING(composition_id,candidate_id)
        WHERE c.task_group <> ''
        ORDER BY c.created_at,c.composition_id,k.candidate_id
    """).fetchall()
    return [(row[0], row[1], json.loads(row[2]), int(row[3]), row[4]) for row in rows]


def _fit(rows: list[tuple[str, str, list[float], int, str]]) -> dict:
    vectors = [row[2] for row in rows]
    means = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(FEATURE_NAMES))]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / len(vectors)
        scales.append(math.sqrt(variance) or 1.0)
    normalized = [[(value - means[i]) / scales[i] for i, value in enumerate(vector)] for vector in vectors]
    weights = [0.0] * len(FEATURE_NAMES)
    bias = 0.0
    rate, regularization = 0.08, 0.02
    for _ in range(500):
        grad = [regularization * weight for weight in weights]
        grad_bias = 0.0
        for vector, row in zip(normalized, rows):
            score = max(-30.0, min(30.0, bias + sum(w * x for w, x in zip(weights, vector))))
            error = 1.0 / (1.0 + math.exp(-score)) - row[3]
            grad_bias += error
            for index, value in enumerate(vector):
                grad[index] += error * value
        count = len(rows)
        bias -= rate * grad_bias / count
        weights = [weight - rate * gradient / count for weight, gradient in zip(weights, grad)]
    return {"feature_names": list(FEATURE_NAMES), "weights": weights, "bias": bias, "means": means, "scales": scales}


def _score(model: dict, vector: list[float]) -> float:
    normalized = [(value - model["means"][i]) / model["scales"][i] for i, value in enumerate(vector)]
    return model["bias"] + sum(weight * value for weight, value in zip(model["weights"], normalized))


def _accuracy(model: dict, rows: list[tuple[str, str, list[float], int, str]]) -> float:
    if not rows:
        return 0.0
    return sum((_score(model, row[2]) >= 0.0) == bool(row[3]) for row in rows) / len(rows)


def _selection_metrics(
    model: dict, rows: list[tuple[str, str, list[float], int, str]], threshold: float,
) -> dict[str, float]:
    selected = [row for row in rows if _score(model, row[2]) >= threshold]
    positives = [row for row in rows if row[3] == 1]
    true_positives = sum(row[3] == 1 for row in selected)
    return {
        "holdout_precision": true_positives / len(selected) if selected else 0.0,
        "holdout_recall": true_positives / len(positives) if positives else 0.0,
        "holdout_abstention": 1.0 - (len(selected) / len(rows)) if rows else 1.0,
    }


def train_selector(*, store: SkillStore | None = None) -> dict:
    """Train a regularized linear ranker with a grouped temporal holdout."""
    db = _store(store)
    _ensure_schema(db)
    rows = _train_rows(db)
    compositions = {row[0] for row in rows}
    if len(compositions) < MIN_COMPOSITIONS:
        return {"trained": False, "reason": "insufficient_compositions", "required": MIN_COMPOSITIONS}
    group_stats: dict[str, dict[str, Any]] = {}
    for composition_id, group, _vector, _label, created_at in rows:
        stats = group_stats.setdefault(group, {"latest": created_at, "compositions": set()})
        stats["latest"] = max(stats["latest"], created_at)
        stats["compositions"].add(composition_id)
    group_order = sorted(group_stats, key=lambda group: (group_stats[group]["latest"], group))
    if len(group_order) < MIN_GROUPS:
        return {"trained": False, "reason": "insufficient_task_groups", "required": MIN_GROUPS}
    target_compositions = max(1, math.ceil(len(compositions) * 0.2))
    holdout_groups: set[str] = set()
    held_compositions: set[str] = set()
    for group in reversed(group_order):
        holdout_groups.add(group)
        held_compositions.update(group_stats[group]["compositions"])
        if len(held_compositions) >= target_compositions:
            break
    holdout_count = len(holdout_groups)
    train = [row for row in rows if row[1] not in holdout_groups]
    holdout = [row for row in rows if row[1] in holdout_groups]
    if len({row[3] for row in train}) < 2 or len({row[3] for row in holdout}) < 2:
        return {"trained": False, "reason": "both_classes_required"}
    model = _fit(train)
    # Zero is the natural decision boundary for logistic logits. It is fixed
    # from the model form and never tuned against the temporal holdout.
    model["selection_threshold"] = 0.0
    model["threshold_provenance"] = {
        "method": "fixed_zero_logit",
        "data": "training_only_no_holdout_tuning",
        "calibrated_probability": False,
    }
    dataset_material = _json([
        (composition_id, group, vector, label)
        for composition_id, group, vector, label, _created_at in rows
    ])
    dataset_hash = hashlib.sha256(dataset_material.encode()).hexdigest()
    model["dataset_hash"] = dataset_hash
    evaluation = {
        "train_accuracy": _accuracy(model, train),
        "holdout_accuracy": _accuracy(model, holdout),
        "known_not_proven": True,
        "note": "Offline label accuracy does not prove end-to-end task or token savings.",
        **_selection_metrics(model, holdout, model["selection_threshold"]),
        "selection_threshold": model["selection_threshold"],
        "threshold_provenance": model["threshold_provenance"],
    }
    split = {"train_groups": len(group_order) - holdout_count, "holdout_groups": holdout_count}
    material = _json({"model": model, "groups": group_order, "labels": [(r[0], r[3]) for r in rows]})
    version = "linear-" + hashlib.sha256(material.encode()).hexdigest()[:16]
    with _LOCK:
        db._conn.execute(
            "INSERT OR IGNORE INTO context_learning_models(version,model,evaluation,split,trained_at) VALUES(?,?,?,?,?)",
            (version, _json(model), _json(evaluation), _json(split), _now()),
        )
        db._conn.commit()
    return {
        "trained": True, "version": version, "weights": model["weights"],
        "evaluation": evaluation, "split": split,
        "split_compositions": {
            "train": len(compositions - held_compositions), "holdout": len(held_compositions),
        },
        "dataset_hash": dataset_hash, "promoted": False,
    }


def _evidence_passes(
    evidence: dict | None, *, version: str, dataset_hash: str,
) -> tuple[bool, list[str], dict[str, Any]]:
    if not isinstance(evidence, dict):
        return False, ["external evidence is required"], {}
    failures: list[str] = []
    if evidence.get("source") != "evaluation_harness":
        failures.append("evidence source must be evaluation_harness")
    if evidence.get("complete") is not True:
        failures.append("evaluation report must be complete")
    if evidence.get("estimated") is not False:
        failures.append("estimated evidence is not eligible for promotion")
    if evidence.get("selector_version") != version:
        failures.append("evidence selector version does not match candidate version")
    if evidence.get("dataset_hash") != dataset_hash:
        failures.append("evidence dataset hash does not match the trained dataset")
    runs = evidence.get("runs")
    if not isinstance(runs, list) or not runs:
        failures.append("complete paired run records are required")
        return False, failures, {}
    expected = evidence.get("expected_runs")
    completed = evidence.get("completed_runs")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        failures.append("expected_runs must be a positive integer")
    if completed != expected or completed != len(runs):
        failures.append("completed_runs must equal expected_runs and stored runs")

    pairs: dict[tuple[str, int], dict[str, dict]] = {}
    tasks: dict[str, set[int]] = {}
    for run in runs:
        if not isinstance(run, dict):
            failures.append("every run must be an object")
            continue
        task_id = run.get("task_id")
        repetition = run.get("repetition", run.get("repeat"))
        condition = run.get("condition")
        if not isinstance(task_id, str) or not task_id or not isinstance(repetition, int) or isinstance(repetition, bool):
            failures.append("every run requires task_id and integer repetition")
            continue
        if condition not in {"baseline", "candidate"}:
            failures.append("every run condition must be baseline or candidate")
            continue
        token_source = run.get("token_source")
        usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
        if token_source is None:
            token_source = usage.get("token_source")
        main_tokens = run.get("main_tokens", usage.get("main_tokens"))
        auxiliary_tokens = run.get("auxiliary_tokens", usage.get("auxiliary_tokens"))
        valid_tokens = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0
            for value in (main_tokens, auxiliary_tokens)
        )
        if token_source != "native" or not valid_tokens:
            failures.append("runs require actual native main and auxiliary token counts")
            continue
        if not isinstance(run.get("success"), bool) or not isinstance(run.get("critical_error"), bool):
            failures.append("runs require boolean success and critical_error outcomes")
            continue
        key = (task_id, repetition)
        if condition in pairs.setdefault(key, {}):
            failures.append("duplicate condition in paired runs")
            continue
        normalized = dict(run)
        normalized["total_tokens"] = float(main_tokens) + float(auxiliary_tokens)
        pairs[key][condition] = normalized
        tasks.setdefault(task_id, set()).add(repetition)
    incomplete = [key for key, conditions in pairs.items() if set(conditions) != {"baseline", "candidate"}]
    if incomplete:
        failures.append("every task repetition must have paired baseline and candidate runs")
    projects = {
        str(run.get("project") or "").strip().lower()
        for run in runs if isinstance(run, dict)
    }
    if len(tasks) < 24 or any(len(repetitions) < 2 for repetitions in tasks.values()):
        failures.append("at least 24 tasks with two complete repetitions are required")
    if not {"skill-hub", "tellurion"}.issubset(projects):
        failures.append("evaluation must include both skill-hub and tellurion projects")
    if failures:
        return False, list(dict.fromkeys(failures)), {}
    reductions = []
    baseline_success = []
    candidate_success = []
    critical_errors = 0
    for conditions in pairs.values():
        baseline = conditions["baseline"]
        candidate = conditions["candidate"]
        if baseline["total_tokens"] <= 0:
            failures.append("baseline token count must be greater than zero")
            continue
        reductions.append((baseline["total_tokens"] - candidate["total_tokens"]) / baseline["total_tokens"])
        baseline_success.append(float(baseline["success"]))
        candidate_success.append(float(candidate["success"]))
        critical_errors += int(candidate["critical_error"])
    if failures:
        return False, failures, {}
    metrics = {
        "median_task_token_reduction": statistics.median(reductions),
        "aggregate_success_regression": (
            sum(baseline_success) / len(baseline_success)
            - sum(candidate_success) / len(candidate_success)
        ),
        "critical_errors": critical_errors,
        "paired_runs": len(pairs),
    }
    if metrics["median_task_token_reduction"] < 0.15:
        failures.append("median task token reduction must be at least 15%")
    if metrics["aggregate_success_regression"] > 0.0:
        failures.append("aggregate success must not regress")
    if metrics["critical_errors"] != 0:
        failures.append("critical errors must be zero")
    return not failures, failures, metrics


def promote_selector(version: str, evidence: dict | None = None, *, store: SkillStore | None = None) -> dict:
    """Promote or roll back to a trained version after external evidence passes."""
    db = _store(store)
    _ensure_schema(db)
    model_row = db._conn.execute(
        "SELECT model FROM context_learning_models WHERE version=?", (str(version),)
    ).fetchone()
    if not model_row:
        raise ValueError(f"unknown selector version: {version}")
    model = json.loads(model_row[0])
    passed, failures, metrics = _evidence_passes(
        evidence, version=str(version), dataset_hash=str(model.get("dataset_hash") or ""),
    )
    if not passed:
        return {"promoted": False, "version": str(version), "state": "needs_review", "failures": failures}
    with _LOCK:
        try:
            db._conn.execute("BEGIN IMMEDIATE")
            db._conn.execute("UPDATE context_learning_models SET promoted=0")
            db._conn.execute(
                "UPDATE context_learning_models SET promoted=1,evidence=?,promoted_at=? WHERE version=?",
                (_json(evidence), _now(), str(version)),
            )
            db._conn.execute(
                "INSERT INTO context_learning_state(key,value) VALUES('promoted_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(version),),
            )
            db._conn.commit()
        except Exception:
            db._conn.rollback()
            raise
    return {
        "promoted": True, "version": str(version), "state": "active",
        "evidence_reference": {
            "source": evidence["source"],
            "selector_version": evidence["selector_version"],
            "dataset_hash": evidence["dataset_hash"],
            "expected_runs": evidence["expected_runs"],
            "completed_runs": evidence["completed_runs"],
        },
        "metrics": metrics,
    }


def rank_candidates(prompt: str, candidates: list[dict], *, store: SkillStore | None = None) -> dict:
    """Rank candidates with the promoted model, or return the safe baseline."""
    db = _store(store)
    _ensure_schema(db)
    active = db._conn.execute(
        "SELECT value FROM context_learning_state WHERE key='promoted_version'"
    ).fetchone()
    if not active:
        return {
            "candidates": candidates, "version": None, "available": False,
            "promoted": False, "selection_threshold": None,
        }
    row = db._conn.execute(
        "SELECT model FROM context_learning_models WHERE version=? AND promoted=1", (active[0],)
    ).fetchone()
    if not row:
        return {
            "candidates": candidates, "version": None, "available": False,
            "promoted": False, "selection_threshold": None,
        }
    model = json.loads(row[0])
    ranked = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["learning_score"] = _score(model, _features(str(prompt), candidate))
        ranked.append((item, index))
    ranked.sort(key=lambda pair: (-pair[0]["learning_score"], pair[1]))
    return {
        "candidates": [pair[0] for pair in ranked], "version": active[0],
        "available": True, "promoted": True,
        "selection_threshold": model["selection_threshold"],
    }


def reset_learning(*, store: SkillStore | None = None) -> dict:
    """Delete the complete local dataset, outcomes, versions, and active state."""
    db = _store(store)
    _ensure_schema(db)
    tables = ("compositions", "candidates", "labels", "outcomes", "models", "state")
    deleted = {}
    with _LOCK:
        try:
            db._conn.execute("BEGIN IMMEDIATE")
            for suffix in tables:
                table = f"context_learning_{suffix}"
                deleted[suffix] = db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                db._conn.execute(f"DELETE FROM {table}")
            composer_tables = (
                "context_composer_compositions",
                "context_composer_candidates",
                "context_composer_drafts",
            )
            existing = {
                row[0] for row in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table in composer_tables:
                if table in existing:
                    key = table.removeprefix("context_")
                    deleted[key] = db._conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                    db._conn.execute(f"DELETE FROM {table}")
            db._conn.commit()
        except Exception:
            db._conn.rollback()
            raise
    return {"reset": True, "deleted": deleted}


def export_learning_data(*, store: SkillStore | None = None) -> dict:
    """Return a local in-memory export. This function performs no upload."""
    db = _store(store)
    _ensure_schema(db)
    result: dict[str, list[dict]] = {}
    specs = {
        "compositions": "SELECT * FROM context_learning_compositions ORDER BY created_at,composition_id",
        "candidates": "SELECT * FROM context_learning_candidates ORDER BY composition_id,candidate_id",
        "labels": "SELECT * FROM context_learning_labels ORDER BY label,composition_id,candidate_id",
        "outcomes": "SELECT * FROM context_learning_outcomes ORDER BY recorded_at,composition_id",
        "models": "SELECT * FROM context_learning_models ORDER BY trained_at,version",
    }
    for key, query in specs.items():
        rows = db._conn.execute(query).fetchall()
        result[key] = [dict(row) for row in rows]
    return result
