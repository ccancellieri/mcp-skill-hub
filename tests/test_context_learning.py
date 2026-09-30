from __future__ import annotations

import copy

import pytest

from skill_hub.context_learning import (
    export_learning_data,
    get_learning_status,
    promote_selector,
    rank_candidates,
    record_composition,
    record_outcome,
    reset_learning,
    train_selector,
)
from skill_hub.store import Skill, SkillStore


@pytest.fixture()
def store(tmp_path):
    value = SkillStore(db_path=tmp_path / "learning.db")
    yield value
    value.close()


def _candidate(candidate_id: str, *, selected: bool = False) -> dict:
    return {
        "candidate_id": candidate_id,
        "kind": "project_memory" if selected else "task",
        "title": "SQLite locking selector" if selected else "unrelated release notes",
        "source": "local",
        "text": "sqlite selector locking deterministic ranking" if selected else "release version archive",
        "project_root": "/work/project",
        "source_hash": f"hash-{candidate_id}",
        "features": {
            "exact_project_match": selected,
            "freshness": 0.9 if selected else 0.1,
            "redundancy": 0.0 if selected else 0.8,
        },
    }


def _composition(index: int, *, task_id: str | None = None) -> dict:
    return {
        "composition_id": f"composition-{index}",
        "original_prompt": "fix sqlite selector locking",
        "candidates": [_candidate(f"good-{index}", selected=True), _candidate(f"bad-{index}")],
        "selected_ids": [f"good-{index}"],
        "rejected_ids": [f"bad-{index}"],
        "excerpts": {f"good-{index}": "sqlite selector locking"},
        "mode": "training",
        "confirmed": True,
        "task_id": task_id,
        "session_id": f"session-{index}",
        "selector_version": None,
    }


def _evidence(version: str, dataset_hash: str, *, reduction: float = 0.20) -> dict:
    runs = []
    for task_index in range(24):
        for repetition in range(2):
            baseline = 1000 + task_index * 10 + repetition
            candidate = int(baseline * (1.0 - reduction))
            for condition, total in (("baseline", baseline), ("candidate", candidate)):
                runs.append({
                    "task_id": f"eval-task-{task_index}",
                    "project": "skill-hub" if task_index % 2 == 0 else "tellurion",
                    "repetition": repetition,
                    "condition": condition,
                    "main_tokens": total - 100,
                    "auxiliary_tokens": 100,
                    "token_source": "native",
                    "success": True,
                    "critical_error": False,
                })
    return {
        "source": "evaluation_harness",
        "complete": True,
        "estimated": False,
        "selector_version": version,
        "dataset_hash": dataset_hash,
        "expected_runs": len(runs),
        "completed_runs": len(runs),
        "runs": runs,
    }


def test_untrained_ranker_is_unavailable_and_preserves_candidates(store):
    candidates = [_candidate("one"), _candidate("two", selected=True)]
    result = rank_candidates("sqlite selector", candidates, store=store)

    assert result == {
        "candidates": candidates, "version": None,
        "available": False, "promoted": False, "selection_threshold": None,
    }
    assert get_learning_status(store=store)["state"] == "untrained"


def test_recording_requires_confirmed_feedback_and_only_labels_explicit_choices(store):
    automatic = _composition(1)
    automatic["confirmed"] = False
    automatic["mode"] = "automatic"
    assert record_composition(automatic, store=store)["recorded"] is False
    assert export_learning_data(store=store)["labels"] == []

    unconfirmed_mixed = _composition(10)
    unconfirmed_mixed["confirmed"] = False
    unconfirmed_mixed["mode"] = "mixed"
    assert record_composition(unconfirmed_mixed, store=store)["recorded"] is False

    confirmed_automatic = _composition(11)
    confirmed_automatic["confirmed"] = True
    confirmed_automatic["mode"] = "automatic"
    assert record_composition(confirmed_automatic, store=store)["recorded"] is False
    assert export_learning_data(store=store)["labels"] == []

    payload = _composition(2)
    payload["candidates"].append(_candidate("unlabelled"))
    result = record_composition(payload, store=store)
    exported = export_learning_data(store=store)

    assert result == {"recorded": True, "composition_id": "composition-2", "labels": 2}
    assert [row["label"] for row in exported["labels"]] == [0, 1]
    assert len(exported["candidates"]) == 3
    assert exported["compositions"][0]["original_prompt"] == payload["original_prompt"]


def test_duplicate_feedback_is_idempotent_but_contradictions_are_rejected(store):
    payload = _composition(3)
    first = record_composition(payload, store=store)
    second = record_composition(copy.deepcopy(payload), store=store)
    assert second == first

    contradictory = copy.deepcopy(payload)
    contradictory["selected_ids"] = ["bad-3"]
    contradictory["rejected_ids"] = ["good-3"]
    with pytest.raises(ValueError, match="contradictory feedback"):
        record_composition(contradictory, store=store)


def test_outcomes_are_independent_and_reset_deletes_everything(store):
    record_composition(_composition(4), store=store)
    record_outcome("composition-4", {"success": True, "critical_error": False}, store=store)
    store._conn.executescript("""
        CREATE TABLE context_composer_drafts (id TEXT);
        CREATE TABLE context_composer_candidates (id TEXT);
        CREATE TABLE context_composer_compositions (id TEXT);
        INSERT INTO context_composer_drafts VALUES ('d');
        INSERT INTO context_composer_candidates VALUES ('c');
        INSERT INTO context_composer_compositions VALUES ('x');
    """)
    assert len(export_learning_data(store=store)["outcomes"]) == 1

    result = reset_learning(store=store)
    assert result["deleted"]["compositions"] == 1
    assert result["deleted"]["composer_drafts"] == 1
    assert store._conn.execute("SELECT COUNT(*) FROM context_composer_drafts").fetchone()[0] == 0
    assert export_learning_data(store=store) == {
        "compositions": [], "candidates": [], "labels": [], "outcomes": [], "models": []
    }


def test_training_waits_for_enough_unique_compositions_and_task_groups(store):
    for index in range(50):
        record_composition(_composition(index, task_id="same-task"), store=store)
    result = train_selector(store=store)
    assert result["trained"] is False
    assert result["reason"] == "insufficient_task_groups"


def test_training_is_deterministic_temporal_and_promotion_requires_evidence(store):
    for index in range(60):
        record_composition(_composition(index, task_id=f"task-{index}"), store=store)

    first = train_selector(store=store)
    second = train_selector(store=store)
    assert first["trained"] is True
    assert first["weights"] == second["weights"]
    assert first["evaluation"]["known_not_proven"] is True
    assert first["evaluation"]["selection_threshold"] == 0.0
    assert first["evaluation"]["threshold_provenance"] == {
        "method": "fixed_zero_logit",
        "data": "training_only_no_holdout_tuning",
        "calibrated_probability": False,
    }
    assert 0.0 <= first["evaluation"]["holdout_precision"] <= 1.0
    assert 0.0 <= first["evaluation"]["holdout_recall"] <= 1.0
    assert 0.0 <= first["evaluation"]["holdout_abstention"] <= 1.0
    assert first["split"] == {"train_groups": 48, "holdout_groups": 12}

    review = promote_selector(first["version"], store=store)
    assert review["promoted"] is False
    assert review["state"] == "needs_review"

    insufficient = promote_selector(
        first["version"],
        evidence=_evidence(first["version"], first["dataset_hash"], reduction=0.14),
        store=store,
    )
    assert insufficient["promoted"] is False

    promoted = promote_selector(
        first["version"], evidence=_evidence(first["version"], first["dataset_hash"]), store=store
    )
    assert promoted["promoted"] is True

    ranked = rank_candidates(
        "fix sqlite selector locking",
        [_candidate("irrelevant"), _candidate("relevant", selected=True)],
        store=store,
    )
    assert ranked["available"] is True
    assert ranked["version"] == first["version"]
    assert ranked["selection_threshold"] == 0.0
    assert ranked["candidates"][0]["candidate_id"] == "relevant"
    assert "learning_score" in ranked["candidates"][0]

    changed = _composition(99, task_id="task-99")
    changed["selected_ids"] = ["bad-99"]
    changed["rejected_ids"] = ["good-99"]
    record_composition(changed, store=store)
    newer = train_selector(store=store)
    assert newer["version"] != first["version"]
    promote_selector(
        newer["version"], evidence=_evidence(newer["version"], newer["dataset_hash"]), store=store
    )
    rollback = promote_selector(
        first["version"], evidence=_evidence(first["version"], first["dataset_hash"]), store=store
    )
    assert rollback["promoted"] is True
    assert get_learning_status(store=store)["promoted_version"] == first["version"]
    summaries = get_learning_status(store=store)["version_summaries"]
    assert {summary["version"] for summary in summaries} == {first["version"], newer["version"]}
    assert next(summary for summary in summaries if summary["active"])["version"] == first["version"]
    assert all(summary["dataset_hash"] for summary in summaries)


def test_missing_task_identity_is_stored_but_excluded_from_training(store):
    for index in range(50):
        payload = _composition(index)
        payload["task_id"] = None
        payload["session_id"] = None
        record_composition(payload, store=store)

    result = train_selector(store=store)
    assert result["trained"] is False
    assert result["reason"] == "insufficient_compositions"
    status = get_learning_status(store=store)
    assert status["compositions"] == 50
    assert status["eligible_compositions"] == 0


def test_promotion_rejects_aggregate_estimates_incomplete_pairs_and_wrong_binding(store):
    for index in range(50):
        record_composition(_composition(index, task_id=f"task-{index}"), store=store)
    trained = train_selector(store=store)

    aggregate_only = {
        "source": "manual", "complete": True, "estimated": True,
        "selector_version": trained["version"], "dataset_hash": trained["dataset_hash"],
        "median_task_token_reduction": 0.20,
        "aggregate_success_regression": 0.0,
        "critical_errors": 0,
    }
    result = promote_selector(trained["version"], aggregate_only, store=store)
    assert result["promoted"] is False
    assert any("estimated" in failure for failure in result["failures"])

    incomplete = _evidence(trained["version"], trained["dataset_hash"])
    incomplete["runs"].pop()
    result = promote_selector(trained["version"], incomplete, store=store)
    assert result["promoted"] is False
    assert any("paired" in failure for failure in result["failures"])

    wrong = _evidence("another-version", trained["dataset_hash"])
    result = promote_selector(trained["version"], wrong, store=store)
    assert result["promoted"] is False
    assert any("version" in failure for failure in result["failures"])

    too_few = _evidence(trained["version"], trained["dataset_hash"])
    too_few["runs"] = [run for run in too_few["runs"] if run["task_id"] != "eval-task-23"]
    too_few["expected_runs"] = too_few["completed_runs"] = len(too_few["runs"])
    result = promote_selector(trained["version"], too_few, store=store)
    assert result["promoted"] is False
    assert any("24 tasks" in failure for failure in result["failures"])

    one_project = _evidence(trained["version"], trained["dataset_hash"])
    for run in one_project["runs"]:
        run["project"] = "skill-hub"
    result = promote_selector(trained["version"], one_project, store=store)
    assert result["promoted"] is False
    assert any("tellurion" in failure for failure in result["failures"])


def test_temporal_holdout_targets_twenty_percent_of_compositions(store):
    for index in range(40):
        record_composition(_composition(index, task_id="old-large-task"), store=store)
    for index in range(40, 61):
        record_composition(_composition(index, task_id=f"recent-{index}"), store=store)

    result = train_selector(store=store)
    assert result["trained"] is True
    assert result["split"] == {"train_groups": 9, "holdout_groups": 13}
    assert result["split_compositions"] == {"train": 48, "holdout": 13}


def test_non_finite_supplied_features_are_replaced_with_safe_defaults(store):
    payload = _composition(1)
    payload["candidates"][0]["features"].update({
        "lexical_relevance": float("nan"),
        "source_kind": float("inf"),
        "token_length": float("-inf"),
        "freshness": 1e308,
    })
    record_composition(payload, store=store)
    vector = export_learning_data(store=store)["candidates"][1]["feature_vector"]
    assert "NaN" not in vector
    assert "Infinity" not in vector
    assert "NaN" not in export_learning_data(store=store)["candidates"][1]["supplied_features"]


def test_skill_only_compositions_keep_authorized_project_scope_in_task_group(store):
    from skill_hub.context_composer import compose_context, prepare_composition

    store.upsert_skill(Skill(
        id="global:sqlite", name="SQLite selector",
        description="Use SQLite selector locking patterns.",
        content="Full global skill content.", file_path="/skills/sqlite/SKILL.md",
        plugin="global",
    ))
    for project_root in ("/projects/alpha", "/projects/beta"):
        draft = prepare_composition(
            "sqlite selector", project_roots=[project_root],
            session_id="shared-native-session", mode="training", store=store,
        )
        assert draft["candidates"]
        assert all(candidate["project_root"] == "" for candidate in draft["candidates"])
        compose_context(
            draft["draft_id"], selected_ids=draft["selected_ids"],
            confirmed=True, store=store,
        )

    groups = [row[0] for row in store._conn.execute(
        "SELECT task_group FROM context_learning_compositions ORDER BY task_group"
    ).fetchall()]
    assert groups == [
        "/projects/alpha\x1fsession:shared-native-session",
        "/projects/beta\x1fsession:shared-native-session",
    ]
