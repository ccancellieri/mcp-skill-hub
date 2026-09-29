from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor


def _use_db(monkeypatch, tmp_path):
    from skill_hub import runtime_context

    monkeypatch.setattr(runtime_context, "RUNTIME_DB_PATH", tmp_path / "runtime.db")
    return runtime_context


def test_observations_are_scoped_by_client_and_native_session(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)

    runtime.observe_runtime({
        "client": {"id": "codex", "version": "1.2"},
        "session": {"id": "codex-session", "turn_id": "turn-1"},
        "model": {"id": "gpt-5", "provider": "openai", "display_name": "GPT-5"},
        "effort": {"value": "high", "scheme": "reasoning_effort"},
    }, default_source="caller_reported")
    runtime.observe_runtime({
        "client": {"id": "pi"},
        "session": {"id": "pi-session"},
        "model": {"id": "claude-sonnet", "provider": "anthropic"},
        "effort": {"value": "medium", "scheme": "thinking_level"},
    }, default_source="native_event")

    rows = runtime.list_runtime_sessions()
    assert {(row["client_id"], row["session_id"]) for row in rows} == {
        ("codex", "codex-session"), ("pi", "pi-session"),
    }
    codex = next(row for row in rows if row["client_id"] == "codex")
    assert codex["model_id"] == "gpt-5"
    assert codex["effort_value"] == "high"
    assert codex["provenance"]["model_id"] == "caller_reported"
    assert codex["source"] == "caller_reported"


def test_concurrent_clients_do_not_overwrite_each_other(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)

    def write(index):
        client = "codex" if index % 2 == 0 else "pi"
        runtime.observe_runtime({
            "client": {"id": client},
            "session": {"id": f"{client}-{index}"},
            "model": {"id": f"model-{index}"},
        }, default_source="native_event")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))

    rows = runtime.list_runtime_sessions(limit=20)
    assert len(rows) == 12
    assert {row["client_id"] for row in rows} == {"codex", "pi"}


def test_new_observation_replaces_model_and_preserves_missing_effort(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    base = {
        "client": {"id": "pi"}, "session": {"id": "same-session"},
        "model": {"id": "model-a", "provider": "old-provider", "display_name": "Old Name"},
        "effort": {"value": "low", "scheme": "thinking_level"},
    }
    runtime.observe_runtime(base, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "same-session"},
        "model": {"id": "model-b"},
    }, default_source="native_event")

    row = runtime.list_runtime_sessions()[0]
    assert row["model_id"] == "model-b"
    assert row["model_provider"] == ""
    assert row["model_display_name"] == ""
    assert row["effort_value"] == ""
    assert "effort_value" not in row["provenance"]


def test_native_same_model_snapshot_without_effort_clears_previous_effort(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "model-a"},
        "effort": {"value": "high", "scheme": "thinking_level"},
    }, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "model-a"},
    }, default_source="native_event")

    row = runtime.list_runtime_sessions()[0]
    assert row["effort_value"] == ""
    assert "effort_value" not in row["observed_at_by_field"]


def test_lower_trust_same_model_snapshot_does_not_clear_native_effort(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "model-a"},
        "effort": {"value": "high", "scheme": "thinking_level"},
    }, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "pi", "version": "adapter-v2"}, "session": {"id": "s"},
        "model": {"id": "model-a"},
    }, default_source="adapter_reported")

    row = runtime.list_runtime_sessions()[0]
    assert row["effort_value"] == "high"
    assert row["provenance"]["effort_value"] == "native_event"


def test_rejected_conflicting_model_rejects_its_dependent_fields(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    runtime.observe_runtime({
        "client": {"id": "codex"}, "session": {"id": "s"},
        "model": {"id": "native-model"},
    }, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "codex", "version": "caller-v2"},
        "session": {"id": "s"},
        "model": {
            "id": "caller-model", "provider": "caller-provider",
            "display_name": "Caller Model",
        },
        "effort": {"value": "high", "scheme": "reasoning_effort"},
    }, default_source="caller_reported")

    row = runtime.list_runtime_sessions()[0]
    assert row["model_id"] == "native-model"
    assert row["model_provider"] == ""
    assert row["model_display_name"] == ""
    assert row["effort_value"] == ""
    assert row["client_version"] == "caller-v2"


def test_identity_update_keeps_model_field_observation_time(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    first = datetime.now(timezone.utc) - timedelta(minutes=5)
    second = datetime.now(timezone.utc)
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "model-a"}, "observed_at": first.isoformat(),
    }, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "pi", "version": "2"},
        "session": {"id": "s"}, "observed_at": second.isoformat(),
    }, default_source="native_event")

    row = runtime.list_runtime_sessions()[0]
    assert row["observed_at"] == second.isoformat()
    assert row["observed_at_by_field"]["model_id"] == first.isoformat()
    assert row["observed_at_by_field"]["client_version"] == second.isoformat()


def test_per_field_native_provenance_wins_over_configured_or_reported(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    normalized = runtime.normalize_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "native-model", "provider": "native-provider"},
        "effort": {"value": "high", "scheme": "thinking_level"},
        "provenance": {
            "model_id": "native_event", "model_provider": "native_event",
            "effort_value": "configured", "effort_scheme": "configured",
        },
    }, default_source="caller_reported")

    assert normalized["model_id"] == "native-model"
    assert normalized["provenance"]["model_id"] == "native_event"
    assert normalized["provenance"]["effort_value"] == "configured"
    assert normalized["source"] == "mixed"


def test_older_and_lower_provenance_observations_do_not_replace_native(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    native = {
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "native-model"}, "observed_at": now.isoformat(),
    }
    runtime.observe_runtime(native, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "stale-model"},
        "observed_at": (now - timedelta(minutes=1)).isoformat(),
    }, default_source="native_event")
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "s"},
        "model": {"id": "configured-model"},
        "observed_at": (now + timedelta(minutes=1)).isoformat(),
    }, default_source="configured")

    row = runtime.list_runtime_sessions()[0]
    assert row["model_id"] == "native-model"
    assert row["provenance"]["model_id"] == "native_event"


def test_listing_missing_store_does_not_create_it(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    assert runtime.list_runtime_sessions() == []
    assert not runtime.RUNTIME_DB_PATH.exists()


def test_unknown_model_and_malformed_telemetry_are_fail_soft(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    assert runtime.observe_runtime("not-an-object") is None
    assert runtime.observe_runtime({"client": {"id": "codex"}}) is None
    assert runtime.list_runtime_sessions() == []

    row = runtime.normalize_runtime({
        "client": {"id": "codex"}, "session": {"id": "s"},
    })
    assert row["model_id"] == ""
    assert row["effort_value"] == ""


def test_stale_records_can_be_excluded(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    runtime.observe_runtime({
        "client": {"id": "pi"}, "session": {"id": "old"},
        "observed_at": old,
    }, default_source="native_event")

    assert runtime.list_runtime_sessions(max_age_seconds=60) == []
    assert runtime.list_runtime_sessions()[0]["session_id"] == "old"


def test_persistence_failure_does_not_raise(monkeypatch, tmp_path):
    runtime = _use_db(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "_connect", lambda: (_ for _ in ()).throw(OSError("readonly")))
    payload = {"client": {"id": "pi"}, "session": {"id": "s"}}
    assert runtime.observe_runtime(payload) is None
    assert runtime.list_runtime_sessions() == []
