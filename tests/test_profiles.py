"""Tests for S3 F-SELECT — profile-based plugin curation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / "skill_hub.db"
    return SkillStore(db_path=db)


@pytest.fixture()
def settings_path(tmp_path, monkeypatch):
    """Redirect plugin_registry's settings file to a temp location."""
    from skill_hub import plugin_registry as pr

    path = tmp_path / "settings.json"
    monkeypatch.setattr(pr, "SETTINGS_PATH", path)
    return path


def test_create_and_list(store, settings_path):
    from skill_hub import profiles as prof

    p = prof.create_profile(store, "minimal", ["superpowers@x"], description="lean")
    assert p["name"] == "minimal"
    assert p["plugins"] == {"superpowers@x": True}
    assert p["description"] == "lean"

    all_profiles = prof.list_profiles(store)
    assert len(all_profiles) == 1
    assert all_profiles[0]["name"] == "minimal"


def test_create_duplicate_rejects(store, settings_path):
    from skill_hub import profiles as prof

    prof.create_profile(store, "x", ["a@b"])
    with pytest.raises(ValueError):
        prof.create_profile(store, "x", ["a@b"])
    prof.create_profile(store, "x", ["c@d"], overwrite=True)
    assert prof.get_profile(store, "x")["plugins"] == {"c@d": True}


def test_switch_profile_writes_settings(store, settings_path):
    from skill_hub import profiles as prof

    settings_path.write_text(json.dumps({
        "enabledPlugins": {"old@src": True, "drop@src": True},
    }))
    prof.create_profile(store, "geoid", {"old@src": True, "new@src": True})

    out = prof.switch_profile(store, "geoid")
    assert out["profile"] == "geoid"
    assert out["needs_restart"] is True
    saved = json.loads(settings_path.read_text())
    # Target overrides previous.
    assert saved["enabledPlugins"] == {"old@src": True, "new@src": True}

    active = prof.get_active_profile(store)
    assert active is not None and active["name"] == "geoid"


def test_switch_dry_run_does_not_write(store, settings_path):
    from skill_hub import profiles as prof

    prof.create_profile(store, "x", ["a@b"])
    out = prof.switch_profile(store, "x", dry_run=True)
    assert out["dry_run"] is True
    assert not settings_path.exists()
    assert prof.get_active_profile(store) is None


def test_switch_unknown_profile_raises(store, settings_path):
    from skill_hub import profiles as prof

    with pytest.raises(KeyError):
        prof.switch_profile(store, "nope")


def test_detect_profile_drift(store, settings_path):
    from skill_hub import profiles as prof

    prof.create_profile(store, "geoid", {"a@x": True, "b@x": True})
    prof.switch_profile(store, "geoid")
    # User manually tampers with settings.json
    settings = json.loads(settings_path.read_text())
    settings["enabledPlugins"] = {"a@x": True, "c@x": True}  # dropped b, added c
    settings_path.write_text(json.dumps(settings))

    drift = prof.detect_profile_drift(store)
    assert drift is not None
    assert "b@x" in drift["missing"]
    assert "c@x" in drift["unexpected"]


def test_auto_curate_flags_stale(store, settings_path):
    from skill_hub import profiles as prof

    settings_path.write_text(json.dumps({
        "enabledPlugins": {"used@x": True, "stale@x": True, "off@x": False},
    }))
    # used@x has recent activity; stale@x and off@x do not.
    store._conn.execute(
        "INSERT INTO session_log (session_id, query, tool_used, plugin_id, created_at)"
        " VALUES (?,?,?,?, datetime('now'))",
        ("s1", "q", "search_skills", "used@x"),
    )
    store._conn.commit()

    candidates = prof.auto_curate_candidates(store, stale_days=14)
    names = {c["plugin_id"] for c in candidates}
    # ``off@x`` is disabled so irrelevant; ``used@x`` had activity; ``stale@x`` flagged.
    assert names == {"stale@x"}
