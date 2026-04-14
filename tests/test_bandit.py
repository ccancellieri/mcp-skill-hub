"""Tests for S4 F-ROUTE — ε-greedy bandit over model tiers."""
from __future__ import annotations

import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from skill_hub.store import SkillStore
    return SkillStore(db_path=tmp_path / "skill_hub.db")


def test_bucket_boundaries():
    from skill_hub.router.bandit import bucket

    assert bucket(0.05, [])[0] == "trivial"
    assert bucket(0.2, [])[0] == "simple"
    assert bucket(0.5, [])[0] == "moderate"
    assert bucket(0.9, [])[0] == "complex"
    assert bucket(0.5, ["debugging", "api"]) == ("moderate", "debugging")
    assert bucket(0.5, [])[1] == "_none"


def test_cold_start_picks_default_and_low_confidence(store, monkeypatch):
    from skill_hub.router import bandit

    monkeypatch.setattr(random, "random", lambda: 1.0)  # disable explore
    d = bandit.select_tier(store, complexity=0.1, domain_hints=[])
    assert d.tier == "tier_cheap"
    assert d.confidence < 0.5
    assert "cold-start" in d.reasoning


def test_record_reward_updates_stats(store):
    from skill_hub.router import bandit

    bandit.record_reward(store, "tier_cheap", "simple", "_none", 1.0)
    bandit.record_reward(store, "tier_cheap", "simple", "_none", 0.5)
    stats = bandit._fetch_stats(store, "simple", "_none")
    assert stats["tier_cheap"]["trials"] == 2
    assert stats["tier_cheap"]["successes"] == pytest.approx(1.5)


def test_exploit_picks_highest_rate(store, monkeypatch):
    from skill_hub.router import bandit

    # Seed enough trials so we leave cold-start mode.
    for _ in range(5):
        bandit.record_reward(store, "tier_cheap", "moderate", "_none", 0.2)
        bandit.record_reward(store, "tier_smart", "moderate", "_none", 1.0)
        bandit.record_reward(store, "tier_mid", "moderate", "_none", 0.5)
    # Disable ε-exploration.
    monkeypatch.setattr(random, "random", lambda: 1.0)
    d = bandit.select_tier(store, complexity=0.5, domain_hints=[])
    assert d.tier == "tier_smart"
    assert "exploit" in d.reasoning


def test_epsilon_explore_when_random_below_epsilon(store, monkeypatch):
    from skill_hub.router import bandit

    # Seed stats past cold-start threshold.
    for _ in range(5):
        bandit.record_reward(store, "tier_cheap", "moderate", "_none", 1.0)
    monkeypatch.setattr(random, "random", lambda: 0.0)  # always explore
    monkeypatch.setattr(random, "choice", lambda seq: "tier_mid")
    d = bandit.select_tier(store, complexity=0.5, domain_hints=[], epsilon=0.1)
    assert d.tier == "tier_mid"
    assert "ε-explore" in d.reasoning


def test_record_reward_rejects_bad_tier(store):
    from skill_hub.router import bandit

    with pytest.raises(ValueError):
        bandit.record_reward(store, "tier_unknown", "simple", "_none", 1.0)


def test_summary_sorted_by_trials(store):
    from skill_hub.router import bandit

    bandit.record_reward(store, "tier_cheap", "simple", "_none", 1.0)
    for _ in range(3):
        bandit.record_reward(store, "tier_mid", "moderate", "_none", 1.0)
    s = bandit.summary(store)
    assert s[0]["tier"] == "tier_mid" and s[0]["trials"] == 3
