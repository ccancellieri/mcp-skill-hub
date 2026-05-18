"""Tests for the credential vault (3-tier backend + config migration)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub import vault as vault_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeKeyringBackend:
    """Stand-in for a real OS keyring backend (class name != 'fail')."""

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service, ref):
        return self._store.get((service, ref))

    def set_password(self, service, ref, value):
        self._store[(service, ref)] = value


class _FailKeyringBackend:
    """Mimics python-keyring's null backend (class name contains 'fail')."""

    def __init__(self):
        self.__class__.__name__ = "fail"


def _install_fake_keyring(monkeypatch, backend):
    """Install a fake `keyring` module into sys.modules."""
    fake = MagicMock()
    fake.get_keyring.return_value = backend
    state = {}

    def _get(service, ref):
        return state.get((service, ref))

    def _set(service, ref, value):
        state[(service, ref)] = value

    fake.get_password.side_effect = _get
    fake.set_password.side_effect = _set
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return fake


def _disable_keyring(monkeypatch):
    """Force tier-A detection to fail."""
    monkeypatch.setattr(vault_mod, "_keyring_available", lambda: False)


# ---------------------------------------------------------------------------
# Tier A — keyring
# ---------------------------------------------------------------------------

def test_tier_a_active_when_keyring_present(monkeypatch):
    monkeypatch.setattr(vault_mod, "_keyring_available", lambda: True)
    fake = _install_fake_keyring(monkeypatch, _FakeKeyringBackend())

    v = vault_mod.Vault.detect(configured_backend=None)
    assert v.backend == "keyring"

    v.set("skill_hub_voyage", "secret-value")
    assert v.get("skill_hub_voyage") == "secret-value"
    fake.set_password.assert_called_once_with(vault_mod._KEYRING_SERVICE, "skill_hub_voyage", "secret-value")


# ---------------------------------------------------------------------------
# Tier B — file-backed with passphrase
# ---------------------------------------------------------------------------

def test_tier_b_fallback_when_keyring_fails_and_env_set(monkeypatch, tmp_path):
    _disable_keyring(monkeypatch)
    monkeypatch.setenv("SKILL_HUB_VAULT_PASSPHRASE", "test-passphrase")
    monkeypatch.setattr(vault_mod, "_file_vault_path", lambda: tmp_path / "vault.age")

    v = vault_mod.Vault.detect(configured_backend=None)
    assert v.backend == "file"

    v.set("skill_hub_voyage", "vy-1234")
    # Round-trip via a fresh Vault instance (proves disk persistence + decrypt).
    v2 = vault_mod.Vault.detect(configured_backend=None)
    assert v2.backend == "file"
    assert v2.get("skill_hub_voyage") == "vy-1234"

    # Wrong passphrase → cannot decrypt.
    monkeypatch.setenv("SKILL_HUB_VAULT_PASSPHRASE", "wrong-pass")
    v3 = vault_mod.Vault.detect(configured_backend=None)
    assert v3.get("skill_hub_voyage") is None


# ---------------------------------------------------------------------------
# Tier C — env-var pass-through
# ---------------------------------------------------------------------------

def test_tier_c_fallback_when_a_and_b_unavailable(monkeypatch):
    _disable_keyring(monkeypatch)
    monkeypatch.delenv("SKILL_HUB_VAULT_PASSPHRASE", raising=False)

    v = vault_mod.Vault.detect(configured_backend=None)
    assert v.backend == "env"

    monkeypatch.setenv("SKILL_HUB_VOYAGE", "env-secret")
    assert v.get("skill_hub_voyage") == "env-secret"


def test_tier_c_explicit_opt_in_short_circuits(monkeypatch):
    # Even with keyring available, explicit env config wins.
    monkeypatch.setattr(vault_mod, "_keyring_available", lambda: True)
    _install_fake_keyring(monkeypatch, _FakeKeyringBackend())

    v = vault_mod.Vault.detect(configured_backend="env")
    assert v.backend == "env"


# ---------------------------------------------------------------------------
# Config migration
# ---------------------------------------------------------------------------

def test_migration_runs_once_then_no_op(monkeypatch, tmp_path):
    # Tier C is fine for this test — we just need a backend whose set() works.
    _disable_keyring(monkeypatch)
    monkeypatch.delenv("SKILL_HUB_VAULT_PASSPHRASE", raising=False)

    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    cfg_path.write_text(json.dumps({"voyage_api_key": "vy-literal-abc"}))

    cfg_mod.reset_vault_migration_flag()
    loaded = cfg_mod.load_config()

    # Literal removed; ref form in place.
    assert "voyage_api_key" not in loaded or not loaded.get("voyage_api_key")
    assert loaded.get("voyage_api_key_ref") == "skill_hub_voyage"

    on_disk = json.loads(cfg_path.read_text())
    assert on_disk.get("voyage_api_key_ref") == "skill_hub_voyage"
    assert not on_disk.get("voyage_api_key")

    # Second load is a no-op — disk content unchanged.
    cfg_mod.reset_vault_migration_flag()
    before = cfg_path.read_text()
    cfg_mod.load_config()
    after = cfg_path.read_text()
    assert before == after


def test_migration_skipped_when_vault_backend_env(monkeypatch, tmp_path):
    _disable_keyring(monkeypatch)
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    cfg_path.write_text(json.dumps({
        "voyage_api_key": "vy-literal",
        "vault_backend": "env",
    }))

    cfg_mod.reset_vault_migration_flag()
    loaded = cfg_mod.load_config()

    # User's explicit env choice respected — literal untouched.
    assert loaded.get("voyage_api_key") == "vy-literal"
    assert loaded.get("voyage_api_key_ref") in (None,)


def test_migration_idempotent_when_ref_already_present(monkeypatch, tmp_path):
    _disable_keyring(monkeypatch)
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    cfg_path.write_text(json.dumps({"voyage_api_key_ref": "skill_hub_voyage"}))

    cfg_mod.reset_vault_migration_flag()
    before = cfg_path.read_text()
    cfg_mod.load_config()
    after = cfg_path.read_text()
    assert before == after


# ---------------------------------------------------------------------------
# status() reports the backend
# ---------------------------------------------------------------------------

def test_status_reports_vault_backend(monkeypatch, tmp_path):
    """The MCP ``status()`` tool must include a Vault backend line."""
    # Force tier C for a predictable label.
    _disable_keyring(monkeypatch)
    monkeypatch.delenv("SKILL_HUB_VAULT_PASSPHRASE", raising=False)

    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    cfg_path.write_text(json.dumps({"vault_backend": "env"}))
    cfg_mod.reset_vault_migration_flag()

    from skill_hub import server as srv

    out = srv.status.fn("summary") if hasattr(srv.status, "fn") else srv.status("summary")
    assert "Vault backend:" in out
    assert "env" in out
