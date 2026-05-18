"""Credential vault — 3-tier backend with explicit opt-in.

Tier A: OS keyring via ``python-keyring`` (preferred).
Tier B: File-backed encrypted vault at ``${XDG_DATA_HOME:-~/.local/share}/
        skill-hub/vault.age`` decrypted via ``${SKILL_HUB_VAULT_PASSPHRASE}``.
        Activated when A is unavailable AND the env var is set. B is
        intentionally simple in v1 — uses a passphrase-derived XOR stream;
        we never write the passphrase to disk.
Tier C: Env-var pass-through. Activated when ``vault_backend: env`` is set
        in config or as the final fallback. Refs are looked up as
        environment variables (uppercased ref name).

Migration: at startup, ``migrate_config_secrets`` reads any literal secret
in the user's config (e.g. ``voyage_api_key``), writes it into the active
backend with a stable ref name (e.g. ``skill_hub_voyage``), replaces the
literal with ``voyage_api_key_ref``, and atomically rewrites config.json.

Idempotent — if the ``_ref`` form is already in place, the migration is a
no-op.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

BackendName = Literal["keyring", "file", "env"]

_KEYRING_SERVICE = "skill-hub"


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".local" / "share"


def _file_vault_path() -> Path:
    return _xdg_data_home() / "skill-hub" / "vault.age"


def _keyring_available() -> bool:
    """Return True if a real OS keyring backend is wired up."""
    try:
        import keyring  # type: ignore
    except Exception:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    # python-keyring's null/fail backend class name contains "fail".
    name = backend.__class__.__name__.lower()
    if "fail" in name:
        return False
    return True


# ---------------------------------------------------------------------------
# Tier B helpers — simple passphrase-derived XOR stream (v1)
# ---------------------------------------------------------------------------

def _derive_keystream(passphrase: str, salt: bytes, length: int) -> bytes:
    """Stretch passphrase + salt into ``length`` bytes via repeated SHA-256."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hashlib.sha256(passphrase.encode("utf-8") + salt + counter.to_bytes(4, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _b_encrypt(plaintext: str, passphrase: str) -> bytes:
    salt = os.urandom(16)
    pt = plaintext.encode("utf-8")
    ks = _derive_keystream(passphrase, salt, len(pt))
    ct = bytes(a ^ b for a, b in zip(pt, ks))
    # MAC: HMAC-ish — sha256(passphrase || salt || ct)
    mac = hashlib.sha256(passphrase.encode("utf-8") + salt + ct).digest()
    return salt + mac + ct


def _b_decrypt(blob: bytes, passphrase: str) -> str | None:
    if len(blob) < 16 + 32:
        return None
    salt = blob[:16]
    mac = blob[16:48]
    ct = blob[48:]
    expected = hashlib.sha256(passphrase.encode("utf-8") + salt + ct).digest()
    if mac != expected:
        return None
    ks = _derive_keystream(passphrase, salt, len(ct))
    pt = bytes(a ^ b for a, b in zip(ct, ks))
    try:
        return pt.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _file_load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _file_save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------

@dataclass
class Vault:
    backend: BackendName
    _passphrase: str | None = None
    _file_path: Path | None = None

    @classmethod
    def detect(cls, configured_backend: str | None = None) -> "Vault":
        """Pick the active backend per A → B → C precedence.

        Explicit ``configured_backend == "env"`` short-circuits to tier C and
        respects the user's choice (no migration nudges).
        """
        if configured_backend == "env":
            return cls(backend="env")

        if _keyring_available():
            return cls(backend="keyring")

        passphrase = os.environ.get("SKILL_HUB_VAULT_PASSPHRASE")
        if passphrase:
            return cls(
                backend="file",
                _passphrase=passphrase,
                _file_path=_file_vault_path(),
            )

        return cls(backend="env")

    # ---- core get / set ---------------------------------------------------

    def get(self, ref: str) -> str | None:
        if self.backend == "keyring":
            try:
                import keyring  # type: ignore
                return keyring.get_password(_KEYRING_SERVICE, ref)
            except Exception:
                return None

        if self.backend == "file":
            assert self._file_path is not None
            assert self._passphrase is not None
            data = _file_load(self._file_path)
            blob_hex = data.get(ref)
            if not blob_hex:
                return None
            try:
                blob = bytes.fromhex(blob_hex)
            except ValueError:
                return None
            return _b_decrypt(blob, self._passphrase)

        # env tier — refs map to upper-cased env vars
        return os.environ.get(ref.upper())

    def set(self, ref: str, value: str) -> None:
        if self.backend == "keyring":
            import keyring  # type: ignore
            keyring.set_password(_KEYRING_SERVICE, ref, value)
            return

        if self.backend == "file":
            assert self._file_path is not None
            assert self._passphrase is not None
            data = _file_load(self._file_path)
            blob = _b_encrypt(value, self._passphrase)
            data[ref] = blob.hex()
            _file_save(self._file_path, data)
            return

        # env tier — best effort: set in current process only. We never
        # write a passphrase / value to disk for tier C.
        os.environ[ref.upper()] = value


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

# Maps literal config key → (ref name, replacement key).
_SECRET_FIELDS: dict[str, tuple[str, str]] = {
    "voyage_api_key": ("skill_hub_voyage", "voyage_api_key_ref"),
    "anthropic_api_key": ("skill_hub_anthropic", "anthropic_api_key_ref"),
}


def migrate_config_secrets(cfg: dict, vault: Vault, save_fn=None) -> bool:
    """Move literal secrets in ``cfg`` into the vault.

    Returns True if a write-back is required. ``save_fn`` is called with the
    mutated cfg dict when changes are made; if omitted, the caller is
    expected to persist via its own save routine.

    Respects the user's choice — if ``cfg.get("vault_backend") == "env"`` we
    do not rewrite the config (refs are resolved via env vars at read time).
    """
    if cfg.get("vault_backend") == "env":
        return False

    changed = False
    for literal_key, (ref_name, ref_key) in _SECRET_FIELDS.items():
        # Already migrated? (truthy ref value, not just a default None placeholder)
        if cfg.get(ref_key):
            # Drop any lingering literal (shouldn't happen, but be defensive).
            if literal_key in cfg and cfg[literal_key]:
                cfg.pop(literal_key, None)
                changed = True
            continue

        value = cfg.get(literal_key)
        if not value or not isinstance(value, str):
            continue

        try:
            vault.set(ref_name, value)
        except Exception:
            # If the active backend fails to store, leave the literal in
            # place so the user notices. Don't crash startup.
            continue

        cfg[ref_key] = ref_name
        cfg.pop(literal_key, None)
        changed = True

    if changed and save_fn is not None:
        save_fn(cfg)

    return changed


def resolve_secret(cfg: dict, vault: Vault, field: str) -> str | None:
    """Resolve a secret either via literal (legacy) or ``<field>_ref`` form."""
    ref_key = f"{field}_ref"
    ref = cfg.get(ref_key)
    if ref:
        return vault.get(ref)
    literal = cfg.get(field)
    if isinstance(literal, str) and literal:
        return literal
    return None
