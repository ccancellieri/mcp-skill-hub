"""Provider health probing — re-enable transiently-disabled providers.

A provider taken offline for a *temporary* upstream outage (e.g. the gateway
blocked the account) can carry ``"auto_reenable": true`` in its
``llm_provider_registry`` record. :func:`reprobe_disabled_providers` probes each
such provider's ``/models`` endpoint and flips ``enabled`` back to ``true`` once
it answers, so the escalation ladder picks it up again without a manual edit.

Providers *without* the flag — including any a user disabled deliberately — are
never touched, and the function only ever re-enables (never disables), so it is
safe to run unattended.
"""
from __future__ import annotations

import urllib.error
import urllib.request

from .. import config as _cfg
from . import credentials as _creds
from .registry import _parse_provider


def probe_models(api_base: str, api_key: str | None, *, timeout: float = 5.0) -> tuple[bool, int]:
    """GET ``{api_base}/models`` and report ``(ok, status)`` where ok is 2xx/3xx.

    A blocked account answers 401/403 (``ok=False``); a recovered one answers
    200 (``ok=True``). Any network/transport error is ``(False, 0)`` — treated
    as "still down", so a flaky probe never wrongly re-enables a provider.
    """
    url = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 0) or 0)
            return (200 <= status < 400, status)
    except urllib.error.HTTPError as exc:
        return (False, int(exc.code))
    except Exception:
        return (False, 0)


def reprobe_disabled_providers(*, timeout: float = 5.0) -> list[dict]:
    """Probe disabled ``auto_reenable`` providers; re-enable the reachable ones.

    Reads the raw ``llm_provider_registry`` (``load_registry`` hides disabled
    records), probes each ``enabled: false`` + ``auto_reenable: true`` provider,
    and sets ``enabled: true`` on a healthy ``/models`` response. Persists the
    config only when something changed. Returns one ``{name, status}`` dict per
    re-enabled provider.
    """
    raw = _cfg.get("llm_provider_registry") or []
    if not isinstance(raw, list):
        return []
    changed: list[dict] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        # Only transiently-disabled, opt-in providers are candidates.
        if rec.get("enabled", True) or not rec.get("auto_reenable"):
            continue
        prov = _parse_provider(rec)
        if prov is None:
            continue
        api_base, api_key = _creds.resolve_credentials(prov)
        if not api_base:
            # Base URL not resolvable (e.g. opencode config absent) — can't probe.
            continue
        ok, status = probe_models(api_base, api_key, timeout=timeout)
        if ok:
            rec["enabled"] = True
            changed.append({"name": rec.get("name"), "status": status})
    if changed:
        _cfg.set("llm_provider_registry", raw)
    return changed
