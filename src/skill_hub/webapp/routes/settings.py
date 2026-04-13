"""Settings route — live config editor grouped by prefix buckets."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import config as _config

router = APIRouter()

_BUCKETS = [
    ("auto_approve", "Auto-approve"),
    ("auto_proceed", "Auto-proceed"),
    ("adaptive_windows", "Adaptive windows"),
    ("prefix_bundles", "Prefix bundles"),
    ("task_type_bundles", "Task-type bundles"),
    ("vector", "Vector"),
    ("router", "Router"),
    ("dashboard", "Dashboard"),
    ("embedding", "Embedding"),
    ("chrome", "Chrome intents"),
    ("questions", "Questions"),
    ("other", "Other"),
]


def _bucket_for(key: str) -> str:
    for prefix, _ in _BUCKETS[:-1]:
        if key.startswith(prefix):
            return prefix
    return "other"


def _field_type(val: Any) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, dict):
        return "dict"
    if isinstance(val, list):
        return "list"
    return "str"


def _group_config(cfg: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {b: [] for b, _ in _BUCKETS}
    for key in sorted(cfg.keys()):
        val = cfg[key]
        bucket = _bucket_for(key)
        entry = {"key": key, "value": val, "type": _field_type(val)}
        if entry["type"] == "dict":
            # Nested dicts (e.g. task_type_bundles with list values) don't
            # flatten well — render them as a JSON textarea instead.
            if any(isinstance(v, (dict, list)) for v in val.values()):
                entry["json_text"] = json.dumps(val, indent=2)
                entry["json_mode"] = True
            else:
                entry["json_mode"] = False
                entry["children"] = [
                    {
                        "key": f"{key}.{k}",
                        "value": v,
                        "type": _field_type(v),
                    }
                    for k, v in val.items()
                ]
        elif entry["type"] == "list":
            # Flat (primitive) lists stay comma-separated; complex lists
            # (e.g. adaptive_windows) become editable JSON textareas.
            if all(isinstance(x, (str, int, float, bool)) for x in val):
                entry["display"] = ", ".join(str(x) for x in val)
                entry["flat_list"] = True
            else:
                entry["json_text"] = json.dumps(val, indent=2)
                entry["flat_list"] = False
        groups[bucket].append(entry)
    return groups


def _coerce(original: Any, raw: str | None) -> Any:
    t = _field_type(original)
    if t == "bool":
        return raw == "on"
    if raw is None:
        return original
    raw = raw.strip()
    if t == "int":
        try:
            return int(raw)
        except ValueError:
            return original
    if t == "float":
        try:
            return float(raw)
        except ValueError:
            return original
    if t == "list":
        if not raw:
            return []
        items = [p.strip() for p in raw.split(",") if p.strip()]
        # Preserve element type if original was numeric
        if original and isinstance(original[0], int):
            out = []
            for p in items:
                try:
                    out.append(int(p))
                except ValueError:
                    pass
            return out
        if original and isinstance(original[0], float):
            out = []
            for p in items:
                try:
                    out.append(float(p))
                except ValueError:
                    pass
            return out
        return items
    return raw


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> Any:
    cfg = _config.load_config()
    groups = _group_config(cfg)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "groups": groups,
            "buckets": _BUCKETS,
            "active_tab": "settings",
        },
    )


@router.post("/settings/save", response_class=HTMLResponse)
async def settings_save(request: Request) -> HTMLResponse:
    form = await request.form()
    form_keys = set(form.keys())
    # __bool__ marks every bool field rendered (checkbox, even if unchecked).
    rendered_bools = set(form.getlist("__bool__")) if hasattr(form, "getlist") else set()
    cfg = _config.load_config()
    # Track every top-level key we actually rendered on the page. Bool fields
    # are absent from the form when unchecked, so we need the "page rendered it"
    # signal separately. We use a hidden marker: every rendered top-level key
    # emits a "__rendered__:<key>" form key? Simpler: detect "touched" by
    # presence of any sub-field for dicts, or presence-of-key-or-known-bool.
    changed = 0
    rendered_roots: set[str] = set()
    for fk in form_keys:
        # __json__.<key> -> treat <key> as the rendered root.
        if fk.startswith("__json__."):
            rendered_roots.add(fk.split(".", 1)[1].split(".", 1)[0])
        else:
            rendered_roots.add(fk.split(".", 1)[0])
    for rb in rendered_bools:
        rendered_roots.add(rb.split(".", 1)[0])

    for key, orig in list(cfg.items()):
        t = _field_type(orig)
        if key not in rendered_roots:
            # Field wasn't on the page (e.g. complex list) — skip.
            continue
        # JSON-textarea override (complex lists / nested dicts).
        json_key = f"__json__.{key}"
        if json_key in form_keys:
            raw = str(form.get(json_key) or "").strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if parsed != orig:
                        cfg[key] = parsed
                        changed += 1
                except json.JSONDecodeError:
                    pass
            continue
        if t == "dict":
            new_dict = dict(orig)
            for sub_key, sub_val in orig.items():
                form_key = f"{key}.{sub_key}"
                if _field_type(sub_val) == "bool":
                    if form_key in rendered_bools:
                        new_dict[sub_key] = form.get(form_key) == "on"
                else:
                    raw = form.get(form_key)
                    if raw is not None:
                        new_dict[sub_key] = _coerce(sub_val, str(raw))
            if new_dict != orig:
                cfg[key] = new_dict
                changed += 1
            continue
        if t == "list" and orig and not all(
            isinstance(x, (str, int, float, bool)) for x in orig
        ):
            continue
        if t == "bool":
            if key not in rendered_bools:
                continue
            new_val = form.get(key) == "on"
        else:
            raw = form.get(key)
            if raw is None:
                continue
            new_val = _coerce(orig, str(raw))
        if new_val != orig:
            cfg[key] = new_val
            changed += 1
    try:
        _config.save_config(cfg)
        msg = f"Saved ✓ ({changed} changed)"
        cls = "ok"
    except OSError as e:
        msg = f"Error: {e}"
        cls = "err"
    return HTMLResponse(f'<span class="status {cls}">{msg}</span>')
