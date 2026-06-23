"""Control Panel — plugins tab (enable/disable, profile activation, git update)."""
from __future__ import annotations

import subprocess
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import config as _cfg
from ... import plugin_registry as _pr

router = APIRouter()


# ---------------------------------------------------------------------------
# Git helpers (factored here to avoid circular imports with server.py)
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str, timeout: float = 60.0) -> tuple[int, str, str]:
    """Run git in *cwd*. Returns (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _default_branch(cwd: Path) -> str:
    rc, out, _ = _git(cwd, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if rc == 0 and out.startswith("origin/"):
        return out.split("/", 1)[1]
    return "main"


def git_pull_plugin(plugin_dir: Path) -> dict[str, Any]:
    """Pull the latest for a git-backed plugin directory.

    Returns a dict with keys:
      status: "updated" | "up_to_date" | "not_git" | "no_origin" | "dirty" | "error"
      message: human-readable one-liner (values are HTML-escaped)
      before_sha, after_sha, commit_count: populated when status=="updated"
    """
    if not plugin_dir.is_dir():
        return {"status": "error", "message": escape(f"Not a directory: {plugin_dir}")}

    if not (plugin_dir / ".git").exists():
        return {"status": "not_git", "message": "Not a git repository — no update available."}

    rc, origin_url, err = _git(plugin_dir, "remote", "get-url", "origin")
    if rc != 0 or not origin_url:
        return {"status": "no_origin",
                "message": escape(f"No origin remote: {err or 'not configured'}")}

    rc, status_out, err = _git(plugin_dir, "status", "--porcelain")
    if rc != 0:
        return {"status": "error", "message": escape(f"git status failed: {err}")}
    if status_out:
        return {"status": "dirty",
                "message": "Working tree has local changes — aborting to protect edits."}

    rc, before_sha, err = _git(plugin_dir, "rev-parse", "HEAD")
    if rc != 0:
        return {"status": "error", "message": escape(f"Cannot read HEAD: {err}")}

    branch = _default_branch(plugin_dir)

    rc, _, err = _git(plugin_dir, "fetch", "origin", branch, timeout=60.0)
    if rc != 0:
        return {"status": "error", "message": escape(f"git fetch failed: {err}")}

    rc, fetched_sha, _ = _git(plugin_dir, "rev-parse", f"origin/{branch}")
    if rc != 0:
        return {"status": "error",
                "message": escape(f"Cannot resolve origin/{branch}")}

    if before_sha == fetched_sha:
        return {"status": "up_to_date",
                "message": (f"Already up to date at "
                             f"{escape(before_sha[:8])} ({escape(branch)}).")}

    # Guard against non-fast-forward.
    rc, ahead_out, _ = _git(
        plugin_dir, "rev-list", "--left-right", "--count",
        f"origin/{branch}...HEAD",
    )
    parts = ahead_out.split() if rc == 0 else ["?", "?"]
    ahead = parts[1] if len(parts) > 1 else "?"
    if ahead not in ("0", "?"):
        return {"status": "error",
                "message": (f"Local branch is {escape(str(ahead))} commit(s) ahead — "
                             "non-fast-forward. Resolve manually.")}

    rc, commits_out, _ = _git(
        plugin_dir, "log", f"{before_sha}..{fetched_sha}", "--oneline",
    )
    commit_count = len(commits_out.splitlines()) if rc == 0 else 0

    rc, _, err = _git(plugin_dir, "merge", "--ff-only", f"origin/{branch}", timeout=60.0)
    if rc != 0:
        return {"status": "error", "message": escape(f"git merge --ff-only failed: {err}")}

    rc, after_sha, _ = _git(plugin_dir, "rev-parse", "HEAD")
    if rc != 0:
        after_sha = fetched_sha

    return {
        "status": "updated",
        "message": (f"{escape(before_sha[:8])} → {escape(after_sha[:8])} "
                    f"({commit_count} commit{'s' if commit_count != 1 else ''})"),
        "before_sha": before_sha[:8],
        "after_sha": after_sha[:8],
        "commit_count": commit_count,
    }


def _group_by_source(plugins: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in plugins:
        grouped[p["source"]].append(p)
    for src in grouped:
        grouped[src].sort(key=lambda p: p["name"])
    return dict(sorted(grouped.items()))


def _detect_active_profile() -> str | None:
    """Return the profile name whose enabled set matches the current settings, if any."""
    cfg = _cfg.load_config()
    profiles = (cfg.get("profiles") or {})
    enabled_now = {
        k.split("@", 1)[0]
        for k, v in _pr._enabled_map().items()
        if v
    }
    for name, prof in profiles.items():
        plugs = prof.get("plugins")
        if plugs == "__all__":
            continue
        if isinstance(plugs, list) and set(plugs) == enabled_now:
            return name
    return None


@router.get("/control/plugins", response_class=HTMLResponse)
def plugins_panel(request: Request) -> Any:
    plugins = list(_pr.iter_all_plugins())
    cfg = _cfg.load_config()
    profiles = (cfg.get("profiles") or {})
    return request.app.state.templates.TemplateResponse(
        request,
        "control_plugins.html",
        {
            "grouped": _group_by_source(plugins),
            "profiles": profiles,
            "active_profile": _detect_active_profile(),
            "total": len(plugins),
            "enabled_total": sum(1 for p in plugins if p["enabled"]),
        },
    )


def _render_plugin_card(request: Request, plugin_id: str) -> HTMLResponse:
    for p in _pr.iter_all_plugins():
        if p["full_key"] == plugin_id or p["name"] == plugin_id:
            return request.app.state.templates.TemplateResponse(
                request, "_plugin_card.html", {"p": p},
            )
    return HTMLResponse(f"<div class='plugin-card error'>unknown plugin: {plugin_id}</div>", status_code=404)


@router.post("/control/plugins/{plugin_id}/toggle", response_class=HTMLResponse)
def toggle(request: Request, plugin_id: str) -> Any:
    # Find current state to flip.
    current = False
    for p in _pr.iter_all_plugins():
        if p["full_key"] == plugin_id or p["name"] == plugin_id:
            current = p["enabled"]
            plugin_id = p["full_key"] or plugin_id
            break
    _pr.toggle(plugin_id, not current)
    return _render_plugin_card(request, plugin_id)


@router.post("/control/plugins/profile/{name}", response_class=HTMLResponse)
def apply_profile(request: Request, name: str) -> Any:
    _pr.apply_profile(name)
    return plugins_panel(request)


@router.post("/control/plugins/reindex", response_class=HTMLResponse)
def reindex(request: Request) -> Any:
    try:
        from ...server import index_plugins as _idx

        _idx()
    except Exception:  # noqa: BLE001
        pass
    return plugins_panel(request)


@router.post("/control/plugins/{plugin_id}/update", response_class=HTMLResponse)
def update_plugin(request: Request, plugin_id: str) -> Any:
    """Pull the latest from a git-backed plugin's source dir and re-index.

    Returns a small HTML fragment swapped into the update-result span of the
    plugin card. Degrades gracefully when the plugin is not git-backed.
    """
    # Resolve the plugin's directory from the registry.
    plugin_dir: Path | None = None
    for p in _pr.iter_all_plugins():
        if p["full_key"] == plugin_id or p["name"] == plugin_id:
            plugin_dir = Path(p["path"])
            plugin_id = p["full_key"] or plugin_id
            break

    if plugin_dir is None:
        return HTMLResponse(
            f'<span class="update-badge update-error" role="status">'
            f'Unknown plugin: {escape(plugin_id)}</span>',
            status_code=404,
        )

    result = git_pull_plugin(plugin_dir)

    if result["status"] == "updated":
        # Best-effort re-index after a successful pull.
        try:
            from ...server import index_plugins as _idx
            _idx()
        except Exception:  # noqa: BLE001
            pass

    status = result["status"]
    message = result["message"]  # already escaped

    if status == "updated":
        css = "update-ok"
    elif status == "up_to_date":
        css = "update-ok"
    elif status in ("not_git", "no_origin"):
        css = "update-na"
    else:
        css = "update-error"

    return HTMLResponse(
        f'<span class="update-badge {css}" role="status">{message}</span>'
    )
