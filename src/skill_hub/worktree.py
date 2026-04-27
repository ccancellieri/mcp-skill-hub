"""Worktree-driven parallel sessions.

Wraps `git worktree` + per-mode session launchers (terminal/tmux/background)
so a skill-hub task can own a long-lived isolated workspace and resume into
it on demand.

Design notes:
- One worktree lives at <repo>/.claude/worktrees/<name> (matches the
  EnterWorktree built-in convention; per-repo, gitignored).
- One branch named cc/<name> (matches global CLAUDE.md "AI-tooling artifacts
  off-limits inside committed content" rule -- the cc/ prefix is a flag).
- Liveness via <worktree>/.claude/session.pid -- written on launch, removed
  by a Stop hook on session exit. Resume reads this to decide focus vs
  relaunch.
- The MCP server runs as a daemon, so os.getcwd() in a tool is not the
  Claude session's cwd. Project resolution is therefore explicit:
  resolve_project(name) searches configured repo roots; the cwt CLI passes
  its actual shell cwd; auto-detect from MCP requires the caller to pass
  cwd= or project= explicitly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

Mode = Literal["terminal", "tmux", "background"]
VALID_MODES: tuple[str, ...] = ("terminal", "tmux", "background")

# Slug rule -- mirrors EnterWorktree's accepted name shape.
_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*$")


class WorktreeError(RuntimeError):
    """Raised for any worktree lifecycle problem -- always with actionable text."""


@dataclass
class WorktreeSpec:
    """Persistent description of a task's worktree + last session."""
    project: str
    name: str
    branch: str
    repo_path: str
    worktree_path: str
    mode: Mode
    pid_file: str
    last_pid: Optional[int] = None
    last_window_id: Optional[str] = None
    log_path: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, blob: str) -> "WorktreeSpec":
        data = json.loads(blob)
        return cls(**data)


# ---------------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------------

def _default_repo_roots() -> list[Path]:
    """Read worktree.repo_roots from skill-hub config; fall back to ~/work/code."""
    try:
        from . import config as _cfg
        roots = (_cfg.load_config().get("worktree") or {}).get("repo_roots")
    except Exception:  # noqa: BLE001
        roots = None
    if not roots:
        roots = ["~/work/code"]
    return [Path(p).expanduser() for p in roots]


def _default_mode() -> Mode:
    try:
        from . import config as _cfg
        mode = (_cfg.load_config().get("worktree") or {}).get("default_mode")
    except Exception:  # noqa: BLE001
        mode = None
    if mode in VALID_MODES:
        return mode  # type: ignore[return-value]
    return "terminal"


def resolve_project(name: str, *, repo_roots: Optional[Iterable[Path]] = None) -> Path:
    """Locate the git repo for project `name` under any configured root.

    Raises WorktreeError if not found or not a git repo.
    """
    if not name or not _NAME_RE.match(name) or any(
        seg in (".", "..") for seg in name.split("/")
    ):
        raise WorktreeError(
            f"invalid project name: {name!r} "
            "(letters, digits, dots, underscores, dashes, /; no .. segments)"
        )
    roots = list(repo_roots) if repo_roots is not None else _default_repo_roots()
    tried: list[str] = []
    for root in roots:
        candidate = (root / name).resolve()
        tried.append(str(candidate))
        if (candidate / ".git").exists():
            return candidate
    raise WorktreeError(
        f"project {name!r} not found as a git repo under any configured root.\n"
        f"  searched: {', '.join(tried) or '(none)'}"
    )


def detect_project_from_cwd(
    cwd: str | Path, *, repo_roots: Optional[Iterable[Path]] = None
) -> Optional[str]:
    """Walk up from `cwd` to the nearest .git, then map back to a project name.

    Returns the project name if `cwd` lives under one of the configured roots,
    else None. Returns None for the root dir itself (e.g. ~/work/code).
    """
    p = Path(cwd).expanduser().resolve()
    repo_root: Optional[Path] = None
    cur = p
    while True:
        if (cur / ".git").exists():
            repo_root = cur
            break
        if cur.parent == cur:
            return None
        cur = cur.parent
    roots = list(repo_roots) if repo_roots is not None else _default_repo_roots()
    for root in roots:
        try:
            rel = repo_root.relative_to(root.expanduser().resolve())
        except ValueError:
            continue
        # Project name is the first path segment under the root.
        parts = rel.parts
        if parts:
            return parts[0]
    return None


# ---------------------------------------------------------------------------
# Worktree creation
# ---------------------------------------------------------------------------

def compute_worktree_path(repo: Path, name: str) -> Path:
    return repo / ".claude" / "worktrees" / name


def compute_branch(name: str) -> str:
    # Global CLAUDE.md: cc/ prefix marks AI-tooling branches.
    return f"cc/{name}"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _registered_worktrees(repo: Path) -> dict[str, str]:
    """Return {worktree_path: branch} from `git worktree list --porcelain`."""
    out = _git(repo, "worktree", "list", "--porcelain").stdout
    result: dict[str, str] = {}
    cur_path: Optional[str] = None
    cur_branch: str = ""
    for line in out.splitlines():
        if line.startswith("worktree "):
            if cur_path is not None:
                result[cur_path] = cur_branch
            cur_path = line[len("worktree "):]
            cur_branch = ""
        elif line.startswith("branch "):
            cur_branch = line[len("branch "):]
    if cur_path is not None:
        result[cur_path] = cur_branch
    return result


def ensure_worktree(
    project: str,
    name: str,
    *,
    mode: Mode | None = None,
    repo_roots: Optional[Iterable[Path]] = None,
) -> WorktreeSpec:
    """Create the worktree if missing; reuse if present. Idempotent.

    Returns a WorktreeSpec without launching a session.
    """
    if mode is None:
        mode = _default_mode()
    if mode not in VALID_MODES:
        raise WorktreeError(f"invalid mode {mode!r}; expected one of {VALID_MODES}")

    repo = resolve_project(project, repo_roots=repo_roots)
    wt_path = compute_worktree_path(repo, name)
    branch = compute_branch(name)
    registered = _registered_worktrees(repo)
    wt_path_str = str(wt_path)

    if wt_path_str not in registered:
        if wt_path.exists():
            raise WorktreeError(
                f"{wt_path} exists on disk but is not a registered git worktree. "
                "Move it aside or run `git worktree prune`."
            )
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        # Create branch from HEAD; if branch already exists (e.g. from a prior
        # cleanup-without-branch-delete), reuse it without -b.
        existing = _git(repo, "branch", "--list", branch, check=False).stdout.strip()
        if existing:
            _git(repo, "worktree", "add", wt_path_str, branch)
        else:
            _git(repo, "worktree", "add", wt_path_str, "-b", branch)

    pid_file = str(wt_path / ".claude" / "session.pid")
    log_path = str(wt_path / ".claude" / "session.log")
    return WorktreeSpec(
        project=project,
        name=name,
        branch=branch,
        repo_path=str(repo),
        worktree_path=wt_path_str,
        mode=mode,
        pid_file=pid_file,
        log_path=log_path,
    )


def teardown_worktree(spec: WorktreeSpec, *, delete_branch: bool = True) -> None:
    """Remove the worktree and (optionally) its branch."""
    repo = Path(spec.repo_path)
    wt = spec.worktree_path
    if wt in _registered_worktrees(repo):
        _git(repo, "worktree", "remove", "--force", wt, check=False)
    # Best-effort branch cleanup; silent if it doesn't exist or has unmerged work.
    if delete_branch:
        _git(repo, "branch", "-D", spec.branch, check=False)


# ---------------------------------------------------------------------------
# Session launch / liveness
# ---------------------------------------------------------------------------

_HOOK_NAME = "worktree-session-stop.sh"


def _claude_binary() -> str:
    """Locate the `claude` CLI on PATH; raise with a clear message if missing."""
    found = shutil.which("claude")
    if not found:
        raise WorktreeError(
            "`claude` CLI not found on PATH. Install Claude Code or add it to PATH."
        )
    return found


def _hook_script_path() -> Path:
    """Path to the bundled Stop hook in the skill-hub repo."""
    return Path(__file__).resolve().parent.parent.parent / "hooks" / _HOOK_NAME


def _ensure_stop_hook(spec: WorktreeSpec) -> None:
    """Write <worktree>/.claude/settings.local.json wiring the pidfile cleanup hook.

    Idempotent: if a Stop hook entry already references the script, do nothing.
    """
    settings_path = Path(spec.worktree_path) / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    hook_cmd = str(_hook_script_path())
    cfg: dict = {}
    if settings_path.exists():
        try:
            cfg = json.loads(settings_path.read_text()) or {}
        except json.JSONDecodeError:
            cfg = {}
    hooks = cfg.setdefault("hooks", {})
    stop = hooks.setdefault("Stop", [])
    for entry in stop:
        for h in entry.get("hooks", []):
            if h.get("command") == hook_cmd:
                return
    stop.append({"hooks": [{"type": "command", "command": hook_cmd, "timeout": 3}]})
    settings_path.write_text(json.dumps(cfg, indent=2))


def _write_pidfile(spec: WorktreeSpec, pid: int, window_id: str = "") -> None:
    Path(spec.pid_file).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "mode": spec.mode,
        "window_id": window_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(spec.pid_file).write_text(json.dumps(payload))


def _read_pidfile(spec: WorktreeSpec) -> Optional[dict]:
    p = Path(spec.pid_file)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user -- treat as alive.
        return True
    except OSError:
        return False


def is_session_alive(spec: WorktreeSpec) -> bool:
    """Check pidfile + process; clean up stale pidfile if process is gone."""
    payload = _read_pidfile(spec)
    if not payload:
        return False
    if _process_alive(int(payload.get("pid", 0))):
        return True
    # Stale: remove and report dead.
    try:
        Path(spec.pid_file).unlink()
    except OSError:
        pass
    return False


def _launch_terminal(spec: WorktreeSpec, *, initial_prompt: Optional[str]) -> tuple[int, str]:
    """macOS only. Opens a new iTerm or Terminal tab cd'd into the worktree.

    Returns (pid, window_id). pid is the AppleScript wrapper's reported PID
    of the launched claude process, window_id is best-effort.
    """
    if sys.platform != "darwin":
        raise WorktreeError(
            "terminal mode is macOS-only. Use mode='tmux' or 'background' on this platform."
        )
    claude = _claude_binary()
    # Compose the shell command. If initial_prompt is set, pipe it in.
    if initial_prompt:
        # `claude --print` exits after the prompt; pipe via heredoc-safe quoting.
        cmd = (
            f"cd {_q(spec.worktree_path)} && "
            f"echo {_q(initial_prompt)} | {_q(claude)} --print"
        )
    else:
        cmd = f"cd {_q(spec.worktree_path)} && exec {_q(claude)}"
    term = os.environ.get("TERMINAL_APP", "iTerm")
    if term.lower() in ("iterm", "iterm2", "iterm.app"):
        script = (
            f'tell application "iTerm" to tell current window to '
            f'set newTab to (create tab with default profile)\n'
            f'tell application "iTerm" to tell current session of current window to '
            f'write text {_qa(cmd)}'
        )
    else:
        script = (
            f'tell application "Terminal" to do script {_qa(cmd)}'
        )
    res = subprocess.run(
        ["osascript", "-e", script],
        check=False, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise WorktreeError(
            f"osascript failed launching {term}: {res.stderr.strip() or res.stdout.strip()}"
        )
    # We don't get a reliable pid from osascript. The Stop hook will clean
    # up the pidfile when this Claude session exits; in the meantime, use a
    # sentinel pid (the parent shell's; treat as "alive" until removed).
    # Best-effort: poll briefly for a `claude` process that wasn't there
    # before (cwd-matched).
    pid = _wait_for_claude_under(spec.worktree_path, timeout=5.0) or 0
    return pid, ""


def _wait_for_claude_under(cwd: str, timeout: float = 5.0) -> Optional[int]:
    """Best-effort: find a `claude` process whose cwd is `cwd`. macOS-only path
    via `lsof`; returns None on failure (caller treats 0 as 'unknown but launched')."""
    if sys.platform != "darwin":
        return None
    deadline = time.monotonic() + timeout
    target = str(Path(cwd).resolve())
    while time.monotonic() < deadline:
        try:
            ps = subprocess.run(
                ["pgrep", "-f", "/claude"], capture_output=True, text=True, check=False
            )
            for pid_s in (ps.stdout or "").split():
                try:
                    pid = int(pid_s)
                except ValueError:
                    continue
                lsof = subprocess.run(
                    ["lsof", "-a", "-d", "cwd", "-p", str(pid), "-Fn"],
                    capture_output=True, text=True, check=False,
                )
                for line in (lsof.stdout or "").splitlines():
                    if line.startswith("n") and line[1:] == target:
                        return pid
        except (FileNotFoundError, OSError):
            return None
        time.sleep(0.2)
    return None


def _launch_tmux(spec: WorktreeSpec, *, initial_prompt: Optional[str]) -> tuple[int, str]:
    if not os.environ.get("TMUX"):
        raise WorktreeError(
            "mode='tmux' requires running inside a tmux session ($TMUX is unset)."
        )
    claude = _claude_binary()
    if initial_prompt:
        cmd = f"echo {_q(initial_prompt)} | {_q(claude)} --print; exec $SHELL"
    else:
        cmd = _q(claude)
    window_name = f"cc-{spec.name}"
    res = subprocess.run(
        ["tmux", "new-window", "-d", "-n", window_name,
         "-c", spec.worktree_path, cmd],
        check=False, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise WorktreeError(
            f"tmux new-window failed: {res.stderr.strip() or res.stdout.strip()}"
        )
    return 0, window_name  # PID unknown via tmux; window name is the handle.


def _launch_background(spec: WorktreeSpec, *, initial_prompt: Optional[str]) -> tuple[int, str]:
    claude = _claude_binary()
    log = Path(spec.log_path) if spec.log_path else Path(spec.worktree_path) / ".claude" / "session.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log, "a", buffering=1)
    log_fp.write(f"\n=== launch at {datetime.now(timezone.utc).isoformat()} ===\n")
    args = [claude, "--print"]
    if initial_prompt:
        args.append(initial_prompt)
    proc = subprocess.Popen(  # noqa: S603 -- claude binary, args composed from typed inputs
        args,
        cwd=spec.worktree_path,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid, ""


def launch_session(spec: WorktreeSpec, *, initial_prompt: Optional[str] = None) -> WorktreeSpec:
    """Spawn a Claude session in the worktree using the spec's mode."""
    _ensure_stop_hook(spec)
    if spec.mode == "terminal":
        pid, window = _launch_terminal(spec, initial_prompt=initial_prompt)
    elif spec.mode == "tmux":
        pid, window = _launch_tmux(spec, initial_prompt=initial_prompt)
    elif spec.mode == "background":
        pid, window = _launch_background(spec, initial_prompt=initial_prompt)
    else:  # pragma: no cover -- ensure_worktree guards this
        raise WorktreeError(f"invalid mode: {spec.mode}")
    if pid > 0:
        _write_pidfile(spec, pid, window_id=window)
    spec.last_pid = pid or None
    spec.last_window_id = window or None
    return spec


def focus_session(spec: WorktreeSpec) -> str:
    """Best-effort bring-to-front. Returns a status string for the caller."""
    if spec.mode == "tmux" and spec.last_window_id:
        subprocess.run(
            ["tmux", "select-window", "-t", spec.last_window_id],
            check=False, capture_output=True,
        )
        return f"tmux window '{spec.last_window_id}' selected"
    if spec.mode == "background":
        return f"background session alive (pid {spec.last_pid}); tail {spec.log_path}"
    if spec.mode == "terminal":
        # AppleScript can't reliably re-focus an arbitrary tab without a saved id.
        # Tell the user where it is.
        return (
            f"terminal session alive (pid {spec.last_pid}); "
            f"switch to the iTerm/Terminal tab cd'd at {spec.worktree_path}"
        )
    return f"session alive (mode {spec.mode}, pid {spec.last_pid})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(s: str) -> str:
    """Single-quote-escape for shell."""
    return "'" + s.replace("'", "'\\''") + "'"


def _qa(s: str) -> str:
    """AppleScript double-quoted string."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
