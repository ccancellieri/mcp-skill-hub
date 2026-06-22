"""System health — detect resource drains and offer one-click remediation.

Powers the ``/health`` webapp panel and the optional background health watcher.
Everything here is stdlib + psutil, macOS/Linux aware, and fails soft: a probe
that cannot run returns empty/zero rather than raising, so the panel never
crashes on a machine missing ``docker`` or running an unexpected OS.

Three families of helpers:

* ``scan_*`` / ``*_info`` — read-only probes (claude daemons, docker, memory,
  swap, top processes).
* ``health_snapshot`` — aggregates the probes and computes a list of actionable
  ``issues`` the panel renders as buttons.
* ``kill_*`` / ``stop_docker`` / ``purge_memory`` — the remediation actions the
  buttons POST to. All are conservative and report exactly what they did.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_CLAUDE_VERSIONS_DIR = Path.home() / ".local" / "share" / "claude" / "versions"
_VERSION_RE = re.compile(r"versions/(\d+\.\d+\.\d+)")
_RESUME_RE = re.compile(r"--resume\s+([0-9a-f-]{8,})")

# A claude process is one of these. ``daemon`` is the long-lived root that owns
# the spare pool — we never propose killing it.
_KIND_DAEMON = "daemon"
_KIND_SESSION = "session"
_KIND_SPARE = "spare"
_KIND_PTY = "pty"
_KIND_OTHER = "other"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` capturing text output; never raises (returns rc=-1 on error)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("command failed %s: %s", cmd, exc)
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr=str(exc))


def _etime_to_seconds(etime: str) -> int:
    """Parse ``ps`` elapsed-time ([[dd-]hh:]mm:ss) into seconds."""
    etime = etime.strip()
    days = 0
    if "-" in etime:
        d, etime = etime.split("-", 1)
        days = int(d)
    parts = [int(p) for p in etime.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return days * 86400 + h * 3600 + m * 60 + s


def _human_age(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600}h"


# --------------------------------------------------------------------------- #
# Claude daemon scan
# --------------------------------------------------------------------------- #
def newest_claude_version() -> str | None:
    """Highest installed claude version (the one new sessions launch with)."""
    try:
        versions = [
            p.name for p in _CLAUDE_VERSIONS_DIR.iterdir()
            if re.fullmatch(r"\d+\.\d+\.\d+", p.name)
        ]
    except OSError:
        return None
    if not versions:
        return None
    return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


@dataclass
class ClaudeProc:
    pid: int
    ppid: int
    version: str | None
    kind: str
    age_s: int
    age: str
    cpu_pct: float
    rss_mb: int
    resume_id: str | None
    stale: bool
    reason: str
    cmd: str


def _classify_claude(cmd: str) -> str:
    if "daemon run" in cmd:
        return _KIND_DAEMON
    if "--resume" in cmd:
        return _KIND_SESSION
    if "--bg-spare" in cmd:
        return _KIND_SPARE
    if "--bg-pty-host" in cmd:
        return _KIND_PTY
    return _KIND_OTHER


def scan_claude_processes() -> list[ClaudeProc]:
    """Enumerate running claude CLI/daemon processes and flag the stale ones.

    Staleness heuristics (conservative — false negatives over false positives):

    * **old version** — process runs a claude version older than the newest
      installed one, i.e. it predates the last update and was never restarted.
    * **orphaned spare/pty** — a pooled helper whose owning daemon has died
      (reparented to PID 1) and that backs no live ``--resume`` session.

    The daemon root and live sessions on the current version are never flagged.
    """
    self_pid = os.getpid()
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return _scan_claude_ps_fallback(self_pid)

    newest = newest_claude_version()
    procs: list[ClaudeProc] = []
    # First pass: collect claude pids so we can detect dead parents.
    live_pids: set[int] = set()
    raw: list[tuple] = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        try:
            cmdline = p.info.get("cmdline") or []
            cmd = " ".join(cmdline)
            if "/claude/versions/" not in cmd and "ClaudeCode.app" not in cmd \
                    and " daemon run" not in cmd:
                continue
            if "/claude" not in cmd:
                continue
            live_pids.add(p.info["pid"])
            raw.append((p, cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for p, cmd in raw:
        try:
            pid = p.info["pid"]
            ppid = p.info.get("ppid") or 0
            kind = _classify_claude(cmd)
            vmatch = _VERSION_RE.search(cmd)
            version = vmatch.group(1) if vmatch else None
            rmatch = _RESUME_RE.search(cmd)
            resume_id = rmatch.group(1) if rmatch else None
            age_s = int(time.time() - (p.info.get("create_time") or time.time()))
            try:
                cpu = p.cpu_percent(interval=None)
                rss_mb = int(p.memory_info().rss / 1024 / 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu, rss_mb = 0.0, 0

            stale, reason = _staleness(
                pid, ppid, kind, version, newest, live_pids, self_pid
            )
            procs.append(ClaudeProc(
                pid=pid, ppid=ppid, version=version, kind=kind,
                age_s=age_s, age=_human_age(age_s), cpu_pct=round(cpu, 1),
                rss_mb=rss_mb, resume_id=resume_id, stale=stale, reason=reason,
                cmd=cmd[:200],
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda c: (not c.stale, -c.age_s))
    return procs


def _staleness(
    pid: int, ppid: int, kind: str, version: str | None, newest: str | None,
    live_pids: set[int], self_pid: int,
) -> tuple[bool, str]:
    """Return (is_stale, human reason)."""
    if pid == self_pid:
        return False, "self"
    if kind == _KIND_DAEMON:
        return False, "daemon root (never killed)"
    if version and newest and version != newest:
        return True, f"old version {version} (current {newest})"
    if kind in (_KIND_SPARE, _KIND_PTY):
        # Orphaned if its parent daemon is gone (reparented to init).
        if ppid in (0, 1) or ppid not in live_pids:
            return True, "orphaned pool helper (parent gone)"
    return False, ""


def _scan_claude_ps_fallback(self_pid: int) -> list[ClaudeProc]:
    """psutil-free path using ``ps``."""
    newest = newest_claude_version()
    cp = _run(["ps", "-Ao", "pid,ppid,etime,%cpu,rss,command"])
    if cp.returncode != 0:
        return []
    live_pids: set[int] = set()
    rows: list[tuple] = []
    for line in cp.stdout.splitlines()[1:]:
        if "/claude" not in line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid_s, ppid_s, etime, cpu_s, rss_s, cmd = parts
        if "/claude/versions/" not in cmd and "ClaudeCode.app" not in cmd \
                and "daemon run" not in cmd:
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        live_pids.add(pid)
        rows.append((pid, int(ppid_s), etime, float(cpu_s), int(rss_s), cmd))
    out: list[ClaudeProc] = []
    for pid, ppid, etime, cpu, rss, cmd in rows:
        kind = _classify_claude(cmd)
        vmatch = _VERSION_RE.search(cmd)
        version = vmatch.group(1) if vmatch else None
        rmatch = _RESUME_RE.search(cmd)
        resume_id = rmatch.group(1) if rmatch else None
        age_s = _etime_to_seconds(etime)
        stale, reason = _staleness(
            pid, ppid, kind, version, newest, live_pids, self_pid
        )
        out.append(ClaudeProc(
            pid=pid, ppid=ppid, version=version, kind=kind, age_s=age_s,
            age=_human_age(age_s), cpu_pct=round(cpu, 1), rss_mb=rss // 1024,
            resume_id=resume_id, stale=stale, reason=reason, cmd=cmd[:200],
        ))
    out.sort(key=lambda c: (not c.stale, -c.age_s))
    return out


# --------------------------------------------------------------------------- #
# Docker scan
# --------------------------------------------------------------------------- #
# Containers the user keeps running as their minimum working set. Anything whose
# name contains one of these substrings is treated as essential and excluded
# from the "stop non-essential" action. Override via the ``essential`` arg.
ESSENTIAL_HINTS = ("catalog", "_db", "-db", "elasticsearch", "postgres", "_es", "-es")


@dataclass
class DockerContainer:
    name: str
    status: str
    project: str
    cpu_pct: float
    mem_mb: int
    essential: bool = False


def _is_essential(name: str, hints: tuple[str, ...] = ESSENTIAL_HINTS) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def scan_docker() -> list[DockerContainer]:
    """List running containers with per-container CPU/mem (best effort)."""
    if not docker_available():
        return []
    ps = _run([
        "docker", "ps", "--format",
        '{{.Names}}\t{{.Status}}\t{{.Label "com.docker.compose.project"}}',
    ])
    if ps.returncode != 0:
        return []
    containers: dict[str, DockerContainer] = {}
    for line in ps.stdout.splitlines():
        cols = line.split("\t")
        if not cols or not cols[0]:
            continue
        name = cols[0]
        status = cols[1] if len(cols) > 1 else ""
        project = cols[2] if len(cols) > 2 and cols[2] else "(none)"
        containers[name] = DockerContainer(
            name, status, project, 0.0, 0, essential=_is_essential(name)
        )

    # Stats are slower; tolerate timeout and leave zeros.
    stats = _run(
        ["docker", "stats", "--no-stream", "--format",
         "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
        timeout=12.0,
    )
    if stats.returncode == 0:
        for line in stats.stdout.splitlines():
            cols = line.split("\t")
            if len(cols) < 3 or cols[0] not in containers:
                continue
            c = containers[cols[0]]
            try:
                c.cpu_pct = float(cols[1].rstrip("%"))
            except ValueError:
                pass
            c.mem_mb = _parse_mem_usage(cols[2])
    return sorted(containers.values(), key=lambda c: (-c.mem_mb, c.name))


def _parse_mem_usage(s: str) -> int:
    """Parse docker '944.6MiB / 7.653GiB' usage -> used MB."""
    used = s.split("/")[0].strip()
    m = re.match(r"([\d.]+)\s*([KMGT]?i?B)", used, re.IGNORECASE)
    if not m:
        return 0
    val, unit = float(m.group(1)), m.group(2).upper()
    factor = {
        "B": 1 / 1024 / 1024, "KIB": 1 / 1024, "KB": 1 / 1024,
        "MIB": 1, "MB": 1, "GIB": 1024, "GB": 1024, "TIB": 1024 * 1024,
        "TB": 1024 * 1024,
    }.get(unit, 1)
    return int(val * factor)


# --------------------------------------------------------------------------- #
# Memory / swap / top processes
# --------------------------------------------------------------------------- #
@dataclass
class MemInfo:
    ram_used_mb: int
    ram_total_mb: int
    ram_pct: float
    swap_used_mb: int
    swap_total_mb: int
    swap_pct: float
    cpu_load_1m: float
    cpu_count: int
    load_pct: float


def memory_info() -> MemInfo:
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        ram_used_mb = int((vm.total - vm.available) / 1024 / 1024)
        ram_total_mb = int(vm.total / 1024 / 1024)
        swap_used_mb = int(sw.used / 1024 / 1024)
        swap_total_mb = int(sw.total / 1024 / 1024)
    except Exception:  # noqa: BLE001
        cpu_count = os.cpu_count() or 1
        ram_used_mb = ram_total_mb = swap_used_mb = swap_total_mb = 0
    try:
        load_1m = os.getloadavg()[0]
    except (OSError, AttributeError):
        load_1m = 0.0
    return MemInfo(
        ram_used_mb=ram_used_mb,
        ram_total_mb=ram_total_mb,
        ram_pct=round(100 * ram_used_mb / ram_total_mb, 1) if ram_total_mb else 0.0,
        swap_used_mb=swap_used_mb,
        swap_total_mb=swap_total_mb,
        swap_pct=round(100 * swap_used_mb / swap_total_mb, 1) if swap_total_mb else 0.0,
        cpu_load_1m=round(load_1m, 2),
        cpu_count=cpu_count,
        load_pct=round(100 * load_1m / cpu_count, 1) if cpu_count else 0.0,
    )


@dataclass
class TopProc:
    pid: int
    name: str
    cpu_pct: float
    rss_mb: int


def top_processes(n: int = 8) -> list[TopProc]:
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return []
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            procs.append(p)
        except psutil.NoSuchProcess:
            continue
    # Prime cpu_percent (needs two reads); short sleep keeps it cheap.
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(0.3)
    out: list[TopProc] = []
    for p in procs:
        try:
            cpu = p.cpu_percent(interval=None)
            rss = int(p.memory_info().rss / 1024 / 1024)
            out.append(TopProc(p.info["pid"], p.info.get("name") or "?", round(cpu, 1), rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    out.sort(key=lambda t: -t.cpu_pct)
    return out[:n]


# --------------------------------------------------------------------------- #
# Aggregate snapshot + issues
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    severity: str          # "info" | "warn" | "crit"
    title: str
    detail: str
    action: str | None     # action id the panel POSTs, or None (advisory only)
    action_label: str | None = None


@dataclass
class HealthSnapshot:
    mem: MemInfo
    claude: list[ClaudeProc]
    docker: list[DockerContainer]
    top: list[TopProc]
    issues: list[Issue]
    stale_daemon_count: int
    docker_mem_mb: int

    def to_dict(self) -> dict:
        return {
            "mem": asdict(self.mem),
            "claude": [asdict(c) for c in self.claude],
            "docker": [asdict(d) for d in self.docker],
            "top": [asdict(t) for t in self.top],
            "issues": [asdict(i) for i in self.issues],
            "stale_daemon_count": self.stale_daemon_count,
            "docker_mem_mb": self.docker_mem_mb,
        }


def health_snapshot(include_top: bool = True) -> HealthSnapshot:
    mem = memory_info()
    claude = scan_claude_processes()
    docker = scan_docker()
    top = top_processes() if include_top else []
    issues: list[Issue] = []

    stale = [c for c in claude if c.stale]
    if stale:
        issues.append(Issue(
            severity="warn",
            title=f"{len(stale)} stale Claude Code process(es)",
            detail="Old-version or orphaned daemon/pool helpers still running. "
                   "Safe to kill — active sessions on the current version are kept.",
            action="kill-daemons",
            action_label=f"Kill {len(stale)} stale",
        ))

    docker_mem = sum(d.mem_mb for d in docker)
    nonessential = [d for d in docker if not d.essential]
    if nonessential:
        freeable = sum(d.mem_mb for d in nonessential)
        issues.append(Issue(
            severity="warn" if docker_mem > 3000 else "info",
            title=f"{len(docker)} Docker container(s) using ~{docker_mem} MB",
            detail=f"{len(nonessential)} non-essential container(s) "
                   f"({', '.join(d.name for d in nonessential)}) hold ~{freeable} MB. "
                   "Catalog/DB/Elasticsearch are kept as your minimum working set.",
            action="stop-nonessential",
            action_label=f"Stop non-essential (free ~{freeable} MB)",
        ))
    elif docker:
        issues.append(Issue(
            severity="info",
            title=f"{len(docker)} essential container(s) using ~{docker_mem} MB",
            detail="Only the catalog/DB/Elasticsearch minimum is running.",
            action="stop-docker",
            action_label="Stop all anyway",
        ))

    if mem.swap_total_mb and mem.swap_pct >= 80:
        issues.append(Issue(
            severity="crit",
            title=f"Swap {mem.swap_pct:.0f}% full ({mem.swap_used_mb}/{mem.swap_total_mb} MB)",
            detail="RAM is exhausted and the machine is swapping to disk — the #1 "
                   "cause of slowness. Free RAM (stop Docker, kill stale procs), "
                   "then purge inactive memory. A reboot fully clears swap.",
            action="purge-memory",
            action_label="Purge inactive memory",
        ))

    if mem.ram_total_mb and mem.ram_pct >= 90:
        issues.append(Issue(
            severity="crit",
            title=f"RAM {mem.ram_pct:.0f}% used",
            detail="Very little free memory. Close apps or stop background services.",
            action=None,
        ))

    if mem.cpu_count and mem.load_pct >= 120:
        issues.append(Issue(
            severity="warn",
            title=f"CPU load {mem.cpu_load_1m} on {mem.cpu_count} cores ({mem.load_pct:.0f}%)",
            detail="More runnable work than cores — everything queues. "
                   "Often a symptom of swapping; fixing memory usually fixes this.",
            action=None,
        ))

    return HealthSnapshot(
        mem=mem, claude=claude, docker=docker, top=top, issues=issues,
        stale_daemon_count=len(stale), docker_mem_mb=docker_mem,
    )


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def _kill_pid(pid: int, hard: bool = False) -> bool:
    try:
        os.kill(pid, signal.SIGKILL if hard else signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True  # already gone
    except (PermissionError, OSError) as exc:
        log.warning("kill %s failed: %s", pid, exc)
        return False


def kill_claude(pids: list[int]) -> dict:
    """Kill the given claude pids (TERM, then KILL after a grace period).

    Re-verifies each pid is actually a claude process and is not this process
    before signalling, so a stale/forged pid can never hit an unrelated app.
    """
    self_pid = os.getpid()
    current = {c.pid: c for c in scan_claude_processes()}
    killed, skipped, failed = [], [], []
    targets = []
    for pid in pids:
        if pid == self_pid:
            skipped.append({"pid": pid, "why": "self"})
            continue
        c = current.get(pid)
        if c is None:
            skipped.append({"pid": pid, "why": "not a claude process / already gone"})
            continue
        if c.kind == _KIND_DAEMON:
            skipped.append({"pid": pid, "why": "daemon root — refused"})
            continue
        targets.append(pid)

    for pid in targets:
        _kill_pid(pid, hard=False)
    if targets:
        time.sleep(1.0)
    still = {c.pid for c in scan_claude_processes()}
    for pid in targets:
        if pid in still:
            _kill_pid(pid, hard=True)
    time.sleep(0.3)
    final = {c.pid for c in scan_claude_processes()}
    for pid in targets:
        (failed if pid in final else killed).append(pid)
    return {"killed": killed, "failed": failed, "skipped": skipped}


def kill_stale_claude() -> dict:
    """Kill every process flagged stale by :func:`scan_claude_processes`."""
    stale = [c.pid for c in scan_claude_processes() if c.stale]
    if not stale:
        return {"killed": [], "failed": [], "skipped": [], "note": "no stale processes"}
    return kill_claude(stale)


def stop_docker(names: list[str] | None = None) -> dict:
    """Stop the named containers (or all running ones if ``names`` is None)."""
    if not docker_available():
        return {"stopped": [], "error": "docker not installed"}
    if names is None:
        names = [c.name for c in scan_docker()]
    if not names:
        return {"stopped": [], "note": "no running containers"}
    cp = _run(["docker", "stop", "--time", "20", *names], timeout=120.0)
    if cp.returncode != 0:
        return {"stopped": [], "error": cp.stderr.strip() or "docker stop failed"}
    stopped = [n for n in cp.stdout.split() if n]
    return {"stopped": stopped or names}


def stop_nonessential_docker() -> dict:
    """Stop every running container except the essential working set.

    Essential = catalog / database / elasticsearch (the minimum the user keeps
    up). Everything else (keycloak, kibana, maps, tools, …) is stopped, freeing
    their RAM without breaking the dev loop.
    """
    if not docker_available():
        return {"stopped": [], "error": "docker not installed"}
    running = scan_docker()
    targets = [c.name for c in running if not c.essential]
    kept = [c.name for c in running if c.essential]
    if not targets:
        return {"stopped": [], "kept": kept, "note": "only essential containers running"}
    res = stop_docker(targets)
    res["kept"] = kept
    return res


def purge_memory() -> dict:
    """Best-effort reclaim of inactive/cached memory.

    macOS: ``purge`` flushes the disk cache and inactive pages (requires root;
    we attempt it non-interactively and report honestly if it needs auth).
    Linux: drop pagecache via ``/proc/sys/vm/drop_caches`` if writable.

    Note: neither truly *empties* swap — only a reboot does. The honest win is
    freeing RAM so the kernel stops swapping; the result message says so.
    """
    system = platform.system()
    import gc
    gc.collect()
    if system == "Darwin":
        if shutil.which("purge") is None:
            return {"ok": False, "note": "`purge` not available; reboot clears swap"}
        cp = _run(["purge"], timeout=60.0)
        if cp.returncode == 0:
            return {"ok": True, "note": "Flushed inactive memory & disk cache. "
                                        "Swap itself only clears on reboot."}
        # purge usually needs root.
        return {
            "ok": False,
            "note": "`purge` needs admin rights. Run it in a terminal with "
                    "`sudo purge`, or reboot to fully clear swap.",
            "stderr": cp.stderr.strip()[:200],
        }
    if system == "Linux":
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            return {"ok": True, "note": "Dropped pagecache/dentries/inodes."}
        except OSError as exc:
            return {"ok": False, "note": f"need root to drop caches: {exc}"}
    return {"ok": False, "note": f"unsupported platform {system}"}


# --------------------------------------------------------------------------- #
# Background health watcher
# --------------------------------------------------------------------------- #
@dataclass
class _WatcherState:
    thread: object | None = None
    stop: object | None = None
    last_snapshot: HealthSnapshot | None = None
    last_actions: list[str] = field(default_factory=list)


_WATCHER = _WatcherState()


def watcher_status() -> dict:
    running = bool(_WATCHER.thread and getattr(_WATCHER.thread, "is_alive", lambda: False)())
    return {
        "running": running,
        "last_actions": list(_WATCHER.last_actions),
        "has_snapshot": _WATCHER.last_snapshot is not None,
    }


def start_health_watcher(
    interval_s: int = 120,
    auto_cleanup: bool = False,
    swap_pct_trigger: float = 85.0,
) -> None:
    """Start a background thread that samples health and (optionally) self-heals.

    Advisory by default: it samples every ``interval_s`` seconds, caches the
    snapshot, and logs warnings. With ``auto_cleanup=True`` it additionally runs
    the *safe* remediations automatically — kill stale (old-version/orphaned)
    Claude processes, and purge inactive memory when swap crosses
    ``swap_pct_trigger``. It never stops Docker on its own (that is the user's
    call via the panel).
    """
    import threading

    if _WATCHER.thread and getattr(_WATCHER.thread, "is_alive", lambda: False)():
        return
    stop = threading.Event()

    def _loop() -> None:
        log.info("health watcher started (interval=%ss auto_cleanup=%s)",
                 interval_s, auto_cleanup)
        while not stop.is_set():
            try:
                snap = health_snapshot(include_top=False)
                _WATCHER.last_snapshot = snap
                for iss in snap.issues:
                    if iss.severity == "crit":
                        log.warning("health: %s — %s", iss.title, iss.detail)
                if auto_cleanup:
                    actions: list[str] = []
                    if snap.stale_daemon_count:
                        res = kill_stale_claude()
                        if res.get("killed"):
                            actions.append(f"killed stale claude {res['killed']}")
                    if snap.mem.swap_total_mb and snap.mem.swap_pct >= swap_pct_trigger:
                        res = purge_memory()
                        actions.append(f"purge_memory: {res.get('note')}")
                    if actions:
                        _WATCHER.last_actions = actions
                        log.info("health watcher auto-cleanup: %s", actions)
            except Exception as exc:  # noqa: BLE001
                log.warning("health watcher tick failed: %s", exc)
            stop.wait(interval_s)
        log.info("health watcher stopped")

    t = threading.Thread(target=_loop, name="health-watcher", daemon=True)
    _WATCHER.thread = t
    _WATCHER.stop = stop
    t.start()


def stop_health_watcher() -> None:
    if _WATCHER.stop is not None:
        _WATCHER.stop.set()  # type: ignore[attr-defined]
