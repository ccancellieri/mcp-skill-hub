"""lint-canary cadence (M3 / issue #17).

Runs ``ruff check --select <next>`` where ``<next>`` rotates through a
config-driven list of selectors. Each invocation:

1. Reads the rotation list (and last cursor) from ``state.json``.
2. Picks the next selector, advances the cursor (mod len).
3. Shells out to ``ruff`` with ``--output-format=json`` and parses findings.
4. Appends a record to the witness log (JSONL).

State + log paths live under ``~/.claude/mcp-skill-hub/state/`` by default but
are overridable to keep tests hermetic.

This module is intentionally dependency-free (only stdlib). ``ruff`` is invoked
as an external binary; if it isn't installed the result records the failure
rather than raising — the cadence keeps advancing so future runs can recover.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Default rotation, lifted verbatim from issue #17:
# F841 / F821 / B023 / S701 / RUF034 / RUF006 / B026 / ...
DEFAULT_SELECTORS: tuple[str, ...] = (
    "F841",   # unused local variable
    "F821",   # undefined name
    "B023",   # function does not bind loop variable
    "S701",   # jinja2 autoescape false
    "RUF034", # useless if-else condition
    "RUF006", # asyncio.create_task return value not stored
    "B026",   # star-arg unpacking after keyword argument
)


def _state_root() -> Path:
    return Path.home() / ".claude" / "mcp-skill-hub" / "state"


def state_path(root: Path | None = None) -> Path:
    return (root or _state_root()) / "lint_canary.json"


def witness_log_path(root: Path | None = None) -> Path:
    return (root or _state_root()) / "witness_log.jsonl"


@dataclass(frozen=True)
class CanaryRun:
    """One lint-canary invocation outcome."""

    selector: str
    findings: int
    findings_sample: list[dict]
    cursor_before: int
    cursor_after: int
    ruff_available: bool
    error: str | None
    target: str

    def to_record(self) -> dict:
        return {
            "kind": "lint_canary",
            "at": int(time.time()),
            "selector": self.selector,
            "findings": self.findings,
            "findings_sample": self.findings_sample,
            "cursor_before": self.cursor_before,
            "cursor_after": self.cursor_after,
            "ruff_available": self.ruff_available,
            "error": self.error,
            "target": self.target,
        }


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _resolve_selectors(state: dict, override: Iterable[str] | None) -> list[str]:
    if override:
        return list(override)
    persisted = state.get("selectors")
    if isinstance(persisted, list) and persisted:
        return [str(s) for s in persisted]
    return list(DEFAULT_SELECTORS)


def _run_ruff(selector: str, target: str) -> tuple[int, list[dict], str | None]:
    """Invoke ruff. Returns ``(count, sample_findings, error_message)``."""
    if shutil.which("ruff") is None:
        return (0, [], "ruff binary not found on PATH")
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv
            ["ruff", "check", "--select", selector, "--output-format", "json", target],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return (0, [], f"ruff invocation failed: {exc!r}")

    # ruff exits non-zero when findings exist; that's expected.
    stdout = (proc.stdout or "").strip()
    if not stdout:
        # ruff may also exit 0 with empty stdout when clean.
        return (0, [], None if proc.returncode in (0, 1) else (proc.stderr or "").strip() or None)
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return (0, [], f"ruff output not JSON: {exc!r}")

    if not isinstance(parsed, list):
        return (0, [], "ruff output unexpected shape (not a list)")

    sample = []
    for item in parsed[:5]:
        if not isinstance(item, dict):
            continue
        sample.append(
            {
                "code": item.get("code"),
                "message": item.get("message"),
                "filename": item.get("filename"),
                "location": item.get("location"),
            }
        )
    return (len(parsed), sample, None)


def run_lint_canary(
    target: str = ".",
    selectors: Iterable[str] | None = None,
    state_file: Path | None = None,
    witness_file: Path | None = None,
    ruff_runner=_run_ruff,
) -> CanaryRun:
    """Advance the rotation by one and record findings.

    Parameters
    ----------
    target:
        Path passed to ``ruff check``. Defaults to ``"."``.
    selectors:
        Optional rotation override. When ``None`` the persisted list (or the
        built-in default) is used. When provided, the list is *also* persisted
        so subsequent runs follow the same order.
    state_file / witness_file:
        Override locations for hermetic tests. Defaults to
        ``~/.claude/mcp-skill-hub/state/lint_canary.json`` and
        ``~/.claude/mcp-skill-hub/state/witness_log.jsonl``.
    ruff_runner:
        Injection point for tests; defaults to the real ruff subprocess call.
    """
    sp = state_file or state_path()
    wp = witness_file or witness_log_path()
    state = _load_state(sp)
    rotation = _resolve_selectors(state, selectors)
    if selectors:
        state["selectors"] = list(rotation)

    cursor_before = int(state.get("cursor", 0)) % len(rotation)
    selector = rotation[cursor_before]
    cursor_after = (cursor_before + 1) % len(rotation)

    findings_count, sample, error = ruff_runner(selector, target)

    state["cursor"] = cursor_after
    state["last_selector"] = selector
    state["last_run_at"] = int(time.time())
    _save_state(sp, state)

    result = CanaryRun(
        selector=selector,
        findings=findings_count,
        findings_sample=sample,
        cursor_before=cursor_before,
        cursor_after=cursor_after,
        ruff_available=(error != "ruff binary not found on PATH"),
        error=error,
        target=target,
    )

    # Append to witness log (JSONL, one record per line). Best-effort:
    # a log-write failure must not break the rotation advance.
    try:
        wp.parent.mkdir(parents=True, exist_ok=True)
        with wp.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_record()) + "\n")
    except OSError:
        pass

    return result


def format_run(run: CanaryRun) -> str:
    """Human-readable single-line summary for the MCP tool return value."""
    head = (
        f"lint_canary: selector={run.selector} "
        f"findings={run.findings} "
        f"cursor {run.cursor_before}->{run.cursor_after}"
    )
    if run.error:
        return f"{head} (error: {run.error})"
    return head
