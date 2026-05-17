"""sync_check — cross-repo stale-import detector (M3 / issue #16).

Encodes the multi-repo sync directive (one logical system across N repos) as a
callable check. Pure grep — no LLM required.

The maintainer's three-repo system (e.g. ``geoid`` SSOT plus ``dynastore`` and
``fao-aip-catalog`` followers) routinely breaks when a primary repo removes or
renames a public symbol but follower repos still reference the old name.
``sync_check`` finds those references mechanically:

1. Diff the *primary* repo against a base ref (default: previous commit on
   ``HEAD``) to obtain identifiers removed from removed/changed lines.
2. Drop any identifier that still appears anywhere in the *current* primary
   working tree — that's a rename or move, not a removal.
3. For each surviving "removed symbol", grep follower repos for references and
   return one structured finding per match::

       stale ref "OldClass" in follower/src/foo.py:42

Pure stdlib — only ``git`` as an external binary, invoked read-only against
the primary repo. Follower scanning uses ``Path.rglob`` + line iteration so
followers don't need to be git repos.

The function is intentionally dependency-free; the MCP server wires it into a
tool wrapper exactly as ``lint_canary`` and ``export_policies`` do.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


# Identifier regex: Python-style names. Two characters minimum to avoid flagging
# every ``x`` / ``i`` loop variable, which would explode false-positive volume.
# (The maintainer's symbols of interest — class names, public functions, env
# vars — comfortably clear that bar.)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b")

# Identifiers we never want to flag, regardless of source. These are language
# keywords and ubiquitous builtins / common locals that would generate noise
# even when removed from a diff (they always exist *somewhere* in any Python
# codebase, but a removed *line* containing ``return self`` shouldn't make
# ``self`` a candidate).
_NOISE_IDENTS: frozenset[str] = frozenset(
    {
        # Python keywords + soft keywords
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield", "match", "case",
        # Common Python builtins / dunders that show up everywhere
        "self", "cls", "print", "len", "str", "int", "float", "bool", "list",
        "dict", "set", "tuple", "bytes", "object", "type", "range", "iter",
        "enumerate", "zip", "map", "filter", "sorted", "reversed", "min",
        "max", "sum", "any", "all", "abs", "round", "open", "input", "id",
        "hash", "repr", "vars", "dir", "isinstance", "issubclass", "getattr",
        "setattr", "hasattr", "delattr", "callable", "super",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "OSError", "NotImplementedError",
        "Any", "Optional", "Union", "List", "Dict", "Tuple", "Set", "Callable",
        "Iterable", "Iterator", "Sequence", "Mapping", "Type",
        # Project-local one/two-letter sentinels that aren't worth flagging.
        # (Two-char minimum already screens single chars; this keeps the list
        # explicit for clarity.)
    }
)


@dataclass(frozen=True)
class StaleRef:
    """One stale reference found in a follower repo."""

    symbol: str
    follower: str   # follower repo root (display string, not necessarily absolute)
    path: str       # path inside the follower repo, posix-style
    line_no: int    # 1-based
    line: str       # raw line content (stripped trailing newline)

    def render(self) -> str:
        return f'stale ref "{self.symbol}" in {self.follower}/{self.path}:{self.line_no}'


@dataclass
class SyncReport:
    """Aggregate result of one ``sync_check`` invocation."""

    primary: str
    base_ref: str
    removed_symbols: list[str] = field(default_factory=list)
    findings: list[StaleRef] = field(default_factory=list)
    followers_scanned: list[str] = field(default_factory=list)
    skipped_followers: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "base_ref": self.base_ref,
            "removed_symbols": list(self.removed_symbols),
            "findings": [
                {
                    "symbol": f.symbol,
                    "follower": f.follower,
                    "path": f.path,
                    "line_no": f.line_no,
                    "line": f.line,
                }
                for f in self.findings
            ],
            "followers_scanned": list(self.followers_scanned),
            "skipped_followers": dict(self.skipped_followers),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Primary-diff extraction
# ---------------------------------------------------------------------------


def _run_git_diff(primary: Path, base_ref: str) -> tuple[str, str | None]:
    """Return ``(unified_diff, error)``. Empty diff + None on clean compare."""
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv
            [
                "git",
                "-C",
                str(primary),
                "diff",
                "--unified=0",
                "--no-color",
                base_ref,
                "--",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "", f"git diff invocation failed: {exc!r}"
    if proc.returncode not in (0, 1):
        # git diff exits 0 (no changes) or 1 (changes present). Anything else is
        # an error (bad ref, not a git repo, etc).
        msg = (proc.stderr or proc.stdout or "").strip() or f"git diff exit {proc.returncode}"
        return "", msg
    return proc.stdout or "", None


def _extract_removed_idents_from_diff(diff_text: str) -> set[str]:
    """Pick identifiers that appeared on removed (``-``) lines of the diff.

    Skips file headers (``--- a/...``) and hunk headers. Drops anything in
    ``_NOISE_IDENTS``.
    """
    removed: set[str] = set()
    for raw in diff_text.splitlines():
        if not raw or not raw.startswith("-"):
            continue
        if raw.startswith("---"):
            continue
        # Strip the leading '-' before tokenizing so we don't include the
        # marker as part of an identifier match.
        line = raw[1:]
        for token in _IDENT_RE.findall(line):
            if token in _NOISE_IDENTS:
                continue
            removed.add(token)
    return removed


def _scan_current_idents(primary: Path, suffixes: Sequence[str]) -> set[str]:
    """Identifiers present anywhere in the primary working tree right now.

    Used to filter out renames/moves: a symbol that still exists somewhere in
    the primary repo isn't *removed*, it just changed location.
    """
    present: set[str] = set()
    for path in _iter_source_files(primary, suffixes):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in _IDENT_RE.findall(text):
            present.add(token)
    return present


# ---------------------------------------------------------------------------
# Follower scanning
# ---------------------------------------------------------------------------


_DEFAULT_SUFFIXES: tuple[str, ...] = (
    ".py", ".pyi",
    ".ts", ".tsx", ".js", ".jsx",
    ".go",
    ".rs",
    ".java", ".kt",
    ".rb",
    ".sql",
    ".toml", ".yaml", ".yml", ".json",
    ".md",
)

# Directories we never descend into when scanning followers. These are huge,
# vendored, or generated trees where a hit is overwhelmingly likely to be
# noise — and the maintainer's stated workflow is "grep src/, not node_modules".
_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "node_modules", ".venv", "venv", "env", ".env",
        "dist", "build", ".tox", ".nox", "site-packages",
        "target",  # rust / java
        ".next", ".nuxt", ".turbo",
        ".idea", ".vscode",
    }
)


def _iter_source_files(root: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    """Yield files under ``root`` matching ``suffixes`` outside excluded dirs.

    Walks via ``os.walk`` so we can prune ``_EXCLUDE_DIRS`` in place — using
    ``Path.rglob`` would descend into every venv before filtering.
    """
    suffix_set = {s.lower() for s in suffixes}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded directories in place so os.walk skips them.
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in suffix_set:
                yield Path(dirpath) / name


def _scan_follower(
    follower: Path,
    symbols: Sequence[str],
    suffixes: Sequence[str],
    display: str,
) -> list[StaleRef]:
    """Grep one follower repo for any of ``symbols``."""
    if not symbols:
        return []
    # Word-boundary regex so ``Foo`` doesn't match ``Foobar``.
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(s) for s in sorted(set(symbols))) + r")\b"
    )
    findings: list[StaleRef] = []
    for path in _iter_source_files(follower, suffixes):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line_no, line in enumerate(fh, start=1):
                    if not pattern.search(line):
                        continue
                    rel = path.relative_to(follower).as_posix()
                    for match in pattern.findall(line):
                        findings.append(
                            StaleRef(
                                symbol=match,
                                follower=display,
                                path=rel,
                                line_no=line_no,
                                line=line.rstrip("\n"),
                            )
                        )
        except OSError:
            continue
    return findings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sync_check(
    primary: str | Path,
    followers: Sequence[str | Path],
    base_ref: str = "HEAD~1",
    suffixes: Sequence[str] | None = None,
    removed_symbols: Sequence[str] | None = None,
) -> SyncReport:
    """Detect stale references in follower repos to symbols removed from primary.

    Parameters
    ----------
    primary:
        Path to the primary (SSOT) repo. Must be a git repo unless
        ``removed_symbols`` is supplied directly.
    followers:
        Paths to follower repos to scan. Followers do *not* need to be git
        repos — they are scanned as plain directories.
    base_ref:
        Git ref to diff against. Default ``HEAD~1`` catches the most recent
        commit. Use ``main`` or a SHA to scan a wider window. Ignored when
        ``removed_symbols`` is provided.
    suffixes:
        File suffixes to scan (case-insensitive). Defaults to a polyglot set
        covering Python / JS / TS / Go / Rust / Java / Kotlin / Ruby / SQL /
        config / docs.
    removed_symbols:
        Explicit list of symbols to grep for. When provided, the git diff
        step is skipped — useful for tests and for callers that derive the
        list elsewhere (e.g. AST diff).

    Returns
    -------
    A :class:`SyncReport`. ``findings`` is empty when no stale refs were
    detected. ``error`` is populated only on git failure when a diff was
    requested — follower IO errors are absorbed silently per-file (matching
    ``lint_canary``'s "keep going" stance).

    Notes
    -----
    Symbols that still exist *anywhere* in the primary working tree are
    filtered out before scanning followers — this is the "no false positives
    for symbols that exist in both" acceptance criterion. A symbol counts as
    "exists" if any source file under ``primary`` (excluding the same
    excluded directories applied to followers) contains it.
    """
    suffix_list = list(suffixes) if suffixes is not None else list(_DEFAULT_SUFFIXES)
    primary_path = Path(os.path.expanduser(str(primary))).resolve()
    report = SyncReport(
        primary=str(primary_path),
        base_ref=base_ref,
    )

    if not primary_path.is_dir():
        report.error = f"primary not a directory: {primary_path}"
        return report

    # Step 1: build the set of removed symbols.
    if removed_symbols is not None:
        candidates = {
            s for s in removed_symbols if s and s not in _NOISE_IDENTS
        }
    else:
        diff_text, err = _run_git_diff(primary_path, base_ref)
        if err:
            report.error = err
            return report
        candidates = _extract_removed_idents_from_diff(diff_text)

    # Step 2: filter out anything still present in the primary working tree.
    if candidates:
        present = _scan_current_idents(primary_path, suffix_list)
        candidates -= present

    report.removed_symbols = sorted(candidates)
    if not candidates:
        return report

    # Step 3: scan each follower.
    for follower in followers:
        follower_str = str(follower)
        follower_path = Path(os.path.expanduser(follower_str)).resolve()
        if not follower_path.is_dir():
            report.skipped_followers[follower_str] = "not a directory"
            continue
        # Display path: keep the caller-provided string when it's short, else
        # fall back to the resolved absolute path. The acceptance example shows
        # ``follower/src/foo.py:42`` so a relative-looking display is preferred.
        display = follower_str if len(follower_str) <= len(str(follower_path)) else str(follower_path)
        report.followers_scanned.append(display)
        report.findings.extend(
            _scan_follower(
                follower=follower_path,
                symbols=sorted(candidates),
                suffixes=suffix_list,
                display=display,
            )
        )

    return report


def format_report(report: SyncReport) -> str:
    """Human-readable multi-line summary for the MCP tool return value."""
    if report.error:
        return f"sync_check error: {report.error}"
    if not report.removed_symbols:
        return (
            f"sync_check: no removed symbols detected in {report.primary} "
            f"(base={report.base_ref})."
        )
    if not report.findings:
        return (
            f"sync_check: {len(report.removed_symbols)} removed symbol(s) "
            f"in primary; 0 stale refs in {len(report.followers_scanned)} "
            f"follower(s). Symbols: {', '.join(report.removed_symbols[:10])}"
            f"{' ...' if len(report.removed_symbols) > 10 else ''}"
        )
    lines = [
        f"sync_check: {len(report.findings)} stale ref(s) across "
        f"{len(report.followers_scanned)} follower(s) "
        f"(symbols removed in primary: {len(report.removed_symbols)})",
    ]
    for f in report.findings:
        lines.append(f.render())
    if report.skipped_followers:
        lines.append("")
        for follower, reason in report.skipped_followers.items():
            lines.append(f"skipped {follower}: {reason}")
    return "\n".join(lines)
