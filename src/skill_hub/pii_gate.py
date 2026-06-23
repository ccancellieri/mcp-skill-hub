"""PII gate — regex scan before save_task / teach when repo is marked public.

Defense-in-depth on top of git pre-commit hooks. Memory / task content can leak
private IPs, GCP project IDs, Cloud Run revision names, or API tokens into a
public repo's ``.skill-hub/`` directory if not gated.

Per-repo opt-in
---------------
The gate activates only when a repo carries ``public: true`` in its
``.skill-hub/policy.yml``::

    public: true
    # optional: override the gate per-call by passing override=True to the
    # caller, but log that override.
    pii_overrides:
      - "10.0.0.1"   # known false positives (optional)

If ``policy.yml`` is missing or ``public`` is falsy, the gate is a no-op so
private repos pay zero cost.

Patterns
--------
We block on these high-confidence patterns (false-positive aware):

  * IPv4 private/public — RFC1918 + general dotted quads
  * GCP project IDs     — ``my-project-12345`` style (lower-kebab, 6-30 chars,
                          ends with digits — common GCP convention)
  * Cloud Run revisions — ``service-name-00001-abc``
  * Anthropic API keys  — ``sk-ant-…`` prefix
  * GitHub tokens       — ``ghp_…`` prefix
  * Email addresses     — ``local@domain.tld`` (personal/work contact PII)
  * Phone numbers       — ``+39 338 200 3690`` style (optional country code,
                          7+ separated digits)
  * Generic bearer tokens that look like long opaque strings are *not* gated
    here — too many false positives. Add to ``pii_overrides`` if needed in
    reverse (extra patterns) — not implemented in M1.

Public surface
--------------
  * ``scan(content)``  → list[Hit]  (always runs; cheap)
  * ``check_for_save(content, repo_root, override=False)``
      Returns ``GateResult`` with allow/block + diagnostic. The caller (server)
      surfaces the block as a refused tool response.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_log = logging.getLogger(__name__)

POLICY_REL = ".skill-hub/policy.yml"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# IPv4 dotted-quad. We tighten with a word-boundary so we don't match version
# strings like "1.2.3.4-rc". This will still catch legitimate IPs in code
# comments — that's the point.
_IPV4 = re.compile(
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"
)

# Anthropic key prefix. The full key is opaque; we match the prefix + at least
# 10 chars of body so we don't false-trigger on the literal word "sk-ant-".
_SK_ANT = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")

# GitHub personal-access token. Same shape.
_GHP = re.compile(r"ghp_[A-Za-z0-9]{20,}")

# GitHub fine-grained, OAuth, and app tokens — same family.
_GH_TOKEN = re.compile(r"(?:github_pat_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]{20,}")

# GCP project IDs: 6-30 chars, lowercase letters/digits/hyphens, must start
# with a letter, and must contain at least one digit. The digit requirement
# avoids matching ordinary kebab-case English ("my-project-name"). We also
# anchor on a left word boundary that excludes hyphens, so we don't match
# inside a longer slug like "feedback-no-intermodule-deps".
_GCP_PROJECT = re.compile(
    r"(?<![\w-])"
    r"(?=[a-z0-9-]{6,30}(?![\w-]))"  # length 6-30 lookahead
    r"[a-z][a-z0-9-]*\d[a-z0-9-]*"   # starts with letter, contains a digit
    r"(?![\w-])"
)

# Cloud Run revision: <service>-<5-digit-rev>-<3-char-suffix>, e.g.
# api-prod-00042-abc. The 5-digit + 3-alpha suffix is very specific to Cloud
# Run and won't match arbitrary strings.
_CLOUD_RUN_REV = re.compile(
    r"(?<![\w-])[a-z][a-z0-9-]{0,40}-\d{5}-[a-z0-9]{3}(?![\w-])"
)

# Email address. Standard local@domain.tld shape. Catches personal/work
# addresses (the kind that leaked via a public career-profile ref). The left
# boundary excludes characters that are themselves valid in a local-part so we
# match the whole address, not a tail of it.
_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w])"
)

# Phone number. Optional ``+`` country code then 7+ phone characters
# (digits, spaces, parens, dashes) ending in a digit — e.g. "+39 338 200 3690".
# We exclude ``.`` from the body so dotted version strings and IPv4 addresses
# (handled by _IPV4) don't double-match. Mirrors the website-generator
# scrubber's phone shape for consistency across the codebase.
_PHONE = re.compile(
    r"(?<![\w])\+?\d[\d\s()-]{7,}\d(?![\w])"
)


@dataclass
class Hit:
    pattern: str
    match: str
    start: int
    end: int


_PATTERN_TABLE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sk_ant_key", _SK_ANT),
    ("github_token", _GHP),
    ("github_token", _GH_TOKEN),
    ("email", _EMAIL),
    ("cloud_run_revision", _CLOUD_RUN_REV),
    ("gcp_project_id", _GCP_PROJECT),
    ("ipv4", _IPV4),
    ("phone", _PHONE),
)


def scan(content: str, *, overrides: Iterable[str] = ()) -> list[Hit]:
    """Run every PII pattern against `content`; return all hits.

    `overrides` is a set of literal substrings that, when matched exactly,
    suppress that hit. Use for documented false positives (e.g. the example
    IP ``1.2.3.4`` in a docstring).

    Order of hits in the result is insertion order across patterns — useful
    for human-readable error messages where the first hit is shown first.
    """
    if not content:
        return []
    override_set = {o.strip() for o in overrides if o and o.strip()}
    out: list[Hit] = []
    seen: set[tuple[int, int]] = set()
    for label, pat in _PATTERN_TABLE:
        for m in pat.finditer(content):
            span = (m.start(), m.end())
            if span in seen:
                continue
            text = m.group(0)
            if text in override_set:
                continue
            seen.add(span)
            out.append(Hit(pattern=label, match=text, start=m.start(), end=m.end()))
    return out


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def _load_policy(repo_root: Path) -> dict[str, Any]:
    """Read ``.skill-hub/policy.yml`` from `repo_root`. Returns {} when missing
    or malformed — the gate is opt-in so silent fallback to "private" is safe.
    """
    policy_path = repo_root / POLICY_REL
    if not policy_path.exists():
        return {}
    try:
        import yaml  # local import — pyyaml is already a dep
    except ImportError:  # pragma: no cover
        _log.warning("pyyaml not installed; PII gate cannot read %s", policy_path)
        return {}
    try:
        text = policy_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, yaml.YAMLError) as exc:
        _log.warning("PII gate: cannot parse %s: %s", policy_path, exc)
        return {}


def is_public(repo_root: Path) -> bool:
    """True if `repo_root/.skill-hub/policy.yml` declares ``public: true``."""
    return bool(_load_policy(repo_root).get("public"))


def _overrides(repo_root: Path) -> list[str]:
    raw = _load_policy(repo_root).get("pii_overrides") or []
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


# ---------------------------------------------------------------------------
# Public gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    allowed: bool
    reason: str = ""
    hits: list[Hit] = field(default_factory=list)
    override_used: bool = False
    public: bool = False

    def format_block_message(self) -> str:
        """Render a refusal message suitable for surfacing through an MCP tool."""
        if self.allowed:
            return ""
        lines = [
            "Refused: content contains likely-private values and this repo is "
            "marked public in .skill-hub/policy.yml.",
            "",
            "Offending substrings:",
        ]
        for h in self.hits[:8]:
            lines.append(f"  - [{h.pattern}] {h.match!r}")
        if len(self.hits) > 8:
            lines.append(f"  - ... and {len(self.hits) - 8} more")
        lines.append("")
        lines.append(
            "To proceed: redact the values, or add them to `pii_overrides:` "
            "in .skill-hub/policy.yml if they are confirmed false positives, "
            "or pass override=True (the override is logged)."
        )
        return "\n".join(lines)


def check(
    content: str,
    repo_root: str | Path | None,
    *,
    override: bool = False,
) -> GateResult:
    """Decide whether `content` may be persisted given `repo_root`'s policy.

    `repo_root` may be None (e.g. caller didn't pass a cwd / repo). In that
    case the gate is a no-op: we cannot know whether the destination is public
    without a policy file, and forcing every caller to set one would break
    private-repo workflows.

    When the repo is public **and** override=True, we still scan and return
    the hits but mark the result allowed; the caller is responsible for
    logging the override (see `log_override`).
    """
    if not repo_root:
        return GateResult(allowed=True, reason="no repo context")
    root = Path(os.path.expanduser(str(repo_root))).resolve()
    if not root.is_dir():
        return GateResult(allowed=True, reason=f"repo_root not a directory: {root}")

    policy = _load_policy(root)
    public = bool(policy.get("public"))
    if not public:
        return GateResult(allowed=True, reason="repo not marked public", public=False)

    overrides = policy.get("pii_overrides") or []
    if not isinstance(overrides, list):
        overrides = []
    hits = scan(content, overrides=[str(o) for o in overrides])

    if not hits:
        return GateResult(allowed=True, reason="no PII detected", public=True)

    if override:
        return GateResult(
            allowed=True,
            reason="PII detected but override=True",
            hits=hits,
            override_used=True,
            public=True,
        )

    return GateResult(
        allowed=False,
        reason="PII detected and repo is public",
        hits=hits,
        public=True,
    )


def resolve_repo_root(
    *, cwd: str = "", project: str = "", repo: str = "",
) -> Path | None:
    """Best-effort: locate the filesystem root of the repo a save/teach call
    targets, for the PII gate's policy.yml lookup.

    Priority:
      1. ``cwd`` walked up to the nearest ``.git`` directory.
      2. ``project`` or ``repo`` name resolved against worktree.repo_roots.
      3. ``None`` when nothing matches — the gate then becomes a no-op.
    """
    if cwd:
        try:
            p = Path(cwd).expanduser().resolve()
            cur = p
            while True:
                if (cur / ".git").exists():
                    return cur
                if cur.parent == cur:
                    break
                cur = cur.parent
        except Exception:  # noqa: BLE001
            pass
    name = (project or repo or "").strip()
    if name:
        try:
            from . import worktree as _wt
            return _wt.resolve_project(name)
        except Exception:  # noqa: BLE001
            return None
    return None


def enforce(
    *, tool: str, content: str, cwd: str = "", project: str = "",
    repo: str = "", override: bool = False,
) -> tuple[bool, str]:
    """Run the per-repo PII gate; on block returns (False, refusal_message).

    On override use, logs the override to ``.skill-hub/pii_overrides.log``.
    Always returns (True, "") for repos without a ``public: true`` policy.
    """
    root = resolve_repo_root(cwd=cwd, project=project, repo=repo)
    if root is None:
        return True, ""
    result = check(content, root, override=override)
    if not result.allowed:
        return False, result.format_block_message()
    if result.override_used:
        try:
            log_override(root, tool, result.hits)
        except OSError:
            pass
    return True, ""


def log_override(
    repo_root: str | Path,
    tool: str,
    hits: list[Hit],
    *,
    log_path: str | Path | None = None,
) -> Path:
    """Append an override entry to ``<repo_root>/.skill-hub/pii_overrides.log``.

    The log is intentionally per-repo and human-readable. It records:
      * ISO timestamp
      * tool name (save_task / teach / …)
      * pattern labels of the hits (never the matched substrings — those
        could themselves be sensitive)

    Returns the path written to.
    """
    import datetime as _dt
    root = Path(os.path.expanduser(str(repo_root))).resolve()
    if log_path is None:
        log_path = root / ".skill-hub" / "pii_overrides.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    labels = ",".join(sorted({h.pattern for h in hits})) or "none"
    line = f"{ts}\t{tool}\t{labels}\t{len(hits)} hit(s)\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
    return log_path
