"""Hard policy guards — glob-based escalation rules checked before LLM dispatch.

These are *deterministic* (unlike semantic teachings). When any file in a
step's ``files`` list matches a guard glob, the step is either:

  - ``escalate``: stop immediately, return status="escalated" (human / Opus must re-plan)
  - ``force_tier_smart``: upgrade routing to Sonnet regardless of kind

Guards are sourced from:
  1. Built-in defaults (bundled with the package — safety-critical rules).
  2. Optional user file at ~/.claude/plan_guards.yaml (project-specific rules,
     merged on top).

Guards are project-agnostic; the schema supports an optional ``repo_glob``
that scopes a rule to a specific repo path when it matters.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


GuardAction = Literal["escalate", "force_tier_smart"]
USER_GUARDS_PATH = Path.home() / ".claude" / "plan_guards.yaml"


@dataclass
class GuardRule:
    name: str
    action: GuardAction
    file_globs: list[str]         # any match in step.files triggers the rule
    content_patterns: list[str]   # any substring found in written content triggers (optional)
    reason: str
    repo_glob: str | None = None  # if set, only applies when repo_path matches


@dataclass
class GuardMatch:
    rule: GuardRule
    matched_file: str | None       # file path that triggered (if glob-based)
    matched_content: str | None    # substring that triggered (if content-based)

    def as_message(self) -> str:
        if self.matched_file:
            return (
                f"[guard:{self.rule.name}] {self.rule.reason} "
                f"(matched file: {self.matched_file})"
            )
        return (
            f"[guard:{self.rule.name}] {self.rule.reason} "
            f"(matched content: {self.matched_content!r})"
        )


# --- Built-in defaults --------------------------------------------------------
# These mirror cross-cutting safety rules every user benefits from. Keep tight;
# per-repo guards go in the user YAML.

DEFAULT_GUARDS: list[GuardRule] = [
    GuardRule(
        name="no-ai-attribution",
        action="escalate",
        file_globs=[],
        content_patterns=["Co-Authored-By: Claude", "Generated with Claude"],
        reason="AI attribution strings are forbidden in commits/code",
    ),
    GuardRule(
        name="no-claude-context-commits",
        action="escalate",
        file_globs=[
            "CLAUDE.md", "**/CLAUDE.md",
            "AGENTS.md", "**/AGENTS.md",
            "GEMINI.md", "**/GEMINI.md",
            ".claude/**",
            ".cursorrules",
            ".aider*",
        ],
        content_patterns=[],
        reason="Agent context files must not be committed to repos",
    ),
    GuardRule(
        name="no-secret-files",
        action="escalate",
        file_globs=[
            "**/.env", "**/.env.*",
            "**/*credentials*", "**/*secret*",
            "**/*.pem", "**/*.key",
        ],
        content_patterns=[],
        reason="Secrets and credentials must never be written by the executor",
    ),
    GuardRule(
        name="no-proactive-migrations",
        action="escalate",
        file_globs=["**/v[0-9][0-9][0-9][0-9]__*.sql"],
        content_patterns=[],
        reason="Migration files must not be generated proactively — use CREATE TABLE IF NOT EXISTS",
    ),
]


def _load_user_guards(path: Path) -> list[GuardRule]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    raw_rules = data.get("guards") if isinstance(data, dict) else None
    if not isinstance(raw_rules, list):
        return []
    out: list[GuardRule] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        name = r.get("name")
        action = r.get("action")
        if not isinstance(name, str) or action not in ("escalate", "force_tier_smart"):
            continue
        out.append(
            GuardRule(
                name=name,
                action=action,
                file_globs=list(r.get("file_globs") or []),
                content_patterns=list(r.get("content_patterns") or []),
                reason=str(r.get("reason") or ""),
                repo_glob=r.get("repo_glob"),
            )
        )
    return out


def load_guards(user_path: Path | None = None) -> list[GuardRule]:
    """Return defaults merged with user YAML. User rules appended (not overriding)."""
    user = _load_user_guards(user_path or USER_GUARDS_PATH)
    return list(DEFAULT_GUARDS) + user


def _match_one(rel_path: str, pattern: str) -> bool:
    """fnmatch plus recursive-glob shim: ``**/x`` also matches bare ``x``."""
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    if pattern.startswith("**/"):
        # Match the bare suffix against the full path (any depth incl. 0).
        suffix = pattern[3:]
        if fnmatch.fnmatch(rel_path, suffix):
            return True
        # Also match against any path component ending with the suffix.
        if fnmatch.fnmatch(rel_path, f"*/{suffix}"):
            return True
    if "/**/" in pattern:
        # Split on `/**/` and require the parts to match as prefix and suffix.
        head, _, tail = pattern.partition("/**/")
        # Greedy: try collapsing `/**/` to `/` and to anything in between.
        if fnmatch.fnmatch(rel_path, f"{head}/{tail}"):
            return True
        if fnmatch.fnmatch(rel_path, f"{head}/*/{tail}"):
            return True
    return False


def _glob_matches(rel_path: str, patterns: list[str]) -> bool:
    return any(_match_one(rel_path, p) for p in patterns)


def check_pre_dispatch(
    step_files: list[str],
    repo_path: Path,
    guards: list[GuardRule],
) -> list[GuardMatch]:
    """Check guards that can fire before the LLM runs — purely file-path based.

    Content-pattern guards are handled separately via ``check_post_dispatch``
    (after the LLM returns proposed file contents).
    """
    matches: list[GuardMatch] = []
    repo_str = str(repo_path)
    for rule in guards:
        if rule.repo_glob and not fnmatch.fnmatch(repo_str, rule.repo_glob):
            continue
        if not rule.file_globs:
            continue
        for f in step_files:
            if _glob_matches(f, rule.file_globs):
                matches.append(GuardMatch(rule=rule, matched_file=f, matched_content=None))
                break
    return matches


def check_post_dispatch(
    changes: dict[str, str],
    repo_path: Path,
    guards: list[GuardRule],
) -> list[GuardMatch]:
    """Check content-pattern guards against the file contents the LLM returned."""
    matches: list[GuardMatch] = []
    repo_str = str(repo_path)
    for rule in guards:
        if rule.repo_glob and not fnmatch.fnmatch(repo_str, rule.repo_glob):
            continue
        if not rule.content_patterns:
            continue
        for _path, content in changes.items():
            for pat in rule.content_patterns:
                if pat in content:
                    matches.append(
                        GuardMatch(rule=rule, matched_file=None, matched_content=pat)
                    )
                    break
            else:
                continue
            break
    return matches


def first_escalation(matches: list[GuardMatch]) -> GuardMatch | None:
    for m in matches:
        if m.rule.action == "escalate":
            return m
    return None


def any_force_smart(matches: list[GuardMatch]) -> bool:
    return any(m.rule.action == "force_tier_smart" for m in matches)
