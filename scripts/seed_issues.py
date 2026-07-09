#!/usr/bin/env python3
"""Idempotent seeder for mcp-skill-hub roadmap milestones + issues.

Run from anywhere; targets ccancellieri/mcp-skill-hub by default (override with $REPO).
Re-runs are safe — skips milestones and issues that already exist (matched by title).

Requires: gh CLI (authenticated).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

REPO = os.environ.get("REPO", "ccancellieri/mcp-skill-hub")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def gh_api(path: str, method: str = "GET", **fields: str) -> dict | list:
    cmd = ["gh", "api", f"repos/{REPO}/{path}", "--method", method]
    for k, v in fields.items():
        cmd.extend(["-f", f"{k}={v}"])
    proc = run(*cmd)
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


# --- Milestones --------------------------------------------------------------

MILESTONES = [
    ("M1: No-LLM", "Useful Without LLM — visibility + pure-stdlib tools"),
    ("M2: Managed Agents", "Managed-Agents architectural refactor (design phase)"),
    ("M3: Worktree Policy", "Worktree + multi-repo policy enforcement"),
]


def ensure_milestones() -> dict[str, int]:
    """Return {title: number}."""
    existing = gh_api("milestones?state=all&per_page=100")
    by_title = {m["title"]: m["number"] for m in existing}
    for title, description in MILESTONES:
        if title in by_title:
            print(f"  milestone exists: {title} (#{by_title[title]})")
            continue
        created = gh_api("milestones", method="POST", title=title, description=description)
        by_title[title] = created["number"]
        print(f"  milestone created: {title} (#{created['number']})")
    return by_title


# --- Issues -----------------------------------------------------------------

@dataclass
class Issue:
    title: str
    milestone: str
    labels: list[str]
    body: str
    slug: str = ""

    def __post_init__(self) -> None:
        if not self.slug:
            self.slug = self.title.lower().split()[0]


def _body(*paragraphs: str) -> str:
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


# ----- M1 issues (8) -----

M1 = "M1: No-LLM"
M1_ISSUES: list[Issue] = [
    Issue(
        title="[m1] no-llm-mode: explicit flag with visible status",
        milestone=M1,
        labels=["m1-no-llm", "kind:feat", "effort:S", "degrades-gracefully", "area:tools"],
        body=_body(
            "## Motivation",
            "Users today perceive skill-hub as 'almost useless' without a local LLM, even though "
            "29 of 40 MCP tools work without any model. The graceful degradation exists but is "
            "invisible and inconsistent.",
            "## Proposal",
            "- Add `no_llm_mode: bool` to `config.json`.\n"
            "- When true: `embed_available()` returns false without probing; `status` reports "
            "`No-LLM mode (29/40 tools available)`; dashboard shows banner.\n"
            "- List exactly which tools are disabled vs available.",
            "## Files",
            "- `src/skill_hub/config.py`\n- `src/skill_hub/server.py:status`\n- `src/skill_hub/dashboard.py`",
            "## Acceptance",
            "- [ ] Flag persists across restarts.\n"
            "- [ ] Disabled tools return a clear user-facing error.\n"
            "- [ ] `status` output matches the tier registry from #m1-2.\n"
            "- [ ] Tests in `tests/test_no_llm_mode.py`.",
        ),
    ),
    Issue(
        title="[m1] tool-capability-matrix: every tool declares its dependency tier",
        milestone=M1,
        labels=["m1-no-llm", "kind:refactor", "effort:S", "area:tools"],
        body=_body(
            "## Motivation",
            "Today the tool/LLM dependency mapping is implicit. A registry makes it queryable and "
            "lets `status` / `list_skills` / `--help` show what works right now.",
            "## Proposal",
            "Add `@requires_capability(\"none\" | \"embedding\" | \"llm\")` decorator. "
            "Decorator records to a module-level registry consulted by other tools.",
            "## Files",
            "- new `src/skill_hub/capabilities.py`\n- decorator applied across `server.py` tool defs",
            "## Acceptance",
            "- [ ] Registry contains all 40 tools — no `unknown` tier.\n"
            "- [ ] Matches the feature inventory used in the roadmap planning.",
        ),
    ),
    Issue(
        title="[m1] degraded-search: FTS5 keyword fallback when embeddings unavailable",
        milestone=M1,
        labels=["m1-no-llm", "kind:feat", "effort:M", "degrades-gracefully", "area:tools"],
        body=_body(
            "## Motivation",
            "When no embedding backend is configured, `search_skills` / `search_context` / "
            "`suggest_plugins` currently error out. SQLite FTS5 is already a sibling of sqlite-vec — "
            "a keyword fallback gives non-zero quality results instead of nothing.",
            "## Proposal",
            "- Build FTS5 indexes alongside existing vector indexes.\n"
            "- When `embed_available()` is false, route searches to FTS5 instead of returning error.\n"
            "- Surface in result envelope which path was used.",
            "## Acceptance",
            "- [ ] With embeddings disabled, search still returns ranked results.\n"
            "- [ ] With embeddings enabled, behaviour unchanged.\n"
            "- [ ] Existing tests pass; new fallback tests cover both modes.",
        ),
    ),
    Issue(
        title="[m1] claims-board: claim / handoff / steal on tasks (no LLM needed)",
        milestone=M1,
        labels=["m1-no-llm", "kind:feat", "effort:M", "area:db", "area:tools"],
        body=_body(
            "## Motivation",
            "Tasks today are single-session bookmarks. A claims layer lets multiple Claude Code "
            "sessions coordinate work-item ownership without an LLM.",
            "## Proposal",
            "- Schema: add `claimed_by`, `claim_token`, `stealable_at` to `tasks`.\n"
            "- Tools: `claim_task(id, agent_id)`, `handoff_task(id, to)`, "
            "`steal_task(id)`, `release_task(id)`.\n"
            "- Pure SQLite — no embeddings.",
            "## Acceptance",
            "- [ ] Claim transitions validated (double-claim rejected).\n"
            "- [ ] Stealable expiry works.\n"
            "- [ ] Existing single-session task flow unchanged when `claimed_by` is null.",
        ),
    ),
    Issue(
        title="[m1] witness-log: append-only fix manifest per repo",
        milestone=M1,
        labels=["m1-no-llm", "kind:feat", "effort:M", "area:tools"],
        body=_body(
            "## Motivation",
            "Capturing 'what fix shipped where' as structured data lets the dashboard show a "
            "real fix log instead of relying on memory files. Append-only keeps the first "
            "implementation simple — no cryptographic signing yet.",
            "## Proposal",
            "- New core_task `witness`. Records `(issue, pr, sha, repo, kind, fix_summary)` "
            "tuples to `~/.skill_hub/witness.jsonl`. Append-only, never edited.\n"
            "- Query tool `list_witness(repo=..., since=...)`.",
            "## Acceptance",
            "- [ ] Append-only enforced (edits raise).\n"
            "- [ ] Filter by repo + since works.\n"
            "- [ ] JSONL parseable by stdlib.",
        ),
    ),
    Issue(
        title="[m1] worktree-aware tasks: capture branch + worktree path on save",
        milestone=M1,
        labels=["m1-no-llm", "m3-worktree-policy", "kind:feat", "effort:S", "area:tools"],
        body=_body(
            "## Motivation",
            "Implements the maintainer's worktree-first workflow as structure. Today the "
            "association between a task and its worktree lives only in conversation context — "
            "tasks survive context resets but their worktree binding does not.",
            "## Proposal",
            "- `save_task` auto-captures `git rev-parse --show-toplevel` + "
            "`git rev-parse --abbrev-ref HEAD` + worktree path.\n"
            "- New filter `list_tasks --worktree-current`.\n"
            "- Optional post-merge git hook (`hooks/post-merge.sh`) closes the matching task "
            "when its branch is deleted.",
            "## Acceptance",
            "- [ ] Worktree + branch recorded on save.\n"
            "- [ ] Mismatch detected when cwd != recorded worktree.\n"
            "- [ ] Post-merge hook closes tasks for deleted branches.",
        ),
    ),
    Issue(
        title="[m1] PII gate: regex scan before save_task / teach when repo is marked public",
        milestone=M1,
        labels=["m1-no-llm", "kind:feat", "effort:M", "area:tools"],
        body=_body(
            "## Motivation",
            "Defence-in-depth on top of git pre-commit hooks. Memory / task content can leak "
            "private IPs, GCP project IDs, Cloud Run revision names, or tokens into a public "
            "repo's `.skill-hub/` directory if not gated.",
            "## Proposal",
            "- Per-repo `.skill-hub/policy.yml` flag `public: true` triggers a regex scan "
            "(IPs, GCP project IDs, Cloud Run revisions, `sk-ant-…`, `ghp_…`) before persisting.\n"
            "- Block on match + show offending substrings.\n"
            "- Override flag for false positives.",
            "## Acceptance",
            "- [ ] Known PII patterns block.\n"
            "- [ ] Allowed content passes.\n"
            "- [ ] Override flag works and is logged.",
        ),
    ),
    Issue(
        title="[m1] dashboard: /status/capabilities view shows what works right now",
        milestone=M1,
        labels=["m1-no-llm", "kind:feat", "effort:S", "area:dashboard"],
        body=_body(
            "## Motivation",
            "Closes the 'is this useless without a local LLM?' question with one URL. Visual "
            "answer to the perception problem M1 is solving.",
            "## Proposal",
            "New route `/status/capabilities` — table of all 40 tools with green / yellow / red "
            "dot per current backend availability. Click a red tool → setup instructions for the "
            "backend it needs.",
            "## Acceptance",
            "- [ ] Renders for current backend state.\n"
            "- [ ] State matches `status` MCP tool output exactly.",
        ),
    ),
]

# ----- M2 issue (1 tracking) -----

M2 = "M2: Managed Agents"
M2_ISSUES: list[Issue] = [
    Issue(
        title="[m2] tracking: Managed-Agents architectural refactor — design phase",
        milestone=M2,
        labels=["m2-managed-agents", "kind:meta", "effort:L"],
        body=_body(
            "## Status",
            "**Design phase.** Sub-issues are filed only after the design doc has settled.",
            "## Motivation",
            "Anthropic's 'Managed Agents' engineering post describes architectural patterns "
            "(decoupled session/harness/sandbox, durable event log, stateless harness recovery) "
            "that map cleanly onto skill-hub's existing surface. Applying them selectively "
            "improves durability without breaking the existing tool surface.",
            "## Candidate workstreams (see `docs/design/managed-agents-refactor.md`)",
            "1. **Event log** — append-only `events` SQLite table; every tool call emits one event "
            "before mutating. Tables become derived projections.\n"
            "2. **Stateless recovery** — `wake_session(session_id)` rehydrates state purely from events.\n"
            "3. **Tool envelope** — uniform `ToolResult(stdout, structured, error)` across all tools.\n"
            "4. **Credential vault** — `python-keyring` for API tokens.\n"
            "5. **Sandbox interface** — optional `provision({resources})` for plan execution.",
            "## Decision gates",
            "- [ ] Design doc reviewed.\n"
            "- [ ] Open questions resolved (see doc).\n"
            "- [ ] Sub-issues filed against this milestone.",
            "## Reference",
            "Design doc: `docs/design/managed-agents-refactor.md` (in the same PR that creates this issue).",
        ),
    ),
]

# ----- M3 issues (5) -----

M3 = "M3: Worktree Policy"
M3_ISSUES: list[Issue] = [
    Issue(
        title="[m3] worktree-policy: pre-flight collision check tool",
        milestone=M3,
        labels=["m3-worktree-policy", "kind:feat", "effort:S", "area:tools"],
        body=_body(
            "## Motivation",
            "Encodes the worktree-naming-collision rule as a callable check rather than a "
            "rule the user re-reads at session start.",
            "## Proposal",
            "`worktree_preflight(issue_number, repo)` tool — runs `git worktree list`, "
            "`git branch --list`, and `gh issue view`. Returns 'safe to start' or "
            "'collision with worktree X / branch Y / open PR Z'.",
            "## Acceptance",
            "- [ ] Detect existing worktree + branch + open PR.\n"
            "- [ ] Clean state passes.\n"
            "- [ ] Sub-second turnaround.",
        ),
    ),
    Issue(
        title="[m3] three-repo-sync: cross-repo stale-import detector",
        milestone=M3,
        labels=["m3-worktree-policy", "kind:feat", "effort:M", "area:tools"],
        body=_body(
            "## Motivation",
            "Encodes the multi-repo sync directive (one logical system across N repos) as a "
            "callable check. Pure grep — no LLM required.",
            "## Proposal",
            "`sync_check(primary, followers=[...])` greps follower repos for symbols recently "
            "removed/renamed in primary. Reports `stale ref \"OldClass\" in follower/src/foo.py:42`.",
            "## Acceptance",
            "- [ ] Synthetic primary diff + follower files → detection accurate.\n"
            "- [ ] No false positives for symbols that exist in both.",
        ),
    ),
    Issue(
        title="[m3] lint-canary cadence: rotate through ruff selectors as a routine",
        milestone=M3,
        labels=["m3-worktree-policy", "kind:feat", "effort:S", "area:tools"],
        body=_body(
            "## Motivation",
            "The lint-as-canary cadence (rotating through ruff selectors F841 / F821 / B023 / "
            "S701 / RUF034 / RUF006 / B026 / …) is a known-good pattern for finding dormant "
            "defects. Making it a button is the durability win.",
            "## Proposal",
            "- New core_task `lint_canary`. Runs `ruff check --select <next>` where `<next>` "
            "rotates through a config-driven list.\n"
            "- Records findings to witness-log (#m1-5).",
            "## Acceptance",
            "- [ ] Rotation advances on each run.\n"
            "- [ ] Findings captured to witness log.",
        ),
    ),
    Issue(
        title="[m3] memory-rule export: render feedback_* memory files as per-repo POLICY.md",
        milestone=M3,
        labels=["m3-worktree-policy", "kind:feat", "effort:S", "area:tools", "area:docs"],
        body=_body(
            "## Motivation",
            "The maintainer's feedback memory rules are a source of policy but live outside the "
            "repo. Paraphrasing them into a per-repo `POLICY.md` (without committing path "
            "references to `~/.claude/`) makes the rules durable and discoverable.",
            "## Proposal",
            "- `export_policies()` reads memory feedback files, renders a per-repo POLICY.md "
            "in `.skill-hub/`.\n"
            "- Re-runs on memory change.\n"
            "- Paraphrases, never path-cites `~/.claude/` (respects the no-AI-paths-in-tracked-content rule).",
            "## Acceptance",
            "- [ ] Round-trip lossless modulo formatting.\n"
            "- [ ] Generated POLICY.md contains zero `~/.claude/` references.",
        ),
    ),
    Issue(
        title="[m3] cross-project task federation: per-repo filter on every task tool",
        milestone=M3,
        labels=["m3-worktree-policy", "kind:feat", "effort:M", "area:db", "area:tools"],
        body=_body(
            "## Motivation",
            "Task DB is process-global today. Without a per-repo filter, the user can't answer "
            "'what tasks are open for repo X right now?' without manually grepping.",
            "## Proposal",
            "- Add `repo` column (auto-captured from worktree, see #m1-6).\n"
            "- `--repo <name>` filter on every task tool.\n"
            "- Dashboard groups tasks by repo.",
            "## Acceptance",
            "- [ ] Filter accuracy across mixed-repo task set.\n"
            "- [ ] Dashboard grouping correct.",
        ),
    ),
]

ALL_ISSUES = M1_ISSUES + M2_ISSUES + M3_ISSUES


# --- Issue creation ---------------------------------------------------------

def existing_issue_titles() -> set[str]:
    titles: set[str] = set()
    page = 1
    while True:
        items = gh_api(f"issues?state=all&per_page=100&page={page}")
        if not items:
            break
        titles.update(item["title"] for item in items if "pull_request" not in item)
        if len(items) < 100:
            break
        page += 1
    return titles


def create_issue(issue: Issue, milestone_number: int) -> None:
    # `gh issue create --milestone` expects the title string (the --number arg
    # exists only on `gh issue edit`). milestone_number is kept on the signature
    # for parity with `ensure_milestones` but unused here.
    del milestone_number
    cmd = [
        "gh", "issue", "create",
        "--repo", REPO,
        "--title", issue.title,
        "--body", issue.body,
        "--milestone", issue.milestone,
    ]
    for label in issue.labels:
        cmd.extend(["--label", label])
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(f"  FAILED: {issue.title}")
        print(proc.stderr.strip())
        return
    print(f"  created: {proc.stdout.strip()}")


def main() -> int:
    if not subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0:
        print("gh is not authenticated. Run `gh auth login`.", file=sys.stderr)
        return 1
    print(f"Target repo: {REPO}")
    print("Ensuring milestones...")
    by_title = ensure_milestones()
    print()
    print("Loading existing issues to skip duplicates...")
    existing = existing_issue_titles()
    print(f"  {len(existing)} pre-existing issue/PR titles found.")
    print()
    print(f"Creating {len(ALL_ISSUES)} issues (skipping duplicates by title)...")
    for issue in ALL_ISSUES:
        if issue.title in existing:
            print(f"  skip (exists): {issue.title}")
            continue
        create_issue(issue, by_title[issue.milestone])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
