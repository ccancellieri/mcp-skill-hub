#!/usr/bin/env bash
# M1 #11 -- optional git post-merge hook.
#
# When a feature branch is merged into the integration branch and then
# deleted (the common `git merge && git branch -d` flow), close any open
# skill-hub task that was recorded against that branch. The task lives on
# in the database (status=closed) so its compaction stays searchable, but
# it stops showing up in `list_tasks` for the now-defunct workspace.
#
# Installation (per repository):
#   ln -s "$(realpath hooks/post-merge.sh)" .git/hooks/post-merge
#   chmod +x .git/hooks/post-merge
#
# Or copy it: `cp hooks/post-merge.sh .git/hooks/post-merge && chmod +x ...`
#
# Behaviour:
#   * Best-effort -- swallows errors and exits 0 so a misconfigured hook
#     never blocks a merge.
#   * Idempotent -- the CLI checks `git rev-parse --verify refs/heads/<b>`
#     and skips when the branch still exists (i.e. merge-without-delete).
#   * Reads the merged branch from ORIG_HEAD's reflog message; falls back
#     to walking every closed reference if that lookup fails.
set -u

# Locate skill-hub-cli; degrade silently when not installed.
CLI="$(command -v skill-hub-cli || true)"
if [ -z "$CLI" ]; then
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
  exit 0
fi

# Best-effort merged-branch discovery. git post-merge does not pass the
# branch name as an argument, so we mine ORIG_HEAD's reflog -- merges write
# entries like "merge feature/foo: ..." or "pull: Fast-forward".
MERGED_BRANCH=""
if git -C "$REPO_ROOT" rev-parse --verify ORIG_HEAD >/dev/null 2>&1; then
  MERGED_BRANCH="$(git -C "$REPO_ROOT" reflog show --no-abbrev -n 1 ORIG_HEAD 2>/dev/null \
    | sed -n -E 's/^.*: (merge|pull) ([^:]+):.*$/\2/p' | head -n 1)"
fi

# Inspect every open task -- the deleted branch may not match ORIG_HEAD
# (e.g. when a maintainer deletes a stale local branch via `git branch -d`
# without an immediate merge). The CLI's existence check guarantees we
# only close tasks whose branch is genuinely gone.
ALL_BRANCHES="$(skill-hub-cli list_tasks open 2>/dev/null | awk -F'#' '/\[OPEN\]/{print $2}' | awk '{print $1}' | tr -d ' ')"

# If we got a hint from ORIG_HEAD, prioritise it; otherwise check every
# branch referenced by an open task.
if [ -n "$MERGED_BRANCH" ]; then
  "$CLI" close_tasks_for_branch "$MERGED_BRANCH" --cwd "$REPO_ROOT" >/dev/null 2>&1 || true
fi

# Walk every open task and let the CLI's existence check decide.
# We list all open tasks' branches via a dedicated SQL helper rather than
# parsing list_tasks output; this is the canonical path.
python3 - "$REPO_ROOT" "$CLI" <<'PYEOF' || true
import os, subprocess, sys
repo, cli = sys.argv[1], sys.argv[2]
try:
    from skill_hub.store import SkillStore
except Exception:
    sys.exit(0)
store = SkillStore()
seen = set()
try:
    for r in store.list_tasks("open"):
        try:
            br = r["branch"]
        except Exception:
            br = None
        if not br or br in seen:
            continue
        seen.add(br)
        # Only act on branches that no longer exist locally.
        rc = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--verify", f"refs/heads/{br}"],
            capture_output=True, text=True, check=False,
        )
        if rc.returncode == 0:
            continue
        subprocess.run(
            [cli, "close_tasks_for_branch", br, "--cwd", repo],
            check=False, capture_output=True,
        )
finally:
    store.close()
PYEOF

exit 0
