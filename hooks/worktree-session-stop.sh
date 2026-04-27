#!/usr/bin/env bash
# Stop hook: remove <cwd>/.claude/session.pid when the Claude session exits.
# Wired into per-worktree .claude/settings.local.json by skill_hub.worktree.
set -eu
PIDFILE="${CLAUDE_PROJECT_DIR:-$PWD}/.claude/session.pid"
[ -f "$PIDFILE" ] && rm -f "$PIDFILE"
exit 0
