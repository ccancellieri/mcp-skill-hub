#!/bin/bash
# Session-start protocol enforcer.
# On the first user prompt of each session, injects a systemMessage
# reminding Claude to execute the mandatory session-start checklist.
#
# Cost: 0 LLM tokens (local or remote). Just a file-existence check.
# Cleanup: flags older than 24h are auto-pruned.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | python3 -c "
import sys, json
print(json.load(sys.stdin).get('session_id', ''))
" 2>/dev/null)

FLAG="/tmp/claude-session-started-${SESSION_ID:-unknown}"
LOG_PATH="$HOME/.claude/mcp-skill-hub/logs/hook-debug.log"

if [ ! -f "$FLAG" ]; then
    touch "$FLAG"
    # Prune stale flags in background
    find /tmp -name "claude-session-started-*" -mtime +1 -delete 2>/dev/null &

    LOG_CMD="tail -f $LOG_PATH"

    python3 -c "
import json
msg = (
    'SESSION START:\n'
    '1. Read project .memory/index.md if it exists in the working directory\n'
    '2. Follow CLAUDE.md multi-level context protocol\n'
    '\n'
    'Hook activity log: $LOG_CMD\n'
    'Mention the log command to the user so they can follow local LLM activity.'
)
print(json.dumps({'decision': 'allow', 'systemMessage': msg}))
"
fi
