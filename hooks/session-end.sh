#!/bin/bash
# Stop hook: save session memory + stats when Claude finishes responding.
#
# Smart routing:
#   1. Local LLM generates memory from session context
#   2. Quality check: if too much detail lost → returns systemMessage
#      asking Claude to write the memory instead
#   3. Logs session stats (messages, interceptions, tokens saved)

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$SCRIPT_DIR/.venv/bin/skill-hub-cli"
DEBUG_LOG="$HOME/.claude/mcp-skill-hub/logs/hook-debug.log"

# Read hook input from stdin
INPUT=$(cat)

echo "[$(date '+%H:%M:%S')] Stop hook raw input: ${INPUT:0:500}" >> "$DEBUG_LOG"

# Extract fields from hook JSON — use IFS=tab to preserve spaces in values
IFS=$'\t' read -r SESSION_ID LAST_MSG TRANSCRIPT STOP_ACTIVE <<< "$(echo "$INPUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
sid = d.get('session_id', '')
last = d.get('last_assistant_message', '')[:2000]
# Escape newlines/tabs for shell
last = last.replace(chr(10), ' ').replace(chr(13), '').replace(chr(9), ' ')
transcript = d.get('transcript_path', '')
active = d.get('stop_hook_active', False)
print(sid + chr(9) + last + chr(9) + transcript + chr(9) + str(active))
" 2>/dev/null)"

echo "[$(date '+%H:%M:%S')] Stop hook — session=$SESSION_ID active=$STOP_ACTIVE" >> "$DEBUG_LOG"

# Don't re-enter if already in a stop hook cycle
if [ "$STOP_ACTIVE" = "True" ]; then
    echo "[$(date '+%H:%M:%S')] Stop hook already active, skipping" >> "$DEBUG_LOG"
    exit 0
fi

# Skip if no session context
if [ -z "$SESSION_ID" ] && [ -z "$LAST_MSG" ]; then
    exit 0
fi

# Run session_end via CLI (async-safe, writes to DB + files)
RESULT=$("$CLI" session_end \
    --session-id "$SESSION_ID" \
    --last-message "$LAST_MSG" \
    --transcript "$TRANSCRIPT" 2>>"$DEBUG_LOG")
CLI_EXIT=$?

echo "[$(date '+%H:%M:%S')] session_end exit=$CLI_EXIT result_len=${#RESULT}" >> "$DEBUG_LOG"

if [ $CLI_EXIT -ne 0 ] || [ -z "$RESULT" ]; then
    exit 0
fi

# Check if there's a systemMessage to forward to Claude
HAS_SYSTEM=$(echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('yes' if d.get('systemMessage') else 'no')
" 2>/dev/null)

if [ "$HAS_SYSTEM" = "yes" ]; then
    echo "$RESULT"
fi

exit 0
