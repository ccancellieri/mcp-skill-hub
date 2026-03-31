#!/bin/bash
# Stop hook: log which MCP tools were used during the session.
# Reads the hook input JSON from stdin, extracts tool usage,
# and calls skill-hub's log_session tool.
#
# Install in ~/.claude/settings.json under hooks.Stop:
# {
#   "type": "command",
#   "command": "/Users/ccancellieri/work/code/mcp-skill-hub/hooks/session-logger.sh",
#   "statusMessage": "Logging session for skill-hub learning..."
# }
#
# The hook receives session context via stdin as JSON.
# We extract tool names and write them to a log file that
# skill-hub can import on next startup.

LOG_DIR="$HOME/.claude/mcp-skill-hub/session-logs"
mkdir -p "$LOG_DIR"

SESSION_ID=$(date +%s)-$$
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="$LOG_DIR/$SESSION_ID.json"

# Read stdin (hook input)
INPUT=$(cat)

# Write raw session data for skill-hub to process
cat > "$LOG_FILE" << HEREDOC
{
  "session_id": "$SESSION_ID",
  "timestamp": "$TIMESTAMP",
  "hook_input": $INPUT
}
HEREDOC

exit 0
