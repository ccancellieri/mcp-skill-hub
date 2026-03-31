#!/bin/bash
# UserPromptSubmit hook: intercept task/memory commands before Claude sees them.
#
# Flow:
#   1. User types "save to memory and close" (or similar)
#   2. This hook fires BEFORE Claude processes the message
#   3. Local LLM (deepseek-r1:1.5b) classifies the intent
#   4. If it's a task command → execute locally, return {"decision":"block"} → 0 Claude tokens
#   5. If not → return nothing, Claude processes normally
#
# Install in ~/.claude/settings.json:
# {
#   "hooks": {
#     "UserPromptSubmit": [{
#       "hooks": [{
#         "type": "command",
#         "command": "/path/to/mcp-skill-hub/hooks/intercept-task-commands.sh",
#         "timeout": 45,
#         "statusMessage": "Checking for task commands..."
#       }]
#     }]
#   }
# }

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$SCRIPT_DIR/.venv/bin/skill-hub-cli"

# Read hook input from stdin
INPUT=$(cat)

# Extract user message from hook JSON
# UserPromptSubmit input format: {"userMessage": "..."}
MESSAGE=$(echo "$INPUT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('userMessage', ''))" 2>/dev/null)

if [ -z "$MESSAGE" ]; then
    exit 0  # No message, allow through
fi

# Quick keyword pre-filter — skip the LLM call for obvious non-commands.
# This saves ~2s for normal messages that clearly aren't task commands.
LOWER_MSG=$(echo "$MESSAGE" | tr '[:upper:]' '[:lower:]')
case "$LOWER_MSG" in
    *"save to memory"*|*"save task"*|*"park this"*|*"remember this"*|\
    *"close task"*|*"done with this"*|*"mark as done"*|*"save and close"*|\
    *"what was i working on"*|*"show tasks"*|*"open tasks"*|*"list tasks"*|\
    *"what did we discuss"*|*"find my previous"*|*"past work"*)
        # Likely a task command — classify with LLM
        ;;
    *)
        # Not a task command — skip LLM, allow through immediately
        exit 0
        ;;
esac

# Classify and optionally execute via local LLM
RESULT=$("$CLI" classify "$MESSAGE" 2>/dev/null)

if [ $? -ne 0 ] || [ -z "$RESULT" ]; then
    exit 0  # CLI failed, allow through
fi

# Check decision
DECISION=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('decision', 'allow'))" 2>/dev/null)

if [ "$DECISION" = "block" ]; then
    # Extract the message to show the user
    FEEDBACK=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Command handled locally.'))" 2>/dev/null)
    # Return block decision with feedback
    echo "{\"decision\": \"block\", \"message\": \"$FEEDBACK\"}"
    exit 0
fi

# Allow through to Claude
exit 0
