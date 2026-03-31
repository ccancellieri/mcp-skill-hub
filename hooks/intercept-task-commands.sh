#!/bin/bash
# UserPromptSubmit hook: intercept task/memory commands before Claude sees them.
#
# Flow:
#   1. Every user message passes through this hook BEFORE Claude sees it
#   2. Python CLI does a fast embedding similarity check (~100ms):
#      - Very long messages (>400 chars) → skip immediately (coding questions)
#      - Short messages → embed and compare to canonical task phrases
#      - Below similarity threshold → allow through immediately
#      - Above threshold → call local LLM for precise classification (~2-5s)
#   3. If it's a task command → execute locally, block Claude (0 tokens used)
#   4. If not → allow through to Claude

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$SCRIPT_DIR/.venv/bin/skill-hub-cli"

# Read hook input from stdin
INPUT=$(cat)

# Extract user message
MESSAGE=$(echo "$INPUT" | python3 -c "
import sys, json
print(json.load(sys.stdin).get('userMessage', ''))
" 2>/dev/null)

if [ -z "$MESSAGE" ]; then
    exit 0
fi

# Classify via local LLM (Python handles the fast semantic prefilter internally)
RESULT=$("$CLI" classify "$MESSAGE" 2>/dev/null)

if [ $? -ne 0 ] || [ -z "$RESULT" ]; then
    exit 0  # CLI failed — allow through
fi

DECISION=$(echo "$RESULT" | python3 -c "
import sys, json
print(json.load(sys.stdin).get('decision', 'allow'))
" 2>/dev/null)

if [ "$DECISION" = "block" ]; then
    # Use python to safely serialise the feedback as JSON (handles quotes/newlines)
    echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(json.dumps({'decision': 'block', 'message': data.get('message', 'Command handled locally.')}))
"
fi

exit 0
