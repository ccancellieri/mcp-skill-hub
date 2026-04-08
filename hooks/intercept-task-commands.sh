#!/bin/bash
# UserPromptSubmit hook: intercept task/memory commands before Claude sees them.
#
# Flow:
#   1. Every user message passes through this hook BEFORE Claude sees it
#   2. Python CLI does a fast embedding similarity check (~100ms):
#      - Very long messages (>max_length) → skip classification (coding questions)
#      - Short messages → embed and compare to canonical task phrases
#      - Below similarity threshold → allow through immediately
#      - Above threshold → call local LLM for precise classification (~2-5s)
#   3. If it's a task command → execute locally, block Claude (0 tokens used)
#   4. If not → dynamic context evaluation + prompt optimization via local LLM

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$SCRIPT_DIR/.venv/bin/skill-hub-cli"

# Debug: log every invocation
DEBUG_LOG="$HOME/.claude/mcp-skill-hub/logs/hook-debug.log"
echo "[$(date '+%H:%M:%S')] Hook fired — SHELL=$SHELL CWD=$(pwd)" >> "$DEBUG_LOG"

# Read hook input from stdin
INPUT=$(cat)
echo "[$(date '+%H:%M:%S')] Input length=${#INPUT}" >> "$DEBUG_LOG"

# Extract user message and session_id — use IFS=tab to preserve spaces
IFS=$'\t' read -r MESSAGE SESSION_ID <<< "$(echo "$INPUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d.get('prompt', '') or d.get('userMessage', '')
# Strip tabs/newlines that would break field splitting
msg = msg.replace(chr(10), ' ').replace(chr(13), '').replace(chr(9), ' ')
sid = d.get('session_id', '')
print(msg + chr(9) + sid)
" 2>/dev/null)"

if [ -z "$MESSAGE" ]; then
    exit 0
fi

# Pre-warm Ollama embed model (fire-and-forget, 2s timeout — just loads model into memory)
curl -s --max-time 2 http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":"warmup","keep_alive":"10m"}' >/dev/null 2>&1 &

echo "[$(date '+%H:%M:%S')] session=$SESSION_ID msg_len=${#MESSAGE}" >> "$DEBUG_LOG"

# Classify via local LLM (Python handles the fast semantic prefilter internally)
RESULT=$("$CLI" classify --session-id "$SESSION_ID" "$MESSAGE" 2>>"$DEBUG_LOG")
CLI_EXIT=$?

echo "[$(date '+%H:%M:%S')] CLI exit=$CLI_EXIT result_len=${#RESULT}" >> "$DEBUG_LOG"

if [ $CLI_EXIT -ne 0 ] || [ -z "$RESULT" ]; then
    echo "[$(date '+%H:%M:%S')] CLI failed or empty — allowing through" >> "$DEBUG_LOG"
    exit 0  # CLI failed — allow through
fi

DECISION=$(echo "$RESULT" | python3 -c "
import sys, json
print(json.load(sys.stdin).get('decision', 'allow'))
" 2>/dev/null)

if [ "$DECISION" = "block" ]; then
    # Use python to safely serialise the feedback as JSON (handles quotes/newlines).
    # Claude Code hook schema uses 'reason' (not 'message') for the text shown to user.
    echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
reason = data.get('reason') or data.get('message', 'Command handled locally.')
print(json.dumps({'decision': 'block', 'reason': reason}))
"
elif [ "$DECISION" = "allow" ]; then
    # Forward the full result — may contain systemMessage, userMessage,
    # or hookSpecificOutput.additionalContext for Claude's context.
    echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
out = {'decision': 'allow'}
if data.get('systemMessage'):
    out['systemMessage'] = data['systemMessage']
if data.get('userMessage'):
    out['userMessage'] = data['userMessage']
hso = data.get('hookSpecificOutput')
if hso:
    out['hookSpecificOutput'] = hso
if len(out) > 1:
    print(json.dumps(out))
" 2>/dev/null
fi

exit 0
