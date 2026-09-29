#!/bin/bash
# UserPromptSubmit adapter for deterministic, bounded context retrieval.
# The Python worker preserves the prompt and makes no L1/L2 model calls.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
HOOK="$SCRIPT_DIR/hooks/prompt_router.py"

# Keep the legacy local-only guard for compatibility. The context worker
# does not perform model or embedding calls.
export SKILL_HUB_LOCAL_ONLY=1

DEBUG_LOG="$HOME/.claude/mcp-skill-hub/logs/hook-debug.log"
echo "[$(date '+%H:%M:%S')] Router hook fired" >> "$DEBUG_LOG"

# Delegate entirely to the Python implementation — it handles JSON I/O
exec "$PYTHON" "$HOOK"
