#!/bin/bash
# UserPromptSubmit hook: prompt router — model/plan-mode/skill orchestration.
#
# Three-tier classification:
#   Tier 1: Heuristics (<5ms, free)
#   Tier 2: Local Ollama (~200-500ms, free) — if Tier-1 confidence < 0.85
#   Tier 3: Claude Haiku batch (~500ms, ~$0.0001) — opt-in, if Tier-2 confidence < 0.7
#
# Output: systemMessage with verdict + preloaded skill hints injected before Claude.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
HOOK="$SCRIPT_DIR/hooks/prompt_router.py"

DEBUG_LOG="$HOME/.claude/mcp-skill-hub/logs/hook-debug.log"
echo "[$(date '+%H:%M:%S')] Router hook fired" >> "$DEBUG_LOG"

# Delegate entirely to the Python implementation — it handles JSON I/O
exec "$PYTHON" "$HOOK"
