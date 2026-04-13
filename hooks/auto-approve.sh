#!/bin/bash
# PreToolUse hook: delegate to auto_approve.py (keeps venv python consistent).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
fi
exec "$PY" "$SCRIPT_DIR/hooks/auto_approve.py"
