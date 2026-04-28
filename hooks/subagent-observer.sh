#!/bin/bash
# SubagentStart / SubagentStop hook: log subagent lifecycle.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
fi
exec "$PY" "$SCRIPT_DIR/hooks/subagent_observer.py"
