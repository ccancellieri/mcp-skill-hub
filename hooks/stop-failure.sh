#!/bin/bash
# StopFailure hook: log API errors to api-errors.jsonl.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
fi
exec "$PY" "$SCRIPT_DIR/hooks/stop_failure.py"
