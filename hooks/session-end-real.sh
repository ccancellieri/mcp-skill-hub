#!/bin/bash
# SessionEnd hook: once-per-session close (persists L1 summary, fires plugin hook).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
fi
exec "$PY" "$SCRIPT_DIR/hooks/session_end_real.py"
