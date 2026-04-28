#!/bin/bash
# PreCompact hook: snapshot routing/tool-chain state before /compact wipes it.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3)"
fi
exec "$PY" "$SCRIPT_DIR/hooks/precompact.py"
