#!/bin/bash
# UserPromptSubmit hook — delegates to Python enforcer (resume-or-create + advisory).
# Prefer the repo's venv Python (has skill_hub installed); fall back to python3.
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$HERE/../.venv/bin/python3"
if [ -x "$VENV_PY" ]; then
    exec "$VENV_PY" "$HERE/session_start_enforcer.py"
fi
exec python3 "$HERE/session_start_enforcer.py"
