"""Lightweight hook worker, deliberately independent of the CLI and LLM stack."""
from __future__ import annotations

import json
import re
import sys

from .router.route import route


def context_output(data: dict) -> dict:
    prompt = data.get("prompt", data.get("userMessage", ""))
    if not isinstance(prompt, str) or not prompt.strip():
        return {}
    # Only accept identity supplied by this hook event. A global active-task
    # marker can belong to another project or simultaneously running session.
    try:
        output = route(
            prompt, session_id=data.get("session_id") or "",
            cwd=data.get("cwd") or "",
        )
    except Exception:
        return {}
    context = output.get("userMessage", "")
    if not context:
        return {}
    context = re.sub(
        r"</?\s*(?:system-reminder|system|assistant|user|tool_use|tool_result|"
        r"function_calls|antml:[a-z_]+)\s*/?>", "", context, flags=re.IGNORECASE,
    )
    return {"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": context,
    }}


def main() -> None:
    try:
        data = json.load(sys.stdin)
        if isinstance(data, dict):
            output = context_output(data)
            if output:
                print(json.dumps(output))
    except Exception:
        return


if __name__ == "__main__":
    main()
