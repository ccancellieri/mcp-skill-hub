"""Portable JSON context adapter for clients without native MCP prompt hooks."""
from __future__ import annotations

import json
import sys

from .context_service import build_context

MAX_INPUT_BYTES = 128 * 1024


def prepare_request(data: dict) -> dict:
    """Validate the common request without depending on any client's events."""
    if not isinstance(data, dict) or not isinstance(data.get("prompt"), str):
        raise ValueError("prompt must be a string")
    for key in ("cwd", "session_id"):
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"{key} must be a string")
    task_id = data.get("task_id")
    if task_id is not None and (type(task_id) is not int or task_id <= 0):
        raise ValueError("task_id must be a positive integer")
    return build_context(data["prompt"], cwd=data.get("cwd", ""),
                         session_id=data.get("session_id", ""), task_id=task_id)


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("context request exceeds input limit")
        result = prepare_request(json.loads(raw))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, TypeError, OSError) as exc:
        print(json.dumps({"context": "", "items": [], "warnings": [str(exc)]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
