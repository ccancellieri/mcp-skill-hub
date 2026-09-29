"""Strict JSON whitespace removal without changing any value token."""

from __future__ import annotations

import json


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def minify_json_preserving_lexemes(text: str) -> str | None:
    """Return compact object/array JSON, or None for unsupported or invalid input.

    Validation runs before scanning so whitespace removal is limited to strict
    JSON. The scanner copies every non-whitespace character verbatim, including
    duplicate keys, escaped strings, and the original spelling of numbers.
    """
    if not text.lstrip().startswith(("{", "[")):
        return None
    try:
        json.loads(text, parse_int=str, parse_float=str, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None

    result: list[str] = []
    quoted = escaped = False
    for char in text:
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            result.append(char)
        elif char not in " \t\r\n":
            result.append(char)
    return "".join(result)
