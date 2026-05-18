"""Tests for the uniform tool envelope — M2 W3.

Cover the contract from `docs/design/managed-agents-refactor.md` §W3:
- happy path: stdout populated, error None, elapsed_ms >= 0
- failure path: error populated, stdout carries "ERROR: ..." rendering,
  no exception bubbles out
- structured: dict returns coerce to JSON stdout + keep the dict on
  ToolResult.structured
- elapsed_ms: always populated
- events_emitted: empty list until W1 wires it
- wrapper preserves __name__ so capability registry stays correct
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.envelope import (  # noqa: E402
    ToolResult,
    get_last_result,
    tool_envelope,
)


def test_happy_path_returns_stdout_and_structured_envelope():
    @tool_envelope
    def my_tool(x: int) -> str:
        return f"got {x}"

    out = my_tool(7)
    assert out == "got 7"

    result = my_tool.envelope(7)
    assert isinstance(result, ToolResult)
    assert result.stdout == "got 7"
    assert result.error is None
    assert result.structured is None
    assert result.elapsed_ms >= 0
    assert result.events_emitted == []


def test_failure_path_surfaces_error_without_raising():
    @tool_envelope
    def broken_tool() -> str:
        raise RuntimeError("boom")

    out = broken_tool()
    assert out.startswith("ERROR: RuntimeError: boom")

    result = broken_tool.envelope()
    assert result.error == "RuntimeError: boom"
    assert result.stdout == "ERROR: RuntimeError: boom"
    assert result.structured is None
    assert result.elapsed_ms >= 0


def test_dict_return_json_serializes_to_stdout_keeps_structured():
    payload = {"alpha": 1, "beta": [2, 3]}

    @tool_envelope
    def dict_tool() -> dict:
        return payload

    out = dict_tool()
    assert json.loads(out) == payload

    result = dict_tool.envelope()
    assert result.structured == payload
    assert json.loads(result.stdout) == payload
    assert result.error is None


def test_elapsed_ms_reflects_real_work():
    @tool_envelope
    def slow_tool() -> str:
        time.sleep(0.02)
        return "ok"

    result = slow_tool.envelope()
    assert result.elapsed_ms >= 15  # 20ms sleep, allow some scheduler slack


def test_events_emitted_is_always_empty_list_pre_w1():
    @tool_envelope
    def t() -> str:
        return "x"

    assert t.envelope().events_emitted == []
    assert t.envelope().events_emitted == []  # idempotent across calls


def test_wrapper_preserves_function_name_for_capability_registry():
    @tool_envelope
    def specific_name() -> str:
        return "ok"

    assert specific_name.__name__ == "specific_name"
    # downstream decorators that key on __name__ (e.g. requires_capability)
    # must see the original symbol.
    specific_name.__capability_tier__ = "none"  # type: ignore[attr-defined]
    assert getattr(specific_name, "__capability_tier__") == "none"


def test_get_last_result_returns_most_recent_per_thread():
    @tool_envelope
    def t(x: int) -> str:
        return str(x)

    t(1)
    last1 = get_last_result()
    assert last1 is not None and last1.stdout == "1"

    t(2)
    last2 = get_last_result()
    assert last2 is not None and last2.stdout == "2"


def test_wrapper_exposes_last_result_attribute():
    @tool_envelope
    def t() -> str:
        return "alpha"

    assert t.last_result is None  # before first call
    t()
    assert t.last_result is not None
    assert t.last_result.stdout == "alpha"


def test_unwrapped_tool_accessible_for_introspection():
    def raw_tool() -> str:
        return "raw"

    wrapped = tool_envelope(raw_tool)
    assert wrapped.__wrapped_tool__ is raw_tool


def test_keyboardinterrupt_still_propagates():
    @tool_envelope
    def t() -> str:
        raise KeyboardInterrupt()

    import pytest
    with pytest.raises(KeyboardInterrupt):
        t()


def test_non_str_non_dict_return_falls_back_to_str():
    @tool_envelope
    def t() -> int:  # type: ignore[return-value]
        return 42  # type: ignore[return-value]

    out = t()
    assert out == "42"
    result = t.envelope()
    assert result.stdout == "42"
    assert result.structured is None


def test_unjsonable_dict_uses_default_str_coercion():
    class Unserializable:
        def __str__(self) -> str:
            return "<unser>"

    payload = {"obj": Unserializable()}

    @tool_envelope
    def t() -> dict:
        return payload

    out = t()
    parsed = json.loads(out)
    assert parsed == {"obj": "<unser>"}
    result = t.envelope()
    assert result.structured is payload


def test_envelope_integrates_with_mcp_tool_monkey_patch():
    """The server.py monkey-patch wraps @mcp.tool() to auto-apply envelope.

    Smoke-test the same wiring pattern without spinning up the real FastMCP
    instance: a fake registry, a patched-style factory, and assert the
    registered function exposes the envelope side-channel.
    """
    registered = {}

    class FakeMCP:
        def tool(self, *args, **kwargs):
            def deco(fn):
                registered[fn.__name__] = fn
                return fn
            return deco

    fake = FakeMCP()
    orig = fake.tool

    def patched(*args, **kwargs):
        raw = orig(*args, **kwargs)
        def deco(fn):
            return raw(tool_envelope(fn))
        return deco

    fake.tool = patched

    @fake.tool()
    def example_tool(x: int) -> str:
        return f"x={x}"

    assert "example_tool" in registered
    fn = registered["example_tool"]
    assert fn(3) == "x=3"
    result = fn.envelope(3)
    assert result.stdout == "x=3"
    assert result.error is None
