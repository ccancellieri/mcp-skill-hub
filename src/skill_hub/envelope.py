"""Uniform tool envelope — M2 W3.

Background
----------
The MCP server (``server.py``) exposes ~78 ``@mcp.tool()`` functions. Each
historically returned a plain ``str`` (or, in a handful of cases, raised
on failure). That made it hard to:

* Time every invocation uniformly.
* Convert exceptions into a structured error rather than a 500 on the wire.
* Attach side-channel data (structured payload, future event IDs from W1).
* Let internal callers (tests, the autopilot loop, the swarm) read the
  same shape the MCP wire layer sees.

The envelope solves all four with a single decorator and a small dataclass.

Back-compat contract
--------------------
MCP clients (Claude Code et al.) must see no wire-format change. That
means the function the FastMCP machinery invokes still returns a plain
``str`` (or whatever the tool natively returns). The envelope is the
*internal* view; the *external* view is unchanged.

How it works
~~~~~~~~~~~~
``@tool_envelope`` returns a wrapper that:

1. Records ``time.monotonic()`` before invoking the wrapped function.
2. Catches every ``Exception`` (not BaseException — we still let
   ``KeyboardInterrupt`` through).
3. Builds a :class:`ToolResult` carrying the stdout, optional structured
   payload, optional error, elapsed milliseconds, and an
   ``events_emitted`` list (always ``[]`` until W1 wires it up).
4. Stamps the :class:`ToolResult` on a thread-local *and* on the wrapper
   itself as ``last_result`` so internal callers can fetch it.
5. Returns ``ToolResult.stdout`` (a plain string) to whatever called the
   wrapper — keeping the FastMCP serialization path identical.

Internal callers that want the structured envelope call the wrapper via
the attached ``.envelope(...)`` callable, which returns the
:class:`ToolResult` directly instead of just ``.stdout``::

    result = my_tool.envelope(arg1, arg2)
    assert isinstance(result, ToolResult)

Integration with ``@mcp.tool()``
--------------------------------
Rather than editing every one of the ~78 tool decorator stacks, the
envelope is wired in at one place in ``server.py`` by monkey-patching
``mcp.tool`` immediately after the ``FastMCP`` instance is created.
The patched factory transparently wraps each registered tool with
``tool_envelope`` before handing it to FastMCP. Existing tool
definitions stay untouched::

    @mcp.tool()
    @requires_capability("embedding")
    def my_tool(...) -> str:
        ...

``@requires_capability(tier)`` only stamps ``__capability_tier__`` on the
function object and records the tier in ``TIER_REGISTRY``. It runs first
(closest to ``def``), so ``TIER_REGISTRY`` is keyed by the original tool
name. The envelope wrapper preserves ``__name__`` via ``functools.wraps``
so FastMCP also sees the original name.

Error surfacing
---------------
When the wrapped tool raises, the envelope writes the formatted error
to ``ToolResult.error`` *and* renders it on ``ToolResult.stdout`` as
``"ERROR: {type}: {message}"``. The stdout rendering preserves the
pre-existing wire-level behavior — several tools already return
``f"ERROR: {e}"`` strings on failure — while ``ToolResult.error`` gives
internal callers a structured handle.

Events emitted (W1 forward-compat)
----------------------------------
The ``events_emitted`` field is reserved for the W1 event-log decorator.
W1 has *not* been implemented yet; for now the envelope always sets it
to an empty list. Once W1 lands, the emit decorator will write
``event_id`` values into a context-local list that the envelope reads
before constructing the :class:`ToolResult`. The dataclass field exists
now so wiring W1 later does not break this module's public surface.
"""
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar


@dataclass
class ToolResult:
    """Uniform return envelope for every ``@mcp.tool()`` invocation.

    Fields
    ------
    stdout:
        The user-facing string. This is what the MCP wire layer sees —
        existing clients keep working because the envelope returns this
        verbatim from the wrapper.
    structured:
        Optional structured payload (``dict``) for internal callers that
        want machine-readable output without re-parsing ``stdout``. The
        MCP wire layer ignores this.
    error:
        ``None`` on success; the formatted error string on failure.
        ``stdout`` and ``error`` may *both* be populated when a tool
        partially succeeded and then failed downstream — current tools
        don't do this, but the shape allows it for future use.
    elapsed_ms:
        Wall-clock milliseconds between entering the wrapper and
        producing the result. Populated on every invocation including
        failures.
    events_emitted:
        List of ``events.id`` integers emitted by the W1 emit decorator
        during this invocation. Always ``[]`` until W1 lands.
    """
    stdout: str
    structured: dict | None = None
    error: str | None = None
    elapsed_ms: int = 0
    events_emitted: list[int] = field(default_factory=list)


F = TypeVar("F", bound=Callable[..., Any])


# Thread-local that the envelope wrapper writes the most recent
# :class:`ToolResult` to. Internal callers that want the structured
# envelope without going through ``.envelope(...)`` (e.g. middleware
# that wraps the wrapper itself) can read it from here. Each tool
# wrapper also stamps the same value on its own ``.last_result``
# attribute for easier access in tests.
_LOCAL = threading.local()


def get_last_result() -> ToolResult | None:
    """Return the last :class:`ToolResult` produced on this thread.

    Returns ``None`` if no envelope-wrapped tool has run on this thread
    yet. Useful in tests and middleware that need the envelope without
    rewiring the call path.
    """
    return getattr(_LOCAL, "last_result", None)


def _coerce_stdout(raw: Any) -> tuple[str, dict | None]:
    """Best-effort split of a tool's native return into stdout/structured.

    Most skill-hub tools already return ``str``. A few may return a dict
    (no current callers do, but the type system allows it). We:

    * Pass strings through unchanged with ``structured=None``.
    * For dicts, JSON-serialize for stdout and keep the dict as
      structured. This is a forward-compat hook — no current tool hits
      this branch.
    * For everything else, fall back to ``str(raw)`` so we never crash
      the wire layer.
    """
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, dict):
        import json
        try:
            return json.dumps(raw, default=str), raw
        except (TypeError, ValueError):
            return str(raw), raw
    return str(raw), None


def tool_envelope(fn: F) -> F:
    """Wrap ``fn`` so every invocation produces a :class:`ToolResult`.

    The wrapper:

    * Returns ``ToolResult.stdout`` (a plain string) to the caller so
      FastMCP's serializer keeps working.
    * Stamps the full :class:`ToolResult` on the thread-local
      ``_LOCAL.last_result`` *and* on the wrapper's ``last_result``
      attribute.
    * Exposes ``wrapper.envelope(*args, **kwargs)`` which invokes the
      same code path but returns the :class:`ToolResult` directly.
    * Catches every ``Exception`` and folds it into
      ``ToolResult(stdout="", error=..., elapsed_ms=...)`` so the MCP
      layer never sees an uncaught exception.

    The wrapper preserves ``__name__``, ``__doc__``, signature, and any
    attributes other decorators have stamped on the inner function.
    """

    def _run(args: tuple, kwargs: dict) -> ToolResult:
        start = time.monotonic()
        try:
            raw = fn(*args, **kwargs)
            stdout, structured = _coerce_stdout(raw)
            elapsed = int((time.monotonic() - start) * 1000)
            result = ToolResult(
                stdout=stdout,
                structured=structured,
                error=None,
                elapsed_ms=elapsed,
                events_emitted=[],
            )
        except Exception as exc:  # noqa: BLE001 — envelope must never re-raise
            elapsed = int((time.monotonic() - start) * 1000)
            err_msg = f"{type(exc).__name__}: {exc}"
            result = ToolResult(
                stdout=f"ERROR: {err_msg}",
                structured=None,
                error=err_msg,
                elapsed_ms=elapsed,
                events_emitted=[],
            )
        _LOCAL.last_result = result
        wrapper.last_result = result  # type: ignore[attr-defined]
        return result

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        return _run(args, kwargs).stdout

    def envelope(*args: Any, **kwargs: Any) -> ToolResult:
        """Invoke the tool and return the full :class:`ToolResult`."""
        return _run(args, kwargs)

    wrapper.envelope = envelope  # type: ignore[attr-defined]
    wrapper.last_result = None  # type: ignore[attr-defined]
    wrapper.__wrapped_tool__ = fn  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


__all__ = ["ToolResult", "tool_envelope", "get_last_result"]
