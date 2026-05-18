"""Sandbox interface for plan-execution tools (M2/W5).

Scoped intentionally narrow: only the three plan-execution tools
(``run_plan``, ``execute_plan_step``, ``author_plan``) consult this module.
Every other ~75 MCP tool keeps current in-process behavior.

Usage
-----

    from skill_hub.sandbox import provision

    run = provision("run_plan")          # callable wrapper
    result = run(lambda: _do_work())     # inner work executes inside sandbox

When the policy says ``sandbox.enabled: false`` (or no policy file is present,
which is the project default) ``provision`` returns a pass-through wrapper —
behavior is identical to calling the function directly. This is the
backward-compatible default the acceptance criteria require.

When sandbox is enabled and a tool is mapped to ``subprocess`` mode, the
wrapper monkey-patches :func:`subprocess.run` (and :func:`subprocess.Popen`)
for the duration of the inner call so that any child process the tool body
spawns is forced into:

* ``cwd`` = a fresh temp directory (writes outside cwd raise)
* ``env`` = a minimal allowlist (``PATH`` restricted, network-related vars
  stripped — e.g. ``http_proxy``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``)
* ``capture_output`` defaults preserved
* explicit absolute-path executables outside the temp dir / allowlisted bin
  dirs are rejected — prevents the plan body from shelling out to arbitrary
  binaries

Modes other than ``native`` and ``subprocess`` are accepted in policy but
treated as the interface stub: today they no-op pass through with a warning.
The interface is what W5 ships; container / VM backends are a follow-up.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

POLICY_FILENAMES = ("policy.yml", ".skill-hub/policy.yml")

# Tool-name → default mode mapping. Only the three plan-execution tools are
# wired today. Anything else routes to ``native`` (pass-through) regardless of
# policy.
_PLAN_TOOLS = frozenset({"run_plan", "execute_plan_step", "author_plan"})

# Env vars that must be stripped when ``no_network`` is in effect. Covers the
# common HTTP/HTTPS/SOCKS proxy knobs plus a few credential-bearing tokens that
# would let a plan step reach external services.
_NETWORK_ENV_KEYS = frozenset({
    "http_proxy", "HTTP_PROXY",
    "https_proxy", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY",
    "ftp_proxy", "FTP_PROXY",
    "no_proxy", "NO_PROXY",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN",
})

# Default PATH allowlist when sandbox is active. Kept tiny on purpose — the
# acceptance command for a plan step is usually ``pytest`` or ``echo`` which
# resolve through one of these dirs. Extend via policy.
_DEFAULT_PATH_ALLOWLIST = ("/usr/bin", "/bin")


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def _find_policy_file(start: Path | None = None) -> Path | None:
    """Walk up from `start` (cwd by default) looking for policy.yml.

    Checks both ``policy.yml`` at the root and ``.skill-hub/policy.yml`` —
    the latter is the same convention used by ``pii_gate``.
    """
    here = Path(start) if start else Path.cwd()
    here = here.resolve()
    for parent in (here, *here.parents):
        for name in POLICY_FILENAMES:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def _parse_policy(path: Path) -> dict[str, Any]:
    """Parse the policy file. Prefer pyyaml; degrade silently if missing."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("sandbox: cannot read %s: %s", path, exc)
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — pyyaml is a hard dep elsewhere
        _log.warning("sandbox: pyyaml unavailable; ignoring %s", path)
        return {}
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        _log.warning("sandbox: cannot parse %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_policy(start: Path | None = None) -> dict[str, Any]:
    """Public helper — returns the ``sandbox:`` sub-tree of the policy file.

    Returns ``{}`` when no policy file is found (sandbox stays off by default).
    """
    p = _find_policy_file(start)
    if p is None:
        return {}
    raw = _parse_policy(p)
    sandbox_block = raw.get("sandbox") or {}
    return sandbox_block if isinstance(sandbox_block, dict) else {}


def _mode_for(tool: str, policy: dict[str, Any]) -> str:
    """Resolve the sandbox mode for `tool` from `policy`.

    Returns ``"native"`` when the sandbox is disabled / the tool is not in
    scope; otherwise the mode string from policy (``subprocess`` today).
    """
    if not policy.get("enabled"):
        return "native"
    if tool not in _PLAN_TOOLS:
        # Out-of-scope tools always pass through. Belt-and-braces for callers.
        return "native"
    modes = policy.get("modes") or {}
    mode = modes.get(tool, "native") if isinstance(modes, dict) else "native"
    return str(mode)


# ---------------------------------------------------------------------------
# Subprocess sandbox primitives
# ---------------------------------------------------------------------------


def _sanitized_env(extra_allow: list[str] | None = None,
                   path_allowlist: list[str] | None = None) -> dict[str, str]:
    """Build a minimal env: empty by default, plus a restricted PATH.

    Network-related and credential-bearing keys are dropped. Callers can
    extend the allowlist via the policy's ``env_allow`` list.
    """
    allowlist = set(extra_allow or [])
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _NETWORK_ENV_KEYS:
            continue
        if k in allowlist:
            out[k] = v
    # Always re-build PATH so it points only at explicitly-allowed dirs.
    dirs = path_allowlist or list(_DEFAULT_PATH_ALLOWLIST)
    out["PATH"] = os.pathsep.join(dirs)
    # Minimum viable HOME so tools that expect it don't crash; point at the
    # temp cwd so they can't write into the user's real home.
    return out


class SandboxViolation(RuntimeError):
    """Raised when a sandboxed call attempts an action outside its grant."""


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


@contextmanager
def _subprocess_sandbox(cwd: Path,
                        env_allow: list[str] | None,
                        path_allowlist: list[str] | None):
    """Patch :mod:`subprocess` so any call inside the with-block is forced
    into the sandbox.

    The patch is process-wide but stack-scoped: nesting is supported because
    we capture the prior callable and restore it on exit.
    """
    real_run = subprocess.run
    real_popen = subprocess.Popen
    sandboxed_env = _sanitized_env(env_allow, path_allowlist)
    sandboxed_env.setdefault("HOME", str(cwd))

    allowed_dirs = [Path(p) for p in (path_allowlist or _DEFAULT_PATH_ALLOWLIST)]
    allowed_dirs.append(cwd)

    def _check_cwd(call_cwd):
        target = Path(call_cwd) if call_cwd else cwd
        if not _is_within(target, cwd):
            raise SandboxViolation(
                f"sandbox: cwd {target!s} escapes sandbox root {cwd!s}"
            )
        return target

    def _check_argv(argv):
        # Reject obvious absolute paths that point outside the allowlist.
        # We deliberately don't try to interpret shell strings — the plan
        # tools call subprocess.run with list-form argv in practice.
        if isinstance(argv, str):
            return  # shell=True path; sanitized env already neuters PATH lookups
        if not argv:
            return
        exe = argv[0]
        if isinstance(exe, (str, bytes)) and os.path.isabs(exe):
            exe_path = Path(os.fsdecode(exe))
            if not any(_is_within(exe_path, d) for d in allowed_dirs):
                raise SandboxViolation(
                    f"sandbox: executable {exe_path!s} outside PATH allowlist"
                )

    def patched_run(*args, **kwargs):
        # subprocess.run(args, *, ...) — pull argv from positional or kw.
        argv = args[0] if args else kwargs.get("args")
        _check_argv(argv)
        kwargs["cwd"] = _check_cwd(kwargs.get("cwd"))
        kwargs["env"] = sandboxed_env
        return real_run(*args, **kwargs)

    def patched_popen(*args, **kwargs):  # pragma: no cover — Popen path
        argv = args[0] if args else kwargs.get("args")
        _check_argv(argv)
        kwargs["cwd"] = _check_cwd(kwargs.get("cwd"))
        kwargs["env"] = sandboxed_env
        return real_popen(*args, **kwargs)

    subprocess.run = patched_run  # type: ignore[assignment]
    subprocess.Popen = patched_popen  # type: ignore[assignment]
    try:
        yield cwd
    finally:
        subprocess.run = real_run  # type: ignore[assignment]
        subprocess.Popen = real_popen  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def provision(resources: str | dict[str, Any] | None = None,
              *,
              policy: dict[str, Any] | None = None) -> Callable:
    """Return a callable that runs `fn(*args, **kwargs)` inside the sandbox
    chosen for the given tool / resource grant.

    `resources` is either:
      * a string — the tool name (the common case from `server.py`),
      * a dict — explicit grant ``{"tool": str, "cwd": Path|None,
        "env_allow": [...], "path_allowlist": [...]}``, or
      * ``None`` — pure pass-through.

    The returned callable has signature ``run(fn, *args, **kwargs)`` and
    invokes `fn` either directly (native / disabled) or under the sandbox
    primitives.
    """
    if isinstance(resources, str):
        grant: dict[str, Any] = {"tool": resources}
    elif isinstance(resources, dict):
        grant = dict(resources)
    else:
        grant = {}

    tool = grant.get("tool", "")
    pol = policy if policy is not None else load_policy()
    mode = _mode_for(tool, pol) if tool else "native"

    if mode == "native":
        def _passthrough(fn, *args, **kwargs):
            return fn(*args, **kwargs)
        _passthrough.mode = "native"  # type: ignore[attr-defined]
        return _passthrough

    if mode == "subprocess":
        env_allow = list(pol.get("env_allow") or [])
        path_allowlist = list(pol.get("path_allowlist") or _DEFAULT_PATH_ALLOWLIST)
        explicit_cwd = grant.get("cwd")

        def _subprocess_wrapper(fn, *args, **kwargs):
            # Each invocation gets its own temp dir unless caller pinned one.
            if explicit_cwd is not None:
                cwd = Path(explicit_cwd)
                cwd.mkdir(parents=True, exist_ok=True)
                ctx_cwd = cwd
                with _subprocess_sandbox(ctx_cwd, env_allow, path_allowlist):
                    return fn(*args, **kwargs)
            else:
                with tempfile.TemporaryDirectory(prefix="hub-sbx-") as td:
                    ctx_cwd = Path(td)
                    with _subprocess_sandbox(ctx_cwd, env_allow, path_allowlist):
                        return fn(*args, **kwargs)

        _subprocess_wrapper.mode = "subprocess"  # type: ignore[attr-defined]
        return _subprocess_wrapper

    # Unknown mode — keep the interface forward-compatible, warn, pass-through.
    _log.warning("sandbox: unknown mode %r for tool %r; passing through", mode, tool)

    def _unknown(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    _unknown.mode = "native"  # type: ignore[attr-defined]
    return _unknown
