"""Small subprocess helpers shared by the service implementations."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def run(argv: Sequence[str], timeout: float = 15.0) -> tuple[int, str]:
    """Run a command, capture combined output, return (returncode, text).

    Never raises — timeouts and missing binaries return a non-zero code
    and a human-readable message.
    """
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return 127, f"not found: {e}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(argv)}"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def popen_detached(argv: Sequence[str], log_path: Path | None = None) -> tuple[bool, str]:
    """Start a long-running subprocess detached from the parent session.

    If ``log_path`` is given, stdout+stderr are redirected to it (append).
    """
    stdout: int | object = subprocess.DEVNULL
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = log_path.open("a", buffering=1)
    try:
        subprocess.Popen(
            list(argv),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except FileNotFoundError as e:
        return False, f"not found: {e}"
    except OSError as e:
        return False, f"spawn failed: {e}"
    return True, "started"


def stream_install(argv: Sequence[str], log_path: Path, timeout: float = 600.0) -> tuple[bool, str]:
    """Run an installer synchronously, streaming output to ``log_path``.

    Used by the Install button route handler.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as fh:
        fh.write(f"$ {' '.join(argv)}\n")
        fh.flush()
        try:
            proc = subprocess.run(
                list(argv),
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
        except FileNotFoundError as e:
            fh.write(f"\n[error] not found: {e}\n")
            return False, f"not found: {e}"
        except subprocess.TimeoutExpired:
            fh.write(f"\n[error] timeout after {timeout}s\n")
            return False, f"timeout after {timeout}s"
    return proc.returncode == 0, "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
