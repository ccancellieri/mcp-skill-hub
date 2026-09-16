#!/usr/bin/env python3
"""Bounded context hook; failures and timeouts leave the user prompt intact."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def main() -> None:
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            return
        from skill_hub.config import load_config

        cfg = load_config()
        if not cfg.get("context_enabled", True) or not cfg.get("hook_enabled", True):
            return
        timeout = min(5.0, max(0.1, float(cfg.get("context_hook_timeout_s", 2.0))))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "skill_hub.context_hook"],
            input=json.dumps(data), capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        if result.returncode == 0 and result.stdout.strip():
            output = json.loads(result.stdout)
            if isinstance(output, dict):
                print(json.dumps(output))
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return


if __name__ == "__main__":
    main()
