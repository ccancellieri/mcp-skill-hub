#!/usr/bin/env python3
"""Compatibility Stop hook for retired per-turn session-end work."""
from __future__ import annotations

import sys


def main() -> int:
    """Leave session maintenance to explicit CLI and SessionEnd workflows."""
    return 0


if __name__ == "__main__":
    sys.exit(main())
