#!/usr/bin/env python3
"""Compatibility Stop hook for the retired generic proceed behavior."""
from __future__ import annotations

import sys


def main() -> int:
    """Leave stop handling to the native client without inspecting hook input."""
    return 0


if __name__ == "__main__":
    sys.exit(main())
