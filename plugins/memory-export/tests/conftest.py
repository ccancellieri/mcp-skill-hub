"""Pytest path bootstrap so tests can ``from memexp import ...``.

Adds the plugin root (parent of ``tests/``) to ``sys.path`` once per session.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
