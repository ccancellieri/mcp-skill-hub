"""Tests for auto_proceed.py — especially the clarifying-question detector.

History of fixes:
- Bare trailing-? pattern removed: fired on every message ending with '?'.
- clarifying_question standalone signal removed: keyword scanning the full
  message body produces false positives (a message explaining the patterns
  will match its own examples). The function is kept for future use as a
  qualifier alongside open_task/plan signals.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
from auto_proceed import last_message_is_clarifying_question  # noqa: E402


def _data(msg: str) -> dict:
    return {"last_assistant_message": msg}


# ── should NOT trigger (plain trailing ? is not enough) ──────────────────────

@pytest.mark.parametrize("msg", [
    "I've pushed the fix. Does this look right?",
    "The tests are all green. Any questions?",
    "Here is the summary of changes. Let me know if you need anything else.",
    "Completed successfully. What would you like to do next?",
    "All 446 tests pass. Anything else?",
    "",
])
def test_non_clarifying_messages_do_not_trigger(msg):
    assert not last_message_is_clarifying_question(_data(msg))


# ── should trigger (explicit mid-task pause asks) ─────────────────────────────

@pytest.mark.parametrize("msg", [
    "I can take two approaches: (a) rewrite the whole module or (b) patch the fix inline.",
    "Should I push this to main or open a PR?",
    "Want me to also update the tests?",
    "Shall I proceed with the migration?",
    "Do you want me to also update the changelog?",
    "Pause here — confirm you want to drop the table?",
    "I can keep going if you want, or push through now.",
    "Ready to proceed? Just confirm.",
    "Want to continue?",
])
def test_clarifying_messages_trigger(msg):
    assert last_message_is_clarifying_question(_data(msg))


def test_empty_message_does_not_trigger():
    assert not last_message_is_clarifying_question({})
    assert not last_message_is_clarifying_question({"last_assistant_message": None})
    assert not last_message_is_clarifying_question({"last_assistant_message": "   "})
