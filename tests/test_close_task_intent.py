"""Tests for close-task intent detection regex patterns in intercept_task_commands.py."""
import importlib.util
import sys
from pathlib import Path

# Load the hook module directly (it lives outside src/)
_HOOK_PATH = Path(__file__).resolve().parent.parent / "hooks" / "intercept_task_commands.py"
_spec = importlib.util.spec_from_file_location("intercept_task_commands", _HOOK_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_CLOSE_TASK_RE = _mod._CLOSE_TASK_RE
_FALSE_POSITIVE_RE = _mod._FALSE_POSITIVE_RE


class TestCloseTaskRe:
    """Positive matches for _CLOSE_TASK_RE."""

    def test_close_this_task(self):
        assert _CLOSE_TASK_RE.search("close this task")

    def test_done_with_this_task(self):
        assert _CLOSE_TASK_RE.search("done with this task")

    def test_finished_this_task(self):
        assert _CLOSE_TASK_RE.search("finished this task")

    def test_finish_the_task(self):
        assert _CLOSE_TASK_RE.search("finish the task")

    def test_wrap_up_the_task(self):
        assert _CLOSE_TASK_RE.search("wrap up the task")

    def test_wrapping_up_task(self):
        assert _CLOSE_TASK_RE.search("I'm wrapping up the task now")

    def test_mark_it_done_task(self):
        assert _CLOSE_TASK_RE.search("mark it done task")

    def test_completed_task(self):
        assert _CLOSE_TASK_RE.search("completed task")

    def test_complete_this_task(self):
        assert _CLOSE_TASK_RE.search("complete this task")

    def test_case_insensitive(self):
        assert _CLOSE_TASK_RE.search("CLOSE THIS TASK")
        assert _CLOSE_TASK_RE.search("Done With This Task")


class TestCloseTaskReNegative:
    """_CLOSE_TASK_RE should NOT match non-task phrases."""

    def test_close_the_file(self):
        assert not _CLOSE_TASK_RE.search("close the file")

    def test_close_connection(self):
        assert not _CLOSE_TASK_RE.search("close connection")

    def test_open_a_task(self):
        # "open" should not match "close/done/finish/etc."
        assert not _CLOSE_TASK_RE.search("open a task")

    def test_plain_close(self):
        assert not _CLOSE_TASK_RE.search("close")

    def test_plain_task(self):
        assert not _CLOSE_TASK_RE.search("task")


class TestFalsePositiveRe:
    """_FALSE_POSITIVE_RE catches accidental matches."""

    def test_close_task_in_the(self):
        assert _FALSE_POSITIVE_RE.search("close task xyz in the kitchen")

    def test_close_task_in_the_pipeline(self):
        assert _FALSE_POSITIVE_RE.search("close task abc in the pipeline")

    def test_close_the_file(self):
        assert _FALSE_POSITIVE_RE.search("close the file")

    def test_close_connection(self):
        assert _FALSE_POSITIVE_RE.search("close connection")

    def test_legitimate_close_not_caught(self):
        # "close this task" should NOT be caught as a false positive
        assert not _FALSE_POSITIVE_RE.search("close this task")

    def test_done_with_task_not_caught(self):
        assert not _FALSE_POSITIVE_RE.search("done with this task")
