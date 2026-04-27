"""cwt -- cold-start CLI for worktree-driven parallel sessions.

Usage:
    cwt <project> <name> [--mode terminal|tmux|background] [--prompt TEXT]
    cwt --resume <task_id>
    cwt --list

Direct counterpart to the MCP tools save_task/reopen_task/list_tasks.
Calls the same skill_hub.worktree helpers; no MCP round-trip.
"""
from __future__ import annotations

import argparse
import sys

from . import worktree as _wt


def _cmd_create(args: argparse.Namespace) -> int:
    try:
        spec = _wt.ensure_worktree(args.project, args.name, mode=args.mode)
        spec = _wt.launch_session(spec, initial_prompt=args.prompt or None)
    except _wt.WorktreeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Persist as a task so /hub-list-tasks and /hub-reopen-task work.
    try:
        from .store import SkillStore
        store = SkillStore()
        store.save_task(
            title=f"{args.project}/{args.name}",
            summary=args.prompt or f"worktree session for {args.project}/{args.name}",
            vector=[],
            context="",
            tags="worktree",
            session_id="cwt",
            cwd=spec.worktree_path,
            branch=spec.branch,
            worktree=spec.to_json(),
        )
    except Exception as e:  # noqa: BLE001 -- never fail the launch on bookkeeping
        print(f"warning: task record not saved: {e}", file=sys.stderr)

    print(f"Worktree: {spec.worktree_path}")
    print(f"Branch:   {spec.branch}")
    print(f"Mode:     {spec.mode}")
    if spec.last_pid:
        print(f"PID:      {spec.last_pid}")
    if spec.log_path and spec.mode == "background":
        print(f"Log:      {spec.log_path}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    from .store import SkillStore
    store = SkillStore()
    task = store.get_task(args.task_id)
    if not task:
        print(f"error: task #{args.task_id} not found", file=sys.stderr)
        return 1
    blob = task["worktree"] if "worktree" in task.keys() else None
    if not blob:
        # Legacy reopen: no worktree attached, just flip status.
        store.reopen_task(args.task_id)
        print(f"Task #{args.task_id} reopened (no worktree attached).")
        return 0
    try:
        spec = _wt.WorktreeSpec.from_json(blob)
    except (ValueError, TypeError) as e:
        print(f"error: task #{args.task_id} worktree spec invalid: {e}", file=sys.stderr)
        return 1
    if _wt.is_session_alive(spec):
        print(_wt.focus_session(spec))
        return 0
    try:
        spec = _wt.launch_session(spec)
    except _wt.WorktreeError as e:
        print(f"error: relaunch failed: {e}", file=sys.stderr)
        return 2
    store.set_task_worktree(args.task_id, spec.to_json())
    store.reopen_task(args.task_id)
    print(f"Resumed task #{args.task_id} in {spec.worktree_path}")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    from .store import SkillStore
    store = SkillStore()
    rows = store.list_tasks("open")
    if not rows:
        print("No open tasks.")
        return 0
    for r in rows:
        # `worktree` column may not surface in this projection; pull from get_task.
        full = store.get_task(r["id"])
        wt_str = ""
        if full is not None and "worktree" in full.keys() and full["worktree"]:
            try:
                spec = _wt.WorktreeSpec.from_json(full["worktree"])
                alive = "✓" if _wt.is_session_alive(spec) else "✗"
                wt_str = f"  [{alive} {spec.mode}] {spec.worktree_path}"
            except (ValueError, TypeError):
                wt_str = "  [worktree spec invalid]"
        print(f"#{r['id']} {r['title']}{wt_str}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cwt",
        description="Worktree-driven parallel Claude sessions.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_create = sub.add_parser("create", help="(default) create a worktree session")
    p_create.add_argument("project")
    p_create.add_argument("name")
    p_create.add_argument("--mode", choices=_wt.VALID_MODES, default=None)
    p_create.add_argument("--prompt", default="")
    p_create.set_defaults(func=_cmd_create)

    p_resume = sub.add_parser("resume", help="resume an existing task by id")
    p_resume.add_argument("task_id", type=int)
    p_resume.set_defaults(func=_cmd_resume)

    p_list = sub.add_parser("list", help="list open tasks")
    p_list.set_defaults(func=_cmd_list)

    # Top-level shorthand flags so `cwt --resume 12` and `cwt --list` work.
    parser.add_argument("--resume", type=int, dest="resume_id", default=None)
    parser.add_argument("--list", action="store_true", dest="do_list")

    # Positional shortcut: `cwt geoid foo` == `cwt create geoid foo`
    args, _unknown = parser.parse_known_args(argv)
    if args.resume_id is not None:
        ns = argparse.Namespace(task_id=args.resume_id)
        return _cmd_resume(ns)
    if args.do_list:
        return _cmd_list(argparse.Namespace())
    if args.cmd is None:
        # Re-parse with implicit "create"
        rest = sys.argv[1:] if argv is None else argv
        args = parser.parse_args(["create", *rest])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
