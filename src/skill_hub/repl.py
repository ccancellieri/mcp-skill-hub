"""Standalone REPL for Skill Hub — works without Claude Code.

Runs entirely locally via Ollama. Use when:
- Claude is rate-limited / exhausted
- You want to interact with local skills and commands
- VS Code extension doesn't show hook output (anthropics/claude-code#42178)

Usage:
    skill-hub-repl              # start interactive REPL
    skill-hub-repl "git status" # run a single command and exit
"""

import json
import os
import readline  # noqa: F401 — enables line editing in input()
import sys

from .activity_log import log_event


# ── ANSI colors ────────────────────────────────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

_BANNER = f"""\
{_BOLD}{_CYAN}╔══════════════════════════════════════════╗
║       Skill Hub — Local REPL             ║
║  All commands run locally via Ollama     ║
║  Type ? for help, Ctrl-C to exit        ║
╚══════════════════════════════════════════╝{_RESET}
"""


def _run_command(message: str) -> str | None:
    """Route a message through the hook pipeline and return the response."""
    from .cli import hook_classify_and_execute, _handle_slash_command

    # Try slash commands and ? help first (fast path)
    stripped = message.strip()
    if stripped.startswith("?") or stripped.startswith("/"):
        result = _handle_slash_command(stripped)
        if result:
            return result.get("reason") or result.get("message", json.dumps(result))

    # Full pipeline: classify, L1-L4, triage
    result = hook_classify_and_execute(message)
    decision = result.get("decision", "allow")

    if decision == "block":
        return result.get("reason") or result.get("message", "(handled locally)")

    # "allow" with systemMessage = context enrichment (show it)
    sys_msg = result.get("systemMessage")
    if sys_msg:
        return f"{_DIM}[Context for Claude — showing locally]{_RESET}\n\n{sys_msg}"

    # Pure pass-through: use local agent as fallback
    from . import config as _cfg
    if _cfg.get("local_execution_enabled"):
        from .local_agent import plan_agent, run_agent

        plan = plan_agent(message)
        if plan.get("can_handle"):
            # Show plan
            plan_steps = plan.get("plan", [])
            skills_needed = plan.get("skills_needed", [])
            commands_needed = plan.get("commands_needed", [])

            lines = [f"{_BOLD}[Local agent — {plan.get('model', '?')}]{_RESET}\n"]
            lines.append(f"Plan for: {message}\n")
            if plan_steps:
                lines.append("Steps:")
                for i, step in enumerate(plan_steps, 1):
                    lines.append(f"  {i}. {step}")
            if skills_needed:
                lines.append(f"\nSkills: {', '.join(skills_needed)}")
            if commands_needed:
                lines.append(f"Commands: {', '.join(commands_needed)}")

            print("\n".join(lines))

            # Ask for confirmation
            try:
                confirm = input(f"\n{_YELLOW}Execute? [y/n]{_RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "\nCancelled."
            if confirm in ("y", "yes", "ok", "go"):
                return run_agent(message)
            return "Cancelled."

        return (f"{_DIM}[Local agent — {plan.get('model', '?')}]{_RESET}\n"
                f"Cannot handle locally: {plan.get('reason', 'unknown')}\n"
                f"This task needs Claude.")

    return f"{_DIM}[Pass-through] This message would go to Claude. Local agent is disabled.{_RESET}"


def _repl_loop() -> None:
    """Interactive REPL loop."""
    from . import config as _cfg
    from .embeddings import embed_available, ollama_available, EMBED_MODEL, RERANK_MODEL

    # Status check
    embed_ok = embed_available()
    reason_ok = ollama_available(RERANK_MODEL)
    models = _cfg.get("local_models") or {}

    print(_BANNER)
    ok = f"{_GREEN}OK{_RESET}"
    missing = f"{_RED}MISSING{_RESET}"
    print(f"  Embed model:  {ok if embed_ok else missing} ({EMBED_MODEL})")
    print(f"  Reason model: {ok if reason_ok else missing} ({RERANK_MODEL})")
    print(f"  L4 model:     {models.get('level_4', 'not configured')}")
    on = f"{_GREEN}ON{_RESET}"
    off = f"{_DIM}OFF{_RESET}"
    print(f"  Local exec:   {on if _cfg.get('local_execution_enabled') else off}")
    print(f"  Working dir:  {os.getcwd()}")
    print()

    log_event("REPL", "session started")

    while True:
        try:
            cwd_short = os.path.basename(os.getcwd()) or "/"
            prompt = f"{_GREEN}{cwd_short}{_RESET} {_BOLD}>{_RESET} "
            message = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Bye.{_RESET}")
            break

        if not message:
            continue

        if message.lower() in ("exit", "quit", "q"):
            print(f"{_DIM}Bye.{_RESET}")
            break

        # cd support — change working directory
        if message.startswith("cd "):
            target = message[3:].strip()
            target = os.path.expanduser(target)
            try:
                os.chdir(target)
                print(f"{_DIM}{os.getcwd()}{_RESET}")
            except OSError as exc:
                print(f"{_RED}cd: {exc}{_RESET}")
            continue

        try:
            response = _run_command(message)
            if response:
                print(response)
                print()
        except Exception as exc:
            print(f"{_RED}Error: {exc}{_RESET}\n")


def main() -> None:
    """Entry point for skill-hub-repl."""
    if len(sys.argv) > 1:
        # Single command mode: skill-hub-repl "git status"
        message = " ".join(sys.argv[1:])
        try:
            response = _run_command(message)
            if response:
                print(response)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        _repl_loop()


if __name__ == "__main__":
    main()
