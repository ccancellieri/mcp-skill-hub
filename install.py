#!/usr/bin/env python3
"""MCP Skill Hub — cross-platform installer for Claude Code.

Works on macOS, Linux, and Windows.

Usage:
    python install.py              # interactive — prompts for optional components
    python install.py --minimal    # core only (no SearXNG, no VPS)
    python install.py --full       # everything auto-configured with defaults
    python install.py --searxng    # core + SearXNG Docker
    python install.py --vps URL    # core + remote VPS Ollama at URL
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"

# Platform paths
HOME = Path.home()
MCP_JSON = HOME / ".mcp.json"
SETTINGS = HOME / ".claude" / "settings.json"
CONFIG_JSON = HOME / ".claude" / "mcp-skill-hub" / "config.json"
HOOKS_DIR = SCRIPT_DIR / "hooks"
DOCKER_DIR = SCRIPT_DIR / "docker"
VENV_DIR = SCRIPT_DIR / ".venv"
BIN_DIR = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
PYTHON = BIN_DIR / ("python.exe" if IS_WINDOWS else "python")
PIP = BIN_DIR / ("pip.exe" if IS_WINDOWS else "pip")
SKILL_HUB_BIN = BIN_DIR / ("skill-hub.exe" if IS_WINDOWS else "skill-hub")

SEARXNG_PORT = 8989


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    """Run a command, print on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"  ERROR: {' '.join(str(c) for c in cmd)}")
        if result.stderr:
            print(f"  {result.stderr.strip()[:300]}")
    return result


def ask(prompt: str, default: str = "") -> str:
    """Prompt user for input with a default value."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def ask_yn(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no with a default."""
    yn = "Y/n" if default else "y/N"
    try:
        answer = input(f"  {prompt} [{yn}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def update_config(updates: dict):
    """Merge updates into ~/.claude/mcp-skill-hub/config.json."""
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            config = json.load(f)
    config.update(updates)
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def probe_url(url: str, timeout: float = 3.0) -> bool:
    """Check if a URL responds with HTTP 200."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_system_ram_gb() -> int:
    """Detect total system RAM in GB. Returns 0 if detection fails."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = run(["sysctl", "-n", "hw.memsize"])
            if result.returncode == 0:
                return int(result.stdout.strip()) // (1024 ** 3)
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024 * 1024)
        elif system == "Windows":
            result = run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"])
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    if "TotalPhysicalMemory" in line:
                        return int(line.split("=")[1]) // (1024 ** 3)
    except Exception:
        pass
    return 0


# Model tiers by RAM
MODEL_TIERS = [
    # (min_ram_gb, reason_model, l4_model, description)
    (64, "qwen2.5-coder:14b", "qwen2.5-coder:32b", "64GB+ — maximum quality"),
    (32, "qwen2.5-coder:7b-instruct-q4_k_m", "qwen2.5-coder:14b", "32GB — high quality"),
    (16, "qwen2.5-coder:7b-instruct-q4_k_m", "qwen2.5-coder:7b-instruct-q4_k_m", "16GB — recommended"),
    (8, "deepseek-r1:1.5b", "qwen2.5-coder:3b", "8GB — minimal"),
]


def recommend_models(ram_gb: int) -> tuple[str, str, str]:
    """Return (reason_model, l4_model, description) based on RAM."""
    for min_ram, reason, l4, desc in MODEL_TIERS:
        if ram_gb >= min_ram:
            return reason, l4, desc
    # Fallback for very low RAM
    return "deepseek-r1:1.5b", "qwen2.5-coder:3b", "<8GB — bare minimum"


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def step_install_package(step: int, total: int):
    """Create venv and install."""
    print(f"[{step}/{total}] Installing Python package...")
    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    run([str(PIP), "install", "-e", ".", "-q"], cwd=str(SCRIPT_DIR))


def step_check_ollama(step: int, total: int, interactive: bool = True) -> str:
    """Check Ollama availability and pull models."""
    print(f"[{step}/{total}] Checking Ollama and models...")
    ollama = shutil.which("ollama")
    if not ollama:
        print("  Warning: Ollama not found. Install from https://ollama.ai then run:")
        print("    ollama pull nomic-embed-text")
        return ""

    # Pull embedding model (required)
    result = run([ollama, "list"])
    installed = result.stdout if result.returncode == 0 else ""

    if "nomic-embed-text" in installed:
        print("  nomic-embed-text already available.")
    else:
        print("  Pulling nomic-embed-text (274 MB, required for embeddings)...")
        run([ollama, "pull", "nomic-embed-text"])

    # Detect RAM and recommend models
    ram_gb = get_system_ram_gb()
    if ram_gb > 0:
        print(f"  Detected RAM: {ram_gb} GB")
    else:
        ram_gb = 16  # safe default
        print("  Could not detect RAM, assuming 16 GB")

    reason_model, l4_model, tier_desc = recommend_models(ram_gb)
    print(f"  Recommended tier: {tier_desc}")
    print(f"    Reasoning model: {reason_model}")
    print(f"    Level 4 model:   {l4_model}")

    # Collect unique models to pull
    models_to_pull = []
    for model in dict.fromkeys([reason_model, l4_model]):  # preserves order, dedupes
        if model not in installed:
            models_to_pull.append(model)

    if not models_to_pull:
        print("  All recommended models already installed.")
    elif interactive:
        print()
        print("  Models to download:")
        for m in models_to_pull:
            print(f"    - {m}")
        if ask_yn("Pull recommended models now?", default=True):
            for m in models_to_pull:
                print(f"  Pulling {m}...")
                run([ollama, "pull", m], timeout=600)
        else:
            print("  Skipped. Pull manually later:")
            for m in models_to_pull:
                print(f"    ollama pull {m}")
    else:
        # Non-interactive (--full): pull automatically
        for m in models_to_pull:
            print(f"  Pulling {m}...")
            run([ollama, "pull", m], timeout=600)

    # Apply config
    local_models = {
        "level_1": "qwen2.5-coder:3b",
        "level_2": "qwen2.5-coder:7b-instruct-q4_k_m" if ram_gb >= 16 else "qwen2.5-coder:3b",
        "level_3": reason_model,
        "level_4": l4_model,
    }
    update_config({
        "reason_model": reason_model,
        "local_models": local_models,
    })
    print(f"  Config updated: reason_model={reason_model}")

    return reason_model


def step_register_mcp(step: int, total: int):
    """Register MCP server in ~/.mcp.json."""
    print(f"[{step}/{total}] Registering MCP server...")
    config = {}
    if MCP_JSON.exists():
        with open(MCP_JSON) as f:
            config = json.load(f)

    servers = config.setdefault("mcpServers", {})
    if "skill-hub" in servers:
        print(f"  Already registered in {MCP_JSON}")
        return

    servers["skill-hub"] = {"type": "stdio", "command": str(SKILL_HUB_BIN)}
    with open(MCP_JSON, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Added skill-hub to {MCP_JSON}")


def _hook_command(script_basename: str) -> str:
    """Return the hook command string, using Python on Windows, bash on Unix."""
    if IS_WINDOWS:
        py_name = script_basename.replace("-", "_").replace(".sh", ".py")
        py_path = HOOKS_DIR / py_name
        if py_path.exists():
            return f"{PYTHON} {py_path}"
        return f"bash {HOOKS_DIR / script_basename}"
    else:
        return str(HOOKS_DIR / script_basename)


def seed_auto_proceed_defaults():
    """Seed auto_proceed_window (23:00-07:00) and auto_proceed_max if absent.
    Only adds keys the user hasn't set; never overwrites."""
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)
    changed = False
    if "auto_proceed" not in cfg:
        cfg["auto_proceed"] = True
        changed = True
    if "auto_proceed_window" not in cfg:
        # Relaxed default: always-on window. Env vars still override.
        cfg["auto_proceed_window"] = {"start_hour": 0, "end_hour": 24}
        changed = True
    if "auto_proceed_max" not in cfg:
        cfg["auto_proceed_max"] = 20
        changed = True
    # Adaptive auto-approve: cache is on by default (cheap, safe); LLM opt-in.
    defaults = {
        "auto_approve_learn": True,           # record user-approved cmds
        "auto_approve_llm": False,            # opt-in: local-LLM classifier
        "auto_approve_confidence": 0.85,
        "auto_approve_verdict_ttl_days": 30,
        "auto_approve_timeout_s": 4.0,
        # Interactive dashboard server (stdlib http.server, loopback only).
        "dashboard_server_enabled": True,
        "dashboard_server_port": 8765,
        "dashboard_auto_open_browser": False,
        # Vector-similarity classifier (faster than LLM, never denies).
        "vector_autoapprove_enabled": True,
        "vector_autoapprove_threshold": 0.88,
        # Learn-from-Claude-sessions: harvest approvals into verdict cache.
        "learn_from_claude_sessions": False,
        # Ask-the-user fallback: when an unknown command isn't classified by
        # cache, vector or LLM, briefly poll the dashboard's question queue.
        "ask_user_on_ambiguous": False,
        "ask_user_timeout_s": 10.0,
        # Ask-on-deny: when a deny_pattern matches, escalate to the user via
        # the dashboard question queue instead of hard-blocking. Catastrophic
        # patterns (fork bombs, dd to raw device) always block immediately.
        "ask_on_deny": True,
        # Adaptive allowance: tiered time windows. First match wins. A window's
        # prefix_bundle is a key into the built-in bundles (read_only, build,
        # deploy) or into user-defined "prefix_bundles". The sentinel
        # "all_non_denied" approves anything not matching a deny_pattern
        # (legacy night-mode behavior).
        "adaptive_windows": [
            {"name": "evening", "start_hour": 18, "end_hour": 23,
             "prefix_bundle": "read_only"},
            {"name": "night", "start_hour": 23, "end_hour": 7,
             "prefix_bundle": "all_non_denied"},
        ],
        # Task-type -> bundle mapping. Active task marker may carry
        # "task_type"; matching prefixes are ADDED additively (never reduce
        # safety). Value may be a bundle name (str) or an inline list.
        "task_type_bundles": {
            "research": "read_only",
            "build": "build",
            "deploy": "deploy",
        },
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            changed = True
    if changed:
        with open(CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Auto-proceed enabled (always-on window; edit {CONFIG_JSON} to change)")
    else:
        print("  auto_proceed settings already present.")


def step_install_commands(step: int, total: int):
    """Copy commands/*.md to ~/.claude/commands/ (idempotent)."""
    print(f"[{step}/{total}] Installing slash commands...")
    src_dir = SCRIPT_DIR / "commands"
    if not src_dir.exists():
        print("  No commands/ directory found — skipped.")
        return
    dst_dir = HOME / ".claude" / "commands"
    dst_dir.mkdir(parents=True, exist_ok=True)
    installed = 0
    for src in sorted(src_dir.glob("*.md")):
        dst = dst_dir / src.name
        shutil.copy(src, dst)
        print(f"  + {src.name}")
        installed += 1
    if installed:
        print(f"  Installed {installed} command(s) to {dst_dir}")
    else:
        print("  No .md files found in commands/")


def step_seed_allowlist(step: int, total: int):
    """Copy examples/skill-hub-allow.yml to ~/.claude/ if not present."""
    print(f"[{step}/{total}] Seeding global allow-list...")
    dst = HOME / ".claude" / "skill-hub-allow.yml"
    src = SCRIPT_DIR / "examples" / "skill-hub-allow.yml"
    if dst.exists():
        print(f"  Already present: {dst}")
        return
    if not src.exists():
        print(f"  Skipped — template missing: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  Wrote {dst}")


def step_install_hooks(step: int, total: int):
    """Register hooks in settings.json (idempotent + upgrades existing entries).

    Each hook definition's ``command`` is its identity key. Re-running this
    step will *replace* an existing entry's other fields (``timeout``, ``if``,
    ``statusMessage``, ...) so changes here propagate without forcing the
    user to hand-edit settings.
    """
    print(f"[{step}/{total}] Installing hooks...")

    if not IS_WINDOWS:
        for sh in HOOKS_DIR.glob("*.sh"):
            sh.chmod(sh.stat().st_mode | 0o755)

    # Event → list of hook definitions.
    event_hooks: dict[str, list[dict]] = {
        "UserPromptSubmit": [
            {
                "type": "command",
                "command": _hook_command("session-start-enforcer.sh"),
                "timeout": 5,
                "statusMessage": "Checking session start protocol...",
            },
            {
                # Prompt router: three-tier classifier (heuristics → Ollama → Haiku)
                # Selects model, plan-mode, preloads skills before Claude responds.
                "type": "command",
                "command": _hook_command("prompt-router.sh"),
                "timeout": 20,
                "statusMessage": "Routing prompt...",
            },
            {
                "type": "command",
                "command": _hook_command("intercept-task-commands.sh"),
                "timeout": 45,
                "statusMessage": "Checking for task commands...",
            },
        ],
        "PreToolUse": [
            {
                # Per-project auto-approve / auto-deny based on
                # <cwd>/.claude/skill-hub-allow.yml. The ``if`` filter
                # (Claude Code 2.1.83) skips the subprocess for non-Bash
                # tool calls — auto-approve only ever acts on Bash anyway.
                "type": "command",
                "command": _hook_command("auto-approve.sh"),
                "if": "Bash(*)",
                "timeout": 5,
                "statusMessage": "Checking allow-list...",
            },
        ],
        "PostToolUse": [
            {
                # Observe successful Bash runs; record as "user_approved"
                # verdicts so future identical/similar commands auto-approve
                # from cache. ``if`` filter avoids per-call subprocess on
                # non-Bash tools (Read/Edit/Grep/...).
                "type": "command",
                "command": _hook_command("post-tool-observer.sh"),
                "if": "Bash(*)",
                "timeout": 5,
                "statusMessage": "Recording approved command...",
            },
        ],
        "PostToolUseFailure": [
            {
                # Negative reinforcement (Claude Code 2.1.119): failed Bash
                # runs flow into the verdict cache as ``status=failed`` so
                # the auto-approve classifier learns to *not* auto-approve
                # flaky commands. Same script, different code branch.
                "type": "command",
                "command": _hook_command("post-tool-observer.sh"),
                "if": "Bash(*)",
                "timeout": 5,
                "statusMessage": "Recording failed command...",
            },
        ],
        "Stop": [
            {
                "type": "command",
                "command": _hook_command("session-end.sh"),
                "timeout": 45,
                "statusMessage": "Saving session memory...",
            },
            {
                # Opt-in overnight auto-proceed (requires SKILL_HUB_AUTO_PROCEED=1).
                # Re-feeds "proceed" to Claude while the active plan still has
                # unchecked items, up to SKILL_HUB_MAX_PROCEEDS (default 20).
                "type": "command",
                "command": _hook_command("auto-proceed.sh"),
                "timeout": 5,
                "statusMessage": "Checking for plan continuation...",
            },
        ],
        "StopFailure": [
            {
                # Append API errors to api-errors.jsonl for the dashboard
                # monitoring tab (Claude Code 2.1.78).
                "type": "command",
                "command": _hook_command("stop-failure.sh"),
                "timeout": 5,
                "statusMessage": "Logging API error...",
            },
        ],
        "SessionEnd": [
            {
                # Real end-of-session work (Claude Code 1.0.85): persists
                # the L1 summary into session:log + dispatches plugin
                # on_session_end. Stop fires per-turn — this fires once.
                "type": "command",
                "command": _hook_command("session-end-real.sh"),
                "timeout": 30,
                "statusMessage": "Closing session...",
            },
        ],
        "PreCompact": [
            {
                # Snapshot routing/tool-chain state before /compact wipes it.
                "type": "command",
                "command": _hook_command("precompact.sh"),
                "timeout": 10,
                "statusMessage": "Snapshotting routing state...",
            },
        ],
        "PostCompact": [
            {
                # Run optimize_memory after compaction trims context.
                "type": "command",
                "command": _hook_command("postcompact.sh"),
                "timeout": 30,
                "statusMessage": "Optimising memory...",
            },
        ],
        "SubagentStart": [
            {
                # Log subagent spawn into session_log (Claude Code 2.1.43).
                "type": "command",
                "command": _hook_command("subagent-observer.sh"),
                "timeout": 5,
                "statusMessage": "Logging subagent start...",
            },
        ],
        "SubagentStop": [
            {
                # Log subagent stop into session_log (Claude Code 1.0.41).
                "type": "command",
                "command": _hook_command("subagent-observer.sh"),
                "timeout": 5,
                "statusMessage": "Logging subagent stop...",
            },
        ],
    }

    settings = {}
    if SETTINGS.exists():
        with open(SETTINGS) as f:
            settings = json.load(f)
    else:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)

    hooks = settings.setdefault("hooks", {})
    changed = False
    added = 0
    updated = 0

    # Upgrade strategy: shallow-merge ONLY missing keys. We never overwrite a
    # field the user has manually customized (e.g. ``statusMessage``,
    # ``timeout``); we only add new keys we ship in this release (e.g. the
    # ``if`` filter introduced in Claude Code 2.1.83).
    for event_name, defs in event_hooks.items():
        event_block = hooks.setdefault(event_name, [{"hooks": []}])
        for new_def in defs:
            existing = None
            for entry in event_block:
                for h in entry.get("hooks", []):
                    if h.get("command") == new_def["command"]:
                        existing = h
                        break
                if existing is not None:
                    break
            if existing is None:
                event_block[0].setdefault("hooks", []).append(new_def)
                added += 1
                changed = True
                print(f"  + {event_name}: {Path(new_def['command']).name}")
            else:
                missing = {k: v for k, v in new_def.items() if k not in existing}
                if missing:
                    existing.update(missing)
                    updated += 1
                    changed = True
                    keys = ", ".join(sorted(missing))
                    print(f"  ~ {event_name}: {Path(new_def['command']).name} "
                          f"(added {keys})")

    if changed:
        with open(SETTINGS, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"  Updated {SETTINGS} ({added} added, {updated} updated)")
    else:
        print("  Hooks already registered.")


# ---------------------------------------------------------------------------
# SearXNG Docker setup
# ---------------------------------------------------------------------------

def docker_available() -> bool:
    """Check if Docker (or Podman) is available and the daemon is running."""
    for cmd in ["docker", "podman"]:
        exe = shutil.which(cmd)
        if exe:
            result = run([exe, "info"], timeout=10)
            if result.returncode == 0:
                return True
    return False


def searxng_running() -> bool:
    """Check if SearXNG is already responding."""
    return probe_url(f"http://localhost:{SEARXNG_PORT}/search?q=test&format=json", timeout=3)


def step_searxng(step: int, total: int):
    """Deploy SearXNG via Docker with minimal resources."""
    print(f"[{step}/{total}] Setting up SearXNG (web search)...")

    # Already running?
    if searxng_running():
        print(f"  SearXNG already responding on port {SEARXNG_PORT}")
        update_config({"searxng_url": f"http://localhost:{SEARXNG_PORT}", "searxng_enabled": True})
        print("  Config updated: searxng_enabled=true")
        return

    if not docker_available():
        print("  Warning: Docker/Podman not found or daemon not running.")
        print("  To install SearXNG manually:")
        print(f"    docker compose -f {DOCKER_DIR}/docker-compose.searxng.yml up -d")
        print("  Or install Docker: https://docs.docker.com/get-docker/")
        return

    # Find docker compose command (v2 plugin or standalone)
    docker = shutil.which("docker") or "docker"
    compose_file = str(DOCKER_DIR / "docker-compose.searxng.yml")

    # Try docker compose (v2) first, fall back to docker-compose (v1)
    compose_cmd = None
    result = run([docker, "compose", "version"])
    if result.returncode == 0:
        compose_cmd = [docker, "compose", "-f", compose_file]
    else:
        dc = shutil.which("docker-compose")
        if dc:
            compose_cmd = [dc, "-f", compose_file]

    if not compose_cmd:
        print("  Warning: docker compose not available.")
        print(f"  Run manually: docker compose -f {compose_file} up -d")
        return

    print(f"  Deploying SearXNG container (port {SEARXNG_PORT}, 128MB limit)...")
    result = run(compose_cmd + ["up", "-d", "--pull", "always"], timeout=120)
    if result.returncode != 0:
        print("  Failed to start SearXNG. Check Docker logs:")
        print(f"    {' '.join(compose_cmd)} logs")
        return

    # Wait for health check
    print("  Waiting for SearXNG to become healthy...", end="", flush=True)
    import time
    for i in range(15):
        time.sleep(2)
        if searxng_running():
            print(" OK")
            update_config({"searxng_url": f"http://localhost:{SEARXNG_PORT}", "searxng_enabled": True})
            print(f"  SearXNG running on http://localhost:{SEARXNG_PORT}")
            print("  Config updated: searxng_enabled=true")
            return
        print(".", end="", flush=True)

    print(" TIMEOUT")
    print("  SearXNG may still be starting. Check:")
    print(f"    curl http://localhost:{SEARXNG_PORT}/search?q=test&format=json")


# ---------------------------------------------------------------------------
# Remote VPS / Ollama configuration
# ---------------------------------------------------------------------------

def step_remote_vps(step: int, total: int, vps_url: str = ""):
    """Configure a remote VPS running Ollama for Level 4 agent."""
    print(f"[{step}/{total}] Configuring remote VPS (Ollama)...")

    if not vps_url:
        vps_url = ask("Remote Ollama URL (e.g. http://myserver:11434)", default="")

    if not vps_url:
        print("  Skipped — no URL provided.")
        return

    # Normalize URL
    vps_url = vps_url.rstrip("/")
    if not vps_url.startswith("http"):
        vps_url = f"http://{vps_url}"
    if ":" not in vps_url.split("//", 1)[-1]:
        vps_url += ":11434"

    # Test connectivity
    print(f"  Testing {vps_url}...")
    if probe_url(f"{vps_url}/api/tags", timeout=5):
        print("  Connected successfully.")
    else:
        print(f"  Warning: Cannot reach {vps_url}/api/tags")
        if not ask_yn("Continue configuring anyway?", default=False):
            return

    # Get model name
    model = ask("Model name on the remote server", default="qwen2.5-coder:32b")

    # API key (for authenticated endpoints like OpenAI-compatible proxies)
    api_key = ask("API key (leave empty for no auth)", default="")

    # Configure remote_llm
    remote_cfg = {
        "base_url": vps_url,
        "model": model,
        "timeout": 120,
    }
    if api_key:
        remote_cfg["api_key"] = api_key

    # Read current config to preserve local_models
    current_config = {}
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            current_config = json.load(f)

    local_models = current_config.get("local_models", {
        "level_1": "qwen2.5-coder:3b",
        "level_2": "qwen2.5-coder:7b-instruct-q4_k_m",
        "level_3": "qwen2.5-coder:14b",
        "level_4": "qwen2.5-coder:32b",
    })

    # Route Level 4 to remote
    local_models["level_4"] = f"remote:{vps_url}"

    update_config({
        "remote_llm": remote_cfg,
        "local_models": local_models,
    })

    print(f"  Config updated:")
    print(f"    remote_llm.base_url = {vps_url}")
    print(f"    remote_llm.model    = {model}")
    print(f"    local_models.level_4 = remote:{vps_url}")

    # Optionally configure SearXNG on the same VPS
    searxng_vps_url = f"{vps_url.rsplit(':', 1)[0]}:{SEARXNG_PORT}"
    if probe_url(f"{searxng_vps_url}/search?q=test&format=json", timeout=3):
        print(f"  Detected SearXNG on VPS at {searxng_vps_url}")
        update_config({"searxng_url": searxng_vps_url, "searxng_enabled": True})
        print(f"    searxng_url = {searxng_vps_url}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def step_summary():
    """Print next steps."""
    print()
    print("=== Setup Complete ===")
    print()
    print("Next steps:")
    print("  1. Restart Claude Code")
    print("  2. In Claude Code, run:  index_skills()  then  index_plugins()")
    print('  3. Try:  search_skills("your task description")')
    print('  4. Teach it:  teach(rule="when I give a URL", suggest="chrome-devtools-mcp")')
    print()

    # Show active config
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)
        extras = []
        if cfg.get("searxng_enabled") and cfg.get("searxng_url"):
            extras.append(f"  SearXNG:    {cfg['searxng_url']}")
        remote = cfg.get("remote_llm", {})
        if remote.get("base_url"):
            extras.append(f"  Remote VPS: {remote['base_url']} ({remote.get('model', '?')})")
        if extras:
            print("Active integrations:")
            for line in extras:
                print(line)
            print()

    # Show model config
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            cfg2 = json.load(f)
        rm = cfg2.get("reason_model")
        lm = cfg2.get("local_models", {})
        if rm:
            print(f"  Models:     reason={rm}, L4={lm.get('level_4', '?')}")

    print()
    print("Monitor hook activity:")
    log_path = HOME / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"
    if IS_WINDOWS:
        print(f"  powershell: Get-Content -Wait {log_path}")
    else:
        print(f"  tail -f {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    print("=== MCP Skill Hub Installer ===")
    print(f"  Platform: {platform.system()} ({platform.machine()})")
    print()

    # Parse flags
    mode = "interactive"
    vps_url = ""
    if "--minimal" in args:
        mode = "minimal"
    elif "--full" in args:
        mode = "full"
    else:
        if "--searxng" in args:
            mode = "searxng"
        if "--vps" in args:
            idx = args.index("--vps")
            vps_url = args[idx + 1] if idx + 1 < len(args) else ""
            mode = "vps" if mode == "interactive" else mode + "+vps"

    # Determine which optional steps to run
    do_searxng = False
    do_vps = False

    if mode == "minimal":
        pass  # core only
    elif mode == "full":
        do_searxng = True
        do_vps = True
    elif mode == "interactive":
        print("Optional components:")
        do_searxng = ask_yn("Deploy SearXNG for web search? (requires Docker)", default=True)
        do_vps = ask_yn("Configure a remote VPS running Ollama?", default=False)
        print()
    else:
        do_searxng = "searxng" in mode
        do_vps = "vps" in mode or bool(vps_url)

    # Calculate total steps
    total = 5  # core steps always (package, ollama, mcp, seed, hooks, commands)
    if do_searxng:
        total += 1
    if do_vps:
        total += 1

    # Core steps
    total += 1  # +1 for allow-list seeding
    step = 1
    step_install_package(step, total); step += 1
    interactive = mode == "interactive"
    step_check_ollama(step, total, interactive=interactive); step += 1
    step_register_mcp(step, total); step += 1
    step_seed_allowlist(step, total); step += 1
    seed_auto_proceed_defaults()
    step_install_hooks(step, total); step += 1
    step_install_commands(step, total); step += 1

    # Optional steps
    if do_searxng:
        step_searxng(step, total); step += 1
    if do_vps:
        step_remote_vps(step, total, vps_url=vps_url); step += 1

    step_summary()


if __name__ == "__main__":
    main()
