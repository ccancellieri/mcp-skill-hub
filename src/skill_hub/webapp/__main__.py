"""Standalone dashboard launcher: `python -m skill_hub.webapp`.

Runs uvicorn in the foreground (blocks) so the dashboard survives independently
of the MCP server. Reads host/port from ~/.claude/mcp-skill-hub/config.json
(dashboard_server_host / dashboard_server_port) with sensible defaults.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import webbrowser

import uvicorn

from ..store import SkillStore
from .main import create_app

CONFIG = pathlib.Path.home() / ".claude" / "mcp-skill-hub" / "config.json"
DB = pathlib.Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"


def _load_cfg() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    cfg = _load_cfg()
    parser = argparse.ArgumentParser(prog="skill-hub-dashboard")
    parser.add_argument("--host", default=cfg.get("dashboard_server_host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(cfg.get("dashboard_server_port", 8765)))
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args()

    app = create_app(SkillStore(DB))
    url = f"http://{args.host}:{args.port}/"
    print(f"skill-hub dashboard: {url}")
    if args.open or cfg.get("dashboard_auto_open_browser"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
