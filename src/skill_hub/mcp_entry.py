"""Lightweight argument gate for the MCP server process."""
from __future__ import annotations

import argparse


def parse_profile() -> str:
    parser = argparse.ArgumentParser(description="Run the Skill Hub MCP server")
    parser.add_argument("--profile", choices=("full", "minimal"), default="full",
                        help="MCP tool surface (default: full)")
    return parser.parse_args().profile


def main() -> None:
    profile = parse_profile()
    from . import mcp_profile_state
    mcp_profile_state.profile = profile
    from .server import main as run_server
    run_server(profile=profile)


if __name__ == "__main__":
    main()
