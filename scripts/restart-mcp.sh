#!/usr/bin/env bash
# Restart the skill-hub MCP server.
#
# skill-hub is a *stdio* MCP server: Claude Code spawns `.venv/bin/skill-hub`
# as a child process and talks to it over stdin/stdout. An external script can
# kill that child, but only Claude Code can re-establish the pipe — so a full
# restart is "kill the old process(es)" here, then `/mcp` in Claude Code to
# respawn one clean instance from current on-disk code.
#
# Repeated `/mcp` reconnects can leave orphaned skill-hub processes behind;
# this kills ALL of them (plus any standalone webapp on :8765) so the next
# `/mcp` yields a single fresh instance that binds the dashboard port.
#
# Usage (instant, no LLM turn):  ! bash scripts/restart-mcp.sh   then run /mcp
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF=$$

kill_matching() {
  local label="$1" pattern="$2"
  # pgrep -f matches the full command line; exclude this script's own PID.
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null | grep -vw "$SELF" || true)"
  if [ -n "$pids" ]; then
    echo "  killing $label: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
  else
    echo "  $label: none running"
  fi
}

echo "Restarting skill-hub MCP server…"
kill_matching "MCP server (.venv/bin/skill-hub)" "${REPO}/.venv/bin/skill-hub"
kill_matching "standalone dashboard (skill_hub.webapp)" "skill_hub.webapp"

# Free the dashboard port in case a process is bound but not pattern-matched.
PORT_PIDS="$(lsof -ti :8765 -sTCP:LISTEN 2>/dev/null | grep -vw "$SELF" || true)"
if [ -n "$PORT_PIDS" ]; then
  echo "  freeing :8765: $PORT_PIDS"
  # shellcheck disable=SC2086
  kill $PORT_PIDS 2>/dev/null || true
fi

sleep 1

# Report leftover survivors (should be none).
LEFT="$(pgrep -f "${REPO}/.venv/bin/skill-hub" 2>/dev/null | grep -vw "$SELF" || true)"
if [ -n "$LEFT" ]; then
  echo "  WARNING: survivors still running ($LEFT) — re-run, or kill -9 manually"
else
  echo "  all skill-hub processes stopped; :8765 $(lsof -ti :8765 -sTCP:LISTEN >/dev/null 2>&1 && echo BUSY || echo free)"
fi

echo
echo "NEXT STEP: run  /mcp  in Claude Code to reconnect — that respawns one"
echo "clean skill-hub instance from current code (re-binds the :8765 dashboard)."
