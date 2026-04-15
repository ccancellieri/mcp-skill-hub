Open the skill-hub control panel in the browser.

Check whether the webapp is already running:

```bash
curl -sf http://127.0.0.1:8765/healthz
```

If that returns `{"ok":true}`, skip to the browser step.

Otherwise start it in the background so it survives this session:

```bash
cd /Users/ccancellieri/work/code/mcp-skill-hub && \
  nohup uv run python -m skill_hub.webapp \
  > /tmp/skill-hub-dashboard.log 2>&1 &
```

Wait up to 5 seconds for the server to become healthy (poll `/healthz`).

Then open the browser at the control panel:

```bash
open -a "Google Chrome" http://127.0.0.1:8765/control
```

Tell the user:
- Control panel: http://127.0.0.1:8765/control
- Dashboard / stats: http://127.0.0.1:8765/
- Log: /tmp/skill-hub-dashboard.log
- To stop: `lsof -ti :8765 | xargs kill`
