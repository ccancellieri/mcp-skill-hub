# Browser automation: sandbox vs. your real Chrome session

Claude Code can drive a real Chrome through the official **Chrome DevTools MCP**
server (`chrome-devtools-mcp`). Skill Hub does not replace that server — it
complements it: the [tooling orchestrator](../design/tooling-orchestrator.md)'s P2
plan is to *steer* URL/UI tasks toward it, and you can add a teaching rule so the hub
suggests it whenever you paste a URL (see [learning](./learning.md)).

This page covers the one configuration question that decides what the browser tools
can actually do: **which Chrome profile they attach to.**

## The two modes

### 1. Isolated sandbox (default)

By default the server launches its **own** throwaway Chrome with a dedicated profile
under `~/.cache/chrome-devtools-mcp/`. It is logged into nothing. Great for scraping,
testing, and visual verification; useless for anything that needs *your* accounts.

```jsonc
// mcpServers entry — explicit sandbox
"args": ["chrome-devtools-mcp@latest", "--isolated=true"]
```

### 2. Your real Chrome session (opt-in)

To reuse your logged-in accounts — e.g. to read an authenticated dashboard, or draft
and publish a post on a site you're signed into — point the server at a real profile.
Two supported ways:

**a) Attach to a running Chrome (recommended).** Start Chrome yourself with the remote
debugging port open, then have the MCP *connect* rather than launch:

```bash
# Launch Chrome with remote debugging (use a profile, see the warning below)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/chrome-automation-profile"
```

```jsonc
// mcpServers entry — connect to the running instance
"args": ["chrome-devtools-mcp@latest", "--browserUrl=http://127.0.0.1:9222"]
```

`--autoConnect` (Chrome 144+) and `--wsEndpoint=ws://…` are alternatives to
`--browserUrl`. Connecting does **not** auto-launch Chrome — start it first.

**b) Let the server launch a real profile.** Pass the profile directory (and *omit*
`--isolated`). Chrome cannot open a profile that another Chrome already has locked, so
the everyday-Chrome profile must be **closed** first — which is exactly why a
dedicated profile (below) is better:

```jsonc
"args": ["chrome-devtools-mcp@latest",
         "--userDataDir=/Users/you/chrome-automation-profile",
         "--channel=stable"]
```

## ⚠️ Security — read before pointing it at your main profile

Giving an agent control of an authenticated browser session means it can act **as
you** on *every* site that profile is logged into — mail, bank, cloud consoles,
source control, social. That is a large blast radius for a model that can be steered
by page content (prompt injection from a web page it visits).

Mitigations, in order of preference:

1. **Use a dedicated "automation" Chrome profile**, signed into *only* the accounts
   you want the agent to touch (e.g. just the one social account). Not your daily
   profile. This is the single most important control.
2. **Constrain the network surface** with `--allowedUrlPattern` (allowlist) or
   `--blockedUrlPattern` (blocklist) so the session can only reach the domains the
   task needs.
3. **Stay in the loop for writes.** Treat navigation/reading as low-risk, but keep a
   human confirmation before any *publish / send / purchase* action — the agent
   drafts, you click submit. Posting to an external service is public and may be
   cached or indexed even if deleted later.
4. Prefer **attach-to-running** (mode 2a) so you can watch what it does and close the
   window the moment you're done.

## Where Skill Hub fits

- **Discovery:** `suggest_plugins` recommends enabling `chrome-devtools-mcp` when a
  task needs a browser; `toggle_plugin` enables it.
- **Steering:** the tooling orchestrator's headless-browser capability (P2) will nudge
  URL/UI turns toward the browser tools instead of guessing.
- **Memory:** a `teach()` rule — *"when I give a URL, use chrome-devtools"* — makes the
  suggestion automatic for your workflow.

The connection mode itself (sandbox vs. real profile) lives in the
`chrome-devtools-mcp` server args above, not in Skill Hub config.
