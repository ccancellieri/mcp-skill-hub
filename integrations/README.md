# Context Adapters

These adapters call the shared JSON interface implemented by
`python -m skill_hub.context_cli`. They pass:

```json
{"prompt":"...","cwd":"/absolute/project","session_id":"...","task_id":null,"runtime":{"client":{"id":"pi"},"session":{"id":"..."},"model":{"id":"...","provider":"...","display_name":"..."},"effort":{"value":"high","scheme":"thinking_level"}}}
```

The command returns the context service result JSON. Each adapter has a hard
two-second timeout, limits stdout to 128 KiB, and treats any failure as no
context. It invokes the command directly, never through a shell.

`runtime` is optional telemetry. Adapters report only fields exposed by their
host event API, so absent model or effort values remain unknown. The installed
adapter selects an out-of-band CLI flag and records these fields as
`adapter_reported`; this label is diagnostic provenance, not authentication or
proof of an untampered native event. Direct CLI and MCP arguments are
`caller_reported`, and JSON payloads cannot promote their own provenance.
Persistence is local, client/session scoped, bounded, and does not participate
in context selection.

The adapters depend on the local `@mcp-skill-hub/context-adapter` workspace
package. From this directory, run `npm install` once. Link package directories
from this checkout rather than copying an adapter directory by itself, so its
local dependency remains resolvable.

## Pi

Pi auto-discovers project extensions from `.pi/extensions/*/index.ts` and
global extensions from `~/.pi/agent/extensions/*/index.ts`. Symlink this
checkout's `integrations/pi` directory into one of those locations, then
reload Pi. Configure a Python executable with `SKILL_HUB_CONTEXT_PYTHON`; by
default the adapter runs:

```text
python3 -m skill_hub.context_cli
```

When the `skill-hub-context` console command is installed, set
`SKILL_HUB_CONTEXT_COMMAND=skill-hub-context` instead. The extension uses
Pi's `before_agent_start` event and returns a custom context message; it does
not rewrite Pi's system prompt. Pi documents extension locations, the event's
`event.prompt`/`ctx.cwd` contract, and custom message returns in its
[extension guide](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md).

## OpenClaw

Use `integrations/openclaw` as the plugin root. Its `package.json` declares
the TypeScript runtime entrypoint and `openclaw.plugin.json` validates three
optional plugin settings:

```json
{
  "pythonCommand": "/absolute/path/to/python3",
  "contextCommand": "skill-hub-context",
  "projectRoot": "/absolute/path/to/project"
}
```

`contextCommand` takes precedence when set; otherwise `pythonCommand` runs
`-m skill_hub.context_cli`. `projectRoot` must be absolute. Without it, the
adapter uses only an absolute `ctx.workspaceDir`; it never falls back to the
host process working directory. Enable the plugin's conversation-access and
prompt-injection permissions required for `before_prompt_build` in OpenClaw's
plugin settings.

The adapter returns `prependContext` from `before_prompt_build`. OpenClaw
documents that hook and its return shape in the
[prompt and session hook reference](https://docs.openclaw.ai/plugins/hooks/prompt-and-session),
and requires a manifest with a JSON schema for native plugins in its
[plugin manifest reference](https://docs.openclaw.ai/plugins/manifest).

## Claude and Codex

Claude and Codex should use the shared MCP `prepare_context` capability rather
than these prompt-hook adapters. Register the Skill Hub MCP server with the
client and call `prepare_context` when context is wanted; no client hook or
automatic prompt interception is required.
