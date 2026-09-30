# MCP Skill Hub

**Alpha · Apache License 2.0 · Author: Carlo Cancellieri**

**Choose the evidence a coding assistant needs, then keep the context small.**

**MCP Skill Hub** is a local Python toolkit for project memory, skill discovery
and reviewable context composition. It started as an MCP server to provide a
standard integration interface across AI clients and agent runtimes
("harnesses"). The name reflects those origins; today the project exposes
several interfaces:

- **[MCP tools](docs/mcp-profiles.md)** for clients that support the Model Context Protocol.
- **CLI tools and a [shared JSON interface](integrations/README.md)** for scripts and direct client integrations.
- **[Plugin extension points](docs/plugin-extension-points.md)** for adding web views, storage and indexing capabilities.
- **Optional [hooks](docs/features/hooks.md) and [native adapters](integrations/README.md)** for supported client events.
- **A web dashboard and [context composer](docs/context-composer.md)** for inspecting sources, reviewing context and copying it into a client.

MCP is one way to access these capabilities. Hooks and native adapters are
client-specific; choose the integration supported by your harness.

The core retrieval path uses SQLite and keyword search: no model download or API
key is needed. The original prompt stays intact.

[Get started](#get-started) · [Context composer](docs/context-composer.md) ·
[Measured results](benchmarks/VERIFIED_RESULTS.md) · [Client integrations](integrations/README.md)

## What works today

| Capability | What you can do | Verification |
| --- | --- | --- |
| **Minimal MCP profile** | Expose eight context tools; load full skill instructions only when needed. | Protocol tests verify the advertised tools, reject excluded calls and prevent background-service startup. |
| **Manual context composer** | Select projects, review candidate sources, choose exact passages, preview and copy a bounded packet. | UI and service checks cover selection, expansion, preview and source changes. |
| **Scoped memory retrieval** | Retrieve project evidence only from explicitly supplied project roots. Missing scope allows global skills only. | Synthetic fixtures and scope regressions check foreign-source rejection. |
| **Deterministic compression** | Compact JSON whitespace while preserving strings, duplicate keys and numeric spelling; identify lossy excerpt selection and truncation. | Fidelity tests cover duplicate members, huge exponents, escapes, Unicode and code. |
| **Source recovery** | Expand a selected candidate or fetch a skill's full indexed text. Changed sources require a new selection. | Fingerprint and stale-source tests; retained originals are used instead of generated digests. |
| **Optional client adapters** | Add bounded evidence while preserving the prompt and native client approvals. MCP and manual copy work without hooks. | Python and Node adapter tests; native-client qualification is tracked separately. |

The full MCP profile remains the compatibility default. The minimal profile is
an explicit choice and does not configure providers, install hooks or start a
model service. Claude, Codex, Pi and OpenClaw integration paths are documented;
adapter tests do not imply every live client configuration has been qualified.

## Measured, with limits

| Comparison | Result | What was measured |
| --- | --- | --- |
| Full → minimal MCP surface | **87 → 8 tools; 92.41% fewer serialized tokens** | Tool schemas and server instructions: 16,362 → 1,242 `o200k_base` tokens, counted once as compact JSON. |
| Project isolation | **0 labeled foreign-source leaks across 144 cases** | Frozen synthetic project-scope fixtures, not an audit of every historical memory record. |

These measurements concern protocol and context payloads. **A reduction in total
tokens per completed task has not been demonstrated.** Clients may cache or defer
tool definitions, and extra context can increase usage. See the
[reproducible report](benchmarks/VERIFIED_RESULTS.md) for versions, commands,
corpus scope and limitations, and the [task evaluation protocol](benchmarks/CONTEXT_VALUE.md).

Local selector studies did not justify promotion. Model-based ranking,
preference learning and automatic prompt rewriting are not part of the normal
composer workflow. A more complex model must earn its place through task-level
results.

## Get started

Requires **Python 3.11+**. For the core workflow, install the Python package:

```sh
git clone https://github.com/ccancellieri/mcp-skill-hub.git
cd mcp-skill-hub
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

On Windows, use the executables in `.venv/Scripts` instead of `.venv/bin`.
The package installation does not register hooks or download model weights.
Index a skill directory you explicitly choose, without generating embeddings:

```sh
.venv/bin/skill-hub-cli index_skills_text --skill-dir /absolute/path/to/skills
```

Repeat `--skill-dir` for additional directories. The command scans their
`SKILL.md` files only. It does not scan other configured locations or index
project memory. See [MCP profiles](docs/mcp-profiles.md) for the tool list.

Configure your MCP client's stdio server with the installed executable and
explicit minimal profile. A typical server entry is:

```json
{
  "command": "/absolute/path/to/mcp-skill-hub/.venv/bin/skill-hub",
  "args": ["--profile", "minimal"]
}
```

Use your client's configuration format and reconnect after changing it.
The server uses the existing local index; select and index your skill sources
before searching. [Installation options](docs/installation.md) cover the full
installer and optional model-backed services.

## Use only the context you need

1. **Discover:** call `search_skills` for short descriptions; use
   `get_skill_content` when a specific skill is needed.
2. **Retrieve:** call `prepare_context` with the original request and the
   absolute project path. Retrieved text is evidence, not an instruction or
   authorization.
3. **Review:** use `prepare_composition`, `expand_context_candidate` and
   `compose_context`, or open the dashboard's `/context` page. Review the sources,
   budget and any lossy transformations before copying the result.

```python
# MCP tool arguments, sent through your client:
prepare_context(text="Review the migration constraints", repo_root="/path/to/project")
search_skills(query="database migration")
```

For the web composer, run `skill-hub-dashboard --no-services` separately and open
`http://127.0.0.1:8765/context`. This option skips background service startup and
maintenance; it does not stop services already running. Starting the minimal MCP server does
not launch the dashboard. Token counts in the
composer are text estimates. Optional prompt compression is an explicit
original/proposal/diff action and never replaces the request automatically.

## Boundaries and compatibility

- Project scope is explicit. Neither ranking history nor client identity grants
  access to another project's memory.
- The automatic context hook is deterministic, has a two-second adapter deadline
  and makes no L1/L2 model call. Failure adds no context; the prompt passes through.
- Client model and effort observations are separate from the server's configured
  providers. Unreported runtime metadata remains unknown.
- Generic auto-proceed, task interception, inferred approvals and hook-driven
  model switching have been retired. See the [migration guide](docs/context-service.md).
- Optional semantic search, provider services and research APIs are documented
  separately. They are not required to use the core context service.

## Development and evidence

Run the isolated Python suite and client-adapter checks:

```sh
uv run --with pytest --with pytest-asyncio --with pytest-timeout \
  python -m pytest -m 'not local_only'
node --test integrations/tests/context-adapters.test.mjs
```

[Development guide](docs/development/README.md) ·
[Documentation index](docs/README.md) · [Roadmap](docs/roadmap.md) ·
[Open issues](https://github.com/ccancellieri/mcp-skill-hub/issues)

## License

Copyright © 2026 Carlo Cancellieri. Licensed under the
[Apache License 2.0](LICENSE).
