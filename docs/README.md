# Skill Hub documentation

Start with the [project introduction](../README.md) and
[verified measurements](../benchmarks/VERIFIED_RESULTS.md).
The core workflow is deterministic retrieval and manual composition. Models,
hooks and background services are optional, separately configured features.

## Core workflow

| Task | Guide |
| --- | --- |
| Install the Python package and configure an MCP client | [Installation](installation.md) |
| Use the eight-tool context surface | [MCP profiles](mcp-profiles.md) |
| Choose sources, shorten passages, preview and copy | [Context composer](context-composer.md) |
| Understand scope, failure behavior and retired hooks | [Context contract and migration](context-service.md) |
| Connect Claude, Codex, Pi or OpenClaw | [Client integrations](../integrations/README.md) |
| Reproduce measurements and understand limits | [Verified results](../benchmarks/VERIFIED_RESULTS.md) |
| Measure total-task usage and correctness | [Evaluation protocol](../benchmarks/CONTEXT_VALUE.md) |

## Reference and operations

- [Tools and CLI](reference/tools.md)
- [Configuration](reference/config.md)
- [Architecture](reference/architecture.md) and [database](reference/database.md)
- [Logs and troubleshooting](reference/logs.md)
- [Dashboard](features/web-control-panel.md)
- [Development invariants](development/README.md) and [roadmap](roadmap.md)

## Optional services and research

These guides describe the broader compatibility surface. They are not
requirements for the minimal MCP profile or promises of improved task outcomes.
The current context contract takes precedence over historical automatic-routing
examples.

- [Semantic search](features/semantic-search.md) and [plugin profiles](features/profiles.md)
- [Optional hook registrations](features/hooks.md)
- [Local execution](features/local-execution.md) and [skill chaining](advanced/skill-chaining.md)
- [Learning mechanisms](features/learning.md), [offline context learning](context-learning.md)
  and [fine-tuning](advanced/fine-tuning.md)
- [Context bridge](advanced/context-bridge.md) and [plugin extension points](plugin-extension-points.md)

Keep documentation focused on observable behavior. Label tokenizer counts,
text estimates and native-client usage separately; never infer task savings
from a smaller context packet alone.
