# Client models and Skill Hub services

Skill Hub distinguishes the coding agent in a client (L3) from its own optional
local models (L1) and remote API models (L2). Connecting Codex, Claude, Pi, or
OpenClaw does not change the server's model configuration or supply API credentials.
The prompt hook remains deterministic and keeps its existing two-second deadline.

## Service model selection

The Models view under Control → Configuration lists configured providers and
models. A configured entry is not proof that its endpoint is reachable. Unknown
prices are displayed as unknown, without borrowing another model's price.

Use `provider-name::model-id` when an ID occurs in multiple provider records.
The provider name selects endpoint and credentials; the stored model ID is sent
through the corresponding transport adapter. For example, a gateway record named
`work` with model `example/model` is selected as `work::example/model`. A plain
ambiguous ID produces an error instead of silently selecting another provider.
The same resolution applies to embedding models. Explicit embedding endpoint
overrides remain supported. An explicit model argument or provider-qualified
tier value is pinned: an unavailable backend produces an error instead of
silently switching providers. Existing automatic escalation still applies to
unpinned service requests.

On the Providers page, **Discover** previews model IDs from a configured
OpenAI-compatible gateway such as OmniRoute (for example,
`api_base: https://gateway.example/v1`). Discovery reads the gateway's
advertised catalog using that provider's configured endpoint and credential,
then proposes IDs absent from its registry entry. An advertised ID does not
establish that it is routable or eligible for the configured account. Discovery
does not save or enable models. Availability and cost remain unknown; a missing
price does not imply free access. Catalogs are capped at 1,000 entries and
256 KiB, with truncation reported.

Legacy tier values and Claude family aliases remain readable. Existing explicit
configuration is retained. Diagnostics record operation, provider, requested and
resolved model, and outcome; the new routing fields contain neither prompts nor
credentials. A reference to Opus in an error may therefore refer to a server
operation even when the connected coding client uses another model.

## Session observations

The Models view also lists client sessions separately. It shows the last reported
client, model, effort and the origin of those values. Missing metadata is shown as
“Not reported”; a configured default is not a detected model. Effort names are
kept in their native format instead of mapping them onto Claude tiers.

`prepare_context` and the JSON context CLI accept an optional `runtime` object:

```json
{
  "client": {"id": "example-client", "version": "1.0"},
  "session": {"id": "example-session", "turn_id": "example-turn"},
  "model": {"id": "example-model", "provider": "example-provider"},
  "effort": {"value": "high", "scheme": "example-client"}
}
```

Normal tool arguments are caller-reported evidence. MCP client information can
identify a client implementation, but does not establish the conversation's model
or reasoning effort. Skill Hub never treats the MCP server process ID as the
native conversation ID, and does not inspect unrelated client histories. Native
adapters use only fields exposed by their supported client events. Pi and
OpenClaw select `--adapter-source` outside the JSON payload; their values are
labeled `adapter_reported`. A payload cannot promote itself to native evidence.
The UI preserves separate observation timestamps for model and effort, so an
identity update does not make retained model information appear freshly observed.
Model changes discard stale display names, providers and effort values.
These observations are for display and diagnosis, never for permission decisions.

Clients that omit this object continue to retrieve context normally. Runtime
observation failures do not suppress an otherwise successful context result.

## Local selector experiment

The optional evaluator in `benchmarks/local_selector.py` compares deterministic
retrieval with local skill scoring. It is not connected to the prompt hook or
enabled by changing a service tier. Its scores are uncalibrated, and neither a
smaller context nor a higher abstention rate alone establishes better selection.
See the benchmark report and reproduction instructions for pinned dependencies,
fixed fixtures, results and limitations.

## Verification

The implementation passed 1,819 Python tests (four tests skipped) and
12 Node adapter tests. Browser checks used isolated synthetic Codex and Pi
session observations, including absent effort and distinct provenance labels.
Adapter tests supply native-shaped host events; they do not establish that a
live Pi or OpenClaw installation was connected. The existing installation and
its configured providers were not changed by this worktree implementation.
