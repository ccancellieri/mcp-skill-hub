# Context composer

To start the UI without automatic background services or maintenance, run
`skill-hub-dashboard --no-services` and open `http://127.0.0.1:8765/context`.
The option is local to that process and does not stop services already running.

Open `/context` to prepare a reviewable context packet with a text-estimated
token budget. The normal workflow is manual. Enter the original prompt and
explicitly select project roots. Only those projects contribute memory and task
evidence; installed skill descriptions are shared catalog entries. No project
scope means global skills only. Scope is never learned.

1. Inspect each candidate's origin, update time, selection reason, and cost.
2. Include or exclude candidates. Expand a source if needed, or paste an exact
   continuous passage to shorten it.
3. Preview the packet, then copy it alongside the original prompt.

Preparing candidates stores a local draft containing the prompt and source
snapshots. Previewing stores the composition but does not record training
labels, start a task, send the packet to a client, or promote a model. The
normal web page has no training, feedback, or automatic-selection controls.
Task and session identities are optional evidence filters and must come from
the client; they never authorize additional project roots.

The server stores references and source fingerprints. Expansion and composition
revalidate indexed sources; changed or missing sources require a new selection.
Posted source bodies cannot replace verified evidence. Refreshing a file on disk
still requires the existing indexing workflow before the index reflects it.

Exact duplicate removal and JSON whitespace compaction are deterministic.
JSON compaction preserves strings, duplicate keys, and numeric spelling.
Passage selection and budget truncation are marked as lossy; no semantic
equivalence is claimed. A skill initially contributes its description rather
than its complete instructions. Full content is fetched only when requested.

Optional prompt compression is a separate explicit action. It conservatively removes
excess blank lines outside fenced code, shows the original, proposal, and diff,
and never silently replaces the submitted prompt. Code, negations, numbers,
paths, and constraints remain intact. Token counts throughout the composer are
text estimates, not measurements of the whole task.

## Experimental learning compatibility

Earlier Training, Automatic, and Mixed modes and their saved local data remain
in the Python API for offline experiments and compatibility. They are not
selected by the normal web route. The web learning, promotion, reset, and
outcome routes have been removed; existing drafts, labels, outcomes, and model
versions are not deleted. See [local learning](context-learning.md) for their
API and evidence contract. Manual composition does not load or use a promoted
selector, even if one is saved.

Indexed memory and wiki sources retain their original text alongside any digest
so ranking, preview, source expansion and freshness checks use the retained
original. Generated digests are not substituted for source evidence. This index
storage is separate from composer drafts and learning data; deleting learning
data does not delete indexed project sources. Legacy digest-only rows cannot
recover an original from the digest alone and are omitted with a warning. They
are rehydrated when an existing
indexing or retrieval path next supplies the verified raw source; otherwise the
digest cache row must be cleared and rebuilt during reindexing before full
expansion is available.

## Integration and limits

MCP exposes preparation, composition, source expansion, and explicit prompt
optimization. Experimental learning is retained for offline Python use.
The older context endpoint is retained. Copying the packet works when a client
has no native integration. Client identity/model observations never select the
server's credentials or providers.

The automatic hook keeps the original prompt, deterministic retrieval, no L1/L2
generation, and its two-second deadline. This composer does not install a new
hook, activate a backend, or change the currently running server.

The evaluation harness separates estimated payload reduction from native task
token usage. Synthetic retrieval tests and native smoke tests are useful
qualification evidence but do not establish savings on real development work.
See [the benchmark](../benchmarks/CONTEXT_VALUE.md). Until measured task evidence
passes the promotion gate, the learned selector remains inactive.
