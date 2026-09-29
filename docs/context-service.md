# Context retrieval

The interactive hook retrieves evidence for the current request. It does not
choose the client's model, rewrite the user's words, create tasks, provision
tools, approve commands, or send automatic continuation messages.

## Entry points

- The Context page previews the original prompt and supplemental evidence
  separately, including sources, omissions and elapsed retrieval time.
- The `prepare_context` MCP tool accepts text, an absolute `repo_root`, and
  optional session/task identity. Call it when starting work or changing topics.
- The prompt hook calls the same deterministic service with the identity supplied
  by the client event. A global active-task marker is not used.
- `improve_prompt` remains a compatibility wrapper. It preserves the original
  text and appends evidence. Language normalization has been retired.

MCP availability does not imply a client has automatic prompt hooks. In clients
without the hook integration, call `prepare_context` explicitly. The service
does not claim it can inspect or change the client's permission settings.

## Boundaries

Project context requires project identity. Missing identity must not fall back
to recently opened tasks in another project. Retrieved text is evidence with a
source, not a new user instruction or authorization. Publication permission
must come from the user's actual instruction with the applicable scope; a
summary, skill or automatic message cannot supply it.

Memory and wiki candidates require retained original indexed text. Generated
digests do not replace that text for relevance or evidence. Digest-only legacy
rows are omitted with a reindex warning; retrieval does not delete them.
Historical task summaries may still contain pollution from retired writers;
project metadata alone is not proof that every sentence belongs to the project.
The bounded memory path searches the first 1600 characters of each source and
reports when a longer tail was not searched. Explicit composition searches the
full indexed source; deeper automatic excerpt selection remains separate work.

Skill descriptions are retrieved first; full skill content remains available
through the existing skill tools. More injected skills is not a success metric.
Relevance, source coverage and correctness matter more than corpus coverage.

The offline selector experiment has a separate, broader shortlist that scans
names and full descriptions in SQLite without loading instruction bodies. A
name match boosts its ranking but does not exclude other description matches.
Token boundaries, accent and English plural normalization, and a small Italian
artifact vocabulary support deterministic matching. This is lexical matching,
not general multilingual semantic understanding. Explicit `$name` and
`$plugin:name` references receive priority. Results have stable ties and a maximum
of 20 candidates, with bounded snippets for rendering.

That broader shortlist and the Qwen reranker are not enabled in the foreground
service, hook, composer, or MCP server. Independent evaluation found that directly
injecting the broader list increased irrelevant context. The existing foreground
retrieval remains in place. The experimental metadata scan costs more as the
catalog grows; a future integration must satisfy the unchanged hook deadline
and whole-task quality/token criteria before it can replace the current path.

Older indexes sometimes store an encoded project name instead of a canonical
path. Configure `context_project_aliases` as an object mapping each absolute
project path to its verified legacy names. An alias assigned to two projects
must not be used to combine their records. Missing or ambiguous provenance is
reported as an omission, not guessed from a directory basename.

The hook has a configurable deadline (`context_hook_timeout_s`, default two
seconds). Failure or timeout emits no supplemental context and lets the user's
prompt pass through. Character and item limits are controlled by
`context_max_chars` and `context_max_items`. Disable enrichment with
`context_enabled=false`. Retrieval needs no model, embedding service or API key.

## Migration

The old auto-proceed hook is a compatibility no-op, including with old enabled
settings. Automatic task interception and session task creation hooks are also
retired. Per-turn Stop memory maintenance is removed from the foreground path;
session-close and explicit curation tools remain available. Reinstalling updates managed hook
registrations while preserving unrelated hooks.

Permission handling defaults to `hook_approval_policy=native`, which leaves
the decision to the client. An explicitly selected deterministic policy can
still enforce configured rules. Time-based relaxation, semantic similarity,
LLM guesses and inferred approval caches do not grant permission.

Legacy settings and historical activity may remain in saved configuration and
analytics for inspection. Their presence does not reactivate retired hooks.
No task, memory, skill source or historical record is deleted by this migration.

## Models and background work

L1 means a local model, L2 a remote model accessed through an API, and L3 the
main coding agent in the client. These describe deployment, not reasoning
difficulty. The interactive context path requires none of L1/L2.

Keep indexing and memory synthesis outside the prompt path. Existing explicit
curation tools remain available. A small shared model for optional reranking
should only be introduced after a comparison demonstrates better relevance
within a fixed latency and token budget. Do not launch an L3 CLI recursively
from a prompt hook or treat a client's subscription as an API credential.

## Verification

Regression cases cover original multiline text, unrelated projects, missing
scope, foreign task identity, unavailable data, disabled enrichment, size limits,
retired continuation and legacy approval settings. Compare retrieval time and
source relevance on representative prompts before enabling optional ranking.
These checks measure the context layer; they do not establish an improvement
in end-to-end coding quality without a separate agent evaluation.
