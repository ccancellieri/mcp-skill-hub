# Deterministic compression and source fidelity

The default compression path removes insignificant whitespace from valid JSON
objects and arrays. It does not parse and reserialize values: duplicate keys,
number spellings, escapes and string contents remain intact. Invalid JSON,
prose, source code and unsupported formats pass through unchanged.

Both `compress_payload` and `maybe_compress` default to `allow_lossy=False`.
They run locally without a model or an optional dependency. This avoids an
auxiliary inference call; it does not make the resulting client input free.
Whole-task token savings must include any subsequent rereads and recovery.

## Lossy transformations

A caller can explicitly request repeated-log-line collapse with
`allow_lossy=True`. Only recognized log-level lines are eligible; code-like,
JSON-like and repetition-marker-bearing input is excluded. This transform is
reported as `lossy=True`: a repetition count is not an exact recovery format.
It does not automatically store the original. Keep it disabled when exact
output is required or the caller has no original to recover.

Selection, excerpts and truncation are also lossy even when deterministic.
Never describe a transformation as lossless merely because it makes no model
call. A recovery marker is usable only if the corresponding original was
actually stored and remains accessible.

The optional `kompress_prose` helper in the webfetch/search lane remains
separate. It uses a model, may delete context that changes meaning, and is not
part of the automatic prompt hook. This change does not activate it or revive
the retired advanced compressor from #119.

## Evidence and accounting

Compression results and events report UTF-8 byte counts. The minimum-size gate
still uses the documented approximate character-to-token estimate; byte counts
are not tokenizer measurements. Historical events retain their old values.

The scoped context service ranks and excerpts retained original memory/wiki
text. It does not inject generated digests as a substitute for the source.
Legacy rows containing only a digest are omitted with a recovery warning;
original rows and generated text remain stored for review and reindexing.

The manual composer reuses the same JSON whitespace compactor. It preserves
source references and distinguishes compaction from exact excerpt selection.
Optional model-based curation remains an explicit/background operation and
cannot confer source authority on generated claims.

## Text and image context across providers

For textual sources, select bounded, verbatim passages with source metadata and
send them as text. This is the portable default across providers and clients.
Rasterizing text into an image is an optional transport experiment, not a
compression rule: image byte size does not predict billed tokens, and image
support, resizing, media types, and usage reporting vary by model and client.

Use an image when the source is inherently visual, or when a comparison on the
same selected passages shows a useful accuracy/cost tradeoff for the specific
provider, model, and client. If image capability or comparable usage is unknown,
keep the text path. Preserve the same source IDs and content hashes in each
format so answers remain attributable and comparisons are auditable.

Context retrieval supplies evidence only. It does not grant outbound-data
permission; any explicit operator policy and the client's native approval
decision remain separate from the material sent to a model.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `compression_enabled` | `True` | Enable the deterministic pass. |
| `compression_min_tokens` | `200` | Approximate minimum input size. |
| `compression_context_aware` | `True` | Retained query-context plumbing; the current JSON transform does not rank content. |

No package or model download is needed for the default pass. Unsupported input
is returned intact rather than forced to meet an advertised compression ratio.
