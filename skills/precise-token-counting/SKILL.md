---
name: precise-token-counting
description: Use whenever calculating, estimating, comparing, or verifying token counts for a prompt, text string, or model context, including during tests and benchmark analysis.
---

# Precise Token Counting

Use a tokenizer, not a character-ratio guess, whenever a token count can affect a claim or test result. Load this skill for every later token-counting test.

## Text counting

- For OpenAI text models, use local `tiktoken` with the model's documented encoding, or `encoding_for_model(model)` when the model is recognized. Record the model/encoding and library version.
- For open-source models, use `transformers.AutoTokenizer.from_pretrained` with the exact model ID and revision. Prefer cached/local files; do not download a tokenizer unless network access is authorized. Record tokenizer ID, revision, and library version.
- A count is exact only for the selected tokenizer and exact input string. Include any added prefixes, separators, and special tokens only when they are actually part of the serialized input. If model/tokenizer mapping is unknown or the library is unavailable, report the count as unavailable; a heuristic may be shown only as a clearly labeled estimate.

## Boundaries

- Text tokenizers do not count image/visual tokens. Image cost depends on provider, model, image dimensions, crops/patches, and transport. Without the provider's documented calculation or authoritative usage metadata, report visual-token count as unsupported; never infer it from pixels or OCR output.
- Plain-text counts do not include provider-specific chat framing, tool schemas, hidden system prompts, or multimodal payload overhead. Count these only from the exact serialized request or authoritative provider usage data. Do not label prompt-text counts as full request counts.
- Keep methods consistent across comparisons and tests. Report exact counts separately from estimates and state the tokenizer and scope beside each result.
