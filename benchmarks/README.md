# Context Comparison

This evaluation tests the claim that the context-service refactor both reduces
input tokens and improves retrieved evidence. It is allowed to reject that
claim. It does not measure final answer quality, billed usage, or complete
coding sessions.

## Comparison Contract

- Compare the actual router at baseline revision `77d5561` with the refactored
  revision `9f74111`, using the same synthetic corpus and independent case labels.
- Include the baseline default configuration and its optional prompt-rewriter
  profile. Do not choose the more favorable baseline after seeing results.
- Disable model inference, network calls and provisioning. Isolate runtime data
  and settings from the user's installation. Document any substituted boundary.
- Preserve the raw outputs and corpus hash so a reviewer can inspect failures,
  not just averages. Never publish private prompts, memory or live database data.
- Keep fixtures fixed after the first measured run. Corrections to invalid
  labels require an explicit explanation and a new corpus version, not silent
  changes that improve a score.

## Token Accounting

Count text using `tiktoken` with `o200k_base`, reporting the tokenizer version.
This is a common comparison tokenizer, not a claim of exact Claude, Pi,
OpenClaw, or every Codex model's billing. Do not substitute character estimates.

The Claude hook's `additionalContext` is model input. Its top-level
`systemMessage` is a user-facing warning and must not be counted as model
context. Record it separately for inspection. The original user prompt is
counted once, plus all supplemental text actually handed off; duplicated prompt
text in supplemental context still costs tokens. See the
[official hook contract](https://code.claude.com/docs/en/hooks).

Also report the original prompt without enrichment. This is a lower-input
baseline, not automatically the best-quality condition. Excluded costs include
conversation history, tool schemas, provider message framing, output/reasoning
tokens, retries, and cache billing.

## Quality And Decision

Quality measures concern source selection: precision, recall and F1 on cases
with expected evidence, no-answer handling separately, and forbidden-source
leakage. These are proxies for useful prompt context, not an evaluation of the
model's eventual answer. Source labels must not be inferred from the router's
own rankings.

A claim of simultaneous improvement requires fewer total measured input tokens
and better evidence-selection scores against both declared historical profiles,
without worse forbidden-source leakage or prompt preservation. Report all cases
and subgroup results, including regressions. Results on this deliberately
constructed corpus do not establish a production-wide average or causality for
coding quality.

If either improvement is absent, do not publish a post asserting both. Record
the negative result and the next experiment instead. A stronger claim requires
a separate paired task evaluation with fixed models, tools, budgets and
independent outcome scoring.
