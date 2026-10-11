# Foreground skill relevance qualification

Measured 2026-10-11. The deterministic repair reduces irrelevant suggestions,
but **does not qualify issue #158 for closure**: both languages fail the
preregistered 90% correct-abstention requirement.

The public fixture contains 48 cases, split into 24 calibration and 24 holdout
cases. Each split has twelve English and twelve Italian prompts, with zero,
one and multiple expected skills. The twelve synthetic skills use English
metadata. Expected source IDs grade actual selected skills, rather than packet
length or similarity scores. Full prompts, expected IDs, selected IDs, packet
contents and measurements are in [the JSON report](FOREGROUND_RELEVANCE_RESULTS.json).

| Holdout metric | English baseline → repair | Italian baseline → repair |
| --- | --- | --- |
| Irrelevant skill injections | 9 → 1 | 7 → 1 |
| Recall, mean across positive cases | 100% → 100% | 66.7% → 91.7% |
| Micro precision | 47.1% → 88.9% | 41.7% → 87.5% |
| Correct abstention, six negative cases | 1/6 → 5/6 | 1/6 → 5/6 |
| Explicit identifier recall | 100% → 100% | 100% → 100% |
| Packet UTF-8 bytes, twelve cases | 3,806 → 2,274 | 2,938 → 2,101 |

The remaining false positive is a PDF-extension definition question in each
language. One Italian request for both presentations and sheets misses the
presentation skill. These observed holdout failures were not used to tune the
definition guard or name-priority rule. Original prompts were preserved and
there were no retrieval failures.

The fixture was frozen before the first ranking change, with SHA-256
`7aa0bf0e77811ec5f8db059244548bec41d9e18d5b1fe8cf432e43256a7cbdb9`.
Fixture labels were known during design: this is a constructed public
qualification, not an independent sealed evaluation. The report retains the
baseline and first negative candidate unchanged. An independent code review
then reproduced two separate regressions: operational acronym-parser requests
were suppressed, and explicit skill names could disappear behind 100 description
matches. Their fixes, including the same saturated lookup edge for legacy unnamed skills, have dedicated regression tests. The subsequent reviewed
rerun happened after holdout inspection, has the same quality counts, and is
not a new independent qualification.

The repair changes skill matching only: generic follow-ups abstain, artifact
terms support Italian requests, metadata matches are bounded, and explicit
IDs/names take priority. Task and memory tokenization retain their prior rules.
No model calls, provider changes or client configuration changes were made.

A separate read-only fresh-process check against the installed catalog returned
zero skills for the current Italian follow-up and for “Run the tests now.”
It returned four for an Airflow DAG request and three for a screen-reader
request; those positive packets were not relevance-graded. Existing connected
MCP workers were not restarted, so this does not qualify actual-client behavior.

Timings measure retrieval inside `build_context`, excluding process startup,
client transport and an enforced hook deadline. Byte counts are not native token
counts or complete-task savings. No production-wide quality or speed claim is
supported. A fresh evaluation is needed before closing #158; provider smoke,
client qualification and complete-task value remain separate gates.

Validation: the offline suite passed 2,138 tests with 18 skips and eight existing
dependency deprecation warnings. That run began before the final legacy-name
lookup refinement; all 130 affected tests then passed on the final source.
Independent final review passed 39 focused tests and found no actionable defect
in the examined invocation, guard and lookup paths.
