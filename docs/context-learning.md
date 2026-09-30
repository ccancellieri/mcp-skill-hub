# Local context selection learning

This is an experimental offline API retained for compatibility with earlier
Training, Automatic, and Mixed composer modes. The normal `/context` web flow
uses manual selection and does not collect learning labels, train or promote a
selector, or record outcomes. The former web learning and outcome routes are
not available. Existing datasets, candidate text snapshots, outcomes, and
model versions remain in the local Skill Hub SQLite database. This subsystem
sends no telemetry, performs no upload, calls no model, and adds no dependency.

## Safety model

Automatic compositions do not become training data. A composition contributes
labels only when a person confirms it and its mode is `training` or `mixed`.
Selected candidates are positive examples, explicitly rejected
candidates are negative examples, and other candidates remain unlabelled.
Repeated submission of the identical composition ID and payload is idempotent;
conflicting feedback for an existing ID is rejected.

Training requires at least 50 unique labelled compositions, both label classes,
and enough distinct task groups for a holdout. A task group includes project
scope and native task identity, with native session identity as the fallback.
Compositions without either identity are retained for local inspection but are
excluded from training; prompt similarity is never used to guess task identity.
Whole groups are selected from newest to oldest until the holdout contains at
least 20 percent of compositions. This targets composition volume even when
group sizes differ, while ensuring a task is never split between training and
holdout. Group chronology uses the latest composition timestamp in each group.

The selector is dependency-free regularized logistic regression over six
features: lexical relevance, source kind, exact project match, freshness,
redundancy, and token length. Training order and inference are deterministic.
The returned `learning_score` is useful only for ordering candidates; it is not
a calibrated probability or a universal relevance threshold.

Each model stores a fixed zero-logit selection threshold. This is the natural
linear-logistic decision boundary and is not tuned on holdout data. Evaluation
reports holdout precision, recall, and abstention at that boundary together
with threshold provenance. Ranking returns the promoted model's threshold so
automatic selection can abstain below it without treating scores as calibrated
confidence.

Newly trained versions are saved for offline evaluation but are not activated.
Their evaluation is marked `known_not_proven` because held-out selection labels
cannot establish end-to-end task success or token savings. When evidence or a
budget boundary is missing, status remains `needs_review`.

Promotion requires a complete report from the evaluation harness. Aggregate
numbers supplied by a caller are insufficient. The report must bind both the
candidate selector version and its training dataset hash, report identical
positive `expected_runs` and `completed_runs`, and include every underlying
run. The benchmark must contain at least 24 distinct tasks, cover both the
Skill Hub and Tellurion projects, and give every task at least two repetitions
with paired `baseline` and `candidate` conditions. Runs must contain native,
non-estimated main and auxiliary token counts plus boolean success and
critical-error outcomes.

Skill Hub derives the acceptance metrics from those paired rows and requires:

- at least 15 percent median task token reduction;
- no aggregate success regression;
- zero critical errors.

This gate validates the report's structure, version and dataset binding, run
completeness, and derived metrics. It does not independently authenticate the
caller or prove that submitted rows came from the named harness. Operators must
treat the report source as locally supplied evidence and protect write access to
the Skill Hub database and promotion API accordingly.

Promotion is explicit and supports rollback by promoting an older stored
version with qualifying evidence. Ranking uses only the promoted version. With
no promoted model, `rank_candidates` returns the original candidate order with
`available: false` and `promoted: false`.

## Python API

```python
from skill_hub.context_learning import (
    export_learning_data,
    get_learning_status,
    promote_selector,
    rank_candidates,
    record_composition,
    record_outcome,
    reset_learning,
    train_selector,
)
```

`record_composition(composition, *, store=None)` accepts the composition ID,
original prompt, candidate snapshots, selected and rejected IDs, excerpts,
mode, confirmation flag, task and session IDs, and selector version. Candidate
snapshots contain candidate ID, kind, title, source, text, project root, source
hash, and optional precomputed features.

`record_outcome(composition_id, outcome, *, store=None)` stores outcome evidence
independently, so a delayed outcome does not rewrite selection feedback.

`train_selector(*, store=None)` returns a readiness reason or a saved version,
split summary, weights, and offline evaluation. `promote_selector(version,
evidence=None, *, store=None)` activates a version only after its bound,
complete paired-run evidence passes the gate. It does not trust caller-supplied
aggregate acceptance fields.

`rank_candidates(prompt, candidates, *, store=None)` returns `candidates`,
`version`, `available`, and `promoted`. It preserves candidate dictionaries and
adds `learning_score` only when a promoted model ranks them.

`get_learning_status`, `export_learning_data`, and `reset_learning` provide
local inspection, local in-memory export, and complete deletion respectively.
Reset removes composition snapshots, labels, outcomes, every model version, and
the promoted-version pointer.
