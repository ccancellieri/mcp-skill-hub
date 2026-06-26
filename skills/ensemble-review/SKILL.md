---
name: ensemble-review
description: Use when the user asks for a "comprehensive code review", "multi-model review", "ensemble review", "deep review", or "parallel model review". Dispatches three specialized Claude model tiers (Opus, Sonnet, Haiku) to review code in parallel, each focusing on different aspects. Aggregates findings into a unified severity-ranked report. Invoke when reviewing critical code paths, production deployments, architectural changes, or when the user wants thorough analysis beyond single-model review.
---

# Ensemble Code Review

Single-model reviews miss things. This methodology orchestrates three Claude model tiers in parallel, each specializing in different review aspects, then aggregates findings into a unified severity-ranked report. The result is a comprehensive review that catches more issues than any single model could.

## When to use

- User explicitly requests "comprehensive review", "ensemble review", or "deep review"
- Reviewing production-critical code paths (auth, payments, data integrity)
- Architectural changes affecting multiple modules
- Before merging high-risk PRs (security-sensitive, breaking changes)
- Post-incident root cause analysis

## Phase 1: File identification

**Scope the review:**
1. Ask user for target files/directories, or use git diff to identify changed files
2. Expand to related files: imports, tests, dependent modules
3. Exclude generated files, vendored code, and lockfiles
4. Confirm scope with user before dispatching models

**Typical scope sizes:**
- Small (1-3 files): Full ensemble review appropriate
- Medium (4-15 files): Focus ensemble on changed files + direct imports
- Large (15+ files): Enforce user to narrow scope, or use this methodology iteratively

## Phase 2: Parallel model dispatch

Each model receives the SAME code but different instructions. Dispatch all three concurrently using Claude's Agent tool or parallel tool calls.

### Opus — Architecture & Design Review

**Focus areas:**
- Architectural patterns and their appropriateness
- Design pattern usage (Factory, Strategy, Observer, etc.)
- Extensibility and future-proofing
- Trade-offs and alternative approaches
- Separation of concerns / module boundaries
- Dependency direction and coupling
- API contract stability

**Prompt template:**
```
You are reviewing code with an ARCHITECTURAL focus. Analyze for:

1. **Design Patterns**: Are patterns used appropriately? Are there better alternatives?
2. **Extensibility**: Can this code adapt to future requirements without major refactors?
3. **Trade-offs**: What design trade-offs were made? Are they documented? Could they be improved?
4. **Coupling**: Are module boundaries clean? Are dependencies flowing in the right direction?
5. **API Contracts**: Are interfaces stable? Are breaking changes hidden or exposed?

For each finding, cite the exact file:line and explain:
- What exists
- Why it's problematic or excellent
- What alternative approach to consider

Format each finding as:
[SEVERITY] file:line — <one-line summary>
  <detailed explanation>
  Alternative: <specific recommendation>

Severity: CRITICAL (architectural flaw), HIGH (design issue), MEDIUM (improvement opportunity), LOW (minor suggestion)
```

### Sonnet — Code Quality & Correctness Review

**Focus areas:**
- Logic errors and off-by-one bugs
- Edge cases and boundary conditions
- Error handling completeness
- Race conditions and concurrency issues
- Security vulnerabilities (injection, auth bypass, data leaks)
- Test coverage gaps
- Resource leaks (connections, file handles, memory)

**Prompt template:**
```
You are reviewing code with a CORRECTNESS focus. Analyze for:

1. **Logic Errors**: Off-by-one errors, incorrect conditionals, missing null checks
2. **Edge Cases**: What inputs break this? What happens at boundaries?
3. **Error Handling**: Are all error paths handled? Are errors propagated correctly?
4. **Concurrency**: Race conditions, deadlocks, thread-safety issues
5. **Security**: Injection vulnerabilities, auth bypasses, data exposure, crypto weaknesses
6. **Resources**: Leaked connections, unclosed files, unbounded memory growth
7. **Test Gaps**: What code paths are untested? What edge cases need test coverage?

For each finding, cite the exact file:line and explain:
- The specific bug or vulnerability
- The failure mode (what goes wrong, how to trigger it)
- The fix (specific code change)

Format each finding as:
[SEVERITY] file:line — <one-line summary>
  Bug: <what's wrong>
  Trigger: <how to reproduce>
  Fix: <specific code change>

Severity: CRITICAL (security/data loss), HIGH (bug), MEDIUM (edge case), LOW (defensive improvement)
```

### Haiku — Style & Maintainability Review

**Focus areas:**
- Naming conventions (clarity, consistency)
- Code formatting and whitespace
- Function/method length and complexity
- Comment quality and necessity
- Dead code and unused imports
- Magic numbers and constants
- Documentation completeness

**Prompt template:**
```
You are reviewing code with a STYLE & MAINTAINABILITY focus. Analyze for:

1. **Naming**: Are names clear, consistent, and self-documenting?
2. **Formatting**: Consistent indentation, spacing, line length?
3. **Complexity**: Functions too long? Too many nesting levels?
4. **Comments**: Useful comments? Missing comments for complex logic?
5. **Dead Code**: Unused imports, unreachable code, obsolete comments?
6. **Constants**: Magic numbers that should be named constants?
7. **Documentation**: Module-level docs present? Public APIs documented?

For each finding, cite the exact file:line and explain:
- The style issue
- Why it matters (readability, maintainability, onboarding)
- The fix (specific refactor)

Format each finding as:
[SEVERITY] file:line — <one-line summary>
  Issue: <what's wrong>
  Impact: <why it matters>
  Fix: <specific change>

Severity: MEDIUM (consistency issue), LOW (minor improvement)
```

## Phase 3: Aggregation methodology

**Do NOT trust single-model findings blindly.** Each finding must survive deduplication and cross-validation.

### Step 3.1: Collect all findings

Gather findings from all three models into a single list. Each finding has:
- Source model (Opus/Sonnet/Haiku)
- Severity (CRITICAL/HIGH/MEDIUM/LOW)
- File:line location
- Summary
- Details

### Step 3.2: Deduplicate

Two findings are duplicates if:
- Same file:line (exact or within 3 lines)
- Same issue category (e.g., both flag missing null check)

**Dedup rules:**
- If multiple models flag the same issue: KEEP, note "cross-validated by X + Y"
- If one model flags an issue another model reviewed but didn't mention: KEEP, but flag for manual review
- Near-duplicates (same root cause, different line): MERGE into single finding with all affected lines

### Step 3.3: Severity adjustment

**Upgrade severity when:**
- Cross-validated by 2+ models (HIGH → CRITICAL, MEDIUM → HIGH)
- Security-related finding confirmed by code path analysis

**Downgrade severity when:**
- Only one model flagged it
- The "issue" is actually intentional design (verify against code comments/commit history)

**Never downgrade:**
- CRITICAL security vulnerabilities
- CRITICAL data loss risks

### Step 3.4: Citation verification

**Every finding MUST cite file:line.** Citations go stale. Verify:
1. The cited line exists
2. The code at that line matches what the model described
3. The line numbers are current (not from an old version)

If citation is invalid:
- If line moved: try to find correct line via git blame or code search
- If code changed: discard finding (stale review)
- If file doesn't exist: discard finding (wrong file path)

## Phase 4: Output format

Generate a structured report in Markdown:

```markdown
# Ensemble Code Review Report

**Scope:** <files/directories reviewed>
**Models:** Claude Opus (architecture), Claude Sonnet (correctness), Claude Haiku (style)
**Date:** <YYYY-MM-DD>

---

## Executive Summary

<2-3 sentences: overall assessment, critical issues count, top recommendation>

---

## Critical Issues (Blocking)

_These issues MUST be addressed before merge. They represent security vulnerabilities, data loss risks, or fundamental architectural flaws._

### 1. [Opus+Sonnet] Authentication bypass in token validation
**Location:** `src/auth/validator.py:142`
**Severity:** CRITICAL (cross-validated)
**Issue:** The token expiration check uses `>=` instead of `>`, allowing tokens that expire exactly at the current timestamp to pass validation.
**Impact:** Attackers can reuse expired tokens within a 1-second window.
**Fix:**
```python
# Before
if token.exp >= now:
    return True
# After
if token.exp > now:
    return True
```

---

## Recommended Improvements

_High-priority issues that should be addressed soon. They represent bugs, design issues, or significant technical debt._

### 1. [Sonnet] Missing error handling in payment processing
**Location:** `src/payments/processor.py:89`
**Severity:** HIGH
**Issue:** The `process_refund` method doesn't handle network timeouts, only HTTP errors.
**Impact:** Transient network issues cause uncaught exceptions, leaving refunds in an undefined state.
**Fix:** Wrap the HTTP call in a retry loop with exponential backoff, and ensure idempotency keys are used.

---

## Minor Suggestions

_Low-priority improvements for code quality and maintainability. Address opportunistically._

### 1. [Haiku] Inconsistent variable naming
**Location:** `src/utils/helpers.py:23`
**Severity:** LOW
**Issue:** Variable `usrId` uses camelCase while the codebase uses snake_case.
**Fix:** Rename to `user_id` for consistency.

---

## Positive Observations

_What the code does well. Acknowledge good practices to reinforce them._

1. **[Opus]** Clean separation of concerns between `PaymentService` and `PaymentGateway` — each has a single responsibility.
2. **[Sonnet]** Comprehensive test coverage for edge cases in `date_utils.py` — leap years, timezone transitions, and invalid inputs are all tested.
3. **[Haiku]** Clear, self-documenting function names throughout `src/api/` — `get_user_by_email`, `validate_payment_amount`, etc.

---

## Review Statistics

- **Total findings:** 12
- **Critical:** 1
- **High:** 3
- **Medium:** 5
- **Low:** 3
- **Cross-validated findings:** 2
- **False positives filtered:** 1

---

## Appendix: Model Coverage

| Aspect | Model | Findings |
|--------|-------|----------|
| Architecture & Design | Opus | 4 |
| Code Quality & Correctness | Sonnet | 6 |
| Style & Maintainability | Haiku | 2 |

_This report was generated by the ensemble-review skill. Each finding has been deduplicated, cross-validated, and citation-verified._
```

## Common mistakes

- **Dispatching models sequentially instead of in parallel.** The whole point is parallelism. Use concurrent tool calls.
- **Accepting model findings without citation verification.** Models hallucinate line numbers. Always verify.
- **Skipping the deduplication step.** Three models will often flag the same issue. The report should show it once, with "cross-validated" notation.
- **Treating all findings as equal.** Critical issues block merges; minor suggestions are nice-to-haves. Make this clear in the report.
- **Not confirming scope with user first.** Reviewing too many files produces noise; reviewing too few misses context.
- **Reviewing generated code.** Lint/format tools handle generated files; don't waste model capacity on them.

## Model selection rationale

Why this tier assignment works:

- **Opus** has the deepest reasoning capacity for architectural trade-offs and long-term design implications. It's expensive but necessary for the "strategic" layer of review.
- **Sonnet** balances reasoning speed with quality for correctness checks. It's the workhorse for finding bugs and security issues.
- **Haiku** is fast and cheap for surface-level checks. Using a larger model for style issues is overkill.

This tiered approach costs ~40% less than running Opus alone while catching more issues through specialization.

## Integration with mcp-skill-hub

The ensemble-review skill can leverage mcp-skill-hub capabilities:

1. **Token tracking:** Each model dispatch is logged via `log_session` for cost analysis
2. **Feedback loop:** Use `record_feedback` after user reviews findings to improve future routing
3. **Teaching rules:** After a successful ensemble review, extract patterns via `teach()`:
   ```
   teach(
       rule="when code has auth/payment/data-integrity logic",
       suggest="always use ensemble-review before merge, focusing Sonnet on security checks"
   )
   ```

## Related skills

- `memory-layer-analysis` — for categorizing findings when they accumulate
- `master-state-compaction` — if findings reveal architectural issues that need snapshot updates
