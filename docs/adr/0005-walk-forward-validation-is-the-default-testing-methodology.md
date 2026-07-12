# Walk-forward validation is the default methodology for any parameter tuning or train/test claim in this repo

## Status

Accepted

## Context

An early visualization script's train/test split was a single static 80/20 chronological cut of the trade window — explicitly documented at the time as a labeled placeholder, not a real methodological split, since nothing in the codebase recorded what date range any strategy's parameters were actually tuned against. That was fine for a first illustrative chart. It stopped being fine the moment real tuning work started on a strategy's own threshold parameter: a single static split only answers "does this parameter look good on one particular boundary," which is exactly the shape of question that produces an overfit parameter that happens to work on that one boundary and nowhere else — the same failure class MCPT/PBO already exist to catch at the strategy level, just unaddressed at the parameter-tuning level.

## Decision

**Walk-forward validation (rolling train/test windows, not one static split) is the default methodology for any parameter tuning or any claim that a result generalizes, repo-wide** — not just for this VRP harvester session. Concretely: `testing/validation/walk_forward_validation.py`'s `run_walk_forward_validation`:

- Splits the full data range into sequential, rolling `(train_window, test_window)` folds (default: 4-year train, 1-year test, 1-year step — non-overlapping test windows tile the whole out-of-sample history; train windows overlap/slide, the classic Pardo-style rolling design).
- Within each fold, every candidate parameterization is run on the train window only; the best-scoring candidate (by Sharpe) is selected, then run *once* on that fold's test window. The test window is never used to pick the parameter — it only ever scores an already-picked one.
- Reports both in-sample and out-of-sample Sharpe per fold, plus a walk-forward efficiency (WFE) ratio — but **WFE is only reported when in-sample Sharpe is positive**. A ratio of two negative Sharpes can look reassuring by pure arithmetic sign cancellation while both periods are actually losing; per this repo's existing convention (`MCPTResult.p_value`, `PerformanceAnalyzer`'s NaN-for-degenerate fields), a misleading number is worse than an explicit `None`.
- The single static 80/20 split in the visualization layer is superseded by this for any actual tuning claim — it remains fine as a purely illustrative marker on a chart that isn't making a generalization claim, but no threshold/parameter decision should be justified by it going forward.

## Consequences

- Tuning a strategy's parameter is now more expensive (N folds × M candidates, vs. one full-sample run) — accepted, since the PRD's own testing decisions already prioritize real robustness over speed, and the alternative (a static split) produces evidence too weak to trust a real decision on.
- A future strategy/tuning session that skips walk-forward and reports only a full-sample or single-split result should be treated as provisional, not a real robustness claim, until run through this module.
- This doesn't replace MCPT/PBO (ADR 0003's graduation-bar criteria) — those test whether one already-chosen configuration's edge is real; walk-forward tests whether the *choice* of configuration itself generalizes across time. Both are needed; neither substitutes for the other.
