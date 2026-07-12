# Everything tested on the fixed historical sample is exploratory, not confirmatory — the real fix is genuinely new data, not a screening step

## Status

Accepted (revised — the initial version of this ADR overclaimed what a full-sample pre-screen can do; see "Superseded reasoning" at the bottom)

*(Redacted for this mirror: strategy names, specific parameter choices, and numeric results have been generalized or removed. The methodological argument is unchanged from the original.)*

## Context

`testing/validation/walk_forward_validation.py` is leak-free *within a single call* (verified directly): each fold's train/test windows are date-disjoint (`test_start == train_end`), candidate selection only ever sees the train slice, and the test slice is scored exactly once, after selection. ADR 0005 was right that this beats a single static split for any one experiment.

What ADR 0005 didn't address, and what this ADR is actually about: a long sequence of research threads — spanning parameter tuning, several structural variants, and multiple candidate signal ideas — has run against the exact same fixed historical sample, a research-licensed options chain dataset covering a multi-decade window with no more recent data available from that source. That's a dozen-plus distinct ideas, all mining the same finite sample.

Two separate mechanisms drive risk here, and they don't respond to the same fix:

1. **Selection bias in an already-computed result.** An adopted configuration was chosen as the best performer among many alternatives. Its reported performance is a "best-of-N" statistic — that bias is fixed the moment the selection happened and does not depend on anyone remembering the other attempts. A future session inheriting "current best is X" from the codebase inherits an already-selected result, not a naive first test.
2. **Researcher-degrees-of-freedom / data-snooping via calibrated priors.** Each successive idea's *design* — not just its score — is shaped by having watched what worked on this exact sample. A dozen threads deep, parameter and structure choices are no longer independent draws from a genuine prior; they're informed by a posterior already conditioned on this data. This makes each new test *more* prone to overfitting than the last, not equally prone — a fundamentally different (and worse) shape of risk than #1.

**Neither of these resets with a memory wipe, a new session, or forgetting.** #1 is a fact about a selection event that already happened, independent of who remembers it. #2 operates through code/config artifacts (adopted defaults, the very existence of a given candidate as "the strategy worth building") that persist regardless of session boundaries — a "fresh" session reading the repo inherits the posterior, not a blank prior. The dataset itself doesn't have a literal problem (querying it doesn't damage the bytes) — what depletes is the family-wise error budget for any claim of significance made against it, which is a fact about the cumulative research process, not about storage.

**ADR 0003's Bonferroni correction does not cover either mechanism** — it only corrects for however many variants were tested within one grid (e.g. one structural sweep), not across the dozen-plus threads that have touched this dataset.

## Decision

**Everything tested against the fixed historical sample — no matter how many statistical screens or corrections it passes — is labeled exploratory, never confirmatory, in any external-facing writeup.** There is no process fix (screening, correction, or otherwise) that changes this while the underlying data stays fixed and finite. Two concrete consequences:

1. **A cheap full-sample screen before spending a walk-forward run stays in place, but only as a compute-cost heuristic, not a bias fix.** Before running a candidate through the full validation pipeline, check full-sample performance against the current baseline and a nominal (uncorrected) significance threshold. This reduces how many ideas reach the more granular fold-by-fold out-of-sample view (which is the richer, more tempting-to-adjust-around information) and saves compute on weak candidates. **It does not reduce the real risk**, because the screen itself queries the same fixed data (full-sample includes every year walk-forward later folds into test windows) — it does not preserve any unspent sample, and it doesn't touch mechanism #2 (researcher priors) at all.

2. **The real fix is genuinely new data, and a plan for it is adopted here, blocked on licensed data access:**
   - The final graduation gate for any strategy is **Shadow's live track record (ADR 0004) AND a walk-forward check against data genuinely never touched by any Testing-wing experiment** — a later window sourced from a properly licensed feed, once that subscription exists. This cannot be an extension of the existing research-only sample (a restricted, personal-license-derived extract that isn't this project's to redistribute or extend, and isn't expected to be refreshed regardless).
   - **Strict protocol, fixed now before any of this data exists, specifically to stop this reserve from being quietly re-mined the same way the original sample was:** each fresh-data slice is spent on exactly one independently-conceived candidate. If that candidate fails (either gate), **the candidate is retired, not tweaked** — the next fresh slice goes to a genuinely different strategy idea, never a revised version of the failed one. Revising a strategy in response to a fresh-data failure and retesting it on a new slice reintroduces the exact researcher-degrees-of-freedom problem this reserve exists to avoid, just relocated onto a much scarcer resource that would then run out fast, each redraw shorter and lower-power than the last.
   - Not yet decided: how the reserve gets partitioned across however many candidates are realistically expected to reach this gate in the near term. Default until decided: draw sequentially, one slice per candidate, only once that candidate has already cleared Testing and Shadow on its own.
   - This complements Shadow rather than replacing it — Shadow's clock is slow (ADR 0004: track-record length scales with a strategy's own trade frequency), so a fast historical check against fresh data is a legitimate, cheaper pre-filter before committing a candidate to Shadow's real-time clock.

## Consequences

- Any number quoted from the fixed historical sample — including any current-best configuration — should be presented (in a post, in outside review, anywhere external) as "what looked best after extensive exploration," never as "a validated result," regardless of which statistical bars it cleared along the way.
- The full-sample pre-screen stays as a cost-saving step only; do not describe it externally as a bias mitigation, since it isn't one.
- The fresh-data gate cannot be exercised until a licensed post-sample data source exists — this ADR fixes the protocol now so that build isn't a re-litigation once the data is available, mirroring how ADR 0004 fixed its framework ahead of `shadow/ledger.py` landing.
- The one-shot-per-candidate rule is a real constraint on research pace once it's active — a tempting revise-and-retry will have to be resisted in the moment, not just on paper.

## Superseded reasoning (kept for record)

The original version of this ADR proposed the full-sample pre-screen as a fix for cross-experiment overfitting risk. On examination, it does not: the screen queries the same fixed dataset a walk-forward run would use, so it doesn't preserve any unspent sample, and it does nothing about the researcher-degrees-of-freedom mechanism (#2 above), which operates through idea design, not through which validation function gets called. Keeping this section rather than deleting it, since the reasoning-correction itself is worth being able to point to later.
