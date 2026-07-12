# Shadow → Pipeline graduation gate: framework fixed now, exact thresholds deferred to issue 11 landing + real data

## Status

Accepted

## Context

Issue 14 needs a gate deciding whether a strategy running in Shadow has earned graduation into a Pipeline as a real Sleeve. Two dimensions, per the PRD: minimum live track-record length, and a check that live Shadow performance hasn't materially diverged from what Testing predicted. Same situation as issue 10's graduation bar (ADR 0003): some of this can be decided in principle now, some genuinely can't without real data that doesn't exist yet. Unlike issue 10, this gate's actual code also has a real blocking dependency — `shadow/ledger.py` (issue 11) is still mid-build in an isolated worktree as this ADR is written, so the gate's checking function isn't implemented yet, only its design.

## Decision

**Fixed now, as mechanism/principle rather than a number:**

- **Track-record length is not a fixed calendar duration or trade count set the same for every strategy.** A strategy trading a few dozen times a year and one trading weekly need genuinely different minimum windows to produce a statistically meaningful sample — a single fixed number would be either too strict for slow strategies or too loose for fast ones. The gate defines this as **"enough live trades/time that a live-vs-backtest divergence check has real statistical power,"** not a specific N. The exact power target (and therefore the actual resulting trade count per strategy) is deferred — see below.
- **The divergence check reuses Testing's existing Monte Carlo machinery, not a new statistical method.** `testing/monte_carlo/mcpt.py`'s null-distribution approach already answers "how does an observed Sharpe compare to what's expected" for a backtest; the same shape of question applies to Shadow — compare live Shadow's realized returns against the distribution Testing's backtest/Monte Carlo predicted, rather than inventing a second, different comparison method. Mirrors ADR 0003's own principle (reuse existing rigor machinery, don't reinvent per-stage).

**Deliberately left open, same reasoning as ADR 0003:** the exact statistical power target for "enough trades" and the exact divergence tolerance (how far live performance can drift from the predicted distribution before it's disqualifying) are both real numbers that need actual Shadow data to calibrate sensibly — setting them now would be a guess dressed up as a bar. Revisit once issue 11 lands and Shadow has run at least one graduated strategy for real.

**Code build order:** `testing/graduation/bar.py` (issue 10) was buildable immediately because its dependencies (`testing/monte_carlo/`) were already merged and stable. This gate's dependency (`shadow/ledger.py`, issue 11) is still in flight as of this ADR — the actual `shadow_graduation_gate.py` checking function is a fast, mechanical follow-up once issue 11 merges and its ledger data shape is final, not a separate open design question. This ADR fixes the framework so that follow-up is a short implementation task, not a re-litigation.

## Consequences

- Same shape of consequence as ADR 0003: a strategy can satisfy every currently-definable criterion and still need Zi's manual sign-off before graduating to a Pipeline, until the power target and divergence tolerance are set from real data.
- The gate's eventual code should report an explicit `not_yet_defined` verdict for both deferred numbers, never a silently-guessed default, matching `testing/graduation/bar.py`'s existing convention.
