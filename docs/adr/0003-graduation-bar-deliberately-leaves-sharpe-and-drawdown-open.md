# Graduation bar: statistical criteria are fixed now, Sharpe/drawdown floors are deliberately left open

## Status

Accepted

## Context

Issue 10 required an explicit bar for what "passing" Testing means before a strategy graduates to Shadow. Two kinds of criteria were on the table: statistical rigor checks (MCPT, PBO) that don't depend on knowing anything about this repo's actual strategies yet, and outcome-level floors (minimum Sharpe, maximum drawdown) that do.

## Decision

**Fixed now, grounded in existing precedent rather than invented:**
- **MCPT**: a strategy's observed Sharpe must survive a **Bonferroni-corrected** p-value, not just uncorrected p<0.05 — the correction denominator is however many strategy/config variants were tested in that session. Matches the multiple-testing discipline already required by `testing/metrics/factor.py` and `alpha-pipeline`'s own Model 5 validation.
- **PBO**: must be **below ~0.26–0.29** — `research-methodology`'s own synthetic-validation reference band for "the search doesn't look overfit," the same band alpha-pipeline's real Model 5 result (PBO=0.168) was judged against.
- **Named source/model**: every strategy under test must cite a real thesis (a paper, a documented market mechanism, or prior ADR reasoning) — not just backtest code with no stated edge hypothesis. Adopted from `oracle3`'s rigor bar.
- **Sensitivity analysis**: any strategy with a calibrated/fitted parameter needs a documented sensitivity check on that parameter before graduating — a "Greeks"-style check on the model's own assumptions. Also adopted from `oracle3`.

**Deliberately left open, not defaulted to a guessed number:** minimum acceptable Sharpe and maximum acceptable backtest drawdown. Zi's explicit reasoning: it's premature to set an outcome floor before this repo has actually run a real body of strategies through Testing — a number picked now would be a guess, not a calibrated bar, and risks being either too strict (nothing graduates) or too loose (something weak graduates) with no empirical grounding either way. Revisit once enough real Testing results exist to see the actual achievable distribution.

## Consequences

- `testing/graduation/bar.py` checks only the two quantitative criteria that are actually defined (MCPT, PBO) programmatically; named-source and sensitivity-analysis are manual checklist items (not automatable — "is there a real thesis" is a judgment call); Sharpe/drawdown are explicitly reported as `not_yet_defined`, not silently skipped or defaulted to a placeholder number that could be mistaken for a real bar.
- A strategy can pass every currently-defined criterion and still require Zi's manual sign-off before graduating, until Sharpe/drawdown floors are set — this is intentional, not a gap to be closed automatically.
- Revisit this ADR once a real distribution of Testing results exists (see the collaborative-workflow convention in `AGENTS.md` for how strategies get run through Testing in the meantime).
