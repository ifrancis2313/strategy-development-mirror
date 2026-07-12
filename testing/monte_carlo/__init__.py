"""Asset-class-agnostic Monte Carlo robustness suite for the Testing wing.

Two independent, complementary overfitting/robustness checks, each asking a
different question of a strategy's returns:

- `mcpt.sign_permutation_mcpt` -- "is this one strategy's edge distinguishable
  from noise?" Sign-permutes a returns/P&L series, rebuilds a null
  distribution of the same Sharpe statistic, and reports where the observed
  Sharpe falls (a p-value). Same Masters/Aronson sign-permutation
  methodology already embedded in `testing.engine.strategy_comparison.
  evaluate_strategy_pnl` (itself ported byte-for-byte from
  `Options_Trading_Strategy`/OFP's `mc_overfitting_mcpt.py` lineage, see ADR
  0011) and conceptually the same family of test as `alpha-pipeline`'s
  `sleeves/pairs_mft/mc_overfitting_mcpt.py` (that one permutes intrabar OHLC
  structure instead of trade signs -- a different, data-shape-specific
  variant of the same MCPT idea; sign-permutation is the one that only needs
  a plain returns series, which is this module's whole point).

- `pbo.probability_of_backtest_overfitting` -- "did the *selection process*
  overfit?" Given a parameter grid's worth of returns (one column per
  variant), runs Combinatorially Symmetric Cross-Validation (CSCV) and
  reports the Probability of Backtest Overfitting (Bailey, Borwein, Lopez de
  Prado & Zhu, 2017) -- the same method `alpha-pipeline`'s
  `research/mc_overfitting_pbo.py` already uses in production.

**Both take nothing but a plain numeric returns series/matrix as input.**
Neither module imports anything from `testing.engine` or
`testing.strategies` -- no options Greeks, no factor-model IC, no
prediction-market calibration. This is deliberate: the same two functions
apply unchanged to an options strategy's trade P&L, an equity factor
portfolio's daily returns, or a prediction-market strategy's settlement P&L.
`testing/strategies/vrp_harvester` is wired against these functions from the
*outside* (see `scripts/monte_carlo_vrp_harvester_demo.py`), not the other
way around.

**License note (`pbo.py` only):** the CSCV/PBO implementation wraps a
vendored, trimmed copy of `esvhd/pypbo` at
`~/Documents/Pycharm/research-methodology/vendored/pypbo-cscv/pbo.py`,
which is **GNU AGPL-3.0**. That's a different tier from every dependency
this repo otherwise uses (MIT/BSD/Apache). AGPL is fine for this repo's
local/personal-research use -- nothing here is served to third parties --
but must be re-checked before this code is ever hosted as a network-
accessible service or redistributed publicly. Same constraint already
flagged in `alpha-pipeline/research/mc_overfitting_pbo.py` and
`pipeline-dashboard/AGENTS.md` (search "pypbo" there); this is the same
library, same constraint, now a third place it applies. `pbo.py` in this
module does not vendor/copy that AGPL source into this repo -- it only
`sys.path`-imports it at runtime from its one existing vendored location,
so the AGPL code itself is never duplicated here.
"""
