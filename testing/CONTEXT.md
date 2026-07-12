# Testing Wing — Engine Architecture

The Testing wing's backtest/statistical-validation engine. `engine/` is strategy-agnostic infrastructure; `strategies/<name>/` is where each individual strategy's signal and entry/exit logic lives. The goal: adding a new strategy should mean writing a new `strategies/<name>/` package, not touching `engine/`.

Root `CONTEXT.md` (Wing/Graduation/Pipeline/Sleeve) is the repo-level glossary; this file is the Testing wing's own architecture map, one level down.

*(This mirror omits the private repo's own "what's ported / what's not" changelog and the numeric results it recorded — see the root README for why. `testing/strategies/toy_example/` is a synthetic fixture demonstrating the extension contract below, not a real strategy.)*

**`monte_carlo/` — asset-class-agnostic:** `mcpt.py` (`sign_permutation_mcpt`) and `pbo.py` (`probability_of_backtest_overfitting`) both take nothing but a plain numeric returns/P&L series (or, for PBO, a returns matrix) and an explicit annualization factor; neither imports anything from `engine/` or `strategies/`. `mcpt.py` implements the Masters/Aronson sign-permutation method as a standalone function that doesn't need a trade-level equity curve to infer its own annualization. `pbo.py` wraps a vendored CSCV/PBO implementation.

**License note:** `pbo.py` depends at runtime on a vendored copy of `esvhd/pypbo`, which is **GNU AGPL-3.0** — a different tier from every other dependency in this repo (MIT/BSD/Apache). Fine for local/personal-research use; re-check before this code is ever hosted as a network-accessible service or redistributed publicly. `pbo.py` does not copy the AGPL source into this repo — it only `sys.path`-imports it at runtime from wherever it's vendored locally.

## Adding a new strategy

A new strategy needs, at minimum:

1. **A config** — a dataclass extending `testing.engine.execution_config.ExecutionConfig`, adding whatever fields the strategy's signal/structure needs (inherit, don't compose, so every field stays flat).
2. **A selector** — any class with a method shaped like:
   ```python
   def select_trade(self, chain_snapshot: pd.DataFrame, entry_date, has_open_position: bool, **kwargs) -> PositionSpec | None
   ```
   It builds a `PositionSpec` (a flat `List[Leg]`, however many the structure needs) using `StrikeSelector` for whatever strikes the strategy targets. Reuse `testing.engine.strike_selector.select_target_expiry(chain, entry_date, min_dte, max_dte)` for the "which expiry do I trade today" step. `testing.engine.backtest_runner.walk_forward` is duck-typed against this — no registration needed, just pass the selector instance in.
3. **A signal, if the strategy is conditioned on one** — compute it however makes sense, then pass it into `walk_forward(chain, config, selector, your_signal_kwarg=signal_series)`. If the value passed is a `pd.Series` keyed by date, `walk_forward` resolves it to that day's value automatically before calling the selector.

Everything downstream — `PositionSimulator`, `SizingEngine`, `PortfolioAggregator`, `PerformanceAnalyzer` — works against `PositionSpec`/`Leg` generically and needs no changes for a new strategy, as long as the strategy's structure is a static, multi-leg position held to an exit trigger. A fundamentally different position lifecycle (e.g. continuous delta-hedging) would need its own simulator — `engine/`'s data layer (`OptionChainRepository`, `SyntheticSpotEngine`, `StrikeSelector`) is still reusable for that, `PositionSimulator` is not.

See `testing/strategies/toy_example/` for a minimal, working, end-to-end example of this contract — used by the graduation-mechanism tests, not a real strategy.

## Data

This engine expects a historical options-chain dataset (not included in this mirror — see the root README). Point `OptionChainRepository` at a local parquet file rather than committing data to the repo.

**Gotcha, worth remembering: never gate entries by pre-filtering the chain DataFrame to a subset condition** (a regime label, a date range that skips days, anything that removes rows a strategy might still hold a position across). `walk_forward`'s `PositionSimulator` walks an open position's `leg_history` day by day using whatever chain rows are present; if the chain has gaps, the simulator silently can't check exit triggers on the missing days — not "time skips forward," but "this position's stop/profit-target check quietly never happens on those days," which can dramatically (and wrongly) change results. **The right pattern**: encode any entry condition in the strategy's own selector logic (resolved per-day via `walk_forward`'s existing `pd.Series` kwarg mechanism) so only *new entries* are gated — an already-open position always gets tracked through the full, continuous, ungated chain.
