# strategy-development (infra mirror)

This is a curated, hand-picked mirror of the research/testing infrastructure from a private working repo — not a strategy, not a track record, not a results post. It exists to show the engineering and statistical-rigor scaffolding a systematic options-trading research pipeline actually needs, for outside review.

**What's deliberately excluded, on purpose:** every strategy implementation, every backtest result, and all market data. No Sharpe ratios, no P&L, no parameter choices that encode a real trading idea. See `architecture.html` for the pipeline shape and `CONTEXT.md` for the vocabulary; `testing/strategies/toy_example/` is a synthetic fixture that exists only to give the graduation mechanism something real to run against — not a strategy.

Still searching for that first strategy validated end-to-end through this pipeline. More to come.

## Quick demo

```sh
python -m examples.synthetic_demo
```

After installing the pinned dependencies (see Running it), this offline command runs in under a minute on seeded synthetic prices: a toy moving-average rule, rolling walk-forward folds, and sign-permutation Monte Carlo tests. It writes only to stdout and needs no data files or credentials. Toy observations omit fills and trading costs.

SYNTHETIC DATA — demonstrates the validation machinery; not a strategy or a result.

By default, PBO and the graduation gate print an explicit skip because the external implementation is excluded. To enable both, point to an existing external checkout through the mechanism in [License](#license).

## What's here

Three sequential "wings," each with a different trust level and data regime:

- **`ingestion/`** — market-data scraping and storage (schema, cleaning, a Supabase client). Serves both this pipeline's own backtesting and any downstream execution system that needs a shared data source.
- **`testing/`** — the backtest/statistical-validation engine: position/leg modeling, strike selection, a strategy-agnostic backtest driver, sign-permutation Monte Carlo (MCPT) and probability-of-backtest-overfitting (PBO) robustness checks, walk-forward validation, and a graduation mechanism that freezes a strategy's validated code into a versioned, standalone snapshot with zero live coupling back to ongoing research.
- **`shadow/`** — live-but-not-real-money proving: runs a graduated strategy against real-time data with simulated fills, building a track record before anything is trusted with real capital. Structurally incapable of placing a real order — no order-placement class exists anywhere under this directory, verified by an AST-walking test, not just documented.

`docs/adr/` records the actual architectural decisions behind this (why walk-forward beats a static train/test split, why Sharpe/drawdown graduation floors are deliberately left unset rather than guessed, why every result against the current historical sample is treated as exploratory rather than confirmatory no matter how many statistical screens it clears). These are real engineering trade-offs made along the way, not after-the-fact justification.

## Methodology highlights

- **Mutation testing**, not just coverage: `mutmut` run against the full framework to distinguish tests that are actually load-bearing from ones that pass regardless of whether the logic they claim to check is right.
- **Property-based testing** (Hypothesis) for the pure-math functions, alongside example-based unit tests.
- **CI** (GitHub Actions): ruff + mypy + pytest + `pip-audit` dependency vulnerability scanning on every push, against a pinned lockfile so runs are reproducible over time.
- **Statistical discipline over convenience**: Bonferroni-corrected significance testing across any multi-variant sweep, an explicit no-lookahead/no-pre-filtering discipline in the backtest engine (a subtle chain-filtering bug that once produced a false-positive "significant" result is exactly the kind of thing this suite now has a regression test for), and a hard boundary between "cleared a statistical bar" and "validated" — see `docs/adr/0006`.

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
ruff check .
mypy .
pytest -q
```

No market data or credentials required — everything here runs against synthetic fixtures.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

The PBO implementation is an external AGPL-3.0 dependency and is not included.
Pass its checkout path as `vendored_pypbo_dir` to
`probability_of_backtest_overfitting()` or set `PYPBO_CSCV_DIR`.
