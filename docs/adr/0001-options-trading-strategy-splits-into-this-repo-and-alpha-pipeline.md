# Options_Trading_Strategy splits: its engine becomes this repo, its live execution becomes an alpha-pipeline sleeve

## Status

Accepted

## Context

`Options_Trading_Strategy` currently does two unrelated jobs in one repo: (1) a mature, strategy-agnostic backtest/research engine (`core/performance.py`, `core/strategy_comparison.py`, `core/position_simulator.py`, `core/backtest_runner.py`, 7 strategies) and (2) live shadow-mode execution against real Tastytrade quotes on a persistent VPS (ADR 0007-0009 in that repo). Running research-mode churn in the same codebase that touches a live broker connection is a real risk, and it conflates two things Zi is trying to separate across his whole project structure: repos that are allowed to be wrong (research) vs. repos that only hold strategies with proven edge (pipelines).

## Decision

Split `Options_Trading_Strategy` in two:

- Its `core/` engine and strategies move into this repo (`strategy-development`), becoming the Testing wing's starting point.
- Its live shadow-mode execution code (Tastytrade client, snapshot store, shadow ledgers) moves into `alpha-pipeline` as a new sleeve, once a strategy has been re-proven through this repo's Shadow wing. `alpha-pipeline` becomes "the" personal pipeline — the landing spot for any strategy that graduates and isn't earmarked for a club pipeline (Options and Futures Pipeline or SIC Research Pipeline).
- `Options_Trading_Strategy` itself is retired once the split is executed.

## Consequences

- `alpha-pipeline`'s ADR 0003 committed to IBKR as sole broker; `Options_Trading_Strategy` moved to Tastytrade specifically because IBKR's unsubscribed delayed data rejects multi-leg combo option orders (a failure ADR 0003 already flagged as a "revisit" trigger). Introducing this repo's Shadow wing as an intermediate stage defers that broker decision: Shadow can hold both IBKR and Tastytrade connections to build a live track record regardless of destination pipeline, so `alpha-pipeline` only has to decide about a second broker once a specific options strategy has actually earned its way to real capital — not upfront.
- Nothing in this repo (Testing or Shadow) ever places a real-money order. That capability only exists in a Pipeline, post-graduation.
