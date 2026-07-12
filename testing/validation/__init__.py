"""Cross-cutting validation methodology (issue: VRP harvester tuning session,
ADR 0005) -- distinct from `testing.engine.backtest_runner.walk_forward`
(the per-day trade-selection driver *within* a single backtest run). This
package's "walk-forward" means the rolling train/test *evaluation*
methodology: does a tuned parameter's apparent edge generalize across time,
or was it picked to fit one particular window. Same name, deliberately
different concept -- see `walk_forward_validation.py`'s own docstring for
the full disambiguation.
"""
