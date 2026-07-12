# Shadow Wing — Architecture

The Shadow wing's live-but-not-real-money proving system: runs a graduated strategy against real-time production data with simulated fills, building a live track record before real capital is ever at risk. Root `CONTEXT.md` (Wing/Graduation/Pipeline/Sleeve) is the repo-level glossary; this file is Shadow's own architecture map, one level down.

*(This mirror omits the private repo's own deploy/composition-script layer and specific graduated-strategy examples — see the root README for why. The architectural reasoning below is unchanged.)*

## What's here

- **`execution.py`** — fill-price, mark-to-market, exit-trigger, and max-loss-based sizing math.
- **`ledger.py`** — the parallel capture-fraction ledger (floor/target/mid, each self-governed, shared vs. independent contract counts stored per trade).
- **`brokers/tastytrade.py`** — read-only Tastytrade client (chain snapshots, leg quotes).
- **`brokers/ibkr.py`** — read-only IBKR client (chain snapshots, leg quotes).

## Why `execution.py` is a second copy of math `testing/engine/` also has

`testing/engine/position_simulator.py` and `testing/engine/sizing.py` already contain byte-for-byte-identical-looking math. Shadow does not import either — and can't, by design (see root `CONTEXT.md`'s Shadow glossary entry).

The reason isn't just a blanket "never import testing.engine" rule — it's structural. A strategy reaches Shadow only via a Graduation snapshot (`testing/graduation/`), and that snapshot's dependency closure (`testing.graduation.closure.compute_closure`) only includes whatever a strategy's own `config`/`signal`/`selector` modules transitively import. Those modules build `PositionSpec`s and pick strikes — they never call `position_simulator`/`sizing` (those are Testing's backtest-driver infrastructure, deliberately left live/un-frozen so re-validation benefits from engine bugfixes made after a strategy graduated). A live shadow ledger still needs to mark a position to market and decide entries/exits, so that logic has to exist *somewhere* Shadow can reach without a live import — `shadow/execution.py` is that somewhere: an independently-ported copy, not derived from `testing/engine/`'s copy, duck-typed against whatever `Leg`/`PositionSpec`/`ExecutionConfig` shape a graduated snapshot happens to carry rather than importing concrete classes.

Verified concretely (`tests/shadow/test_graduated_snapshot_integration.py`): loads a real graduated snapshot via `testing.graduation.loader`, drives its own selector against a synthetic chain to get a real `PositionSpec` built from the snapshot's own frozen classes, and feeds that into `shadow.ledger.run_ledgers` — proving the duck-typed boundary actually works end to end, not just against hand-built test doubles. (This test skips gracefully in this mirror, since no graduated snapshot is present.)

## The one permitted dependency on `testing/`

`testing.graduation` (manifest format + `load_snapshot`/`load_snapshot_module`) is the sole exception to "never import testing.*" — it's the format a graduated strategy arrives in, not live engine code. Nothing under `shadow/` imports it directly (none of `execution.py`/`ledger.py`/`brokers/*.py` need to know a manifest exists); a composition script that actually drives "load today's graduated strategy + connect to a broker + run the ledger" would be the piece that imports it, and would live outside `shadow/` proper. Verified concretely, not just asserted: `tests/shadow/test_zero_testing_coupling.py` AST-walks every `.py` file under `shadow/` and fails if anything imports `testing.engine`, `testing.strategies`, or even a *different* part of `testing.*` than `testing.graduation`.

## Defense in depth: no order-placement capability exists

Shadow's core safety property: `TastytradeOrderManager` (or any equivalent) is not ported into this codebase at all, for either broker. There's no class or function anywhere under `shadow/` that can place, modify, or cancel a real order — not gated behind a config flag, not reachable via any code path. Verified structurally (not just documented) by `tests/shadow/test_tastytrade_broker.py`/`test_ibkr_broker.py`'s AST checks: no order-placement-shaped class/function is defined anywhere under `shadow/`, and `shadow/brokers/tastytrade.py` never even imports the Tastytrade SDK's order-construction types (`NewOrder`, `OrderAction`, `Account`). If a strategy ever needs to place a real order, that capability is added only in a downstream Pipeline, post-graduation — nothing in this repo (Testing or Shadow) ever places a real-money order.

## Broker environments

- **Tastytrade**: production (`api.tastyworks.com`, `is_test=False`) is real money/real-time data — what a meaningful live shadow track record should actually read from (sandbox is explicitly rejected for that purpose: fake/delayed data makes the track record meaningless). Sandbox/cert (`api.cert.tastyworks.com`, `is_test=True`) is fake money/delayed data, safe for connectivity verification. `shadow.brokers.tastytrade.TastytradeClient` defaults to `is_test=True` in this mirror, since Shadow has no single fixed deployment target here.
- **IBKR**: `market_data_type=3` (delayed) by default, works without a paid subscription.

## Deliberately not built here

The Shadow → Pipeline graduation gate (minimum track record length, live-vs.-Testing-Monte-Carlo divergence check) has a mechanism (`shadow/graduation_gate.py`) but requires real accumulated Shadow data to actually run against, which this mirror doesn't include.
