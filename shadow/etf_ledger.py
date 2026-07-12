"""Single-instrument long/flat position tracker for ETF-carry strategies --
for a strategy class that doesn't fit `shadow.ledger`'s options-shaped
capture-fraction concept (floor/target/mid exists specifically to bracket
bid/ask spread-capture uncertainty on a multi-leg options fill; a
buy-and-hold ETF has no such ambiguity, just a real round-trip cost already
priced into the strategy's own config). Same "give each genuinely different
asset class its own path rather than a forced shared interface" principle
the rest of this repo follows.

Mirrors a vectorized backtest's exact no-lookahead math (`signal.shift(1)`
deciding today's position, `position.diff().abs()` charging the round-trip
cost only on a position change) as a day-by-day live process instead of a
vectorized pass over a full history: each call stores today's own signal so
tomorrow's call can use it as "yesterday's" lagged value. Persists to the same
`data/generated/shadow/` root `shadow.ledger` uses, one JSON file per
strategy, so the two never collide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SHADOW_ROOT = "data/generated/shadow"


@dataclass(frozen=True)
class EtfDayOutcome:
    action: str  # "entered" | "exited" | "holding" | "flat" | "first_run"
    position: int
    signal: float
    price: float
    pnl_dollars: float
    equity: float


def _path_for(strategy: str, root: str = SHADOW_ROOT) -> Path:
    return Path(root) / f"{strategy}_etf.json"


def _load(path: Path) -> dict:
    if not path.exists():
        return {"position": None, "last_signal": None, "last_price": None, "entry_price": None,
                 "equity": 0.0, "history": []}
    with open(path) as f:
        return json.load(f)


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(strategy: str, root: str = SHADOW_ROOT) -> dict:
    return _load(_path_for(strategy, root))


def _decide_position(config, last_signal: float | None) -> int:
    if last_signal is None:
        return 0  # no prior signal to act on yet -- matches backtest.py's position.iloc[0] = 0
    if config.contango_threshold is None:
        return 1  # naive/unconditioned mode: always long once a prior signal exists
    return int(last_signal > config.contango_threshold)


def run_daily(strategy: str, config, signal_today: float, price_today: float, today, root: str = SHADOW_ROOT) -> EtfDayOutcome:
    """Process one trading day. `signal_today` is `df[config.signal_tenor] -
    df[config.reference_tenor]` computed from TODAY's close-equivalent
    quotes; it is stored for tomorrow's call, never used to decide today's
    own position (that would be lookahead -- today's position comes from
    the signal stored by *yesterday's* call).
    """
    path = _path_for(strategy, root)
    state = _load(path)

    prev_position = state["position"]
    position_today = _decide_position(config, state["last_signal"])

    instrument_return = 0.0 if state["last_price"] is None else (price_today - state["last_price"]) / state["last_price"]
    position_change = abs(position_today - prev_position) if prev_position is not None else position_today

    gross_pnl = position_today * instrument_return * config.fixed_notional
    cost = position_change * (config.round_trip_cost_bps / 10_000.0) * config.fixed_notional
    pnl_dollars = gross_pnl - cost
    new_equity = state["equity"] + pnl_dollars

    if prev_position is None:
        action = "first_run"
    elif position_change == 0:
        action = "holding" if position_today == 1 else "flat"
    elif position_today == 1:
        action = "entered"
    else:
        action = "exited"

    entry_price = state["entry_price"]
    if position_today == 1 and (prev_position != 1):
        entry_price = price_today
    elif position_today == 0:
        entry_price = None

    state["position"] = position_today
    state["last_signal"] = signal_today
    state["last_price"] = price_today
    state["entry_price"] = entry_price
    state["equity"] = new_equity
    state["history"].append({
        "date": pd.Timestamp(today).isoformat(),
        "position": position_today,
        "signal": signal_today,
        "price": price_today,
        "pnl_dollars": pnl_dollars,
        "equity": new_equity,
    })
    _save(path, state)

    return EtfDayOutcome(
        action=action, position=position_today, signal=signal_today,
        price=price_today, pnl_dollars=pnl_dollars, equity=new_equity,
    )
