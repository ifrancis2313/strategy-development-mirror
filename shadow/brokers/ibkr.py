"""IBKR read-only client for Shadow -- new (issue 11), not a port. Shape
mirrors `Options_Trading_Strategy`'s `core/broker/ibkr_client.py::IBKRClient`
(connect/disconnect/context-manager, sparse-probe-then-dense-band chain
discovery, `get_chain_snapshot`'s `OptionChainRepository`-shaped output) and
`alpha-pipeline`'s `sleeves/pairs_mft/execution/ibkr_client.py` (env-var
host/port/client_id fallback pattern) as read-only references -- neither
file is imported or modified.

Built on `ib_async` (the maintained successor to `ib_insync`), same library
already proven against a real IBKR paper account in both sibling projects.

Shadow's whole point is holding **multiple broker connections
simultaneously** (ADR 0001's consequence: defers `alpha-pipeline` ADR
0003's IBKR-vs-Tastytrade decision) -- this client is IBKR's side of that,
independent of `shadow/brokers/tastytrade.py`.

**No order-placement capability anywhere in this module, or anywhere in
`shadow/`.** Unlike Tastytrade (where `Options_Trading_Strategy` already
has a working order manager this repo deliberately declines to port -- see
`shadow/brokers/tastytrade.py`'s module docstring), IBKR order placement
was never even a candidate for this repo: the issue that created this
module scopes it explicitly to "quotes/chain data only, no order
placement" -- Shadow never places a real-money order regardless of broker
(`CONTEXT.md`'s Pipeline glossary entry, ADR 0001's consequence). There is
no `placeOrder`/`whatIfOrder` call, and no method that could reach one, in
this file.

`market_data_type=3` (delayed) is the default -- works without a paid
subscription (confirmed by both reference implementations above); pass 1
(live) once a real-time subscription exists on the connected account.
"""

from __future__ import annotations

import datetime
import os

import pandas as pd
from ib_async import IB, Contract, Index, Option


class IBKRClient:
    """Generic, read-only IBKR connectivity for Shadow: option chain
    snapshots and leg quote refreshes, nothing else. `host`/`port`/
    `client_id` fall back to `IBKR_HOST`/`IBKR_PORT`/`IBKR_CLIENT_ID` env
    vars if not passed explicitly (matches `alpha-pipeline`'s
    `pairs_mft/execution/ibkr_client.py` convention). Default `client_id`
    (15) is deliberately distinct from `Options_Trading_Strategy`'s (10)
    and `alpha-pipeline`'s (11) defaults so Shadow can hold its own TWS/IB
    Gateway session concurrently with either sibling project's, without a
    clientId collision disconnecting one of them.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        market_data_type: int = 3,
    ):
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(port if port is not None else os.environ.get("IBKR_PORT", 7497))
        self.client_id = int(client_id if client_id is not None else os.environ.get("IBKR_CLIENT_ID", 15))
        self.market_data_type = market_data_type
        self.ib = IB()

    def connect(self) -> None:
        self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=8)
        self.ib.reqMarketDataType(self.market_data_type)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self) -> IBKRClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def account_summary(self) -> dict:
        """Read-only account metadata (e.g. confirming this is a paper
        account, `AccountType`/`NetLiquidation`) -- never used for sizing
        or order construction, since Shadow places no orders.
        """
        values = self.ib.accountSummary()
        return {v.tag: v.value for v in values}

    def _probe_deltas(self, symbol: str, exchange: str, currency: str, strikes: list, expiry: str, right: str, trading_class: str):
        """Sparse probe across the strike range for one side (P or C),
        returns [(strike, delta), ...] -- used to locate roughly where a
        strategy's target deltas sit before a dense, narrowly-targeted fetch.
        """
        probe_strikes = strikes[:: max(1, len(strikes) // 24)]
        probes = [
            Option(symbol, expiry, strike, right, exchange, tradingClass=trading_class, currency=currency)
            for strike in probe_strikes
        ]
        self.ib.qualifyContracts(*probes)
        probes = [c for c in probes if c.conId]
        tickers = [self.ib.reqMktData(c, "106", False, False) for c in probes]
        self.ib.sleep(4)

        deltas = [(c.strike, t.modelGreeks.delta) for c, t in zip(probes, tickers, strict=True) if t.modelGreeks]
        for c in probes:
            self.ib.cancelMktData(c)
        return deltas

    def get_chain_snapshot(
        self,
        symbol: str,
        exchange: str,
        currency: str,
        secid: int,
        min_dte: int,
        max_dte: int,
        target_deltas: list[tuple[str, float]],
        dense_window: int = 8,
        trading_class: str | None = None,
    ) -> pd.DataFrame:
        """One expiry (nearest the midpoint of [min_dte, max_dte]), in the
        same DataFrame shape as `testing.graduation`'s frozen
        `engine.chain_repository.OptionChainRepository` output (secid, date,
        exdate, cp_flag, strike_price, best_bid, best_offer, delta,
        impl_volatility, optionid) -- a graduated snapshot's own selector
        works unchanged against this. `target_deltas` is a list of
        (right, delta) pairs the calling strategy's config actually needs.
        """
        underlying = Index(symbol, exchange, currency)
        self.ib.qualifyContracts(underlying)
        trading_class = trading_class or symbol

        chains = self.ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)
        chain = next(c for c in chains if c.tradingClass == trading_class)

        today = pd.Timestamp(datetime.date.today())
        target_dte = (min_dte + max_dte) / 2
        expirations = sorted(chain.expirations)
        chosen_expiry = min(expirations, key=lambda e: abs((pd.Timestamp(e) - today).days - target_dte))
        strikes = sorted(chain.strikes)

        selected_strikes_by_right: dict = {}
        for right in {r for r, _ in target_deltas}:
            probe = self._probe_deltas(symbol, exchange, currency, strikes, chosen_expiry, right, trading_class)
            if not probe:
                continue
            strikes_for_right = selected_strikes_by_right.setdefault(right, set())
            for _, target_delta in [(r, d) for r, d in target_deltas if r == right]:
                nearest_strike = min(probe, key=lambda sd: abs(sd[1] - target_delta))[0]
                idx = strikes.index(nearest_strike)
                lo, hi = max(0, idx - dense_window), min(len(strikes), idx + dense_window)
                strikes_for_right.update(strikes[lo:hi])

        contracts = [
            Option(symbol, chosen_expiry, strike, right, exchange, tradingClass=trading_class, currency=currency)
            for right, strikes_for_right in selected_strikes_by_right.items()
            for strike in sorted(strikes_for_right)
        ]
        self.ib.qualifyContracts(*contracts)
        contracts = [c for c in contracts if c.conId]

        tickers = [self.ib.reqMktData(c, "106", False, False) for c in contracts]
        self.ib.sleep(8)

        rows = []
        for contract, ticker in zip(contracts, tickers, strict=True):
            greeks = ticker.modelGreeks
            rows.append({
                "secid": secid,
                "date": today,
                "exdate": pd.Timestamp(chosen_expiry),
                "cp_flag": contract.right,
                "strike_price": contract.strike,
                "best_bid": ticker.bid if ticker.bid and ticker.bid > 0 else float("nan"),
                "best_offer": ticker.ask if ticker.ask and ticker.ask > 0 else float("nan"),
                "delta": greeks.delta if greeks else float("nan"),
                "impl_volatility": greeks.impliedVol if greeks else float("nan"),
                "optionid": contract.conId,
            })

        for contract in contracts:
            self.ib.cancelMktData(contract)

        return pd.DataFrame(rows)

    def get_daily_closes(self, contract: Contract, lookback_days: int) -> pd.Series:
        """Daily close history via `reqHistoricalData` (a separate data
        permission from live/delayed streaming quotes), for a realized-vol
        input.
        """
        self.ib.qualifyContracts(contract)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{lookback_days} D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return pd.Series(dtype=float)
        return pd.Series({pd.Timestamp(bar.date): bar.close for bar in bars}).sort_index().rename("close")

    def get_daily_closes_with_fallback(
        self, primary_contract: Contract, fallback_contract: Contract, lookback_days: int
    ) -> pd.Series:
        """Tries `primary_contract` first; retries with `fallback_contract`
        if no bars come back (e.g. an index-quote subscription wall).
        """
        closes = self.get_daily_closes(primary_contract, lookback_days)
        if not closes.empty:
            return closes
        return self.get_daily_closes(fallback_contract, lookback_days)

    def get_leg_quotes(self, legs: list) -> dict[object, tuple[float, float]]:
        """Cheap current bid/ask refresh for a *known* set of already-traded
        contracts (an open shadow position's legs, identified by conId).
        `legs` is duck-typed (needs only `.optionid`) -- see
        `shadow.execution`'s module docstring.
        """
        contracts = [Option(conId=leg.optionid, exchange="SMART") for leg in legs]
        self.ib.qualifyContracts(*contracts)
        tickers = [self.ib.reqMktData(c, "", False, False) for c in contracts]
        self.ib.sleep(4)

        # Keyed by the original leg.optionid, not contract.conId -- contracts
        # and tickers stay index-aligned with legs (qualifyContracts mutates
        # in place, doesn't reorder), so this doesn't depend on qualification
        # correctly preserving conId for every contract. Matches
        # TastytradeClient.get_leg_quotes' own keying (by the original leg,
        # never a re-derived field) -- without this, a leg whose contract
        # failed to qualify (plausible for an aging/near-expiry position,
        # exactly what this method exists to refresh) could silently key
        # under the wrong id, either KeyError-ing inside shadow.execution's
        # `quotes[leg.optionid]` lookup or bypassing shadow/ledger.py's
        # intended graceful "missing quote, hold" degradation entirely.
        # Found during the front-to-back review (2026-07-12).
        quotes: dict[object, tuple[float, float]] = {
            leg.optionid: (
                ticker.bid if ticker.bid and ticker.bid > 0 else float("nan"),
                ticker.ask if ticker.ask and ticker.ask > 0 else float("nan"),
            )
            for leg, ticker in zip(legs, tickers, strict=True)
        }

        for contract in contracts:
            self.ib.cancelMktData(contract)

        return quotes
