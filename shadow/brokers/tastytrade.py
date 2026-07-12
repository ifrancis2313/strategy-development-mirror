"""Tastytrade read-only client for Shadow -- ported from
`Options_Trading_Strategy`'s `core/broker/tastytrade_client.py`
(`TastytradeClient`), preserving its behavior (ADR 0001's "migration, not a
rewrite" framing, same principle Testing's engine port already established).

`get_chain_snapshot` returns a DataFrame in the same shape as
`testing.graduation`'s frozen `engine.chain_repository.OptionChainRepository`
output (secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
delta, impl_volatility, optionid) -- a graduated snapshot's own selector
works unchanged against this live data, no separate "live version" of a
strategy's selection logic needed.

**Defense-in-depth safety property (ADR 0008, `Options_Trading_Strategy`
`docs/adr/0008-shadow-mode-track-record-before-live.md`):** "the shadow
path never instantiates `TastytradeOrderManager` (the only thing that can
place an order)... even a mis-set flag can't trade on a read token." Shadow
goes one step further than that source repo: `TastytradeOrderManager` (or
any order-placement equivalent) simply **does not exist anywhere in this
codebase**. There is no class here capable of submitting a Tastytrade order
-- not gated behind a config flag, not reachable via any code path,
regardless of what a config or environment variable says. If a strategy
ever needs to place a real order, that capability is added only in a
Pipeline, post-graduation (see `CONTEXT.md`'s Pipeline glossary entry and
ADR 0001's consequence: "Nothing in this repo (Testing or Shadow) ever
places a real-money order").

Environments (see ADR 0008's own description):
  - Production  (api.tastyworks.com,      is_test=False) -- real money,
    real-time data. This is what a real Shadow deployment should read from
    for a meaningful live track record (ADR 0008 explicitly rejects sandbox
    for that purpose: "fake/delayed data makes the track record
    meaningless"). The VPS running `Options_Trading_Strategy`'s own live
    shadow system is the sole owner of that production OAuth grant --
    never point this client at the same grant from another machine (the
    refresh token rotates on every use; a second concurrent user
    invalidates it). Use a separate, dedicated grant for any real Shadow
    deployment.
  - Sandbox/cert (api.cert.tastyworks.com, is_test=True)  -- fake money,
    delayed/fake data; a developer test environment. Safe for connectivity
    verification and local development against a throwaway grant.
`is_test` defaults to `True` (sandbox) here -- unlike the source repo's
hardcoded `is_test=False`, since Shadow has no single fixed deployment
target yet and a safe-by-default connection target is preferable to an
implicit production default.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import time

import httpx
import pandas as pd
import yfinance as yf
from tastytrade import DXLinkStreamer, Session
from tastytrade.dxfeed import Greeks, Quote
from tastytrade.instruments import Option, get_option_chain

_TT_PROD_API = "https://api.tastyworks.com"
_TT_CERT_API = "https://api.cert.tastyworks.com"
_ENV_FILE = pathlib.Path(__file__).parents[2] / ".env"


class TastytradeClient:
    """Read-only Tastytrade connectivity: option chain snapshots and leg
    quote refreshes only. No method on this class (or anywhere in
    `shadow/`) can place, modify, or cancel an order -- see module
    docstring.
    """

    def __init__(
        self,
        secret: str | None = None,
        refresh_token: str | None = None,
        account_number: str | None = None,
        is_test: bool = True,
    ):
        self._secret = secret or os.environ.get("TT_SECRET")
        self._refresh_token = refresh_token or os.environ.get("TT_REFRESH")
        self._account_number = account_number or os.environ.get("TT_ACCOUNT")
        self._is_test = is_test

    def __enter__(self) -> TastytradeClient:
        return self

    def __exit__(self, *exc) -> None:
        pass  # Sessions are per-operation; no persistent socket to close.

    # ------------------------------------------------------------------
    # Token rotation — Tastytrade returns a new refresh token on every
    # /oauth/token call; the SDK silently drops it. Exchange first,
    # persist any rotation, then pass the current token into Session.
    # ------------------------------------------------------------------

    def _exchange_and_persist(self) -> str:
        if self._secret is None or self._refresh_token is None:
            raise ValueError(
                "TastytradeClient needs a secret and refresh_token -- pass them explicitly or "
                "set TT_SECRET/TT_REFRESH (TT_ACCOUNT is also read from the environment if not passed)."
            )
        self._refresh_token = exchange_refresh_token(self._secret, self._refresh_token, is_test=self._is_test)
        return self._refresh_token

    # ------------------------------------------------------------------
    # Public interface (read-only: quotes/chain data only)
    # ------------------------------------------------------------------

    def get_chain_snapshot(
        self,
        symbol: str,
        secid: int,
        min_dte: int,
        max_dte: int,
        settlement_type: str = "PM",
    ) -> pd.DataFrame:
        """Returns a DataFrame in `OptionChainRepository`'s format for the
        expiry nearest the midpoint of [min_dte, max_dte]. All strikes at
        the chosen expiry are returned -- the caller's (graduated
        snapshot's own) StrikeSelector handles delta-targeting from there.
        """
        return asyncio.run(self._chain_async(symbol, secid, min_dte, max_dte, settlement_type))

    def get_leg_quotes(self, legs: list) -> dict[object, tuple[float, float]]:
        """Refreshes bid/ask for a known set of already-traded contracts
        (identified by their OCC `optionid`) via the DXLink streamer.
        `legs` is duck-typed (needs only `.optionid`) -- see `shadow.execution`'s
        module docstring for why Shadow doesn't import a concrete `Leg` class.
        """
        return asyncio.run(self._leg_quotes_async(legs))

    def get_daily_closes(self, lookback_days: int, ticker: str = "SPY") -> pd.Series:
        """Daily closes via yfinance (Tastytrade's API doesn't expose a
        historical bars endpoint), used as the realized-vol input.
        """
        df = yf.download(ticker, period=f"{lookback_days + 20}d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        closes = df["Close"].squeeze()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes.sort_index().rename("close").tail(lookback_days + 10)

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    @staticmethod
    def _choose_expiry(raw_chain: dict, today_date, target_dte: float, settlement_type: str):
        if settlement_type == "standard":
            candidates = raw_chain
        else:
            filtered = {
                d: [o for o in opts if o.settlement_type == settlement_type]
                for d, opts in raw_chain.items()
                if any(o.settlement_type == settlement_type for o in opts)
            }
            candidates = filtered if filtered else raw_chain

        chosen_exp = min(candidates, key=lambda d: abs((d - today_date).days - target_dte))
        return chosen_exp, candidates[chosen_exp]

    async def _chain_async(
        self, symbol: str, secid: int, min_dte: int, max_dte: int, settlement_type: str
    ) -> pd.DataFrame:
        self._exchange_and_persist()
        async with Session(self._secret, self._refresh_token, is_test=self._is_test, timeout=60.0) as session:
            raw_chain = await get_option_chain(session, symbol)

            today = pd.Timestamp.today().normalize()
            today_date = today.date()
            target_dte = (min_dte + max_dte) / 2

            chosen_exp, options_at_exp = self._choose_expiry(raw_chain, today_date, target_dte, settlement_type)
            streamer_syms = [o.streamer_symbol for o in options_at_exp if o.streamer_symbol]

            data_by_sym = await self._stream_quotes_and_greeks(session, streamer_syms)

        rows = []
        for opt in options_at_exp:
            ss = opt.streamer_symbol
            d = data_by_sym.get(ss, {})
            rows.append({
                "secid": secid,
                "date": today,
                "exdate": pd.Timestamp(opt.expiration_date),
                "cp_flag": opt.option_type.value,
                "strike_price": float(opt.strike_price),
                "best_bid": d.get("bid", float("nan")),
                "best_offer": d.get("ask", float("nan")),
                "delta": d.get("delta", float("nan")),
                "impl_volatility": d.get("iv", float("nan")),
                "optionid": opt.symbol,
            })
        return pd.DataFrame(rows)

    async def _leg_quotes_async(self, legs: list) -> dict[object, tuple[float, float]]:
        occ_to_ss = {leg.optionid: Option.occ_to_streamer_symbol(str(leg.optionid)) for leg in legs}
        streamer_syms = list(occ_to_ss.values())
        self._exchange_and_persist()
        async with Session(self._secret, self._refresh_token, is_test=self._is_test, timeout=60.0) as session:
            data_by_sym = await self._stream_quotes_and_greeks(session, streamer_syms, timeout=10.0)
        return {
            leg.optionid: (
                data_by_sym.get(occ_to_ss[leg.optionid], {}).get("bid", float("nan")),
                data_by_sym.get(occ_to_ss[leg.optionid], {}).get("ask", float("nan")),
            )
            for leg in legs
        }

    @staticmethod
    async def _stream_quotes_and_greeks(
        session: Session,
        streamer_syms: list[str],
        timeout: float = 15.0,
    ) -> dict[str, dict]:
        result: dict[str, dict] = {}
        all_syms = set(streamer_syms)
        greeks_seen: set = set()
        quotes_seen: set = set()

        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Greeks, streamer_syms)
            await streamer.subscribe(Quote, streamer_syms)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                g = streamer.get_event_nowait(Greeks)
                if g and g.event_symbol in all_syms:
                    entry = result.setdefault(g.event_symbol, {})
                    entry["delta"] = float(g.delta)
                    entry["iv"] = float(g.volatility)
                    greeks_seen.add(g.event_symbol)

                q = streamer.get_event_nowait(Quote)
                if q and q.event_symbol in all_syms:
                    entry = result.setdefault(q.event_symbol, {})
                    bid = float(q.bid_price)
                    ask = float(q.ask_price)
                    entry["bid"] = bid if bid > 0 else float("nan")
                    entry["ask"] = ask if ask > 0 else float("nan")
                    quotes_seen.add(q.event_symbol)

                if greeks_seen >= all_syms and quotes_seen >= all_syms:
                    break
                if not g and not q:
                    await asyncio.sleep(0.05)

        return result


def exchange_refresh_token(secret: str, refresh_token: str, is_test: bool = True) -> str:
    """Exchange a Tastytrade refresh token; persist and return any rotated
    one. Tastytrade rotates the refresh token on every `/oauth/token` call
    and the SDK silently drops it (captures only `access_token`).
    `is_test` selects the sandbox/cert vs. production OAuth endpoint --
    exchanging a sandbox token against the production endpoint (or vice
    versa) fails, so this must match the `Session`'s own `is_test`.
    """
    api_base = _TT_CERT_API if is_test else _TT_PROD_API
    resp = httpx.post(
        f"{api_base}/oauth/token",
        json={
            "grant_type": "refresh_token",
            "client_secret": secret,
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    # A failed exchange (e.g. a revoked/expired grant -- a real, already-
    # documented failure mode, see AGENTS.md's "Grant revoked" incident)
    # has no "refresh_token" key in its error body, so the check below used
    # to silently fall through to `return refresh_token` -- the stale,
    # invalid old token, no error raised -- and the caller would only find
    # out much later inside Session() with a far less diagnostic error.
    # raise_for_status() surfaces the real failure immediately instead.
    # Found during the front-to-back review (2026-07-12).
    resp.raise_for_status()
    data = resp.json()
    if "refresh_token" in data and data["refresh_token"] != refresh_token:
        new_token = data["refresh_token"]
        os.environ["TT_REFRESH"] = new_token
        _update_env_file(_ENV_FILE, "TT_REFRESH", new_token)
        return new_token
    return refresh_token


def _update_env_file(env_path: pathlib.Path, key: str, value: str) -> None:
    """Rewrite a single KEY=value line in the .env file in place."""
    if not env_path.exists():
        return
    text = env_path.read_text()
    pattern = rf"^{re.escape(key)}=.*$"
    replacement = f"{key}={value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    else:
        text = text.rstrip("\n") + f"\n{replacement}\n"
    env_path.write_text(text)
