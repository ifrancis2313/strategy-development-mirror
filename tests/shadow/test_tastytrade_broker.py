"""Mocked-broker integration tests for shadow/brokers/tastytrade.py --
style matches alpha-pipeline's sleeves/pairs_mft/tests (mocked SDK objects,
no real network/session), plus ports the no-live-session unit tests from
Options_Trading_Strategy's tests/core/broker/test_tastytrade_client.py
(`_choose_expiry` per-book settlement routing, `get_daily_closes` ticker
routing).

Also verifies the ADR 0008 defense-in-depth safety property this repo
strengthens: no order-placement class exists anywhere in shadow/brokers/
(not just "gated behind a read-scope token," as the source repo does --
the capability is structurally absent here, see module docstring).
"""

from __future__ import annotations

import ast
import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from shadow.brokers import tastytrade as tt
from shadow.brokers.tastytrade import TastytradeClient

REPO_ROOT = Path(__file__).resolve().parents[2]
SHADOW_ROOT = REPO_ROOT / "shadow"


def _fake_frame():
    idx = pd.date_range("2026-01-01", periods=40, tz="UTC")
    return pd.DataFrame({"Close": range(40)}, index=idx)


def test_get_daily_closes_routes_the_requested_ticker_to_yfinance(monkeypatch):
    seen = []

    def fake_download(ticker, **kwargs):
        seen.append(ticker)
        return _fake_frame()

    monkeypatch.setattr(tt.yf, "download", fake_download)
    client = TastytradeClient.__new__(TastytradeClient)  # no session/creds needed

    client.get_daily_closes(lookback_days=21)
    client.get_daily_closes(lookback_days=21, ticker="QQQ")
    client.get_daily_closes(lookback_days=21, ticker="IWM")

    assert seen == ["SPY", "QQQ", "IWM"]


TODAY = datetime.date(2020, 1, 15)


def _opt(settlement, sym):
    return SimpleNamespace(settlement_type=settlement, streamer_symbol=sym)


def _raw_chain():
    return {
        TODAY + datetime.timedelta(days=30): [_opt("PM", "a")],
        TODAY + datetime.timedelta(days=37): [_opt("AM", "b")],
        TODAY + datetime.timedelta(days=40): [_opt("PM", "c")],
    }


def test_pm_route_filters_to_pm_settled_expiries():
    chosen, opts = TastytradeClient._choose_expiry(_raw_chain(), TODAY, 37, "PM")
    assert chosen == TODAY + datetime.timedelta(days=40)
    assert [o.streamer_symbol for o in opts] == ["c"]


def test_standard_route_ignores_settlement_filter():
    chosen, opts = TastytradeClient._choose_expiry(_raw_chain(), TODAY, 37, "standard")
    assert chosen == TODAY + datetime.timedelta(days=37)
    assert [o.streamer_symbol for o in opts] == ["b"]


def test_pm_route_falls_back_to_all_when_no_pm_expiries():
    chain = {
        TODAY + datetime.timedelta(days=35): [_opt("AM", "x")],
        TODAY + datetime.timedelta(days=44): [_opt("AM", "y")],
    }
    chosen, opts = TastytradeClient._choose_expiry(chain, TODAY, 37, "PM")
    assert chosen == TODAY + datetime.timedelta(days=35)


def test_client_defaults_to_sandbox_is_test_true():
    client = TastytradeClient(secret="x", refresh_token="y", account_number="z")
    assert client._is_test is True


def test_client_can_be_pointed_at_production_explicitly():
    client = TastytradeClient(secret="x", refresh_token="y", account_number="z", is_test=False)
    assert client._is_test is False


def test_exchange_refresh_token_uses_cert_endpoint_for_sandbox(monkeypatch):
    seen_urls = []

    class FakeResponse:
        def json(self):
            return {}

        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        seen_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(tt.httpx, "post", fake_post)
    tt.exchange_refresh_token("secret", "refresh", is_test=True)
    assert seen_urls == [f"{tt._TT_CERT_API}/oauth/token"]


def test_exchange_refresh_token_uses_production_endpoint_when_not_sandbox(monkeypatch):
    seen_urls = []

    class FakeResponse:
        def json(self):
            return {}

        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        seen_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(tt.httpx, "post", fake_post)
    tt.exchange_refresh_token("secret", "refresh", is_test=False)
    assert seen_urls == [f"{tt._TT_PROD_API}/oauth/token"]


def test_client_falls_back_to_env_vars_when_credentials_not_passed_explicitly(monkeypatch):
    """`secret or os.environ.get("TT_SECRET")` (same for refresh_token,
    account_number) -- mutation testing (2026-07-12) found all three
    fallbacks survive being dropped/changed to `and`, since no prior test
    actually constructed a client with credentials=None relying on the env
    var. This is exactly the credential-wiring path behind this repo's own
    documented "Grant revoked" incident."""
    monkeypatch.setenv("TT_SECRET", "env-secret")
    monkeypatch.setenv("TT_REFRESH", "env-refresh")
    monkeypatch.setenv("TT_ACCOUNT", "env-account")

    client = TastytradeClient()

    assert client._secret == "env-secret"
    assert client._refresh_token == "env-refresh"
    assert client._account_number == "env-account"


def test_client_explicit_credentials_take_precedence_over_env_vars(monkeypatch):
    monkeypatch.setenv("TT_SECRET", "env-secret")
    client = TastytradeClient(secret="explicit-secret", refresh_token="r", account_number="a")
    assert client._secret == "explicit-secret"


def test_exchange_refresh_token_returns_the_rotated_token_when_it_differs(monkeypatch, tmp_path):
    """`data["refresh_token"] != refresh_token` -- mutation testing
    (2026-07-12) found this survives being flipped to `==`. No prior test
    supplied a response body that actually carries a "refresh_token" key
    (only the empty-`{}`-response and error-response cases were covered)."""
    class FakeResponse:
        def json(self):
            return {"refresh_token": "rotated-token"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tt.httpx, "post", lambda url, **kwargs: FakeResponse())
    monkeypatch.setattr(tt, "_ENV_FILE", tmp_path / "fake.env")  # never touch the real repo .env
    monkeypatch.setenv("TT_REFRESH", "old-token")

    result = tt.exchange_refresh_token("secret", "old-token", is_test=True)

    assert result == "rotated-token"
    assert os.environ["TT_REFRESH"] == "rotated-token"


def test_exchange_refresh_token_returns_the_same_token_when_unrotated(monkeypatch, tmp_path):
    class FakeResponse:
        def json(self):
            return {"refresh_token": "same-token"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tt.httpx, "post", lambda url, **kwargs: FakeResponse())
    monkeypatch.setattr(tt, "_ENV_FILE", tmp_path / "fake.env")
    monkeypatch.setenv("TT_REFRESH", "same-token")

    result = tt.exchange_refresh_token("secret", "same-token", is_test=True)

    assert result == "same-token"


def test_exchange_refresh_token_raises_on_a_revoked_or_expired_grant(monkeypatch):
    # A real OAuth failure response (e.g. "Grant revoked", see AGENTS.md's
    # already-documented incident) is a non-2xx status with no
    # "refresh_token" key -- must raise, not silently return the stale old
    # token as if nothing went wrong.
    request = httpx.Request("POST", f"{tt._TT_CERT_API}/oauth/token")
    fake_response = httpx.Response(400, json={"error": "invalid_grant"}, request=request)

    def fake_post(url, **kwargs):
        return fake_response

    monkeypatch.setattr(tt.httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        tt.exchange_refresh_token("secret", "refresh", is_test=True)


# ---------------------------------------------------------------------
# Defense-in-depth: no order-placement capability exists in shadow/
# ---------------------------------------------------------------------

_ORDER_PLACEMENT_NAMES = {
    "TastytradeOrderManager", "OrderManager", "NewOrder", "place_order",
    "submit_entry", "submit_exit",
}


def test_no_order_manager_class_or_symbol_defined_anywhere_in_shadow():
    """Structural (not just behavioral) proof of ADR 0008's safety
    property: grep the AST of every module under shadow/ for any class or
    top-level function whose name suggests order placement. None should
    exist -- the capability isn't gated behind a flag, it's absent."""
    violations = []
    for path in sorted(SHADOW_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in _ORDER_PLACEMENT_NAMES:
                violations.append((str(path.relative_to(REPO_ROOT)), node.name))
    assert violations == []


def test_tastytrade_module_never_imports_order_type_or_newer_order_symbols():
    """The tastytrade SDK's own order-placement surface (NewOrder,
    OrderAction, Account.place_order) is never imported by shadow/brokers/
    tastytrade.py -- confirms this isn't just "the class wasn't written,"
    the SDK types needed to construct an order aren't even reachable here.
    """
    source = (SHADOW_ROOT / "brokers" / "tastytrade.py").read_text()
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
    forbidden = {"NewOrder", "OrderAction", "Account", "OrderType", "OrderTimeInForce"}
    assert imported_names & forbidden == set()
