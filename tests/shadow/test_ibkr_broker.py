"""Mocked-broker integration tests for shadow/brokers/ibkr.py -- style
matches alpha-pipeline's sleeves/pairs_mft/tests/test_ibkr_client.py
(mocked `ib_async.IB` object, no real TWS/Gateway connection here).

Read-only surface only: connect/disconnect, market-data-type, env-var
host/port/client_id fallback, and `get_leg_quotes`'s bid/ask refresh.
`get_chain_snapshot`'s sparse-probe-then-dense-band flow is exercised by
the live verification script (scripts/shadow/verify_ibkr_paper_connection.py)
against a real paper TWS session instead of being mocked here -- it's
IBKR-specific chain-discovery logic with too many live-session-shaped calls
(qualifyContracts/reqSecDefOptParams/reqMktData/ib.sleep) to usefully mock
without just re-asserting the mock's own scripted behavior back at itself.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

from shadow.brokers.ibkr import IBKRClient

REPO_ROOT = Path(__file__).resolve().parents[2]
SHADOW_ROOT = REPO_ROOT / "shadow"


class TestIBKRClientConfig:
    def test_defaults_when_no_env_or_args(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        assert client.host == "127.0.0.1"
        assert client.port == 7497
        assert client.client_id == 15
        assert client.market_data_type == 3

    def test_env_vars_override_defaults(self):
        env = {"IBKR_HOST": "10.0.0.5", "IBKR_PORT": "4002", "IBKR_CLIENT_ID": "42"}
        with patch.dict("os.environ", env, clear=True):
            client = IBKRClient()
        assert client.host == "10.0.0.5"
        assert client.port == 4002
        assert client.client_id == 42

    def test_explicit_args_win_over_env(self):
        env = {"IBKR_HOST": "10.0.0.5", "IBKR_PORT": "4002", "IBKR_CLIENT_ID": "42"}
        with patch.dict("os.environ", env, clear=True):
            client = IBKRClient(host="192.168.1.1", port=7496, client_id=99)
        assert client.host == "192.168.1.1"
        assert client.port == 7496
        assert client.client_id == 99

    def test_default_client_id_distinct_from_sibling_projects(self):
        """Options_Trading_Strategy defaults to 10, alpha-pipeline's
        pairs_mft to 11 -- Shadow's own default (15) must not collide with
        either, so all three can hold a TWS/Gateway session concurrently."""
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        assert client.client_id not in (10, 11)

    def test_connect_sets_market_data_type(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()
        client.connect()
        client.ib.connect.assert_called_once()
        client.ib.reqMarketDataType.assert_called_once_with(3)

    def test_disconnect_only_when_connected(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()
        client.ib.isConnected.return_value = False
        client.disconnect()
        client.ib.disconnect.assert_not_called()

        client.ib.isConnected.return_value = True
        client.disconnect()
        client.ib.disconnect.assert_called_once()

    def test_context_manager_connects_and_disconnects(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()
        client.ib.isConnected.return_value = True
        with client as c:
            assert c is client
            client.ib.connect.assert_called_once()
        client.ib.disconnect.assert_called_once()


class TestGetLegQuotes:
    def test_returns_bid_ask_keyed_by_conid(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()

        leg = MagicMock(optionid=12345)
        contract = MagicMock(conId=12345)
        client.ib.qualifyContracts.return_value = None

        with patch("shadow.brokers.ibkr.Option", return_value=contract):
            ticker = MagicMock(bid=1.5, ask=1.7)
            client.ib.reqMktData.return_value = ticker
            quotes = client.get_leg_quotes([leg])

        assert quotes == {12345: (1.5, 1.7)}
        client.ib.cancelMktData.assert_called_once()

    def test_keyed_by_original_leg_optionid_even_if_contract_conid_diverges(self):
        # A qualified contract's own .conId doesn't have to equal the
        # leg.optionid it was constructed from qualifyContracts is a mock
        # here and can't be relied on to preserve it (and in a real,
        # unmocked failure-to-qualify case, might not). The returned dict
        # must still be keyed by leg.optionid -- what shadow.execution's
        # `quotes[leg.optionid]` lookup actually looks for -- not whatever
        # contract.conId happens to be after qualification.
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()

        leg = MagicMock(optionid=12345)
        contract = MagicMock(conId=0)  # diverges from leg.optionid on purpose
        client.ib.qualifyContracts.return_value = None

        with patch("shadow.brokers.ibkr.Option", return_value=contract):
            ticker = MagicMock(bid=1.5, ask=1.7)
            client.ib.reqMktData.return_value = ticker
            quotes = client.get_leg_quotes([leg])

        assert quotes == {12345: (1.5, 1.7)}

    def test_nan_when_quote_missing_or_nonpositive(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()

        leg = MagicMock(optionid=999)
        contract = MagicMock(conId=999)

        with patch("shadow.brokers.ibkr.Option", return_value=contract):
            ticker = MagicMock(bid=-1, ask=None)
            client.ib.reqMktData.return_value = ticker
            quotes = client.get_leg_quotes([leg])

        bid, ask = quotes[999]
        assert bid != bid  # NaN
        assert ask != ask  # NaN
        # `ticker.bid and ticker.bid > 0` (same for ask) -- mutation testing
        # (2026-07-12) flagged the `> 0` as surviving being loosened to
        # `>= 0`. Checked by hand rather than chased: the leading `ticker.bid
        # and ...` truthiness check already short-circuits at exactly 0.0
        # (falsy in Python), the one value where `>0` and `>=0` would ever
        # disagree -- confirmed equivalent by applying the mutant directly
        # (mutmut apply) and re-running this suite, no new failure. Not a
        # real gap.


class TestGetDailyCloses:
    def test_parses_bars_into_series(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()
        bar1 = MagicMock(date="2024-01-02", close=10.0)
        bar2 = MagicMock(date="2024-01-03", close=10.5)
        client.ib.reqHistoricalData.return_value = [bar1, bar2]

        series = client.get_daily_closes(MagicMock(), lookback_days=5)
        assert len(series) == 2
        assert series.name == "close"

    def test_empty_bars_returns_empty_series(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()
        client.ib.reqHistoricalData.return_value = []

        series = client.get_daily_closes(MagicMock(), lookback_days=5)
        assert series.empty

    def test_fallback_used_when_primary_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            client = IBKRClient()
        client.ib = MagicMock()
        bar = MagicMock(date="2024-01-02", close=10.0)
        client.ib.reqHistoricalData.side_effect = [[], [bar]]

        series = client.get_daily_closes_with_fallback(MagicMock(), MagicMock(), lookback_days=5)
        assert len(series) == 1


# ---------------------------------------------------------------------
# Defense-in-depth: no order-placement capability exists in shadow/
# ---------------------------------------------------------------------

_ORDER_PLACEMENT_METHOD_NAMES = {"placeOrder", "whatIfOrder", "place_order", "submit_entry", "submit_exit"}


def test_ibkr_module_never_calls_placeorder():
    """Structural proof: shadow/brokers/ibkr.py never calls IB.placeOrder
    (or any order-placement-shaped method) -- the read-only surface is all
    there is, not a behaviorally-unused-but-present capability."""
    source = (SHADOW_ROOT / "brokers" / "ibkr.py").read_text()
    tree = ast.parse(source)
    called_attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            called_attrs.add(node.attr)
    assert called_attrs & _ORDER_PLACEMENT_METHOD_NAMES == set()
