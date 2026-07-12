"""End-to-end tests of the graduation mechanism against a synthetic
`toy_example` strategy fixture (see testing/strategies/toy_example) -- the
acceptance criteria this mechanism is actually graded on:

1. A graduation produces a frozen, versioned copy + manifest (what/when/params).
2. The graduated copy has zero live import/dependency back to testing/'s
   ongoing code -- proven here by mutating the *live* strategy code after
   graduating and confirming the frozen copy is unaffected, not merely by
   checking for absent import strings (that's `test_snapshot.py`'s job).
3. Re-running a graduated snapshot's own recorded params back through
   Testing on fresh data actually works.
4. Graduating is a manual function call, never triggered as a side effect
   of anything else (see `test_graduate_is_never_triggered_automatically`).
"""

import pandas as pd

from testing.engine.backtest_runner import walk_forward
from testing.engine.strike_selector import StrikeSelector
from testing.graduation.graduate import graduate
from testing.graduation.loader import load_snapshot_module
from testing.graduation.manifest import GraduationManifest
from testing.graduation.revalidate import rehydrate_config
from testing.strategies.toy_example.config import ToyConfig
from testing.strategies.toy_example.trade_selector import ToyTradeSelector


def _chain_row(date, exdate, cp_flag, strike, bid, offer, delta, optionid, secid=108105):
    return {
        "secid": secid,
        "date": pd.Timestamp(date),
        "exdate": pd.Timestamp(exdate),
        "cp_flag": cp_flag,
        "strike_price": strike,
        "best_bid": bid,
        "best_offer": offer,
        "delta": delta,
        "impl_volatility": 0.20,
        "optionid": optionid,
    }


def _one_trade_chain(entry_date: str, exit_check_date: str, exdate: str) -> pd.DataFrame:
    """One entry day at ToyConfig's default short/long put deltas (-0.25/-0.10,
    31 DTE, inside the default [30, 45] window), plus one later day whose
    tighter bid/ask clears the default 50% profit target -- same trade shape
    `tests/testing/engine/test_backtest_runner.py` already establishes.
    """
    return pd.DataFrame(
        [
            _chain_row(entry_date, exdate, "P", 95, 3.0, 3.2, -0.25, "short"),
            _chain_row(entry_date, exdate, "P", 90, 1.0, 1.2, -0.10, "long"),
            _chain_row(exit_check_date, exdate, "P", 95, 0.9, 1.0, -0.10, "short"),
            _chain_row(exit_check_date, exdate, "P", 90, 0.2, 0.3, -0.05, "long"),
        ]
    )


def _run(chain: pd.DataFrame, config: ToyConfig):
    selector = ToyTradeSelector(config, StrikeSelector())
    return walk_forward(chain, config, selector)


class TestGraduationProducesFrozenSnapshotAndManifest:
    def test_manifest_and_frozen_files_written_to_disk(self, tmp_path):
        config = ToyConfig()

        manifest = graduate(
            "toy_example", config, destination_root=tmp_path, notes="graduation end-to-end test"
        )

        destination = tmp_path / manifest.strategy_name / manifest.graduation_id
        assert (destination / "manifest.json").exists()
        assert (destination / "strategies" / "toy_example" / "config.py").exists()
        assert (destination / "strategies" / "toy_example" / "trade_selector.py").exists()
        assert (destination / "engine" / "execution_config.py").exists()
        assert (destination / "engine" / "leg.py").exists()
        assert (destination / "engine" / "position.py").exists()
        assert (destination / "engine" / "strike_selector.py").exists()

        read_back = GraduationManifest.read(destination)
        assert read_back == manifest
        assert read_back.strategy_name == "toy_example"
        assert read_back.params["short_put_delta"] == -0.25
        assert read_back.notes == "graduation end-to-end test"

    def test_second_graduation_of_same_strategy_increments_version(self, tmp_path):
        first = graduate("toy_example", ToyConfig(), destination_root=tmp_path)
        second = graduate("toy_example", ToyConfig(long_put_delta=-0.05), destination_root=tmp_path)

        assert first.version == 1
        assert second.version == 2
        assert first.graduation_id != second.graduation_id


class TestZeroLiveDependency:
    def test_frozen_snapshot_unaffected_by_mutating_the_live_strategy_code(self, tmp_path, monkeypatch):
        chain = _one_trade_chain("2020-01-15", "2020-01-20", "2020-02-15")
        config = ToyConfig()

        trades_before = _run(chain, config)
        assert len(trades_before) == 1
        _, result_before = trades_before[0]
        assert result_before.exit_reason == "profit_target"

        manifest = graduate("toy_example", config, destination_root=tmp_path)
        destination = tmp_path / manifest.strategy_name / manifest.graduation_id

        # Simulate "the source changed tomorrow": the live selector now
        # rejects every trade unconditionally.
        monkeypatch.setattr(
            ToyTradeSelector, "select_trade", lambda self, *args, **kwargs: None
        )
        live_selector = ToyTradeSelector(config, StrikeSelector())
        assert live_selector.select_trade(chain, "2020-01-15", has_open_position=False) is None

        # The frozen copy is a separate class object loaded from its own
        # file, not the monkeypatched live class -- it must still work.
        frozen_config_cls = load_snapshot_module(destination, "strategies.toy_example.config").ToyConfig
        frozen_selector_cls = load_snapshot_module(
            destination, "strategies.toy_example.trade_selector"
        ).ToyTradeSelector
        frozen_strike_selector_cls = load_snapshot_module(
            destination, "engine.strike_selector"
        ).StrikeSelector

        assert frozen_selector_cls is not ToyTradeSelector  # genuinely different class object

        frozen_config = frozen_config_cls(**manifest.params)
        frozen_selector = frozen_selector_cls(frozen_config, frozen_strike_selector_cls())

        trades_after = walk_forward(chain, frozen_config, frozen_selector)

        assert len(trades_after) == 1
        spec_after, result_after = trades_after[0]
        spec_before, _ = trades_before[0]
        assert spec_after.entry_date == spec_before.entry_date
        assert result_after.exit_reason == result_before.exit_reason
        assert result_after.pnl == result_before.pnl
        assert result_after.entry_credit == result_before.entry_credit

    def test_frozen_copy_files_contain_no_testing_import_strings(self, tmp_path):
        manifest = graduate("toy_example", ToyConfig(), destination_root=tmp_path)
        destination = tmp_path / manifest.strategy_name / manifest.graduation_id

        for path in destination.rglob("*.py"):
            text = path.read_text()
            assert "from testing." not in text
            assert "import testing" not in text


class TestRevalidationOnFreshData:
    def test_rehydrated_params_reproduce_original_config(self, tmp_path):
        original = ToyConfig(short_put_delta=-0.25, long_put_delta=-0.10)

        manifest = graduate("toy_example", original, destination_root=tmp_path)
        rehydrated = rehydrate_config(manifest, ToyConfig)

        assert rehydrated == original

    def test_rehydrated_params_run_through_live_testing_engine_on_fresh_data(self, tmp_path):
        """The actual supported re-validation path: graduate a config, pull
        its params back out of the manifest, reconstruct a live config from
        them, and run it through the live `walk_forward` on data the
        original graduation never saw. Must reproduce exactly what directly
        constructing the same config and running it would produce -- proof
        the round trip through the manifest doesn't lose or corrupt
        anything.
        """
        original = ToyConfig(short_put_delta=-0.25, long_put_delta=-0.10)
        manifest = graduate("toy_example", original, destination_root=tmp_path)

        # "Fresh data": different calendar dates than the graduation ever ran on.
        fresh_chain = _one_trade_chain("2024-06-03", "2024-06-08", "2024-07-03")

        rehydrated = rehydrate_config(manifest, ToyConfig)
        trades_rehydrated = _run(fresh_chain, rehydrated)

        directly_constructed = ToyConfig(short_put_delta=-0.25, long_put_delta=-0.10)
        trades_direct = _run(fresh_chain, directly_constructed)

        assert len(trades_rehydrated) == 1
        spec_r, result_r = trades_rehydrated[0]
        spec_d, result_d = trades_direct[0]
        assert spec_r.entry_date == spec_d.entry_date == pd.Timestamp("2024-06-03")
        assert result_r.exit_reason == result_d.exit_reason == "profit_target"
        assert result_r.pnl == result_d.pnl
        assert result_r.entry_credit == result_d.entry_credit


class TestGraduationIsManualOnly:
    def test_graduate_is_a_plain_function_not_wired_to_any_trigger(self):
        """Guards against a future regression where graduation gets called
        as a side effect of running Testing (e.g. from within
        `walk_forward` or a comparison script) -- it must only ever be
        reachable via an explicit, top-level call.
        """
        import testing.engine.backtest_runner as backtest_runner
        import testing.engine.strategy_comparison as strategy_comparison

        assert "graduate" not in vars(backtest_runner)
        assert "graduate" not in vars(strategy_comparison)
