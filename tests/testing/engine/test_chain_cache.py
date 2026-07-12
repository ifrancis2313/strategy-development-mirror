import pandas as pd
from pandas.testing import assert_frame_equal

from testing.engine.chain_cache import load_or_build_spx_chain


class _FakeRepo:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = 0

    def get_chain(self, secid, start_date, end_date, columns=None):
        self.calls += 1
        return self.frame


def _sample_chain() -> pd.DataFrame:
    return pd.DataFrame({
        "secid": [108105, 108105],
        "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        "exdate": pd.to_datetime(["2020-02-21", "2020-02-21"]),
        "cp_flag": ["P", "C"],
        "strike_price": [3200.0, 3300.0],
        "best_bid": [10.0, 8.0],
        "best_offer": [10.5, 8.5],
        "delta": [-0.20, 0.30],
        "impl_volatility": [0.18, 0.17],
        "optionid": [1, 2],
    })


def test_cache_miss_queries_repo_and_writes_cache_file(tmp_path):
    repo = _FakeRepo(_sample_chain())
    path = str(tmp_path / "spx_chain_cache.parquet")

    result = load_or_build_spx_chain(repo, path=path)

    assert repo.calls == 1
    assert (tmp_path / "spx_chain_cache.parquet").exists()
    assert_frame_equal(result, _sample_chain())


def test_cache_hit_does_not_query_repo_again(tmp_path):
    repo = _FakeRepo(_sample_chain())
    path = str(tmp_path / "spx_chain_cache.parquet")

    first = load_or_build_spx_chain(repo, path=path)
    second = load_or_build_spx_chain(repo, path=path)

    assert repo.calls == 1
    assert_frame_equal(first, second)


def test_cache_hit_returns_frame_equal_to_a_direct_repo_call(tmp_path):
    repo = _FakeRepo(_sample_chain())
    path = str(tmp_path / "spx_chain_cache.parquet")

    load_or_build_spx_chain(repo, path=path)
    cached = load_or_build_spx_chain(repo, path=path)
    direct = repo.get_chain(108105, "1996-01-01", "2019-12-31", columns=None)

    assert_frame_equal(cached, direct)
