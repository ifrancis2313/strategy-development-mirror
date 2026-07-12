from collections.abc import Sequence

import pandas as pd
import pyarrow.dataset as ds


class OptionChainRepository:
    """Queries the IvyDB option chain Parquet file without loading it fully into memory.

    Filters by secid/date push down into pyarrow's row-group skipping (the file's
    row groups are date-contiguous), and only the requested columns are read.
    Parameterized by secid -- never hardcodes SPX -- so it stays usable for any
    other underlying in the same file.
    """

    #: OptionMetrics stores strike_price as an integer scaled by 1000
    #: (e.g. a 1149.92 strike is stored as 1149920) -- confirmed against real
    #: bid/ask magnitudes on 2010-06-30 SPX. Rescaled here so every downstream
    #: module gets real dollar strikes without re-deriving this fix.
    _STRIKE_SCALE = 1000.0

    def __init__(self, parquet_path: str):
        self._dataset = ds.dataset(parquet_path, format="parquet")

    @classmethod
    def _rescale(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "strike_price" in df.columns:
            df["strike_price"] = df["strike_price"] / cls._STRIKE_SCALE
        return df

    def get_chain(
        self,
        secid: int,
        start_date,
        end_date,
        columns: Sequence[str] | None = None,
        cp_flag: str | None = None,
        optionids: Sequence | None = None,
    ) -> pd.DataFrame:
        filter_expr = (
            (ds.field("secid") == secid)
            & (ds.field("date") >= pd.Timestamp(start_date))
            & (ds.field("date") <= pd.Timestamp(end_date))
        )
        if cp_flag is not None:
            filter_expr &= ds.field("cp_flag") == cp_flag
        if optionids is not None:
            filter_expr &= ds.field("optionid").isin(list(optionids))

        table = self._dataset.to_table(filter=filter_expr, columns=columns)
        return self._rescale(table.to_pandas())

    def get_snapshot(
        self,
        secid: int,
        date,
        min_dte: int | None = None,
        max_dte: int | None = None,
        cp_flag: str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        df = self.get_chain(secid, date, date, columns=columns, cp_flag=cp_flag)
        if df.empty or (min_dte is None and max_dte is None):
            return df
        dte = (df["exdate"] - df["date"]).dt.days
        if min_dte is not None:
            df = df[dte >= min_dte]
            dte = dte[df.index]
        if max_dte is not None:
            df = df[dte <= max_dte]
        return df
