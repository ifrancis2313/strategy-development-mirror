import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_realized_vol(spot: pd.Series, window: int) -> pd.Series:
    """Rolling annualized realized vol (stdev of log returns) on a spot
    series. Generic -- any strategy comparing implied vol against realized
    vol can use this, not just the VRP harvester's own signal framing.
    """
    log_returns = np.log(spot / spot.shift(1))
    rv = log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rv.rename(f"rv_{window}d")
