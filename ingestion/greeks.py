"""Vectorized Black-Scholes greeks and implied vol, put/call aware, operating
over a whole chain at once (arrays/Series in, arrays out) rather than one
contract at a time. Ported verbatim from Options and Futures Pipeline's
`pipeline/block1_ingestion/greeks.py`.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, right: np.ndarray) -> np.ndarray:
    """right: array of 'C'/'P', same length as the other inputs."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(np.asarray(right) == "C", call, put)


def delta(S, K, T, r, sigma, right) -> np.ndarray:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    call_delta = norm.cdf(d1)
    return np.where(np.asarray(right) == "C", call_delta, call_delta - 1)


def gamma(S, K, T, r, sigma, right=None) -> np.ndarray:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, right=None) -> np.ndarray:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1 vol point (1%)


def theta(S, K, T, r, sigma, right) -> np.ndarray:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    call = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    put = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    raw = np.where(np.asarray(right) == "C", call, put)
    return raw / 365  # per calendar day


def all_greeks(S, K, T, r, sigma, right) -> dict[str, np.ndarray]:
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    right = np.asarray(right)
    return {
        "delta": delta(S, K, T, r, sigma, right),
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, right),
        "vega": vega(S, K, T, r, sigma),
    }


def implied_vol(price: float, S: float, K: float, T: float, r: float, right: str,
                 lo: float = 1e-4, hi: float = 5.0) -> float | None:
    """Brent's method root-find on BS price. Not vectorized -- only used as a
    per-row fallback when a vendor doesn't supply IV, which is the rare case.
    Returns None if no sign change in [lo, hi] (e.g. price outside no-arbitrage
    bounds) rather than raising, since one bad quote shouldn't kill a chain.
    """
    def f(sigma):
        return bs_price(np.array([S]), np.array([K]), np.array([T]),
                         np.array([r]), np.array([sigma]), np.array([right]))[0] - price

    try:
        return brentq(f, lo, hi)
    except ValueError:
        return None
