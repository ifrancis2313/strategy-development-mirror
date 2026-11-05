"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV), Bailey, Borwein, Lopez de Prado & Zhu (2017) --
asset-class-agnostic.

Where `mcpt.sign_permutation_mcpt` asks "is this one strategy's edge real vs.
random noise," PBO asks a different question: given a matrix of N strategy
variants' returns over T periods (e.g. a parameter sweep), did the process of
*picking the best-looking one in-sample* itself overfit? CSCV splits the T
periods into S contiguous chunks, forms every combinatorial in-sample/
out-of-sample split of size S/2 vs S/2, and for each split checks whether the
in-sample-best variant's out-of-sample rank beats the median. PBO is the
fraction of splits where it doesn't -- high on a grid of pure noise, low when
a real edge exists among the candidates (see this module's own tests, and
the reference benchmark noted below).

**License note.** This wraps a vendored, trimmed copy of `esvhd/pypbo`
(https://github.com/esvhd/pypbo) from a local checkout, which is **GNU
AGPL-3.0** -- not the MIT/Apache/BSD every other dependency in this repo
uses. Reviewed and vendored 2026-07-07 (see that directory's own
`SOURCE.md` for origin commit, trim notes, and a synthetic noise-vs-real-edge
benchmark: pure noise scores PBO ~0.52, a panel with genuine edge among the
candidates scores ~0.26-0.29, both at the literature-standard S=16). AGPL is
fine for this repo's local/personal-research use -- nothing here is served
to third parties -- but must be re-checked before this code is ever hosted
as a network-accessible service or redistributed publicly. Same constraint
already flagged in `alpha-pipeline/research/mc_overfitting_pbo.py` and
`pipeline-dashboard/AGENTS.md` (search "pypbo" there) -- same library, same
constraint, now flagged a third time. This module does not copy that AGPL
source into this repo; it only `sys.path`-imports it at runtime from its one
existing vendored location, so the AGPL code itself is never duplicated
here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

_DEFAULT_VENDORED_PYPBO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendored", "pypbo-cscv")
)
_VENDORED_PYPBO_DIR = os.path.expanduser(
    os.environ.get("PYPBO_CSCV_DIR", _DEFAULT_VENDORED_PYPBO_DIR)
)


def _load_vendored_pypbo(vendored_dir: str | None = None):
    """Imports the vendored `pbo` module from a configurable location,
    inserting that directory onto `sys.path` on first use only. Raises a
    clear `ImportError` (rather than a confusing `ModuleNotFoundError` deep
    in `pypbo`'s own code) if that vendored checkout isn't present.

    The location can be passed explicitly or set through `PYPBO_CSCV_DIR`;
    otherwise the repository-relative `vendored/pypbo-cscv` directory is
    used.
    """
    vendored_dir = os.path.abspath(
        os.path.expanduser(
            vendored_dir or os.environ.get("PYPBO_CSCV_DIR", _VENDORED_PYPBO_DIR)
        )
    )
    if not os.path.isdir(vendored_dir):
        raise ImportError(
            f"Vendored pypbo-cscv not found at {vendored_dir!r}. "
            "See research-methodology/vendored/pypbo-cscv/SOURCE.md for provenance; "
            "this module intentionally does not vendor a second copy (AGPL-3.0, see this "
            "module's docstring)."
        )
    if vendored_dir not in sys.path:
        sys.path.insert(0, vendored_dir)
    import pbo as vendored_pbo

    return vendored_pbo


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    prob_oos_loss: float
    n_chunks: int
    n_combinations: int
    n_parameterizations: int
    n_periods_used: int
    n_periods_dropped: int


def _annualized_sharpe_metric(periods_per_year: float):
    """Builds a `metric_func` for the vendored `pbo()`: column-wise
    annualized Sharpe of a `(sub_T, N)` returns chunk -- the standard PBO
    metric (Bailey et al. use Sharpe by default). A zero-variance column
    scores `0.0` (neither over- nor under-performing) rather than `nan`/`inf`,
    since PBO's rank-based comparison needs a finite value for every column
    in every chunk.
    """

    def _metric(chunk: np.ndarray) -> np.ndarray:
        mean = chunk.mean(axis=0)
        std = chunk.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(std > 0, mean / std * np.sqrt(periods_per_year), 0.0)
        return sharpe

    return _metric


def probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    s_chunks: int = 16,
    periods_per_year: float = 252.0,
    threshold: float = 0.0,
    vendored_pypbo_dir: str | None = None,
) -> PBOResult:
    """CSCV/PBO over a plain returns matrix.

    `returns` -- wide `DataFrame`, rows = periods (any cadence: daily
    returns, per-trade P&L on a common date grid, etc.), columns = the
    parameter variants/strategies being compared. No column needs any
    asset-class-specific meaning; this function only ever calls `.mean()`/
    `.std()` on numeric values. Rows containing a `NaN` in *any* column are
    dropped before chunking (CSCV needs a fully-populated matrix), matching
    `alpha-pipeline/research/mc_overfitting_pbo.py`'s existing convention --
    typically the warmup rows of whichever variant has the longest lookback,
    not a leakage concern since this is a robustness check, not a live
    signal.

    `s_chunks` must be even (literature-standard default 16, `C(16, 8) =
    12,870` combinations); `threshold` is the out-of-sample cutoff for
    `prob_oos_loss` (0.0 for Sharpe = "probability of an out-of-sample
    loss). `vendored_pypbo_dir` selects the external pypbo-cscv checkout;
    when omitted, `PYPBO_CSCV_DIR` or the repository-relative default is used.

    Raises `ValueError` if fewer than 2 columns are given, or too few clean
    rows remain for the requested `s_chunks`.
    """
    if returns.shape[1] < 2:
        raise ValueError(f"PBO/CSCV needs at least 2 parameterizations (columns), got {returns.shape[1]}")
    if s_chunks % 2 != 0:
        raise ValueError(f"s_chunks must be even, got {s_chunks}")

    clean = returns.dropna(how="any")
    dropped = len(returns) - len(clean)
    if len(clean) < s_chunks * 2:
        raise ValueError(
            f"Only {len(clean)} clean periods after dropping {dropped} rows with any NaN; "
            f"need >= {s_chunks * 2} for s_chunks={s_chunks}"
        )

    vendored_pbo = _load_vendored_pypbo(vendored_pypbo_dir)
    metric = _annualized_sharpe_metric(periods_per_year)
    result = vendored_pbo.pbo(clean, S=s_chunks, metric_func=metric, threshold=threshold)

    return PBOResult(
        pbo=round(float(result.pbo), 6),
        prob_oos_loss=round(float(result.prob_oos_loss), 6),
        n_chunks=s_chunks,
        n_combinations=comb(s_chunks, s_chunks // 2),
        n_parameterizations=int(returns.shape[1]),
        n_periods_used=int(len(clean)),
        n_periods_dropped=int(dropped),
    )
