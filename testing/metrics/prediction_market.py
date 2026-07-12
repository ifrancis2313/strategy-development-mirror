"""Prediction-market metrics: calibration, Brier score, log loss,
edge-vs-market-implied-probability, and Kelly/fractional-Kelly position
sizing for binary-outcome markets (Kalshi/PredictIt-style YES/NO contracts).

Deliberately independent of `testing.engine` -- a prediction-market bet's
payoff shape (buy a $0-$1 contract that resolves to exactly $0 or $1) has
nothing in common with an options position's multi-leg payoff diagram, so
this module does not import `engine.payoff` or `engine.sizing`. Reusing
options-style max-loss/notional sizing here would be structurally wrong: an
options spread's max loss is a property of its strikes, not of a
forecaster's belief, whereas a prediction-market bet's correct stake size is
a function of the edge between a predicted probability and the market's own
implied probability (Kelly) -- there is no options analogue to "Kelly-size a
directional probability estimate". See the repo `AGENTS.md`'s asset-class
metrics decision (Greeks for options; IC/cross-sectional for factor models;
calibration/Brier/Kelly for prediction markets -- one module per asset class
rather than a premature shared interface) for why this module stands alone.
"""

from dataclasses import dataclass

import numpy as np


def _validate_probs_and_outcomes(predicted_probs: np.ndarray, outcomes: np.ndarray) -> None:
    if predicted_probs.shape != outcomes.shape:
        raise ValueError(
            f"predicted_probs and outcomes must be the same length, "
            f"got {predicted_probs.shape} vs {outcomes.shape}"
        )
    if predicted_probs.size == 0:
        raise ValueError("predicted_probs/outcomes must not be empty")
    if np.any((predicted_probs < 0.0) | (predicted_probs > 1.0)):
        raise ValueError("predicted_probs must all be in [0, 1]")
    if np.any((outcomes != 0.0) & (outcomes != 1.0)):
        raise ValueError("outcomes must be binary (0/1 or False/True)")


@dataclass(frozen=True)
class CalibrationBin:
    bin_low: float
    bin_high: float
    predicted_mean: float | None
    empirical_freq: float | None
    count: int


@dataclass(frozen=True)
class CalibrationCurve:
    bins: list[CalibrationBin]

    @property
    def calibration_error(self) -> float:
        """Mean absolute gap between predicted mean and empirical frequency,
        weighted by bin count and taken only over occupied bins -- the
        standard scalar summary of a calibration curve (expected calibration
        error / ECE), for when one number is more useful than the full
        per-bin breakdown. Returns 0.0 if every bin is empty (degenerate,
        e.g. n_bins larger than the number of distinct predictions).
        """
        occupied = [b for b in self.bins if b.count > 0]
        if not occupied:
            return 0.0
        total = sum(b.count for b in occupied)
        weighted_gap = sum(
            b.count * abs((b.predicted_mean or 0.0) - (b.empirical_freq or 0.0)) for b in occupied
        )
        return weighted_gap / total


def calibration_curve(predicted_probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> CalibrationCurve:
    """Buckets predictions into `n_bins` equal-width probability bins over
    [0, 1] and compares each bin's mean predicted probability against its
    empirical outcome frequency -- the standard reliability-diagram
    construction (Murphy & Winkler 1977; DeGroot & Fienberg 1983). A
    perfectly-calibrated forecaster's points sit on the y=x diagonal: among
    all bets a forecaster called "70% likely", roughly 70% should resolve
    YES.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    p = np.asarray(predicted_probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    _validate_probs_and_outcomes(p, o)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Interior boundaries only (edges[1:-1]): np.digitize with right=False
    # puts x==1.0 in the last bin (index n_bins - 1) since it's >= every
    # interior boundary, so no separate clip/clamp is needed for the p=1.0
    # edge case.
    bin_idx = np.digitize(p, edges[1:-1], right=False)

    bins = []
    for i in range(n_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        predicted_mean = float(p[mask].mean()) if count else None
        empirical_freq = float(o[mask].mean()) if count else None
        bins.append(
            CalibrationBin(
                bin_low=float(edges[i]),
                bin_high=float(edges[i + 1]),
                predicted_mean=predicted_mean,
                empirical_freq=empirical_freq,
                count=count,
            )
        )
    return CalibrationCurve(bins=bins)


def brier_score(predicted_probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probability and realized binary
    outcome (Brier, 1950) -- the standard proper scoring rule for
    probabilistic binary forecasts. Lower is better; an always-predict-0.5
    forecaster scores exactly 0.25 against a 50/50 base rate, which is the
    natural "no skill" reference point.
    """
    p = np.asarray(predicted_probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    _validate_probs_and_outcomes(p, o)
    return float(np.mean((p - o) ** 2))


def log_loss(predicted_probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the realized outcomes under the
    predicted probabilities -- the other standard proper scoring rule
    alongside Brier score, and more sharply punishing confident-and-wrong
    calls (a 0.99 prediction that resolves NO costs far more log loss than
    Brier's fixed quadratic penalty implies). Probabilities are clipped to
    `[eps, 1-eps]` first since log(0) is undefined -- a forecaster who is
    ever exactly 0% or 100% sure and wrong would otherwise produce infinite
    loss from a single bad call, which would silently break any aggregate
    (e.g. average log loss across a whole strategy) computed downstream.
    """
    p = np.asarray(predicted_probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    _validate_probs_and_outcomes(p, o)
    p_clipped = np.clip(p, eps, 1.0 - eps)
    losses = -(o * np.log(p_clipped) + (1.0 - o) * np.log(1.0 - p_clipped))
    return float(np.mean(losses))


def edge_vs_market(predicted_probs: np.ndarray, market_prices: np.ndarray) -> np.ndarray:
    """Elementwise edge between a predicted probability and the market's own
    implied probability. A Kalshi/PredictIt YES contract's price *is* its
    implied probability -- it costs `price` and pays $1 on YES, so a
    rational market price equals the market's consensus P(YES). Positive
    edge means the predicted probability exceeds the market's (a reason to
    buy YES); negative means a reason to buy NO. This is the same
    `predicted_prob - market_price` quantity `kelly_fraction_binary_market`
    uses internally to size a bet, exposed separately here since a strategy
    review often wants the raw edge distribution (mean edge, hit rate of
    edge sign vs. realized outcome, edge decile bucketing) independent of
    any specific sizing decision.
    """
    p = np.asarray(predicted_probs, dtype=float)
    m = np.asarray(market_prices, dtype=float)
    if p.shape != m.shape:
        raise ValueError(
            f"predicted_probs and market_prices must be the same length, got {p.shape} vs {m.shape}"
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("predicted_probs must all be in [0, 1]")
    if np.any((m <= 0.0) | (m >= 1.0)):
        raise ValueError("market_prices must all be in (0, 1) -- a market price of 0 or 1 has no edge to take")
    return p - m


def kelly_fraction_from_odds(prob: float, b: float) -> float:
    """Classic Kelly criterion for a bet that pays `b` units of profit per
    unit staked on a win (net decimal odds b:1) and loses the entire stake
    on a loss:

        f* = p - q/b = (b*p - q) / b,  where q = 1 - p

    This is the textbook closed form (Kelly, 1956; the standard reference
    form used throughout Thorp's applied-Kelly papers and most
    gambling/quant-finance texts) -- verified in the test suite against the
    canonical worked example (60% win probability, even-money payoff b=1.0
    gives f*=0.2).
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if b <= 0.0:
        raise ValueError(f"b (net odds) must be positive, got {b}")
    q = 1.0 - prob
    return (b * prob - q) / b


@dataclass(frozen=True)
class KellySizing:
    side: str  # "yes", "no", or "none" (no edge either direction)
    edge: float
    full_kelly_fraction: float


def kelly_fraction_binary_market(predicted_prob: float, market_price_yes: float) -> KellySizing:
    """Kelly sizing for a binary prediction-market contract (Kalshi/PredictIt
    style): buying 1 YES contract at `market_price_yes` in (0, 1) costs that
    price and pays out $1 if the market resolves YES, $0 otherwise -- so a
    win nets `(1 - price)` profit per unit staked and a loss costs the full
    stake. That is exactly the classic Kelly bet shape with net odds
    `b = (1 - price) / price`, so substituting into
    `kelly_fraction_from_odds` and simplifying gives

        f*_yes = (predicted_prob - price) / (1 - price)

    A NO contract is priced at `(1 - price)` and resolves on the
    complementary event, so the same derivation applied to
    `(1 - predicted_prob, 1 - price)` gives the NO side's fraction. This
    function picks whichever side has positive edge (a rational bettor only
    ever takes the side the market is mispricing in their favor) and returns
    `side="none", full_kelly_fraction=0.0` when the predicted probability
    exactly matches the market price.
    """
    if not 0.0 < market_price_yes < 1.0:
        raise ValueError(f"market_price_yes must be in (0, 1), got {market_price_yes}")
    if not 0.0 <= predicted_prob <= 1.0:
        raise ValueError(f"predicted_prob must be in [0, 1], got {predicted_prob}")

    edge_yes = predicted_prob - market_price_yes
    if edge_yes == 0.0:
        return KellySizing(side="none", edge=0.0, full_kelly_fraction=0.0)

    if edge_yes > 0.0:
        b = (1.0 - market_price_yes) / market_price_yes
        fraction = kelly_fraction_from_odds(predicted_prob, b)
        return KellySizing(side="yes", edge=edge_yes, full_kelly_fraction=max(fraction, 0.0))

    price_no = 1.0 - market_price_yes
    prob_no = 1.0 - predicted_prob
    b_no = (1.0 - price_no) / price_no
    fraction_no = kelly_fraction_from_odds(prob_no, b_no)
    return KellySizing(side="no", edge=-edge_yes, full_kelly_fraction=max(fraction_no, 0.0))


def fractional_kelly(sizing: KellySizing, fraction: float = 0.5, cap: float = 1.0) -> float:
    """Scales a full-Kelly fraction down by `fraction` (half-Kelly, i.e.
    `fraction=0.5`, is standard practice in applied Kelly betting: full Kelly
    is only growth-optimal if the input probability is exactly correct, and
    overshoots badly -- with much higher variance for a small extra edge in
    growth rate -- whenever the estimate is even slightly off, which a
    prediction-market forecast always is to some degree). The result is then
    capped at `cap` (default 1.0 = never stake more than the entire
    bankroll on one bet, regardless of what the raw formula says).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if cap <= 0.0:
        raise ValueError(f"cap must be positive, got {cap}")
    return min(sizing.full_kelly_fraction * fraction, cap)
