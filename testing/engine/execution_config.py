from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExecutionConfig:
    """The genuinely strategy-agnostic entry/exit/sizing knobs: which
    underlying, the DTE entry window, the exit rule (profit target / hard
    stop / DTE stop), the fill-price assumption, and notional/contract sizing.

    Every strategy-specific config (e.g. VRPConfig) composes this rather than
    duplicating these fields.
    """

    secid: int = 108105  # SPX

    # Per-book identity (ADR 0010). A "book" is one (symbol, secid)-bound
    # instance of a strategy: its own broker chain symbol, its own realized-vol
    # ticker, and its own settlement/expiry convention. Defaults reproduce the
    # SPX book exactly, so a bare ExecutionConfig()/VRPConfig() behaves as it did
    # before multi-index -- the live SPX shadow is untouched.
    symbol: str = "SPX"  # broker chain symbol (get_chain_snapshot)
    rv_ticker: str = "SPY"  # realized-vol input ticker (get_daily_closes); SPY is SPX's proxy
    settlement_type: str = "PM"  # "PM" = SPXW weeklies; "standard" = standard listed (QQQ/IWM)

    # Live deployment mode (ADR 0008). "shadow" (default): record simulated
    # fills to the shadow ledgers, never place a broker order -- the daily-check
    # script must not even construct the order manager. "live": additionally
    # place real orders into a separate real-fill ledger, while the shadow
    # ledgers keep running. Default is "shadow" so a forgotten setting, a new
    # strategy, or a fresh checkout can never place a real order by accident;
    # going live is a deliberate, git-tracked edit of this one field.
    execution_mode: Literal["shadow", "live"] = "shadow"

    min_dte: int = 30
    max_dte: int = 45

    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    exit_dte: int = 21
    fill_mode: str = "bid_ask"  # or "mid" -- mid is optimistic, for slippage comparison only

    # What profit_target_pct/stop_loss_multiple are a fraction OF (ADR 0011).
    # "entry_credit" (default, unchanged): abs(entry_credit), as it always
    # was -- every currently-live/committed strategy keeps this exact
    # behavior. "max_loss": core.payoff.max_loss_at_expiration(legs,
    # entry_credit) instead -- for a low-or-negative-credit multi-leg
    # structure (gamma_hedged_vrp's 3-put tail leg, a near-zero-cost risk
    # reversal), entry_credit collapses toward 0 and both triggers fire on
    # ordinary daily noise; max_loss scales with the position's actual
    # worst-case risk instead. Opt-in per strategy, not a global default,
    # because switching it changes exit timing (and therefore every
    # backtested number) for whatever strategy sets it -- never flip this on
    # an already-shadow-live config without a full regression re-run.
    exit_trigger_basis: Literal["entry_credit", "max_loss"] = "entry_credit"

    fixed_notional: float = 100_000.0
    contract_multiplier: int = 100

    # Live order limit-price target, as a spread-capture fraction (same scale
    # as fill_mode's numeric form: 0.0=cross the spread, 1.0=mid). 0.65 is
    # the issue #8 follow-up sweep's finding -- Sharpe crossed from negative
    # to positive between 60% and 70% capture on the same 493 backtest
    # trades -- not an arbitrary "be a bit nicer than market order" guess.
    live_target_capture: float = 0.65
