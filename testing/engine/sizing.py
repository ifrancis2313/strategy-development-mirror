from dataclasses import dataclass

from testing.engine.execution_config import ExecutionConfig
from testing.engine.payoff import max_loss_at_expiration
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import TradeResult


@dataclass(frozen=True)
class SizedTradeResult:
    trade_result: TradeResult
    contracts: int
    pnl_dollars: float


def contracts_for_entry(spec: PositionSpec, entry_credit: float, config: ExecutionConfig, notional: float | None = None) -> int:
    """Max-loss-based contract count for a single position, given just its
    spec and entry credit -- no `TradeResult` (i.e. no exit) required yet.
    `size_fixed_notional`/`size_vol_scaled` call this per already-simulated
    trade; the live daily-check script calls it directly at entry time,
    before any exit has happened, against the exact same formula so live
    sizing can't drift from backtest sizing.

    `notional` defaults to `config.fixed_notional` when not given (the
    fixed-notional case); callers doing vol-scaled sizing pass the
    already-VIX-adjusted notional instead.

    Max loss per contract comes from `core.payoff.max_loss_at_expiration`
    (see ADR 0011 and that module's docstring for why it replaced the old
    group-by-right width heuristic that used to live here -- identical
    numbers for a plain vertical/iron condor/calendar spread, correct
    instead of silently wrong for a 3+-strike same-right structure or a
    naked single leg). An `inf` max loss (uncovered short call) makes
    `notional // inf` collapse to 0, so `max(1, ...)` correctly floors to
    the same maximally-conservative 1-contract answer this function already
    gives for any max loss that exceeds notional -- no special-casing
    needed for the unbounded case.
    """
    if notional is None:
        notional = config.fixed_notional
    max_loss_per_contract = max_loss_at_expiration(spec.legs, entry_credit) * config.contract_multiplier
    return max(1, int(notional // max_loss_per_contract)) if max_loss_per_contract > 0 else 1


class SizingEngine:
    """Applies position sizing to already-simulated trades, after the fact --
    sizing only changes contract count, never the per-contract trade economics,
    so it never needs to re-run PositionSimulator.
    """

    def size_fixed_notional(
        self,
        trades: list[tuple[PositionSpec, TradeResult]],
        config: ExecutionConfig,
    ) -> list[SizedTradeResult]:
        sized = []
        for spec, result in trades:
            contracts = contracts_for_entry(spec, result.entry_credit, config)
            pnl_dollars = result.pnl * config.contract_multiplier * contracts
            sized.append(SizedTradeResult(trade_result=result, contracts=contracts, pnl_dollars=pnl_dollars))
        return sized

    def size_vol_scaled(
        self,
        trades: list[tuple[PositionSpec, TradeResult]],
        config: ExecutionConfig,
        vol_by_date: dict,
        reference_vol: float,
    ) -> list[SizedTradeResult]:
        """Same max-loss-based contract count as size_fixed_notional, but the
        notional itself shrinks as the reference vol series rises relative to
        `reference_vol` (selling into an already-elevated-vol environment gets
        less capital, not more, even though the premium looks more
        attractive) -- per the handoff doc's section 4.3. Falls back to the
        unscaled notional on entry dates with no reading in `vol_by_date`.

        Deliberately vol-source-agnostic: `vol_by_date`/`reference_vol` were
        originally named for VIX (the handoff doc's calibration case) but the
        math is just a ratio, so it works identically for any same-unit vol
        series keyed by entry date -- e.g. `testing.engine.realized_vol
        .compute_realized_vol`'s trailing RV, first wired up for VRP
        harvester's own vol-aware sizing (2026-07-10, no VIX data sourced
        yet -- see AGENTS.md's hedge-candidate-search section) before any
        VIX-based caller existed. Renamed from vix_by_date/reference_vix at
        that point since this was still unused by any real strategy and the
        VIX-specific names would have been actively misleading for an RV
        caller. Whatever series is passed, `reference_vol` must be on the
        same scale (e.g. both fractional annualized vol, or both VIX points)
        -- only the ratio matters, so scale-mismatches are silent, not
        checked.
        """
        sized = []
        for spec, result in trades:
            vol_at_entry = vol_by_date.get(result.entry_date)
            notional = config.fixed_notional * (reference_vol / vol_at_entry) if vol_at_entry else config.fixed_notional
            contracts = contracts_for_entry(spec, result.entry_credit, config, notional=notional)
            pnl_dollars = result.pnl * config.contract_multiplier * contracts
            sized.append(SizedTradeResult(trade_result=result, contracts=contracts, pnl_dollars=pnl_dollars))
        return sized
