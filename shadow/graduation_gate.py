"""The Shadow -> Pipeline graduation gate (issue 14 / ADR 0004).

Same shape as `testing.graduation.bar` (issue 10 / ADR 0003), one stage
later: checks what's actually computable now, reports `not_yet_defined`
for what needs real data to calibrate rather than guessing a number.

Track-record length is deliberately not a fixed trade count baked into this
module -- per ADR 0004, how many live trades count as "enough" depends on
each strategy's own trade cadence, and the caller (the collaborative
workflow, per `AGENTS.md`) decides what target to check against, if any.
The divergence check compares a Shadow ledger's live Sharpe against a
reference Sharpe the caller supplies -- typically produced by re-running the
graduated snapshot's own params through Testing on fresh data via
`testing.graduation.revalidate` -- rather than this module recomputing
anything from `testing/`. Zero import from `testing.engine`/`testing.strategies`,
matching the rest of `shadow/` (see `tests/shadow/test_zero_testing_coupling.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["pass", "fail", "not_yet_defined"]


@dataclass(frozen=True)
class ShadowTrackRecord:
    """Extracted from `shadow.ledger.load_state(strategy)[ledger]` -- a
    plain summary of one ledger's closed-trade history, not a copy of the
    raw state dict.
    """

    strategy: str
    ledger: str
    n_closed_trades: int
    pnl_per_contract: tuple[float, ...]
    first_entry_date: str | None
    last_exit_date: str | None

    @property
    def live_sharpe(self) -> float:
        """Mean/std (ddof=1) of per-contract P&L, unannualized -- a Shadow
        ledger's trades aren't evenly time-spaced the way backtest bars are,
        so there's no single `periods_per_year` to annualize against here.
        Comparisons against a reference Sharpe should use the same
        convention on both sides (see `check_shadow_graduation_gate`).
        Returns `nan` for fewer than 2 trades or zero variance -- same
        degenerate-input convention `testing.monte_carlo.mcpt.annualized_sharpe`
        uses, not 0 and not a raised exception.
        """
        import math

        n = len(self.pnl_per_contract)
        if n < 2:
            return float("nan")
        mean = sum(self.pnl_per_contract) / n
        variance = sum((x - mean) ** 2 for x in self.pnl_per_contract) / (n - 1)
        std = math.sqrt(variance)
        if std == 0:
            return float("nan")
        return mean / std


def shadow_track_record(state: dict[str, Any], strategy: str, ledger: str) -> ShadowTrackRecord:
    """`state` is `shadow.ledger.load_state(strategy)`'s raw output."""
    closed = state.get(ledger, {}).get("closed", [])
    pnl = tuple(trade["pnl_per_contract"] for trade in closed)
    first_entry = closed[0]["entry_date"] if closed else None
    last_exit = closed[-1]["exit_date"] if closed else None
    return ShadowTrackRecord(
        strategy=strategy,
        ledger=ledger,
        n_closed_trades=len(closed),
        pnl_per_contract=pnl,
        first_entry_date=first_entry,
        last_exit_date=last_exit,
    )


@dataclass(frozen=True)
class ShadowGraduationGateReport:
    track_record_verdict: Verdict
    n_closed_trades: int
    min_trades_target: int | None
    divergence_verdict: Verdict
    live_sharpe: float | None
    reference_sharpe: float | None
    max_divergence: float | None

    @property
    def ready_to_graduate(self) -> bool:
        """Same convention as `testing.graduation.bar.GraduationBarReport`:
        `not_yet_defined` criteria never block (there is no bar to have
        cleared or missed yet), only an explicit `fail` does.
        """
        return self.track_record_verdict != "fail" and self.divergence_verdict != "fail"

    def summary(self) -> str:
        lines = [
            f"Track record: {self.track_record_verdict} ({self.n_closed_trades} closed trades"
            + (f", target {self.min_trades_target})" if self.min_trades_target is not None else ", no target set)"),
            f"Live-vs-backtest divergence: {self.divergence_verdict}"
            + (
                f" (live_sharpe={self.live_sharpe:.4f}, reference_sharpe={self.reference_sharpe:.4f}, "
                f"max_divergence={self.max_divergence:.4f})"
                if self.live_sharpe is not None and self.reference_sharpe is not None and self.max_divergence is not None
                else " (no reference/tolerance set)"
            ),
            f"Ready to graduate: {self.ready_to_graduate}",
        ]
        return "\n".join(lines)


def check_shadow_graduation_gate(
    track_record: ShadowTrackRecord,
    *,
    min_trades_target: int | None = None,
    reference_sharpe: float | None = None,
    max_divergence: float | None = None,
) -> ShadowGraduationGateReport:
    """`min_trades_target`/`max_divergence` have no default -- omitting
    either means that criterion is `not_yet_defined`, never a silently
    guessed number, per ADR 0004. `reference_sharpe` is the caller's own
    responsibility to produce (typically via `testing.graduation.revalidate`
    re-run on fresh data) -- this module never reaches into `testing/`.
    """
    if min_trades_target is None:
        track_record_verdict: Verdict = "not_yet_defined"
    else:
        track_record_verdict = "pass" if track_record.n_closed_trades >= min_trades_target else "fail"

    live_sharpe = track_record.live_sharpe if track_record.n_closed_trades >= 2 else None

    # Normalize to None whenever non-finite, not just when already None:
    # ShadowTrackRecord.live_sharpe can itself return NaN (its own
    # zero-variance guard) even with >= 2 trades, and `reference_sharpe` is
    # caller-supplied with no guarantee either. Without this, a NaN value
    # would pass the `is None` check below and fall through to the pass/fail
    # comparison, where `nan <= max_divergence` is always False in Python --
    # silently producing "fail" instead of this module's own
    # "not_yet_defined" convention for a genuinely uninformative input.
    # Found during the front-to-back review (2026-07-12).
    live_sharpe = live_sharpe if live_sharpe is not None and math.isfinite(live_sharpe) else None
    reference_sharpe = reference_sharpe if reference_sharpe is not None and math.isfinite(reference_sharpe) else None

    if reference_sharpe is None or max_divergence is None or live_sharpe is None:
        divergence_verdict: Verdict = "not_yet_defined"
    else:
        divergence_verdict = "pass" if abs(live_sharpe - reference_sharpe) <= max_divergence else "fail"

    return ShadowGraduationGateReport(
        track_record_verdict=track_record_verdict,
        n_closed_trades=track_record.n_closed_trades,
        min_trades_target=min_trades_target,
        divergence_verdict=divergence_verdict,
        live_sharpe=live_sharpe,
        reference_sharpe=reference_sharpe,
        max_divergence=max_divergence,
    )
