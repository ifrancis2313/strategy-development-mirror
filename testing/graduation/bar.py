"""The Testing -> Shadow graduation bar (issue 10 / ADR 0003).

Checks only what's actually defined right now: MCPT (Bonferroni-corrected
p-value) and PBO (below the research-methodology reference band). Named-
source and sensitivity-analysis criteria are manual checklist items -- "is
there a real thesis behind this strategy" isn't something code can verify --
represented here as fields the caller must attest to, not computed. Sharpe
and drawdown floors are deliberately undefined per ADR 0003 (premature to
set a number before a real body of Testing results exists) and are reported
as `not_yet_defined`, never silently skipped or defaulted to a guessed
number that could be mistaken for a real bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from testing.monte_carlo.mcpt import MCPTResult
from testing.monte_carlo.pbo import PBOResult

#: research-methodology's own synthetic-validation reference band -- PBO
#: below this doesn't look overfit. Alpha-pipeline's real Model 5 result
#: (PBO=0.168) was judged against this same band; reused here rather than
#: inventing a separate one, so the bar stays consistent across repos.
PBO_REFERENCE_BAND_UPPER = 0.26

#: Family-wise significance level the Bonferroni correction targets.
DEFAULT_ALPHA = 0.05

Verdict = Literal["pass", "fail", "not_yet_defined", "not_attested"]


@dataclass(frozen=True)
class GraduationBarReport:
    """One criterion per field. `Verdict` is `"not_attested"` for the two
    manual criteria when the caller didn't explicitly attest to them (the
    safe default -- silence is not evidence of a real thesis or a real
    sensitivity check), and `"not_yet_defined"` for Sharpe/drawdown per
    ADR 0003.
    """

    mcpt_verdict: Verdict
    mcpt_p_value: float | None
    mcpt_bonferroni_alpha: float
    pbo_verdict: Verdict
    pbo_value: float | None
    named_source_verdict: Verdict
    sensitivity_analysis_verdict: Verdict
    sharpe_verdict: Verdict = "not_yet_defined"
    drawdown_verdict: Verdict = "not_yet_defined"

    @property
    def ready_to_graduate(self) -> bool:
        """`True` only if every criterion that's actually defined and
        checkable passed. Sharpe/drawdown being `not_yet_defined` does NOT
        block this -- per ADR 0003, those floors don't exist yet, so a
        strategy can't be held to a bar that was deliberately never set.
        `not_attested` criteria DO block it -- an un-attested manual check
        is not the same as a passed one.
        """
        blocking = (self.mcpt_verdict, self.pbo_verdict, self.named_source_verdict, self.sensitivity_analysis_verdict)
        return all(v == "pass" for v in blocking)

    def summary(self) -> str:
        lines = [
            f"MCPT (Bonferroni-corrected): {self.mcpt_verdict}"
            + (f" (p={self.mcpt_p_value:.4f}, alpha={self.mcpt_bonferroni_alpha:.4f})" if self.mcpt_p_value is not None else ""),
            f"PBO (< {PBO_REFERENCE_BAND_UPPER}): {self.pbo_verdict}" + (f" (pbo={self.pbo_value:.4f})" if self.pbo_value is not None else ""),
            f"Named source/thesis: {self.named_source_verdict}",
            f"Sensitivity analysis on calibrated params: {self.sensitivity_analysis_verdict}",
            f"Minimum Sharpe: {self.sharpe_verdict} (ADR 0003 -- no floor set yet)",
            f"Maximum drawdown: {self.drawdown_verdict} (ADR 0003 -- no floor set yet)",
            f"Ready to graduate: {self.ready_to_graduate}",
        ]
        return "\n".join(lines)


def check_graduation_bar(
    mcpt: MCPTResult,
    pbo: PBOResult,
    n_tests: int,
    *,
    has_named_source: bool,
    has_sensitivity_analysis: bool,
    alpha: float = DEFAULT_ALPHA,
) -> GraduationBarReport:
    """`n_tests` is however many strategy/config variants were tested in
    this session -- the Bonferroni denominator, matching the session-wide
    correction convention `testing.metrics.factor` already uses. `mcpt`/
    `pbo` are this module's own real results (`sign_permutation_mcpt`/
    `probability_of_backtest_overfitting`'s output), never recomputed here.

    `has_named_source`/`has_sensitivity_analysis` are the caller's explicit
    attestation for the two manual criteria -- there is no default that
    means "yes"; omitting them means "not_attested", not "pass".
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1 (the Bonferroni denominator)")

    corrected_alpha = alpha / n_tests
    if mcpt.p_value is None:
        mcpt_verdict: Verdict = "fail"
    else:
        mcpt_verdict = "pass" if mcpt.p_value <= corrected_alpha else "fail"

    pbo_verdict: Verdict = "pass" if pbo.pbo < PBO_REFERENCE_BAND_UPPER else "fail"

    return GraduationBarReport(
        mcpt_verdict=mcpt_verdict,
        mcpt_p_value=mcpt.p_value,
        mcpt_bonferroni_alpha=corrected_alpha,
        pbo_verdict=pbo_verdict,
        pbo_value=pbo.pbo,
        named_source_verdict="pass" if has_named_source else "not_attested",
        sensitivity_analysis_verdict="pass" if has_sensitivity_analysis else "not_attested",
    )
