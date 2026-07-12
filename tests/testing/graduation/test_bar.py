import numpy as np
import pytest

from testing.graduation.bar import DEFAULT_ALPHA, PBO_REFERENCE_BAND_UPPER, check_graduation_bar
from testing.monte_carlo.mcpt import MCPTResult
from testing.monte_carlo.pbo import PBOResult


def _mcpt(p_value: float | None) -> MCPTResult:
    return MCPTResult(observed_sharpe=1.0, p_value=p_value, n_permutations=1000, null_distribution=np.array([0.0]))


def _pbo(value: float) -> PBOResult:
    return PBOResult(
        pbo=value,
        prob_oos_loss=0.5,
        n_chunks=16,
        n_combinations=12870,
        n_parameterizations=9,
        n_periods_used=1000,
        n_periods_dropped=0,
    )


def test_passes_every_criterion() -> None:
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(0.1), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.mcpt_verdict == "pass"
    assert report.pbo_verdict == "pass"
    assert report.named_source_verdict == "pass"
    assert report.sensitivity_analysis_verdict == "pass"
    assert report.ready_to_graduate is True


def test_mcpt_fails_when_p_value_exceeds_corrected_alpha() -> None:
    # p=0.03 fails uncorrected 0.05 too, but definitely fails once corrected
    # across 10 tests (alpha/10 = 0.005).
    report = check_graduation_bar(
        _mcpt(0.03), _pbo(0.1), n_tests=10, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.mcpt_verdict == "fail"
    assert report.ready_to_graduate is False


def test_mcpt_p_value_none_is_a_fail_not_a_pass() -> None:
    # None means every permutation draw was degenerate -- must never be
    # silently treated as "no evidence against, so pass".
    report = check_graduation_bar(
        _mcpt(None), _pbo(0.1), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.mcpt_verdict == "fail"
    assert report.ready_to_graduate is False


def test_pbo_fails_above_reference_band() -> None:
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(PBO_REFERENCE_BAND_UPPER + 0.01), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.pbo_verdict == "fail"
    assert report.ready_to_graduate is False


def test_pbo_passes_at_the_alpha_pipeline_model5_reference_value() -> None:
    # Real value from alpha-pipeline's own validated Model 5 (PBO=0.168) --
    # anchors this test to the actual precedent the bar was built to match.
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(0.168), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.pbo_verdict == "pass"


def test_unattested_manual_criteria_block_graduation() -> None:
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(0.1), n_tests=1, has_named_source=False, has_sensitivity_analysis=False
    )
    assert report.named_source_verdict == "not_attested"
    assert report.sensitivity_analysis_verdict == "not_attested"
    assert report.ready_to_graduate is False


def test_sharpe_and_drawdown_never_block_graduation_per_adr_0003() -> None:
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(0.1), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.sharpe_verdict == "not_yet_defined"
    assert report.drawdown_verdict == "not_yet_defined"
    assert report.ready_to_graduate is True


def test_mcpt_p_value_exactly_at_corrected_alpha_passes() -> None:
    """mcpt_verdict uses `<=`, not `<` -- mutation testing (2026-07-12) found
    the boundary itself untested. n_tests=1 keeps corrected_alpha == alpha
    (0.05) exact in float64, so p_value=0.05 lands bit-identical to it."""
    report = check_graduation_bar(
        _mcpt(DEFAULT_ALPHA), _pbo(0.1), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.mcpt_p_value == report.mcpt_bonferroni_alpha  # confirms the boundary actually lands exactly
    assert report.mcpt_verdict == "pass"


def test_pbo_exactly_at_reference_band_upper_fails() -> None:
    """pbo_verdict uses `<`, not `<=` -- a PBO exactly at the reference band
    edge does NOT clear it (mutation testing, 2026-07-12: boundary untested)."""
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(PBO_REFERENCE_BAND_UPPER), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    assert report.pbo_verdict == "fail"


def test_zero_tests_rejected() -> None:
    with pytest.raises(ValueError):
        check_graduation_bar(_mcpt(0.001), _pbo(0.1), n_tests=0, has_named_source=True, has_sensitivity_analysis=True)


def test_summary_mentions_every_criterion() -> None:
    report = check_graduation_bar(
        _mcpt(0.001), _pbo(0.1), n_tests=1, has_named_source=True, has_sensitivity_analysis=True
    )
    text = report.summary()
    for keyword in ("MCPT", "PBO", "Named source", "Sensitivity", "Sharpe", "drawdown", "Ready to graduate"):
        assert keyword in text
