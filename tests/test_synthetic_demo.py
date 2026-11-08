"""Exercise the synthetic demo through its real validation entry points."""

import os

import pytest

from examples.synthetic_demo import DISCLAIMER, SKIP, main
from testing.monte_carlo.pbo import _VENDORED_PYPBO_DIR


def test_demo_is_deterministic_and_labeled(capsys):
    main(["--seed", "42"])
    first = capsys.readouterr().out
    main(["--seed", "42"])
    assert capsys.readouterr().out == first
    assert first.splitlines()[0] == DISCLAIMER
    assert (
        "Prices are a zero-drift random walk, so any apparent performance above is noise; "
        "the Monte Carlo p-values are the check that flags it."
    ) in first.splitlines()
    assert "Folds: 5" in first
    assert first.count("MCPT ma_") == 3
    if not os.path.isdir(_VENDORED_PYPBO_DIR):
        assert first.count(SKIP) == 2


def test_default_path_uses_loader_import_error(monkeypatch, capsys):
    import testing.monte_carlo.pbo as pbo_module

    monkeypatch.setattr(pbo_module, "_VENDORED_PYPBO_DIR", "/nonexistent/path/pypbo-cscv")
    monkeypatch.delenv("PYPBO_CSCV_DIR", raising=False)
    main([])
    output = capsys.readouterr().out
    assert output.startswith(DISCLAIMER + "\n")
    assert output.count(SKIP) == 2
    assert "Graduation gate ma_" not in output


@pytest.mark.skipif(
    not os.path.isdir(_VENDORED_PYPBO_DIR),
    reason=f"vendored pypbo-cscv not found at {_VENDORED_PYPBO_DIR} on this machine/runner",
)
def test_full_path(capsys):
    main([])
    output = capsys.readouterr().out
    assert SKIP not in output
    assert "PBO: " in output
    assert output.count("Graduation gate ma_") == 3
    assert "named source=not_attested; sensitivity=not_attested" in output
