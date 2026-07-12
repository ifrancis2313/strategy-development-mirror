"""Tests `compute_closure`'s dependency-graph mechanism in isolation, using
a small synthetic `testing`-shaped package tree under `tmp_path` -- not the
real repo -- so these tests exercise the AST-analysis logic itself rather
than doubling as an integration check. A separate real-`toy_example` test
below confirms the closure computed against the actual codebase matches
what was hand-verified by reading the source.
"""

from pathlib import Path

import pytest

from testing.graduation.closure import UnsupportedImportError, compute_closure


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _synthetic_repo(tmp_path: Path) -> Path:
    """A minimal `testing/engine/` + `testing/strategies/widget/` tree,
    mirroring the real package's shape: a strategy module that imports one
    engine module, which itself imports a second engine module.
    """
    root = tmp_path / "repo"
    _write(root / "testing" / "__init__.py")
    _write(root / "testing" / "engine" / "__init__.py")
    _write(
        root / "testing" / "engine" / "alpha.py",
        "from testing.engine.beta import helper\n\n\ndef alpha_fn():\n    return helper()\n",
    )
    _write(root / "testing" / "engine" / "beta.py", "def helper():\n    return 42\n")
    _write(root / "testing" / "strategies" / "__init__.py")
    _write(root / "testing" / "strategies" / "widget" / "__init__.py")
    _write(
        root / "testing" / "strategies" / "widget" / "config.py",
        "from testing.engine.alpha import alpha_fn\n\n\ndef use_it():\n    return alpha_fn()\n",
    )
    return root


def test_closure_includes_strategy_module_and_transitive_engine_deps(tmp_path):
    repo_root = _synthetic_repo(tmp_path)

    closure = compute_closure("testing.strategies.widget", repo_root)

    assert set(closure) == {
        "strategies.widget.config",
        "engine.alpha",
        "engine.beta",
    }


def test_closure_module_source_paths_point_at_real_files(tmp_path):
    repo_root = _synthetic_repo(tmp_path)

    closure = compute_closure("testing.strategies.widget", repo_root)

    assert closure["engine.beta"].source_path == repo_root / "testing" / "engine" / "beta.py"
    assert closure["engine.beta"].absolute_dotted == "testing.engine.beta"


def test_closure_rejects_bare_import_of_testing_package(tmp_path):
    root = tmp_path / "repo"
    _write(root / "testing" / "__init__.py")
    _write(root / "testing" / "engine" / "__init__.py")
    _write(root / "testing" / "engine" / "gamma.py", "def noop():\n    return None\n")
    _write(root / "testing" / "strategies" / "__init__.py")
    _write(root / "testing" / "strategies" / "widget" / "__init__.py")
    _write(
        root / "testing" / "strategies" / "widget" / "config.py",
        "import testing.engine.gamma\n",
    )

    with pytest.raises(UnsupportedImportError):
        compute_closure("testing.strategies.widget", root)


def test_closure_rejects_bare_from_testing_import(tmp_path):
    root = tmp_path / "repo"
    _write(root / "testing" / "__init__.py")
    _write(root / "testing" / "strategies" / "__init__.py")
    _write(root / "testing" / "strategies" / "widget" / "__init__.py")
    _write(
        root / "testing" / "strategies" / "widget" / "config.py",
        "from testing import engine\n",
    )

    with pytest.raises(UnsupportedImportError):
        compute_closure("testing.strategies.widget", root)


def test_closure_ignores_non_testing_imports(tmp_path):
    root = tmp_path / "repo"
    _write(root / "testing" / "__init__.py")
    _write(root / "testing" / "strategies" / "__init__.py")
    _write(root / "testing" / "strategies" / "widget" / "__init__.py")
    _write(
        root / "testing" / "strategies" / "widget" / "config.py",
        "import pandas as pd\nfrom dataclasses import dataclass\n",
    )

    closure = compute_closure("testing.strategies.widget", root)

    assert set(closure) == {"strategies.widget.config"}


def test_closure_rejects_non_testing_strategy_package(tmp_path):
    with pytest.raises(ValueError):
        compute_closure("not_testing.strategies.widget", tmp_path)


def test_real_toy_example_closure_is_exactly_config_selector_plus_deps():
    """Confirms the closure computed against the actual repo matches the
    minimal set hand-derived by reading toy_example's own imports --
    config/trade_selector, plus execution_config/leg/position/
    strike_selector from engine/. Deliberately excludes backtest_runner,
    position_simulator, performance, sizing, portfolio, chain_repository,
    chain_cache, spot, strategy_comparison -- none of those are imported by
    the strategy's own decision-logic modules, so none belong in a frozen
    snapshot meant only to run that decision logic standalone.
    """
    repo_root = Path(__file__).resolve().parents[3]

    closure = compute_closure("testing.strategies.toy_example", repo_root)

    assert set(closure) == {
        "strategies.toy_example.config",
        "strategies.toy_example.trade_selector",
        "engine.execution_config",
        "engine.leg",
        "engine.position",
        "engine.strike_selector",
    }
