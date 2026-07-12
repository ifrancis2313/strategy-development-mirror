"""Tests the frozen-copy mechanism itself in isolation, using a small
synthetic package tree (not the real `testing/` package) so these tests
exercise the copy+import-rewrite logic on its own -- this is the genuinely
novel part of the graduation mechanism per the issue's acceptance criteria.
"""

from pathlib import Path

import pytest

from testing.graduation.closure import ModuleSource, compute_closure
from testing.graduation.loader import load_snapshot_module
from testing.graduation.snapshot import (
    hash_snapshot,
    relative_import,
    verify_no_testing_imports,
    write_snapshot,
)


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _synthetic_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root / "testing" / "__init__.py")
    _write(root / "testing" / "engine" / "__init__.py")
    _write(
        root / "testing" / "engine" / "alpha.py",
        "from testing.engine.beta import helper\n\n\ndef alpha_fn():\n    return helper() + 1\n",
    )
    _write(root / "testing" / "engine" / "beta.py", "def helper():\n    return 42\n")
    _write(root / "testing" / "strategies" / "__init__.py")
    _write(root / "testing" / "strategies" / "widget" / "__init__.py")
    _write(
        root / "testing" / "strategies" / "widget" / "config.py",
        "from testing.engine.alpha import alpha_fn\n\n\ndef use_it():\n    return alpha_fn()\n",
    )
    return root


class TestRelativeImport:
    def test_reaches_sibling_module_same_directory(self):
        assert relative_import("engine.leg", "engine.strike_selector") == ".strike_selector"

    def test_reaches_across_top_level_packages(self):
        assert (
            relative_import("strategies.vrp_harvester.trade_selector", "engine.leg") == "...engine.leg"
        )

    def test_reaches_sibling_in_same_strategy_package(self):
        assert (
            relative_import("strategies.vrp_harvester.trade_selector", "strategies.vrp_harvester.config")
            == ".config"
        )

    def test_config_reaching_execution_config(self):
        assert (
            relative_import("strategies.vrp_harvester.config", "engine.execution_config")
            == "...engine.execution_config"
        )


class TestWriteSnapshot:
    def test_copies_every_module_in_closure(self, tmp_path):
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"

        written = write_snapshot(closure, destination)

        assert set(written) == {"strategies.widget.config", "engine.alpha", "engine.beta"}
        assert (destination / "engine" / "alpha.py").exists()
        assert (destination / "engine" / "beta.py").exists()
        assert (destination / "strategies" / "widget" / "config.py").exists()

    def test_generates_init_py_at_every_package_level(self, tmp_path):
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"

        write_snapshot(closure, destination)

        assert (destination / "__init__.py").exists()
        assert (destination / "engine" / "__init__.py").exists()
        assert (destination / "strategies" / "__init__.py").exists()
        assert (destination / "strategies" / "widget" / "__init__.py").exists()

    def test_rewrites_absolute_testing_imports_to_relative(self, tmp_path):
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"

        write_snapshot(closure, destination)

        config_text = (destination / "strategies" / "widget" / "config.py").read_text()
        alpha_text = (destination / "engine" / "alpha.py").read_text()
        assert "from testing." not in config_text
        assert "from ...engine.alpha import alpha_fn" in config_text
        assert "from testing." not in alpha_text
        assert "from .beta import helper" in alpha_text

    def test_verify_no_testing_imports_passes_on_clean_snapshot(self, tmp_path):
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"
        write_snapshot(closure, destination)

        verify_no_testing_imports(destination)  # must not raise

    def test_verify_no_testing_imports_catches_a_leftover_absolute_import(self, tmp_path):
        destination = tmp_path / "leaky_snapshot"
        destination.mkdir()
        _write(destination / "__init__.py")
        _write(destination / "leaky.py", "from testing.engine.beta import helper\n")

        with pytest.raises(RuntimeError, match="still imports"):
            verify_no_testing_imports(destination)

    def test_verify_no_testing_imports_catches_a_bare_import_testing_statement(self, tmp_path):
        """`alias.name == TESTING_ROOT_PACKAGE or alias.name.startswith(TESTING_ROOT_PACKAGE + ".")`
        (the `ast.Import`, not `ast.ImportFrom`, branch) -- mutation testing
        (2026-07-12) found this survives being flipped to `and`. A bare
        `import testing` (exact match, no dot) satisfies the first
        condition but not `.startswith("testing.")`, so `and` would let it
        through silently -- this repo's own explicit zero-live-coupling
        guarantee (AGENTS.md), untested against this exact import shape."""
        destination = tmp_path / "leaky_snapshot_bare_import"
        destination.mkdir()
        _write(destination / "__init__.py")
        _write(destination / "leaky.py", "import testing\n")

        with pytest.raises(RuntimeError, match="still imports"):
            verify_no_testing_imports(destination)

    def test_snapshot_is_frozen_against_later_source_mutation(self, tmp_path):
        """The core guarantee: once written, a snapshot must be unaffected
        by the source module changing afterwards.
        """
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"
        write_snapshot(closure, destination)

        before = (destination / "engine" / "beta.py").read_text()

        # Mutate the *source* file after the snapshot was taken.
        (repo_root / "testing" / "engine" / "beta.py").write_text("def helper():\n    return 999999\n")

        after = (destination / "engine" / "beta.py").read_text()
        assert after == before
        assert "999999" not in after

    def test_hash_snapshot_changes_when_snapshot_content_changes(self, tmp_path):
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"
        write_snapshot(closure, destination)

        original_hash = hash_snapshot(destination)

        (destination / "engine" / "beta.py").write_text("def helper():\n    return -1\n")
        tampered_hash = hash_snapshot(destination)

        assert original_hash != tampered_hash

    def test_hash_snapshot_is_deterministic_for_identical_content(self, tmp_path):
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)

        dest_a = tmp_path / "snapshot_a"
        dest_b = tmp_path / "snapshot_b"
        write_snapshot(closure, dest_a)
        write_snapshot(closure, dest_b)

        assert hash_snapshot(dest_a) == hash_snapshot(dest_b)


class TestFrozenSnapshotIsExecutable:
    def test_loaded_snapshot_runs_standalone_and_produces_correct_result(self, tmp_path):
        """Not just "no testing.* import strings remain" -- the frozen
        copy must actually import and execute correctly on its own, using
        only its own files.
        """
        repo_root = _synthetic_repo(tmp_path)
        closure = compute_closure("testing.strategies.widget", repo_root)
        destination = tmp_path / "snapshot"
        write_snapshot(closure, destination)

        config_module = load_snapshot_module(destination, "strategies.widget.config")

        assert config_module.use_it() == 43  # helper() == 42, alpha_fn() adds 1

    def test_manual_module_source_list_produces_same_closure_shape(self, tmp_path):
        # Sanity: ModuleSource is a plain, comparable dataclass -- useful
        # for constructing ad-hoc closures directly in a test without
        # going through compute_closure.
        module = ModuleSource(
            relative_dotted="engine.beta", absolute_dotted="testing.engine.beta", source_path=tmp_path / "beta.py"
        )
        assert module.relative_dotted == "engine.beta"
