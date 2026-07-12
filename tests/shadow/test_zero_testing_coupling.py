"""Concretely verifies shadow/'s zero-live-coupling guarantee: no module
under shadow/ imports testing.engine or testing.strategies, anywhere,
in any import shape. Same style issue 09 used for
`testing.graduation.snapshot.verify_no_testing_imports` -- an AST walk of
every .py file, not just "I didn't write one" / a naive grep that a
multi-line or parenthesized import could dodge.

testing.graduation (manifest/loader) is the one explicitly-allowed
dependency (CONTEXT.md's Graduation entry: a strategy enters Shadow only
via a frozen snapshot, never a live import) -- this test asserts shadow/
imports at most that one testing.* subpackage, never engine or strategies.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHADOW_ROOT = REPO_ROOT / "shadow"

FORBIDDEN_PREFIXES = ("testing.engine", "testing.strategies")


def _shadow_py_files() -> list[Path]:
    return sorted(p for p in SHADOW_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(tree: ast.AST) -> list[str]:
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def test_shadow_has_python_files_to_check():
    # Sanity: guards against this test silently checking zero files if
    # shadow/ ever gets restructured.
    files = _shadow_py_files()
    assert len(files) >= 5
    names = {p.name for p in files}
    assert "ledger.py" in names
    assert "execution.py" in names
    assert "tastytrade.py" in names
    assert "ibkr.py" in names


def test_no_module_under_shadow_imports_testing_engine_or_strategies():
    violations = []
    for path in _shadow_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _imported_modules(tree):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES):
                violations.append((str(path.relative_to(REPO_ROOT)), module))
    assert violations == [], (
        "shadow/ must never live-import testing.engine or testing.strategies "
        f"(CONTEXT.md's Shadow entry, ADR 0001) -- found: {violations}"
    )


def test_no_module_under_shadow_imports_ingestion():
    """Shadow's only permitted upstream dependency is testing.graduation's
    frozen-snapshot format -- ingestion/ is a separate wing entirely and
    shadow has no legitimate reason to import it."""
    violations = []
    for path in _shadow_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _imported_modules(tree):
            if module == "ingestion" or module.startswith("ingestion."):
                violations.append((str(path.relative_to(REPO_ROOT)), module))
    assert violations == []


def test_only_testing_dependency_allowed_is_graduation():
    """Any testing.* import found under shadow/ must be testing.graduation
    (or a strict submodule of it) -- never a bare `testing` import and
    never anything else under testing/."""
    violations = []
    for path in _shadow_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _imported_modules(tree):
            if module == "testing" or module.startswith("testing.") and not module.startswith("testing.graduation"):
                violations.append((str(path.relative_to(REPO_ROOT)), module))
    assert violations == []
