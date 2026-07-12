"""Static dependency-closure computation for a graduation.

A strategy package (e.g. `testing.strategies.vrp_harvester`) only needs a
*subset* of `testing/engine/` to run its own decision logic standalone --
whatever its config/signal/selector modules actually import, transitively.
This module computes exactly that subset via AST analysis, so a graduated
snapshot never has to carry the whole engine (most of which -- the
backtest driver, position simulator, performance analyzer -- is Testing's
own reusable infrastructure, not part of what gets frozen; see
`testing/graduation/graduate.py` module docstring for why).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

TESTING_ROOT_PACKAGE = "testing"


class UnsupportedImportError(RuntimeError):
    """Raised when a module's imports can't be safely resolved into the
    frozen-snapshot's dependency closure: a bare `import testing.x.y`
    statement, or a relative import reaching into `testing/`. Neither form
    is used anywhere in `testing/` today (every internal import is an
    absolute `from testing.a.b import X`) -- flagged loudly here rather
    than silently mishandled, since a missed dependency would silently
    produce a broken frozen snapshot.
    """


@dataclass(frozen=True)
class ModuleSource:
    """One module in a graduation's dependency closure.

    `relative_dotted` is the module's dotted path with the leading
    `testing.` package segment stripped -- this becomes both its location
    inside the frozen snapshot and the identity used to compute relative
    imports (see `snapshot.relative_import`), so the snapshot's own code
    never has to know it was produced from a package literally named
    `testing`.
    """

    relative_dotted: str
    absolute_dotted: str
    source_path: Path


def _module_to_path(dotted: str, repo_root: Path) -> Path:
    return (repo_root / Path(*dotted.split("."))).with_suffix(".py")


def _package_files(package_dotted: str, repo_root: Path) -> list[str]:
    """Every non-`__init__.py` module inside a package directory (e.g.
    `testing.strategies.vrp_harvester`), as dotted module paths. Recurses
    into sub-packages so a strategy organized into subdirectories later
    doesn't need this to change.
    """
    package_dir = repo_root / Path(*package_dotted.split("."))
    modules = []
    for path in sorted(package_dir.rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_root).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def _testing_imports(source: str, module_dotted: str) -> list[str]:
    """Static-analysis pass over one module's `from testing... import ...`
    statements. Returns the absolute dotted module path of each one.

    Deliberately narrow: only handles the exact import shape used
    everywhere in `testing/` today -- see `UnsupportedImportError`.
    """
    tree = ast.parse(source, filename=module_dotted)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == TESTING_ROOT_PACKAGE:
                raise UnsupportedImportError(
                    f"{module_dotted}: bare 'from testing import ...' is unsupported -- "
                    "import a specific submodule instead"
                )
            if node.module and node.module.startswith(TESTING_ROOT_PACKAGE + "."):
                if node.level != 0:
                    raise UnsupportedImportError(
                        f"{module_dotted}: relative import reaching {node.module!r} is unsupported"
                    )
                found.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TESTING_ROOT_PACKAGE or alias.name.startswith(TESTING_ROOT_PACKAGE + "."):
                    raise UnsupportedImportError(
                        f"{module_dotted}: bare 'import {alias.name}' is unsupported -- "
                        "use 'from ... import ...'"
                    )
    return found


def compute_closure(strategy_package: str, repo_root: Path) -> dict[str, ModuleSource]:
    """The full set of `testing/` modules a strategy package needs to run
    its decision logic (config + signal + selector) standalone -- the
    strategy's own modules, plus the transitive closure of whatever
    `testing.engine` (or other `testing.*`) modules they import.

    `strategy_package` is a dotted path, e.g. `testing.strategies.vrp_harvester`.
    Returns a mapping of frozen-snapshot-relative dotted path -> `ModuleSource`.
    """
    if not strategy_package.startswith(TESTING_ROOT_PACKAGE + "."):
        raise ValueError(f"strategy_package must be under 'testing.', got {strategy_package!r}")

    discovered: dict[str, ModuleSource] = {}
    queue: list[str] = list(_package_files(strategy_package, repo_root))
    seen: set[str] = set()

    while queue:
        dotted = queue.pop()
        if dotted in seen:
            continue
        seen.add(dotted)

        source_path = _module_to_path(dotted, repo_root)
        source = source_path.read_text()
        relative_dotted = dotted[len(TESTING_ROOT_PACKAGE) + 1 :]
        discovered[relative_dotted] = ModuleSource(
            relative_dotted=relative_dotted,
            absolute_dotted=dotted,
            source_path=source_path,
        )

        for dep in _testing_imports(source, dotted):
            if dep not in seen:
                queue.append(dep)

    return discovered
