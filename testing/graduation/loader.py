"""Loads an already-written frozen graduation snapshot as a standalone
Python package -- without requiring it to be importable via any dotted
name or placed on `sys.path`. This is the mechanism Shadow (or a
re-validation script, or a test proving the zero-dependency guarantee)
uses to actually run a graduated snapshot's own code.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType

#: Memoizes `load_snapshot` per resolved directory. Without this, two
#: `load_snapshot_module` calls against the same snapshot directory (e.g.
#: `graduated_options_daily_check.py` loading config/trade_selector/
#: strike_selector separately) would each independently re-exec the package
#: under a fresh UUID name, producing two non-identical class objects for
#: what should be "the same" class -- harmless today only because no caller
#: does an `isinstance`/dataclass-identity check across the boundary, but a
#: real landmine for the next one that does. Found and fixed during the
#: front-to-back review (2026-07-12). Also bounds `sys.modules` growth to
#: the number of distinct snapshot directories actually loaded in a process
#: lifetime, rather than growing unboundedly per call.
_loaded_snapshots: dict[Path, ModuleType] = {}


def load_snapshot(snapshot_dir: Path) -> ModuleType:
    """Loads a frozen snapshot directory (one containing a top-level
    `__init__.py`, as `write_snapshot` produces) as its own uniquely-named
    package. Memoized per resolved `snapshot_dir` -- repeated calls against
    the same directory return the SAME loaded package (see
    `_loaded_snapshots`), so a class defined in the snapshot is the same
    object everywhere it's loaded from. Two different snapshot directories
    (e.g. two graduations of the same strategy) still never collide, since
    each gets its own cache entry and its own fresh `sys.modules` name.
    """
    resolved = snapshot_dir.resolve()
    if resolved in _loaded_snapshots:
        return _loaded_snapshots[resolved]

    package_name = f"_graduation_snapshot_{uuid.uuid4().hex}"
    init_file = resolved / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name, init_file, submodule_search_locations=[str(resolved)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load snapshot package at {snapshot_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    _loaded_snapshots[resolved] = module
    return module


def load_snapshot_module(snapshot_dir: Path, relative_dotted: str) -> ModuleType:
    """Loads one submodule (e.g. `'strategies.vrp_harvester.trade_selector'`)
    out of a frozen snapshot directory, resolving that module's own
    internal relative imports against the snapshot's package root -- see
    `load_snapshot` for why this doesn't need `sys.path` or a dotted
    top-level import.
    """
    root = load_snapshot(snapshot_dir)
    return importlib.import_module(f"{root.__name__}.{relative_dotted}")
