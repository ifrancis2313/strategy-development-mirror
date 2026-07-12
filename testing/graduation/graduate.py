"""The one manual act that moves a strategy's validated code + parameters
out of Testing as a frozen, versioned, standalone snapshot. Never
triggered automatically -- always an explicit call (or CLI invocation),
matching `alpha-pipeline`'s "Promotion is deliberately manual" rule
(compounding-overfitting risk from auto-reselection).

What gets frozen is deliberately narrow: a strategy's config + signal +
selector modules (its actual decision logic), plus whatever `testing.engine`
modules those transitively import (see `testing.graduation.closure`) --
not the whole engine. Testing's backtest driver
(`testing.engine.backtest_runner.walk_forward`), position simulator,
sizing, and performance analyzer stay live and un-frozen: they're Testing's
own reusable infrastructure, used identically whether validating a
strategy for the first time or re-validating an already-graduated one's
recorded params later (see `testing/graduation/revalidate.py`). Freezing
them too would be pointless (they're never strategy-specific) and would
give re-validation no way to benefit from Testing engine bugfixes made
after a strategy graduated.

Usage as a script (the deliberate, manual action this is meant to be):

    python -m testing.graduation.graduate toy_example \\
        testing.strategies.toy_example.config.ToyConfig \\
        --params-json '{"long_put_delta": -0.05}' \\
        --notes "Passed Testing's Sharpe/MCPT bar"

Or from a script/notebook, which avoids JSON-quoting an arbitrary config:

    from testing.strategies.toy_example.config import ToyConfig
    from testing.graduation.graduate import graduate

    graduate("toy_example", ToyConfig(long_put_delta=-0.05), notes="...")
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from testing.engine.execution_config import ExecutionConfig
from testing.graduation.closure import compute_closure
from testing.graduation.manifest import GraduationManifest
from testing.graduation.snapshot import hash_snapshot, verify_no_testing_imports, write_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION_ROOT = REPO_ROOT / "graduated"


def _next_version(strategy_root: Path) -> int:
    if not strategy_root.exists():
        return 1
    versions = []
    for entry in strategy_root.iterdir():
        if not entry.is_dir():
            continue
        head = entry.name.split("_", 1)[0]  # "v3" out of "v3_20260709T153000Z"
        if head.startswith("v") and head[1:].isdigit():
            versions.append(int(head[1:]))
    return max(versions, default=0) + 1


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def graduate(
    strategy_name: str,
    config: ExecutionConfig,
    *,
    destination_wing: str = "shadow",
    destination_root: Path = DEFAULT_DESTINATION_ROOT,
    repo_root: Path = REPO_ROOT,
    notes: str = "",
) -> GraduationManifest:
    """Graduates `strategy_name` (must exist under `testing/strategies/`)
    with the already-validated `config`. Computes the dependency closure,
    writes a frozen, import-rewritten copy under
    `destination_root/strategy_name/v{n}_{timestamp}/`, verifies the
    zero-live-dependency guarantee, and writes the manifest alongside it.

    Returns the written `GraduationManifest`.
    """
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance (extends ExecutionConfig)")

    strategy_package = f"testing.strategies.{strategy_name}"
    closure = compute_closure(strategy_package, repo_root)

    strategy_root = destination_root / strategy_name
    version = _next_version(strategy_root)
    timestamp = datetime.now(UTC)
    graduation_id = f"v{version}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    destination = strategy_root / graduation_id

    written_modules = write_snapshot(closure, destination)
    verify_no_testing_imports(destination)
    source_sha256 = hash_snapshot(destination)

    config_cls = type(config)
    manifest = GraduationManifest(
        strategy_name=strategy_name,
        graduation_id=graduation_id,
        version=version,
        source_wing="testing",
        destination_wing=destination_wing,
        graduated_at=timestamp.isoformat(),
        config_type=f"{config_cls.__module__}.{config_cls.__qualname__}",
        params=asdict(config),
        source_modules=written_modules,
        source_sha256=source_sha256,
        source_commit=_git_commit(repo_root),
        notes=notes,
    )
    manifest.write(destination)
    return manifest


def _load_config_class(dotted: str) -> type[ExecutionConfig]:
    module_name, _, class_name = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"expected a dotted module.ClassName path, got {dotted!r}")
    module = importlib.import_module(module_name)
    config_cls: type[ExecutionConfig] = getattr(module, class_name)
    return config_cls


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Manually graduate a Testing-wing strategy as a frozen, versioned snapshot. "
            "Never run automatically -- this is the deliberate action Zi takes once a "
            "strategy has passed Testing's validation bar."
        ),
    )
    parser.add_argument("strategy_name", help="e.g. vrp_harvester (must exist under testing/strategies/)")
    parser.add_argument(
        "config_class",
        help="dotted path to the validated config dataclass, "
        "e.g. testing.strategies.vrp_harvester.config.VRPConfig",
    )
    parser.add_argument(
        "--params-json",
        default="{}",
        help="JSON object of field overrides on top of the config class's defaults",
    )
    parser.add_argument("--destination-wing", default="shadow", choices=["shadow", "pipeline"])
    parser.add_argument("--destination-root", default=str(DEFAULT_DESTINATION_ROOT))
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    config_cls = _load_config_class(args.config_class)
    overrides: dict[str, Any] = json.loads(args.params_json)
    config = config_cls(**overrides)

    manifest = graduate(
        args.strategy_name,
        config,
        destination_wing=args.destination_wing,
        destination_root=Path(args.destination_root),
        notes=args.notes,
    )
    destination = Path(args.destination_root) / manifest.strategy_name / manifest.graduation_id
    print(f"Graduated {manifest.strategy_name} as {manifest.graduation_id} -> {destination}")
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
