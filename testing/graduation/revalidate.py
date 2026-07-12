"""Re-validation: re-running a graduated snapshot's own recorded
parameters back through Testing on fresh data (CONTEXT.md's **Graduation**
entry). This deliberately uses the *live*, current `testing.engine` and
`testing.strategies.<name>` code -- re-validation is a check of "does this
strategy still work," which should benefit from any Testing engine
bugfixes made since the strategy graduated. Only the parameter *values*
come from the frozen manifest; the frozen snapshot's own copied code is
not involved in re-validation (it exists so Shadow can run standalone,
never for this).
"""

from __future__ import annotations

from testing.engine.execution_config import ExecutionConfig
from testing.graduation.manifest import GraduationManifest


def rehydrate_config(manifest: GraduationManifest, config_cls: type[ExecutionConfig]) -> ExecutionConfig:
    """Reconstructs the exact config a graduation validated, from its
    manifest's recorded `params`. The supported re-validation path: build
    a fresh selector bound to the result, and pass both into
    `testing.engine.backtest_runner.walk_forward` against fresh data --
    exactly what the original graduation's own validation run did, driven
    only by the frozen params.

    `config_cls` should be the strategy's own live config class (e.g. the
    current `testing.strategies.vrp_harvester.config.VRPConfig`), not
    anything loaded from the frozen snapshot.
    """
    return config_cls(**manifest.params)
