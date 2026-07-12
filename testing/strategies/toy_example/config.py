from dataclasses import dataclass

from testing.engine.execution_config import ExecutionConfig


@dataclass(frozen=True)
class ToyConfig(ExecutionConfig):
    """Fixture-only config for a bare put credit spread. Not a real strategy
    -- this package exists solely to give the graduation mechanism and the
    "adding a new strategy" contract (see testing/CONTEXT.md) a working,
    end-to-end example to run against, with no relation to anything actually
    researched in this pipeline. Field values are arbitrary placeholders.
    """

    short_put_delta: float = -0.25
    long_put_delta: float = -0.10
