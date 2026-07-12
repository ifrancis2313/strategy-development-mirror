"""The manifest format: what was graduated, when, and with what validated
parameters -- enough to (a) audit later why a given Sleeve exists, and (b)
reconstruct the exact config object to re-run through Testing on fresh
data (see `testing/graduation/revalidate.py`).

One `manifest.json` is written alongside every graduation's frozen code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class GraduationManifest:
    """Immutable record of one graduation.

    `params` is the strategy config's own field values (via
    `dataclasses.asdict`) -- the validated parameters, not just their
    class name. `source_modules` is the exact set of frozen-snapshot-relative
    module paths written (see `testing.graduation.closure.compute_closure`),
    an audit trail of precisely what code this graduation contains.
    `source_sha256` is an integrity fingerprint over that same code (see
    `testing.graduation.snapshot.hash_snapshot`). `source_commit` ties the
    snapshot to the exact `strategy-development` repo state it was cut from,
    when available.
    """

    strategy_name: str
    graduation_id: str
    version: int
    source_wing: str
    destination_wing: str
    graduated_at: str
    config_type: str
    params: dict[str, Any]
    source_modules: tuple[str, ...]
    source_sha256: str
    source_commit: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "graduation_id": self.graduation_id,
            "version": self.version,
            "source_wing": self.source_wing,
            "destination_wing": self.destination_wing,
            "graduated_at": self.graduated_at,
            "config_type": self.config_type,
            "params": self.params,
            "source_modules": list(self.source_modules),
            "source_sha256": self.source_sha256,
            "source_commit": self.source_commit,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraduationManifest:
        return cls(
            strategy_name=data["strategy_name"],
            graduation_id=data["graduation_id"],
            version=data["version"],
            source_wing=data["source_wing"],
            destination_wing=data["destination_wing"],
            graduated_at=data["graduated_at"],
            config_type=data["config_type"],
            params=dict(data["params"]),
            source_modules=tuple(data["source_modules"]),
            source_sha256=data["source_sha256"],
            source_commit=data.get("source_commit"),
            notes=data.get("notes", ""),
        )

    def write(self, directory: Path) -> Path:
        path = directory / MANIFEST_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def read(cls, path_or_directory: Path) -> GraduationManifest:
        path = path_or_directory / MANIFEST_FILENAME if path_or_directory.is_dir() else path_or_directory
        return cls.from_dict(json.loads(path.read_text()))
