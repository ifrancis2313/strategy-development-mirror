"""Tests the manifest format in isolation: round-trip serialization and
that it actually carries what CONTEXT.md's Graduation definition requires
-- what was graduated, when, and with what validated parameters.
"""

import json

from testing.graduation.manifest import GraduationManifest


def _manifest(**overrides):
    defaults = dict(
        strategy_name="toy_example",
        graduation_id="v1_20260709T120000Z",
        version=1,
        source_wing="testing",
        destination_wing="shadow",
        graduated_at="2026-07-09T12:00:00+00:00",
        config_type="testing.strategies.toy_example.config.ToyConfig",
        params={"short_put_delta": -0.25, "long_put_delta": -0.10},
        source_modules=("engine.leg", "strategies.toy_example.config"),
        source_sha256="a" * 64,
    )
    defaults.update(overrides)
    return GraduationManifest(**defaults)


def test_manifest_records_what_when_and_params():
    manifest = _manifest()

    assert manifest.strategy_name == "toy_example"
    assert manifest.graduated_at == "2026-07-09T12:00:00+00:00"
    assert manifest.params["short_put_delta"] == -0.25
    assert manifest.version == 1


def test_to_dict_from_dict_round_trip_is_lossless():
    manifest = _manifest(notes="passed Sharpe/MCPT bar", source_commit="deadbeef")

    round_tripped = GraduationManifest.from_dict(manifest.to_dict())

    assert round_tripped == manifest


def test_to_dict_is_json_serializable():
    manifest = _manifest()

    serialized = json.dumps(manifest.to_dict())
    restored = GraduationManifest.from_dict(json.loads(serialized))

    assert restored == manifest


def test_source_modules_survives_json_as_tuple_not_list():
    manifest = _manifest()

    restored = GraduationManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

    assert isinstance(restored.source_modules, tuple)
    assert restored.source_modules == manifest.source_modules


def test_write_then_read_round_trip(tmp_path):
    manifest = _manifest(notes="disk round trip")

    written_path = manifest.write(tmp_path)
    assert written_path == tmp_path / "manifest.json"
    assert written_path.exists()

    read_back = GraduationManifest.read(tmp_path)
    assert read_back == manifest

    # read() also accepts the manifest file path directly, not just its directory
    read_back_by_file = GraduationManifest.read(written_path)
    assert read_back_by_file == manifest


def test_default_notes_and_commit_are_optional():
    manifest = _manifest()

    assert manifest.notes == ""
    assert manifest.source_commit is None
