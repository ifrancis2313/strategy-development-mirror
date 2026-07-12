"""Shadow (Wing 3): live-but-not-real-money proving. See `CONTEXT.md`'s
Shadow glossary entry and `docs/adr/0001`/`docs/adr/0002`.

Structurally isolated from `testing/`/`ingestion/`: no module under
`shadow/` imports `testing.engine` or `testing.strategies` (verified by
`tests/shadow/test_zero_testing_coupling.py`). The only permitted
dependency on `testing/` is `testing.graduation` (the frozen-snapshot
manifest/loader format) -- a strategy enters Shadow only via Graduation,
never a live code import.

Own `requirements.txt`, own subdirectory, no live imports elsewhere in this
repo -- structurally compatible with the pull-before-run deploy mechanism
ADR 0002 describes (issue 13, not built here).
"""
