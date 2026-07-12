"""Graduation: the manual, one-way mechanism that copies a strategy's
validated code + parameters out of Testing as a frozen, versioned snapshot
(Testing -> Shadow, and later Shadow -> a Pipeline). See CONTEXT.md's
**Graduation** entry and ADR 0001 for why this must never be a live import
back to `testing/`'s ongoing code.

Public entry points:
- `testing.graduation.graduate.graduate` -- perform a graduation
- `testing.graduation.revalidate.rehydrate_config` -- reconstruct a
  graduated snapshot's own params for re-running through Testing's live
  engine on fresh data
- `testing.graduation.loader.load_snapshot` / `load_snapshot_module` --
  dynamically load a frozen snapshot's own code, standalone
- `testing.graduation.bar.check_graduation_bar` -- the Testing -> Shadow
  graduation bar itself (issue 10 / ADR 0003): checks MCPT + PBO
  quantitatively, requires manual attestation for named-source/sensitivity-
  analysis, and deliberately reports Sharpe/drawdown as undefined rather
  than a guessed floor
"""
