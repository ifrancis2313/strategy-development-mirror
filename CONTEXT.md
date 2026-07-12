# Strategy Development

The single place a trading idea is born, tested, and proven before it's trusted with real capital. Adds a live-but-not-real-money proving stage on top of historical backtesting. Contrasts with a downstream **Pipeline**, which holds only strategies that have already graduated — this repo is where things are allowed to be wrong.

## Language

**Wing**:
One of the three sequential stages a strategy passes through inside this repo: **Ingestion**, **Testing**, **Shadow**. Each wing has a different trust level and a different data regime (historical vs. live).

**Ingestion** (Wing 1):
The wing responsible for pulling and storing market data (scraping + Supabase), serving both this repo's own Testing/Shadow wings and any downstream pipeline that needs a shared data source.

**Testing** (Wing 2):
The wing responsible for historical backtesting and statistical validation of a strategy — Sharpe, drawdown, win-rate, Greek exposures, Monte Carlo/MCPT null-distribution testing, multi-strategy comparison. Strategy-agnostic: applies equally to options strategies, equity factor models, and prediction-market strategies, though each asset class needs its own metrics module (Greeks don't apply to a factor model; calibration/Brier score don't apply to an options spread). See `testing/CONTEXT.md` for that code's own architecture map and the "adding a new strategy" extension contract — this file stays at the repo-glossary level, not the wing's internal implementation.
_Avoid_: Backtesting (too narrow — this wing also covers Monte Carlo robustness testing and cross-strategy comparison, not just a single equity-curve run)

**Shadow** (Wing 3):
The wing responsible for live-but-not-real-money proving: running a graduated-from-Testing strategy against real-time production data with simulated fills (a **Shadow ledger**), building a live track record before a strategy is trusted with a Pipeline's real capital. Runs unattended, never on a laptop. Deliberately isolated from Wings 1/2's code — a strategy enters this wing only via **Graduation**, never a live import, so ongoing research churn in Wings 1/2 can't destabilize a process touching live broker connections.

**Graduation**:
The one-way, manual act of copying a strategy's validated code as a frozen snapshot from one wing to the next (Testing → Shadow) or out of this repo entirely (Shadow → a Pipeline, as a new **Sleeve**). A graduated copy is never live-coupled back to its source — the source can keep changing without affecting anything already graduated. Re-validation later means re-running the graduated snapshot's own params back through Testing on fresh data, not maintaining a code link.
_Avoid_: Promotion (a distinct term used elsewhere for a narrower, single-repo concept — moving a candidate into a registry within one sleeve. Graduation is the cross-repo/cross-wing act.)

**Confirmation** (ADR 0007):
The one-shot, no-tuning evaluation of an already-frozen strategy snapshot against a slice of data no Testing-wing experiment has ever touched — required alongside the graduation bar's statistical criteria before Graduation (Testing → Shadow) may proceed. Everything tested against this repo's fixed historical research sample is exploratory, never confirmatory (ADR 0006), no matter how many statistical screens it clears — Confirmation is the first genuinely confirmatory check a candidate faces, and Shadow's own gate (ADR 0004) is the second and stronger one. A Confirmation failure retires the candidate outright; the next reserve slice goes only to an independently-conceived new candidate, never a revised resubmission.
_Avoid_: conflating this with `testing/validation/walk_forward_validation.py`'s walk-forward (a rolling-refit tuning methodology with a train window) — Confirmation has no train window at all, since the candidate's parameters are already frozen by the time it reaches this gate.

**Pipeline**:
A repo holding only strategies that have already graduated out of this repo's Shadow wing — the only place real capital is ever at risk. This repo never places a real-money order.

**Sleeve**:
An independently-managed sub-strategy within a Pipeline. Referenced here only because Graduation's terminal step is a strategy becoming a new Sleeve in a Pipeline.

## Relationships

- A strategy moves through exactly one path: **Ingestion** supplies data to → **Testing** validates historically → **Confirmation** checks it once against untouched data → **Graduation** copies it into → **Shadow** proves it live → **Graduation** copies it into → a **Pipeline** as a new **Sleeve**
- **Ingestion** also serves data directly to any **Pipeline** that needs it, independent of whether a specific strategy is mid-graduation
- Each **Graduation** is a frozen, one-directional copy — never a live dependency in either direction
- **Shadow** never live-imports **Testing** or **Ingestion**'s code; it only receives what's been explicitly graduated to it
- **Confirmation** is a gate, not a wing — it doesn't move code anywhere, it only decides whether Graduation (Testing → Shadow) is allowed to happen

## Example dialogue

> **Dev:** "I tweaked a strategy's exit logic in Testing — does that change what's running in Shadow right now?"
> **Domain expert:** "No. Shadow is running whatever was graduated as of its last Graduation — your Testing changes only affect Shadow once you graduate the new version deliberately."

---

**About this mirror:** this is a curated, hand-picked copy of the research/testing infrastructure from a private working repo — not the repo itself. Strategy implementations, backtest results, and market data are deliberately excluded; see the root README for what's in scope and why.
