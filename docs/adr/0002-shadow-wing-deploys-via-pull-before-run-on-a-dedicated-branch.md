# Shadow wing deploys via pull-before-run on a dedicated branch, not manual SSH or a webhook

## Status

Accepted

## Context

Shadow (Wing 3) runs unattended on the same droplet already hosting `Options_Trading_Strategy` and Options and Futures Pipeline (`vrp-scheduler`, 1GB RAM). Zi needs new graduated strategies to reach the droplet without manually SSHing in each time — the existing `Options_Trading_Strategy`/OFP convention (`git clone` + manual `git pull`, per that repo's ADR 0009) doesn't scale to Shadow's expected graduation frequency.

Three options considered:

- **Manual SSH + `git pull`** (today's convention) — maximum control, zero new infra, but exactly the recurring manual chore Zi wants removed.
- **Webhook-triggered deploy** (GitHub webhook → an always-on listener on the droplet pulls immediately on push) — near-instant deploy, but requires a persistent, internet-reachable service and a public HTTPS endpoint on a box that also holds live broker credentials. Buys deploy speed nothing in this system currently needs (every strategy trades daily-or-slower); the added attack surface and a second thing-that-can-crash-and-needs-monitoring aren't justified by that.
- **Pull-before-run on a dedicated branch** — chosen.

## Decision

- A `wing3-deploy` branch receives a merge only when Zi has deliberately decided a strategy is ready (Graduation stays a manual human act — only the *propagation* of an already-decided graduation is automated, mirroring `alpha-pipeline`'s own "Promotion is deliberately manual" rule).
- The droplet's cron wrapper runs `git fetch && git reset --hard origin/wing3-deploy` as its first step before each scheduled Shadow run. Pushing a merge to that branch is the entire deploy action — no SSH required for routine updates.
- Deploy latency equals whatever cron cadence the trading job itself runs on — not a fixed constant. A future higher-frequency strategy needs its own faster cron entry regardless of deploy mechanism (you can't trade 5-minute bars off a once-daily loop either way); once that entry exists, its pull-before-run step inherits the same tighter cadence automatically. Mirrors OFP's existing pattern of a once-daily job and a 30-minute intraday job coexisting on one box.
- Shadow's ledgers/state live in a separate private state repo, decoupled from the code repo, exactly as `Options_Trading_Strategy` ADR 0009 already established — `git reset --hard` on the code repo must never risk live position state.
- SSH access remains available as the escape hatch for same-tick hotfixes and diagnosing a bad deploy (the dead-man's-switch, inherited from ADR 0009, flags that a run failed but not why).

## Consequences

- A bad merge to `wing3-deploy` breaks the next scheduled run automatically, with no CI gate in between — the safety net is the healthchecks.io dead-man's-switch noticing a missed run, not a pre-merge check. Worth a lightweight local smoke-test discipline before merging, though not mandatory.
- Rollback is a `git revert` on `wing3-deploy`, same latency as a forward deploy (up to one cron cycle) unless done via emergency manual SSH.
- Resource contention on the 1GB droplet gets more pressing, not less, as any future strategy's required cadence tightens — a capacity check is needed before adding any sub-hourly job, separate from this decision.
- This governs Shadow's deploy only. The destination pipeline (e.g. `alpha-pipeline`) makes its own, independent deploy-cadence decision once a strategy actually graduates into it — this ADR doesn't constrain that.
- Reversible if requirements change: a webhook listener could be added later for one specific high-frequency need while B stays the default elsewhere. Not a one-way door, unlike ADR 0001's repo split.
