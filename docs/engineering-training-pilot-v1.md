# Engineering training pilot v1

This is newly generated engineering evidence, not recovered historical assets or
a corrected learned baseline. No claim about method superiority, real traffic,
fare, full route-cardinality coverage or paper-result reproduction is permitted.

## Frozen stages

1. **Labels:** use only the existing30-station file, SHA256
   `8b6135f9ab112953b4f8731463c3d5dc1a22bfea1085b48849ff3c96538bd965`.
   Depot104.06,30.67; capacity30; N5 and6;32 train+8 validation perN; generation
   seed20261001. Existing passenger allocation algorithm, explicitly30 rather
   than generator's default50. Zero-weight destinations are allowed and retained.
   Canonical station-ID/passenger mappings must be globally unique, including
   across splits. Permuting an instance does not create a new identity.
2. **Label target:** sum of passenger-weighted cumulative haversine kilometres,
   excluding a return-to-depot segment. Solver local labels are0..N-1. Independently
   enumerate every permutation for each N≤6 instance, recompute cost and compare
   objective values (ties may have different valid routes). This is passenger-km;
   dividing by30 gives weighted average passenger distance in km, not minutes.
   Fleet operating cost continues to use closed-tour kilometres separately.
3. **Router:** AM64/8/3, initialization seed0;20 total Adam updates, lr1e-4,
   weight_decay1e-4, batch8. Alternate N5/6 and sample within the frozen training
   group using seed5000. No scheduler, early stop, tuning or additional permutation
   augmentation. Normalization follows existing compute_normalization_stats on
   the64 training instances only, including each training instance's depot node;
   validation instances never enter statistics or gradients. Save the final step,
   not a checkpoint selected on validation. Validation is a diagnostic of mean-token
   NLL grouped byN and route validity, not a quality pass/fail criterion.
4. **SAC:** seeds0/1, ST-SACA and SACA, same trained pilot routing assets,
   runtime30 stations/10 buses/capacity30, A0,8 natural transitions, replay size8,
   batch4. Explicit select_action(deterministic=False,greedy_samples=0). Exactly
   one SAC.update per method/seed: three actual Adam steps (actor,critic,alpha),
   plus one target soft update. Four SAC updates across the batch therefore mean
   twelve optimizer steps; with AM20, the complete pilot has32 optimizer steps.

## Required evidence

Finite losses, parameters, nonzero finite gradients; actual optimizer state.step
equals20 for every AM parameter and1 for each SAC optimizer parameter; actor and
critic changed; alpha changed and stays finite/positive; target matches its tau
formula; route state unchanged with no gradients. Save/load includes actor,
critic,target,log_alpha and all three optimizer states/moments/parameter groups.
Deep-copy the checkpoint snapshot, compare recursively after weights_only=True
CPU loading, then compare same-mode actor forward on one fixed state.

This is not a full replay/RNG/config restart test. Deterministic actor.forward
and stochastic sample use different transformations in existing code; never
compare those paths as a checkpoint round-trip assertion or silently change them.

## Assets and isolation

All outputs go into a new immutable per-job directory. Distinct names:
pilot_labels.json, pilot_router_state.pt, pilot_router_stats.pt,
pilot_router_training_state.pt, pilot_sac_<method>_seed<seed>.pt and
pilot_manifest.json. Every manifest declares engineering_pilot_only and
scientific_result_verified=false. Input manifests and artifact hashes are
verified before/after consumers; source HEAD must agree across stages.

The test-scoped constructor adapter maps only the two exact expected production
asset lookups to these hash-bound pilot files. It never changes production
constructors or installs a fallback. Runtime assets are shared identically by all
four method/seed pairs, but cannot be promoted as a validated production baseline.

No raw orders, raw-field interpretation, package installation on4090, GPU use,
historical output replacement, or oldP1 reopening. Train/validation losses and
small fixture metrics are not reported as algorithm rankings.

## Resource contract

Four known Control jobs in one batch, dependency chain labels→router→two SAC
seeds. CPU2 each; walltime180/300/180/180 seconds; hard8GiB address space each;
no automatic retry. Existing conda3.11.11 only. Checkpoint file≤32MiB;
per-stage artifacts≤128MiB and whole planned batch≤256MiB; compact summary≤8KiB.
Control must verify total output budget before declaring the batch complete.

Hosted CPU CI uses the same top-level dependency versions as the prior fixture,
with torch2.6.0+cpu instead of the existing4090 +cu124 build. It runs one complete
bounded pipeline in temporary directories. Its success is implementation evidence,
not the4090 terminal receipt. No new environment or dependency installation on4090.

## Remaining scientific design

N5/6 labels are intentionally tiny and do not cover runtime dispatch routes up
to30 distinct destinations. Before a meaningful learned baseline, freeze a route
distribution/quality test, policy training/evaluation seeds and budget, common
evaluation action path, and fair input/parameter-matched attention controls.
The unresolved raw schema does not prevent this synthetic engineering pilot, but
still prevents claims of reconstructed real-world Chengdu demand or actual fare.
