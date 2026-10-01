# Environment consistency and fail-fast audit repair

This change starts a new engineering baseline from submission commit
`dfb698a62c6b2a9329d7b190fa6173a2841120b1`. It does not reproduce, validate,
or overwrite the submitted paper's numerical results.

## Slot boundary and accounting contract

- A state represents the start of slot t. Only buses with zero remaining trip
  time at that boundary contribute seats and can depart.
- The existing demand, supply-price, allocation normalization, rounding,
  demand-phase expression, ORR denominator, distance and learning objectives
  are preserved.
- Dispatch proposals are distinct from actual service. Only validated
  passengers in bus_orders contribute revenue, ORR and orders_accepted.
  orders_proposed records the pre-dispatch candidate count.
- Routes retain their supplied bus IDs. Newly departed and already travelling
  buses both advance by one slot duration. A trip shorter than one slot is
  available again at the next slot boundary, never re-dispatched within a slot.
- This explicitly corrects the former extra-slot delay for newly dispatched
  trips. Results from the corrected dynamics are a new baseline.
- Zero-order slots follow the same demand update, clock and terminal path.
  Stepping an already completed episode is rejected until reset.
- Invalid assignments (busy/unknown bus, capacity, destination coverage, or
  overserving) fail before mutating the fleet clock.
- JDRL retains its own bus-profit objective and return signature. Its bus
  profits and the common global net profit now account for the same service.
- The GRC/JDRL fleet-size configuration defaults remain 10, but reset now
  respects the explicit num_buses value consistently with ST-SACA and SACA.

## Deliberately blocked experiment configurations

The current GRC-ELG and JDRL-POMO implementations are not accepted as verified
baselines: GRC's routing weights are untrained and replaced on evaluation;
JDRL loads AM assets while POMO input, normalization and augmentation contracts
remain inconsistent. Training, environment construction and routing construction
fail with actionable errors. No provenance flag or filename change bypasses
these errors. A separately reviewed implementation must resolve the blockers.

The wo-route and wo-ORR default configuration drift is diagnosed, not silently
rewritten. Their configurations must match the full control except for the
declared intervention; wo-ORR specifically requires lambda_or=0. This check
does not certify the scientific validity of the model or ablation implementation.
The actual wo-ORR defaults also lack the full control's num_buses field.
Even a caller-supplied aligned wo-ORR configuration remains blocked: that
ablation owns a separate legacy environment not covered by these four repairs.
It requires reviewed integration before enabling.

The current entrypoint wrappers compare against a fresh full-model default
Config. They are intentionally not a general paired non-default protocol
validator: alternate matched episode/demand/seed settings require an explicit
reviewed reference contract rather than silently bypassing the default guard.

No checkpoint is generated or substituted, no objective is replaced, and no
manuscript source is changed. Normal legacy module imports still require their
framework dependencies; the guards prevent experiment execution, not all
module-level device probing.

## Synthetic verification

Run from the repository root:

`PYTHONPATH=src python3 -m unittest discover -s tests -v`

The suite uses only Python's standard library. It extracts the actual reset,
demand, supply, step, reward and vehicle-update methods from all four source
ASTs, bypassing constructors that load models. Small deterministic vector/RNG
stubs and dispatch fixtures exercise the same cases for each method: busy
capacity, actual service, zero orders, time/done, bus IDs, trip completion,
invalid assignments and configurable fleet size.

These tests establish control-flow/accounting invariants. They are not a full
NumPy/PyTorch integration test, numerical policy evaluation, training smoke
test, timing benchmark, or validation of paper results. Actual model-dependent
integration remains blocked by missing/invalid scientific asset bindings.

The existing experiments/speed.py end-to-end benchmark repeatedly calls step
without episode resets (30 warmup plus 200 measured calls). It is not compatible
with the explicit terminal boundary and remains unsupported until its reset/
episode measurement protocol is reviewed. No timing claim is made by this PR.

Configuration tests check preserved bad defaults fail, matching controls pass,
both baseline blockers aggregate before a partial all-method suite, and guard
call sites precede experiment side effects.

The approved execution target is 4090 via existing Control. Tests must not use
home-mini, real order data, GPUs, external tracking services or training entrypoints.
