# Synthetic fixture integration v1 — review draft

This test checks software wiring and accounting only. A success is not a learned
policy evaluation, corrected scientific baseline, training result, or evidence
that production routing assets exist. No production files are changed.

## Source and inputs

- Research base: `GaoYucen/ST-SACA@0b73d3b0f5677e855aa03cbf02b900b4087ad0f8`.
- Add `tests/fixture_integration.py`, dependency-free CLI/contract tests,
  `.github/workflows/fixture-integration.yml`, and this contract. No production edits.
- Input geometry is 30 deterministic synthetic points near the configured depot.
- Two exact constructor asset lookups are intercepted in test scope. The real
  constructor creates AM and loads an in-memory randomly initialized state dict;
  fixed finite means and positive standard deviations are synthetic fixtures.
- No checkpoint file is opened or deserialized. Every other torch load/save
  fails. The original production constructor and guards remain unchanged.
- ST-SACA SpatialActor and SACA Actor execute real forward/sample operations.
  Final price/allocation heads are initialized to zero weights/biases to avoid
  saturation hiding dispatch coverage. Earlier layers and AM remain random.
- Real dispatch and AM routing, environment step, fleet clock, and reward code
  execute. Only station input and checkpoint inputs are replaced. No numerical
  dependency is mocked and no AST extraction is used.

## Cases and assertions

For each seed 0 and 1, each method executes exactly four injected fixture slots:
mixed idle/busy service, all busy, zero demand, terminal service. Non-contiguous
bus IDs exercise identity. Demand and fleet are deliberately injected before
each slot; these are not uninterrupted natural episodes. Existing Poisson demand
advance executes with A=0 and a reseed after the constructor's hard-coded seed.

Assertions cover finite actor output and probabilities; action bounds; AM
one-through-N route permutation; actual AM calls; idle-only assignment; passenger
coverage; capacity; served counts not exceeding proposals; independent scalar
haversine closed-tour distance; gross collected prices minus distance cost;
ORR using effective demand plus epsilon; reward; all old/new trip clock advances;
stable bus IDs/capacities; four-step done boundary and rejected fifth step;
unchanged model state hashes and absent gradients. Adam steps, tensor backward,
checkpoint save and unexpected checkpoint load are blocked in test scope.

## Proposed execution contract — not yet approved or dispatched

- One batch with exactly two seed jobs, no retries or replacement IDs.
- CPU only, two threads, max 180 seconds per seed job. No GPU allocation.
- Candidate interpreter: existing `/opt/conda/bin/python` 3.11.11 on 4090.
- Observed metadata versions: torch 2.6.0+cu124, NumPy 1.26.4, SciPy 1.16.3,
  Matplotlib 3.10.7, pandas 2.3.3, tqdm 4.67.1, scikit-learn 1.7.2.
  Metadata discovery has passed; actual import compatibility remains untested.
- No installation or environment mutation. Set CUDA_VISIBLE_DEVICES empty,
  OMP/MKL/OpenBLAS threads 2, WANDB_MODE disabled, MPLBACKEND Agg,
  PYTHONDONTWRITEBYTECODE 1. Set MPLCONFIGDIR to a job-local empty directory.
- Proposed address-space cap 8 GiB needs contract review; it is not inherited
  from the raw audit's 512 MiB limit. Record peak RSS and elapsed time.
- Run from an immutable reviewed checkout. Freeze final source commit, harness
  SHA256, contract SHA256, and exact dependency inventory in the request.
- Emit one bounded JSON result per seed, capped 32 KiB. Preserve stderr/failure
  stage and exit code in the receipt. Never interpret a timeout as a pass.
- Actual command after approval: `/opt/conda/bin/python -B tests/fixture_integration.py --seed 0`
  and the equivalent seed 1 command, in separate bounded seed jobs.
- No raw order data, preprocessing, labels, optimizer update, training entry
  point, model checkpoint creation, production ledger activation, or old P1 gate
  change. Critic/backward integration is outside this protocol.

## Review and validation status

This draft is prepared locally for review. Publication, CI execution and 4090
execution have not occurred. Seven dependency-free CLI/static checks passed.
Syntax/static validation is separate from actual PyTorch integration.
The draft hosted CI uses Python 3.11.11 and torch 2.6.0 CPU from the official
PyTorch registry, with the seven metadata-matched package versions pinned from
PyPI. Its two seed jobs each cap the fixture command at 180 seconds; each total
job is capped at 12 minutes including dependency installation. It requests no
self-hosted runner and has read-only repository permissions. This hosted CPU
build differs from the 4090 CUDA-enabled torch build; CI success cannot replace
the later reviewed 4090 compatibility check. It does not impose the proposed
8 GiB 4090 address-space cap on hosted runners.
