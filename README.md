# A Spatio-Temporal Adaptive Framework for Dispatching, Routing, and Pricing of Bus-Booking

This repository contains the code for **ST-SACA**, a spatio-temporal adaptive extension of SACA for the joint order dispatching, route planning, and trajectory pricing problem in bus-booking systems.

The paper studies two settings:

- **JDRP**: joint dispatching, routing, and pricing under a stationary environment.
- **R-JDRP**: the robust variant under non-stationary demand uncertainty.

ST-SACA keeps the end-to-end SAC plus attention-routing design of SACA, and adds a **Spatio-Temporal Adaptive Actor** that combines station coordinates and real-time demand with self-attention. The goal is to make pricing and dispatching robust to spatial heterogeneity and temporal demand shifts.

## Repository Layout

```text
.
├── src/st_saca/
│   ├── agents/          # ST-SACA and SACA implementations
│   ├── baselines/       # GRC-ELG and JDRL-POMO baselines
│   ├── routing/         # Attention routing model and POMO training code
│   ├── experiments/     # Training, robustness, ablation, sensitivity, speed
│   ├── analysis/        # Plotting, route visualization, case study
│   └── preprocessing/   # DBSCAN/PAM station generation utilities
├── data/stations/       # Included 30 Chengdu destination station medoids
├── checkpoints/routing/ # Local routing checkpoints, ignored by git
├── outputs/             # Local logs and figures, ignored by git
└── SACA.py, train.py    # Thin compatibility wrappers
```

The DiDi Chengdu GAIA raw order dataset is **not bundled**. The repository includes only the 30 station medoids used by the simulator.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional dependencies:

```bash
pip install wandb scikit-learn-extra jupyter
```

The routing module expects local files:

```text
checkpoints/routing/best_model.pth
checkpoints/routing/normalization_stats.pt
```

These files are ignored by git. You can place them there, or override paths with:

```bash
set ST_SACA_ROUTING_CKPT_DIR=C:\path\to\routing_checkpoints
set ST_SACA_OUTPUT_DIR=C:\path\to\outputs
set ST_SACA_DATA_DIR=C:\path\to\data
```

## Quick Start

Train ST-SACA:

```bash
python -m st_saca.experiments.train --method st-saca
```

Run the four-method non-stationary comparison:

```bash
python -m st_saca.experiments.train --method all --uncertainty-scale 5 --uncertainty-frequency 1
```

Run ablations:

```bash
python -m st_saca.experiments.ablation --variant wo-route
python -m st_saca.experiments.ablation --variant wo-orr
python -m st_saca.experiments.ablation --variant wo-spatial
```

Plot comparison curves:

```bash
python -m st_saca.analysis.plot_comparison
```

Benchmark inference speed:

```bash
python -m st_saca.experiments.speed
```

Legacy commands such as `python SACA.py`, `python SACA_baseline.py`, `python train.py`, and `python plot.py` are preserved as wrappers.

## Paper Results

Stationary setting:

| Bus Count | Method | Profit | ORR | Length |
| --- | --- | ---: | ---: | ---: |
| 10 | DPEA+SubBus | 342.2 | 0.13 | 57.1 |
| 10 | PODP+L2i | 606.1 | 0.26 | 113.1 |
| 10 | SACA | 23.25 | 0.10 | 12.17 |
| 10 | ST-SACA | 38.00 | 0.18 | 17.73 |
| 20 | DPEA+SubBus | 948.9 | 0.33 | 150.6 |
| 20 | PODP+L2i | 1410.9 | 0.53 | 242.0 |
| 20 | SACA | 53.97 | 0.21 | 15.85 |
| 20 | ST-SACA | 73.35 | 0.28 | 18.51 |

Non-stationary setting with `A=5`, `omega=1`:

| Method | Reward | Profit | Cost | ORR |
| --- | ---: | ---: | ---: | ---: |
| SACA | 21.70 | 21.33 | 12.11 | 0.09 |
| GRC-ELG | 33.34 | 32.68 | 19.21 | 0.16 |
| JDRL-POMO | 24.86 | 34.14 | 9.65 | 0.09 |
| ST-SACA | 38.70 | 38.00 | 17.73 | 0.18 |

Ablation:

| Variant | Reward | Profit | Cost | ORR | Distance (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ST-SACA (Full) | 38.70 | 38.00 | 17.73 | 0.18 | 118.23 |
| w/o Spatial Attention | 20.49 | 20.12 | 12.03 | 0.09 | 80.22 |
| w/o Learned Route | 26.82 | 26.62 | 9.71 | 0.09 | 64.73 |
| w/o ORR Reward | 25.98 | 25.98 | 9.91 | 0.09 | 66.05 |

Average inference time per time slot:

| Order Batch Size | 50 | 100 | 200 | 500 |
| --- | ---: | ---: | ---: | ---: |
| Inference Time (ms) | 12 | 18 | 26 | 45 |

## Citation

```bibtex
@misc{gao_st_saca,
  title  = {A Spatio-Temporal Adaptive Framework for Dispatching, Routing, and Pricing of Bus-Booking},
  author = {Gao, Yucen and Du, Pengfei and Gao, Xiaofeng and Wang, Bin and Yang, Xiaochun and Chen, Guihai},
  note   = {Manuscript}
}
```

## License

No license file is currently declared for this repository. Please add a license before public redistribution or reuse.
