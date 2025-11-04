# SACA-BusBooking: Implementation of "SACA: An End-to-End Method for Dispatching, Routing, and Pricing of Online Bus-Booking"

This repository contains the official code implementation of the paper **"SACA: An End-to-End Method for Dispatching, Routing, and Pricing of Online Bus-Booking"**. SACA solves the joint problem of **order dispatching**, **route planning**, and **trajectory pricing** for online bus-booking platforms in an end-to-end manner, aiming to maximize long-term platform revenue.

## 1. Project Introduction

Bus-booking is a novel urban transportation service that provides customized routes for passengers (e.g., airport midnight shuttles). The core challenge lies in the tight coupling of three tasks:



* **Order Dispatching**: Assign passenger orders to buses.

* **Route Planning**: Plan optimal routes for each bus (returning to the departure station).

* **Trajectory Pricing**: Dynamically set order prices to balance demand-supply and revenue.

Traditional multi-stage methods fail to capture interactions between these tasks. SACA addresses this by integrating:



* **Soft Actor-Critic (SAC)** for continuous-space pricing (maximizing long-term revenue).

* **Attention Mechanism** (Transformer-based) for efficient dispatching and route planning.

The method is proven effective on real-world ride-hailing data, outperforming baseline methods in revenue and route efficiency.

## 2. Method Overview

SACA consists of three core modules (aligned with Section 4 of the paper):

### 2.1 Candidate Order Set Generation



* Models demand-supply relationships using distance-aware functions:


  * **Demand Function**: $D_k^t(p_k^t) = N_{pk}^t \cdot F_{pk}(p)$ (passengers willing to pay price $p$).

  * **Supply Function**: $S_k^t(p_k^t) = N_s^t \cdot a_k^t \cdot F_{sk}(p)$ (available seats allocated to destination $k$).

* Determines accepted orders as $O_k^t = \min(D_k^t, S_k^t)$.
* In the provided SACA-ori.py simulation, we instantiate:
  * $w_k = \dfrac{\text{dist\_max} + \text{dist\_min} - \text{dist}_k}{\text{dist\_max}}$
  * Demand probability: $F_{pk}(p) = 1 - w_k \cdot p^2$, hence $D_k^t = N_{pk}^t \cdot F_{pk}(p)$
  * Supply factor: $F_{sk}(p) = w_k \cdot p^2$, hence $S_k^t = N_s^t \cdot a_k^t \cdot F_{sk}(p)$

### 2.2 Trajectory Pricing with SAC



* **State**: Encodes demand ($N_{pk}^t$, potential passengers per destination) and bus status ($\tau_i^t$, remaining return time).

* **Action**: Continuous price vector ($P^t$) and seat allocation vector ($A^t$, $\sum a_k^t = 1$).

* **Reward**: Combines immediate revenue ($R^t$) and order response rate (ORR) to ensure convergence:$Reward^t = R^t + \lambda \cdot ORR^t$ (λ adjusts ORR importance).

### 2.3 Order Dispatching & Route Planning with Attention



* **Encoder**: Embeds destination coordinates into hidden states.
* **Attention Layers**: Uses Multi-Head Attention (MHA) + Feed-Forward (FF) layers with Layer Normalization (LayerNorm).
* **Decoder/Router**: Assigns the closest orders within capacity, then plans a greedy route by ranking destinations using self-attention similarity; buses return to the departure station.

## 3. Dataset

We use the **Didi Chuxing GAIA Initiative** dataset (Chengdu city ride-hailing orders):



* **Raw Data**: 30 days of records (7,065,937 entries), including order ID, time, and coordinates.

* **Preprocessing**:

1. Filter missing values and outliers.

2. Retain only **nighttime orders** (to match bus-booking scenarios like airport shuttles).

3. Interpolate to expand sparse order data.

* **Station Selection**:


  * DBSCAN: Eliminate noise data.

  * PAM Clustering: Generate 30 destination stations (see Figure 3 in the paper).

> **Note**
>
> : Access the dataset via the 
>
> [Didi GAIA Initiative](https://outreach.didichuxing.com/)
>
> The provided SACA-ori.py uses a synthetic simulation environment:
> - Destination coordinates are sampled around a fixed departure station.
> - Potential demand per destination is generated from a Poisson process with mild temporal fluctuation.
> You may still follow the data processing pipeline to reproduce paper-level experiments with the real dataset; however, running SACA-ori.py does not require the dataset.

## 4. Environment Setup

### Hardware
- Apple Silicon (Metal Performance Shaders, MPS) supported by default
- CPU or NVIDIA GPU (CUDA) also supported
- 16GB+ RAM recommended

### Software
- Python 3.11
- PyTorch 2.9.0
- Required Libraries:
```
pip install torch
pip install numpy scipy
pip install matplotlib
```
- Optional (for dataset preprocessing & extended experiments):
```
pip install pandas scikit-learn seaborn tqdm
```

## 5. Experimental Results

Key results (reproduced from Table 1 in the paper) for 10/20 buses:



| Model       | Num Buses | Revenue (Rev) | Order Response Rate (ORR) | Total Distance (Length) |
| ----------- | --------- | ------------- | ------------------------- | ----------------------- |
| DPEA+SubBus | 10        | 342.21        | 0.13                      | 57.09                   |
| PODP+L2i    | 10        | 606.11        | 0.26                      | 113.13                  |
| **SACA**    | 10        | 675.78        | 0.13                      | 66.01                   |
| DPEA+SubBus | 20        | 948.98        | 0.33                      | 150.65                  |
| PODP+L2i    | 20        | 1410.91       | 0.53                      | 242.04                  |
| **SACA**    | 20        | 1639.95       | 0.34                      | 173.22                  |



* SACA achieves **higher revenue** than baselines by balancing demand-supply and route efficiency.

* SACA reduces total travel distance compared to PODP+L2i (avoids detours from improper orders).
> Note: The numbers in the table are from the paper. The default simulation in SACA-ori.py will not match them exactly without careful calibration of the environment and hyperparameters.

## 6. How to Run
- Quick start (simulation):
```
python SACA-ori.py
```
- The training log prints per-episode averages:
  - Avg Reward, Avg Revenue, Avg ORR, Avg Distance (km)
- Early stopping is enabled based on a moving-window criterion.

## 7. Configuration
- Edit the Config class in SACA-ori.py to change:
  - num_destinations, bus_capacity, bus_speed, beta_d
  - time_slots_per_episode, lambda_or (reward weighting)
  - SAC hyperparameters: gamma, tau, lr, hidden_dim, batch_size, etc.
- Device selection is automatic:
  - MPS > CUDA > CPU (depending on availability)

## 8. Citation

If you use this code or reference the paper, please cite:

```
@inproceedings{DBLP:conf/dasfaa/GaoSWGC23,
  author       = {Yucen Gao and
                  Yulong Song and
                  Xikai Wei and
                  Xiaofeng Gao and
                  Guihai Chen},
  title        = {{SACA:} An End-to-End Method for Dispatching, Routing, and Pricing
                  of Online Bus-Booking},
  booktitle    = {Database Systems for Advanced Applications (DASFAA)},
  volume       = {13946},
  pages        = {303--313},
  year         = {2023}
}
```

## 9. Acknowledgements

This work is supported by:



* National Key R\&D Program of China (2020YFB1707900)

* National Natural Science Foundation of China (62272302)

* Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102)

* DiDi GAIA Research Collaboration Plan (202204)

Thanks to Didi Chuxing for providing the GAIA dataset.