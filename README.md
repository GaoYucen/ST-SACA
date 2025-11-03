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

### 2.2 Trajectory Pricing with SAC



* **State**: Encodes demand ($N_{pk}^t$, potential passengers per destination) and bus status ($\tau_i^t$, remaining return time).

* **Action**: Continuous price vector ($P^t$) and seat allocation vector ($A^t$, $\sum a_k^t = 1$).

* **Reward**: Combines immediate revenue ($R^t$) and order response rate (ORR) to ensure convergence:$Reward^t = R^t + \lambda \cdot ORR^t$ (λ adjusts ORR importance).

### 2.3 Order Dispatching & Route Planning with Attention



* **Encoder**: Embeds destination coordinates into hidden states.

* **Attention Layers**: Uses Multi-Head Attention (MHA) + Feed-Forward (FF) layers, with Batch Normalization (BN) replacing Layer Normalization.

* **Decoder**: Selects destinations via softmax-weighted relevance scoring, satisfying capacity/loop constraints (buses return to departure station).

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

## 4. Environment Setup

### Hardware (as in the paper)



* CPU: AMD Ryzen 7-5800H (8 cores, 3.20 GHz)

* GPU: NVIDIA GeForce RTX 3060 Ti

* RAM: 16GB+ (recommended)

### Software



* Python 3.8

* Required Libraries:



```
pip install torch torchvision torchaudio  # For SAC and Transformer

pip install numpy pandas scikit-learn    # Data processing & clustering

pip install matplotlib seaborn           # Visualization

pip install tqdm                         # Progress tracking
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

## 6. Citation

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

## 7. Acknowledgements

This work is supported by:



* National Key R\&D Program of China (2020YFB1707900)

* National Natural Science Foundation of China (62272302)

* Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102)

* DiDi GAIA Research Collaboration Plan (202204)

Thanks to Didi Chuxing for providing the GAIA dataset.