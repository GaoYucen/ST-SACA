# ST-SACA: Spatial-Temporal Attention-based Soft Actor-Critic for Bus Booking Systems

This repository contains the enhanced code implementation for the journal version (TMC) of the paper **"SACA: An End-to-End Method for Dispatching, Routing, and Pricing of Online Bus-Booking"**. 

Building upon the conference version (DASFAA), this implementation introduces a **Spatial-Temporal Attention-based State Representation (ST-SAC)** mechanism to significantly improve the agent's ability to capture demand distribution patterns and optimize long-term revenue.

## 1. Technique Improvement: ST-SAC

In the original SACA (Baseline), the Actor network used a simple Multi-Layer Perceptron (MLP) to process the state vector. This approach treated each destination station as an independent feature, ignoring the geometric structure of the transportation network.

**The new ST-SAC (implemented in `SACA.py`) introduces:**
*   **Spatial-Aware State Representation**: Explicitly models the spatial coordinates $(x, y)$ of each destination along with its real-time demand.
*   **Self-Attention Mechanism**: Uses a Transformer-based attention layer to extract spatial dependencies between stations, allowing the agent to identify "hotspot clusters" dynamically.
*   **Performance Gains**: Compared to the baseline, ST-SAC achieves **~33% higher average reward** and **~39.5% higher Order Response Rate (ORR)**.

## 2. Project Structure

The repository is organized as follows:

```
.
├── SACA.py              # [Main] The proposed ST-SAC model with Spatial-Aware Actor
├── SACA_baseline.py     # [Baseline] The original SAC model with MLP Actor
├── compare_models.py       # [Script] Automates training of both models and generates comparison plots
├── routing_model/          # Pre-trained Attention-based Routing Model (AM)
│   ├── AM.py               # Attention Model architecture
│   └── model/              # Saved weights for the routing model
├── find_station/           # Station clustering scripts (DBSCAN, PAM)
├── log/                    # [Output] Stores training logs (.csv) and result plots (.png)
├── param/                  # [Output] Stores trained model parameters (.pth)
└── README.md               # This file
```

## 3. Key Components

### 3.1 Environment (`BusBookingEnv`)
*   **Dynamic Demand**: Simulates passenger demand using Poisson distribution with time-varying intensity.
*   **Realistic Routing**: Integrates a pre-trained `AttentionRouteModel` to solve the Vehicle Routing Problem (VRP) for dispatched buses.
*   **Cost Model**: Calculates operational costs based on real-world Haversine distance (`beta_d=0.1`).

### 3.2 Spatial Actor (`SpatialActor`)
*   **Input**: State vector + Station Coordinates.
*   **Architecture**: `Input Embedding` -> `Self-Attention Layer` -> `Global Pooling` -> `Fusion with Bus State` -> `Policy Head`.
*   **Output**: Continuous actions for Pricing ($P$) and Seat Allocation ($A$).

## 4. Usage

### Prerequisites
*   Python 3.8+
*   PyTorch
*   NumPy, Pandas, Matplotlib, Scipy

### Running the Models

**1. Train the Proposed ST-SAC Model:**
```bash
python SACA.py
```
*   Results will be saved to `log/training_results_spatial_[timestamp].png` and `log/training_log_spatial_[timestamp].csv`.
*   Model weights will be saved to `param/sac_bus_booking_spatial_[timestamp].pth`.

**2. Train the Baseline Model:**
```bash
python SACA_baseline.py
```

**3. Run Comparison Experiment:**
To automatically train both models and generate a comparison plot:
```bash
python compare_models.py
```
*   This will generate `log/comparison_result.png` showing the performance gap between ST-SAC and the Baseline.

## 5. Performance Comparison

Recent experiments (100 episodes) show significant improvements:

| Metric | Baseline (MLP) | **ST-SAC (Spatial)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg Reward** | 42.08 | **55.97** | **+33.0%** |
| **Revenue** | 41.48 | **55.12** | **+32.9%** |
| **ORR** | 15.2% | **21.2%** | **+39.5%** |
| **Cost** | 10.04 | 12.09	| +20.4% (Reasonably increasing) |

*ST-SAC converges faster and learns a more efficient policy that balances high revenue with reasonable operational costs.*
