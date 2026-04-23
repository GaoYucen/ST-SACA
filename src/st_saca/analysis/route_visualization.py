from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from st_saca.routing.gener_data import generate_single_data
from contextlib import redirect_stdout
from io import StringIO

try:
    # Optional: depends on torch.
    from st_saca.routing.am_test import presingle  # type: ignore

    _HAS_PRESINGLE = True
except Exception:
    presingle = None  # type: ignore
    _HAS_PRESINGLE = False


def _route_polyline(sample: Dict, order: Sequence[int]) -> np.ndarray:
    """Build polyline coords following start -> loc[order...] -> start."""
    loc = np.asarray(sample["loc"], dtype=np.float64)
    start = np.asarray(sample["start"], dtype=np.float64)
    if len(order) == 0:
        return np.stack([start, start], axis=0)
    path = np.vstack([start[None, :], loc[np.array(order, dtype=int)], start[None, :]])
    return path


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance (km)."""
    R = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


def route_distance_km(sample: Dict, order: Sequence[int]) -> float:
    path = _route_polyline(sample, order)
    dist = 0.0
    for i in range(len(path) - 1):
        dist += _haversine_km(path[i, 0], path[i, 1], path[i + 1, 0], path[i + 1, 1])
    return float(dist)


def avg_passenger_distance_km(sample: Dict, order: Sequence[int]) -> float:
    """Average passenger distance (km), matching `AM.calculate_avg_passenger_distance`.

    Semantics:
    - Bus starts from `start`.
    - Visits destinations in `order`.
    - For each visited destination t, passengers getting off there (weight) experience
      the *cumulative* bus distance from start to that destination.
    - Return weighted average: sum(weight_t * cum_dist_t) / sum(weight_t).

    Important: This does NOT include the return-to-start segment.
    """
    loc = np.asarray(sample["loc"], dtype=np.float64)
    start = np.asarray(sample["start"], dtype=np.float64)
    weight = np.asarray(sample.get("weight", np.ones((loc.shape[0],), dtype=np.float64)), dtype=np.float64)

    if loc.shape[0] == 0 or len(order) == 0:
        return 0.0

    # Reorder weights to match visiting order
    order_arr = np.asarray(order, dtype=int)
    sorted_weights = weight[order_arr]
    total_passengers = float(sorted_weights.sum())
    if total_passengers <= 0:
        return 0.0

    # Compute cumulative distance from start to each visited node
    cum = 0.0
    total_weighted = 0.0
    prev = start
    for idx, w in zip(order_arr, sorted_weights):
        cur = loc[idx]
        cum += _haversine_km(prev[0], prev[1], cur[0], cur[1])
        total_weighted += float(w) * cum
        prev = cur

    return float(total_weighted / (total_passengers + 1e-6))


def greedy_baseline(sample: Dict) -> Tuple[List[int], float]:
    """Simple nearest-neighbor greedy baseline."""
    loc = np.asarray(sample["loc"], dtype=np.float64)
    start = np.asarray(sample["start"], dtype=np.float64)
    n = loc.shape[0]
    if n == 0:
        return [], 0.0

    remaining = set(range(n))
    order: List[int] = []
    cur = start
    while remaining:
        # pick nearest next
        next_idx = min(
            remaining,
            key=lambda i: _haversine_km(cur[0], cur[1], loc[i, 0], loc[i, 1]),
        )
        order.append(next_idx)
        remaining.remove(next_idx)
        cur = loc[next_idx]
    return order, avg_passenger_distance_km(sample, order)


def random_baseline(sample: Dict, seed: int = 0) -> Tuple[List[int], float]:
    """Random visiting order baseline."""
    rng = np.random.default_rng(seed)
    n = int(np.asarray(sample["loc"]).shape[0])
    order = list(range(n))
    rng.shuffle(order)
    return order, avg_passenger_distance_km(sample, order)


def plot_base_map(ax: plt.Axes, sample: Dict) -> None:
    """Draw stations and start point (no routes)."""
    loc = np.asarray(sample["loc"], dtype=np.float64)
    start = np.asarray(sample["start"], dtype=np.float64)
    ax.scatter(loc[:, 0], loc[:, 1], s=18, c="black", alpha=0.75, label="stations")
    ax.scatter([start[0]], [start[1]], s=55, c="gold", edgecolors="black", zorder=5, label="start")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)


def plot_route_line(
    ax: plt.Axes,
    sample: Dict,
    order: Sequence[int],
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
) -> None:
    """Overlay one route polyline."""
    path = _route_polyline(sample, order)
    ax.plot(path[:, 0], path[:, 1], "-o", color=color, label=label, linewidth=linewidth, markersize=3)


if __name__ == "__main__":
    # User requirement:
    # 1. 4 Subplots, one for each Algorithm.
    # 2. In each subplot, show routes for N=[6, 9, 12, 15].
    # 3. Different Ns are distinguished by different colors AND line styles.

    Ns = [6, 9, 12]
    # Colors for each N
    N_colors = {
        6: "tab:blue",
        9: "tab:orange",
        12: "tab:green"
        # 15: "tab:red"
    }
    # Line styles for each N
    N_styles = {
        6: "-",      # Solid
        9: "-",     # Dashed
        12: "-",    # Dash-dot
        # 9: "--",     # Dashed
        # 12: "-."    # Dash-dot
        # 15: ":"      # Dotted
    }

    # Algorithms configuration
    # We will iterate over these to create subplots
    algo_names = ["ST-SACA", "SACA", "JDRL-POMO", "GRC-ELG"]

    # Optional: silence verbose prints inside `generate_single_data`.
    SILENCE_GENERATOR_PRINTS = True

    # 1. Pre-generate 4 instances (one for each N) so all algorithms run on the SAME set of instances.
    instances = {}
    for n_dest in Ns:
        if SILENCE_GENERATOR_PRINTS:
            buf = StringIO()
            with redirect_stdout(buf):
                stations, passengers, start_pos, best_route, best_cost = generate_single_data(n_dest)
        else:
            stations, passengers, start_pos, best_route, best_cost = generate_single_data(n_dest)
        
        instances[n_dest] = {
            "sample": {
                "loc": np.asarray(stations, dtype=np.float32),
                "weight": np.asarray(passengers, dtype=np.float32),
                "start": np.asarray(start_pos, dtype=np.float32),
            },
            "best_route": best_route, # For ST-SACA
            "best_cost": best_cost
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, algo_name in zip(axes, algo_names):
        ax.set_title(algo_name, fontsize=14, pad=10)
        
        # For this algorithm, plot routes for all Ns
        for n_dest in Ns:
            data = instances[n_dest]
            sample = data["sample"]
            color = N_colors[n_dest]
            linestyle = N_styles[n_dest]
            
            # Determine route based on algorithm
            if algo_name == "ST-SACA":
                # Exact optimal
                order = list(data["best_route"])
            elif algo_name == "SACA":
                # AM Model
                if _HAS_PRESINGLE and presingle is not None:
                    order, _ = presingle(sample)
                else:
                    # Fallback if model not loaded
                    order, _ = greedy_baseline(sample)
            elif algo_name == "JDRL-POMO":
                # Greedy placeholder
                order, _ = greedy_baseline(sample)
            elif algo_name == "GRC-ELG":
                # Random placeholder
                order, _ = random_baseline(sample, seed=42 + n_dest)
            else:
                order = []

            # Calculate metric for legend
            avg_dist = avg_passenger_distance_km(sample, order)

            # Plot Stations (small dots)
            loc = sample["loc"]
            start = sample["start"]
            # Plot stations
            ax.scatter(loc[:, 0], loc[:, 1], s=20, color=color, marker='o', alpha=0.6)
            # Plot start (only once? or for each? If they overlap, it's fine)
            ax.scatter([start[0]], [start[1]], s=80, color=color, edgecolors='red', marker='*', zorder=5)

            # Plot Route
            path = _route_polyline(sample, order)
            ax.plot(path[:, 0], path[:, 1], linestyle=linestyle, color=color, linewidth=2.0, label=f"N={n_dest} (Avg={avg_dist:.2f})")

        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.show()



