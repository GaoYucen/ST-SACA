"""Case study: visualize pricing policies under a fixed demand pattern.

Goal
----
Generate a *fixed* potential demand vector (N_p) and compare the pricing heatmaps
between multiple methods:

- ST-SACA (SACA.py)
- SACA baseline (SACA_baseline.py)
- GRC-ELG (GRC_ELG.py)
- JDRL-POMO (JDRL_POMO.py)

This script produces figures similar to the paper's case study:
- Potential demand heatmap
- Pricing heatmaps for each method

Notes
-----
- This script does NOT train models. It instantiates agents with random weights.
  For paper-quality plots, you should load trained checkpoints and then call
  select_action on the same fixed state.
- "Demand" shown here is the *potential demand* N_p. The actual realized demand
  depends on the price via demand_function.

Outputs
-------
- Always: show a grouped figure (Potential demand + 4 pricing heatmaps).
- Optionally: save *each* heatmap as a separate PDF under `log/`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class CaseCfg:
    seed: int = 34
    grid_shape: tuple[int, int] = (6, 5)  # 6 rows x 5 cols (30 stations)
    # potential demand generation
    demand_min: int = 0
    demand_max: int = 8
    # where to save
    out_dir: str = "log"
    # saving
    save_pdf: bool = True  # default OFF; keep showing grouped figure

    # plotting / paper style
    fontsize: int = 20  # 期刊字体通常较小 (8-10pt)
    
    # 核心修改：使用 Matplotlib 的 RdYlBu_r (红-黄-蓝，翻转)
    cmap: str = "RdYlBu_r" 
    
    annotate: bool = False
    grid_lines: bool = True   # 建议开启，用白色分割线增加高级感
    
    # Heatmap appearance
    cell_gap: float = 2     # 线条改细，精致感更强
    center_color: str = "#FFFFFF" # 回归白色或极浅灰，不要用有色背景干扰数据阅读


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # allow plotting-only environments (but agents need torch)
        pass


def _cell_edges(grid_shape: tuple[int, int]):
    """Return pcolormesh edge grids so cells are not clipped.

    pcolormesh expects corner coordinates (H+1, W+1). Using explicit edges
    avoids half-cell clipping on the first row/col.
    """

    h, w = grid_shape
    x = np.arange(w + 1)
    y = np.arange(h + 1)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _set_plot_style(cfg: CaseCfg) -> None:
    """Paper-style plotting defaults (conference-friendly)."""

    # Avoid global grid lines (they can visually split each cell)
    try:
        plt.style.use("seaborn-v0_8-white")
    except Exception:
        # fallback to default
        pass

    # Font: prefer Times New Roman; fallback to DejaVu Sans
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": cfg.fontsize,
            "axes.labelsize": cfg.fontsize,
            "xtick.labelsize": cfg.fontsize - 1,
            "ytick.labelsize": cfg.fontsize - 1,
            "legend.fontsize": cfg.fontsize - 1,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # PDF/PS font embedding
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "none",
        }
    )


def _build_diverging_cmap(cfg: CaseCfg):
    """Blue->(not pure white)->Red diverging colormap."""

    colors = [
        "#2c7bb6",  # blue
        cfg.center_color,  # near-zero: light grey (not pure white)
        "#d7191c",  # red
    ]
    return LinearSegmentedColormap.from_list("PaperRdBu", colors, N=256)


def _generate_fixed_Np(cfg: CaseCfg) -> np.ndarray:
    """Generate a fixed 30-dim potential demand (N_p).

    We use a deterministic random pattern with a couple of high-demand hotspots.
    """

    h, w = cfg.grid_shape
    n = h * w

    # base random field
    base = np.random.randint(cfg.demand_min, cfg.demand_max + 1, size=n)

    # add a few hotspots (deterministic given seed)
    hotspot_idx = [3, 4, 9, 10, 21, 22]  # arbitrary, makes visible structure
    for idx in hotspot_idx:
        if 0 <= idx < n:
            base[idx] = max(base[idx], cfg.demand_max)

    return base.astype(float)


def _reshape_to_grid(x: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    if x.size != h * w:
        raise ValueError(f"Expected {h*w} items, got {x.size}")
    return x.reshape(h, w)


def _infer_price_vectors(N_p: np.ndarray):
    """Build env + agent for each method and infer price vectors p."""

    results: dict[str, np.ndarray] = {}

    # ST-SACA
    import SACA as STSACA

    cfg_st = STSACA.Config()
    env_st = STSACA.BusBookingEnv(cfg_st)
    env_st.reset()
    env_st.N_p = N_p.copy()
    state_st = env_st.get_state()
    state_dim_st = cfg_st.num_destinations + len(env_st.buses)
    action_dim_st = 2 * cfg_st.num_destinations
    agent_st = STSACA.SAC(cfg_st, state_dim_st, action_dim_st, env_st.dispatcher, env_st.dest_coords)
    p_st, _a_st = agent_st.select_action(state_st, deterministic=True)
    results["ST-SACA"] = np.asarray(p_st, dtype=float)

    # SACA baseline
    import SACA_baseline as SACA

    cfg_sa = SACA.Config()
    env_sa = SACA.BusBookingEnv(cfg_sa)
    env_sa.reset()
    env_sa.N_p = N_p.copy()
    state_sa = env_sa.get_state()
    state_dim_sa = cfg_sa.num_destinations + len(env_sa.buses)
    action_dim_sa = 2 * cfg_sa.num_destinations
    agent_sa = SACA.SAC(cfg_sa, state_dim_sa, action_dim_sa, env_sa.dispatcher)
    p_sa, _a_sa = agent_sa.select_action(state_sa, deterministic=True)
    results["SACA"] = np.asarray(p_sa, dtype=float)

    # GRC-ELG
    import GRC_ELG as GRC

    cfg_grc = GRC.Config()
    env_grc = GRC.BusBookingEnv(cfg_grc)
    env_grc.reset()
    env_grc.N_p = N_p.copy()
    state_grc = env_grc.get_state()
    state_dim_grc = cfg_grc.num_destinations + len(env_grc.buses)
    action_dim_grc = 2 * cfg_grc.num_destinations
    agent_grc = GRC.GRC_SAC(cfg_grc, state_dim_grc, action_dim_grc, env_grc.dispatcher, env_grc.dest_coords)
    p_grc, _a_grc = agent_grc.select_action(state_grc, deterministic=True)
    results["GRC-ELG"] = np.asarray(p_grc, dtype=float)

    # JDRL-POMO
    import JDRL_POMO as JDRL

    cfg_jdrl = JDRL.Config()
    env_jdrl = JDRL.BusBookingEnv(cfg_jdrl)
    env_jdrl.reset()
    env_jdrl.N_p = N_p.copy()
    state_jdrl = env_jdrl.get_state()
    state_dim_jdrl = cfg_jdrl.num_destinations + env_jdrl.num_buses
    action_dim_jdrl = 2 * cfg_jdrl.num_destinations
    agent_jdrl = JDRL.JDRLAgent(cfg_jdrl, state_dim_jdrl, action_dim_jdrl)
    p_jdrl, _a_jdrl, *_rest = agent_jdrl.select_action(state_jdrl, deterministic=True)
    results["JDRL-POMO"] = np.asarray(p_jdrl, dtype=float)

    return results


def _annotate_cells(ax: plt.Axes, grid: np.ndarray, fmt: str = "{:.2f}") -> None:
    # Smaller text to avoid clutter
    for (i, j), val in np.ndenumerate(grid):
        ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7, color="white")


def _style_heatmap_axes(ax: plt.Axes, grid_shape: tuple[int, int], grid_lines: bool) -> None:
    h, w = grid_shape
    # Put ticks at cell centers for pcolormesh with corner coords
    ax.set_xticks(np.arange(w) + 0.5)
    ax.set_yticks(np.arange(h) + 0.5)
    # Labels start from 1 (not 0)
    ax.set_xticklabels([str(i + 1) for i in range(w)])
    ax.set_yticklabels([str(i + 1) for i in range(h)])
    ax.set_xlabel("Station")
    ax.set_ylabel("Station")
    ax.set_xlim(0, w)
    # Y-axis from bottom to top (row 0 at bottom)
    ax.set_ylim(0, h)

    # Clean spines (remove all borders)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Force-disable any background grids from global styles unless explicitly enabled
    ax.grid(False)
    ax.tick_params(length=0)

    # Keep each cell square (important for PDF export)
    ax.set_aspect("equal", adjustable="box")

    if grid_lines:
        # Draw only cell boundaries (no extra background grid)
        ax.set_xticks(np.arange(0, w + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, h + 1, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)


def _save_single_heatmap_pdf(
    title: str,
    grid: np.ndarray,
    out_pdf: str,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    cfg: CaseCfg,
) -> None:
    h, w = cfg.grid_shape
    # figsize proportional to grid: width ~ w, height ~ h
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6 * h / w), constrained_layout=False)
    # pcolormesh with edgecolors creates visible gaps between cells
    X, Y = _cell_edges(cfg.grid_shape)
    m = ax.pcolormesh(
        X,
        Y,
        grid,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="white",
        linewidth=cfg.cell_gap,
        antialiased=True,
    )
    # No title for single PDF
    _style_heatmap_axes(ax, cfg.grid_shape, cfg.grid_lines)
    # Ensure full cells are visible; y-axis bottom-to-top
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    if cfg.annotate:
        _annotate_cells(ax, grid)
    # Use make_axes_locatable to create colorbar that exactly matches axes height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(m, cax=cax)
    cbar.ax.tick_params(labelsize=cfg.fontsize - 1)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_grouped_heatmaps(demand_grid: np.ndarray, pricing_grids: dict[str, np.ndarray], cfg: CaseCfg) -> None:
    # Potential demand + 4 pricing heatmaps
    order = ["ST-SACA", "SACA", "GRC-ELG", "JDRL-POMO"]
    h, w = cfg.grid_shape
    # figsize: total width ~ num_plots * cell_width, height ~ cell_height ratio
    fig, axes = plt.subplots(1, 1 + len(order), figsize=(18, 18 / (1 + len(order)) * h / w), constrained_layout=True)
    if 1 + len(order) == 1:
        axes = [axes]

    # shared pricing range for fair visual comparison
    all_pricing = np.concatenate([pricing_grids[name].ravel() for name in order])
    p_vmin = float(np.nanmin(all_pricing))
    p_vmax = float(np.nanmax(all_pricing))

    cmap = _build_diverging_cmap(cfg) if cfg.cmap == "RdBu_r" else cfg.cmap

    X, Y = _cell_edges(cfg.grid_shape)

    # Potential demand
    m0 = axes[0].pcolormesh(
        X,
        Y,
        demand_grid,
        cmap=cmap,
        shading="flat",
        edgecolors="white",
        linewidth=cfg.cell_gap,
        antialiased=True,
    )
    # No title for grouped figure
    _style_heatmap_axes(axes[0], cfg.grid_shape, cfg.grid_lines)
    h, w = cfg.grid_shape
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(0, h)
    if cfg.annotate:
        _annotate_cells(axes[0], demand_grid, fmt="{:.0f}")
    cbar0 = fig.colorbar(m0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=cfg.fontsize - 1)

    pricing_images = []
    for i, name in enumerate(order, start=1):
        grid = pricing_grids[name]
        m = axes[i].pcolormesh(
            X,
            Y,
            grid,
            cmap=cmap,
            vmin=p_vmin,
            vmax=p_vmax,
            shading="flat",
            edgecolors="white",
            linewidth=cfg.cell_gap,
            antialiased=True,
        )
        pricing_images.append(m)
        # No title for grouped subplots
        _style_heatmap_axes(axes[i], cfg.grid_shape, cfg.grid_lines)
        h, w = cfg.grid_shape
        axes[i].set_xlim(0, w)
        axes[i].set_ylim(0, h)
        if cfg.annotate:
            _annotate_cells(axes[i], grid)

    # One shared colorbar for all pricing plots
    cbar = fig.colorbar(pricing_images[-1], ax=axes[1:], fraction=0.02, pad=0.01)
    cbar.set_label("Price", fontsize=cfg.fontsize)
    cbar.ax.tick_params(labelsize=cfg.fontsize - 1)

    plt.show()


def main() -> None:
    cfg = CaseCfg()

    # ---- Switch here ----
    save_pdf = cfg.save_pdf  # default OFF

    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX")
    print(f"Python: {sys.executable}")
    print(f"Conda env: {conda_env}")
    print("Tip: for agents, run this script after activating your (pytorch) env, e.g. `conda activate pytorch`.\n")

    _set_seeds(cfg.seed)
    _set_plot_style(cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)
    N_p = _generate_fixed_Np(cfg)
    price_vectors = _infer_price_vectors(N_p)

    demand_grid = _reshape_to_grid(N_p, cfg.grid_shape)
    pricing_grids = {k: _reshape_to_grid(v, cfg.grid_shape) for k, v in price_vectors.items()}

    # Build colormap once (keep consistent across grouped + saved plots)
    cmap = _build_diverging_cmap(cfg) if cfg.cmap == "RdBu_r" else cfg.cmap

    # Grouped figure (always show)
    _plot_grouped_heatmaps(demand_grid, pricing_grids, cfg)

    # Save each heatmap as separate PDF (optional)
    if save_pdf:
        # fixed filename prefix (no timestamp)
        base = os.path.join(cfg.out_dir, "case_study_pricing")
        # shared pricing range for fair visual comparison
        order = ["ST-SACA", "SACA", "GRC-ELG", "JDRL-POMO"]
        all_pricing = np.concatenate([pricing_grids[name].ravel() for name in order])
        p_vmin = float(np.nanmin(all_pricing))
        p_vmax = float(np.nanmax(all_pricing))

        _save_single_heatmap_pdf(
            "Potential demand",
            demand_grid,
            base + "_demand.pdf",
            cmap=cmap,
            vmin=None,
            vmax=None,
            cfg=cfg,
        )
        for name, grid in pricing_grids.items():
            safe = name.lower().replace(" ", "_").replace("-", "_")
            _save_single_heatmap_pdf(
                name,
                grid,
                base + f"_{safe}.pdf",
                cmap=cmap,
                vmin=p_vmin,
                vmax=p_vmax,
                cfg=cfg,
            )
    print(f"Saved PDFs under: {cfg.out_dir} (prefix: case_study_pricing_*.pdf)")


if __name__ == "__main__":
    main()
