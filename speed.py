"""Benchmark inference speed @dataclass
class BenchConfig:
    warmup: int = 30
    iters: int = 200
    greedy_samples: int = 0  # for SAC-style agents; 0 means normal sampling
    deterministic: bool = True
    seed: int = 42
    # 参数扫描：不同的车辆数 (E2E 在大规模时很慢，使用较小范围)
    num_buses_list: tuple = (10, 50, 100, 200, 500)ple algorithms.

Measures pure action-selection latency (i.e., agent.select_action on a fixed state).
This does NOT include environment stepping.

Algorithms:
- ST-SACA: `SACA.py` (SpatialActor)
- SACA: `SACA_baseline.py`
- GRC-ELG: `GRC_ELG.py`
- JDRL-POMO: `JDRL_POMO.py`

Output:
- Prints a summary table (ms) to console
- Saves per-run raw timings and summary to `log/` as CSV
"""

from __future__ import annotations

import os
import time
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import pandas as pd


@dataclass
class BenchConfig:
    warmup: int = 30
    iters: int = 200
    greedy_samples: int = 0  # for SAC-style agents; 0 means normal sampling
    deterministic: bool = True
    seed: int = 42
    # 参数扫描：不同的车辆数
    num_buses_list: tuple = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * q
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _now() -> float:
    # high-res timer
    return time.perf_counter()


def _bench(name: str, fn: Callable[[], Any], cfg: BenchConfig) -> dict:
    # Warmup
    for _ in range(cfg.warmup):
        fn()

    times = []
    for _ in range(cfg.iters):
        t0 = _now()
        fn()
        t1 = _now()
        times.append((t1 - t0) * 1000.0)  # ms

    return {
        "method": name,
        "iters": cfg.iters,
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times)),
        "p50_ms": float(_percentile(times, 0.50)),
        "p95_ms": float(_percentile(times, 0.95)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "raw_ms": times,
    }


def _bench_e2e(
    name: str,
    action_fn: Callable[[], Any],
    step_fn: Callable[[Any], Any],
    cfg: BenchConfig,
) -> dict:
    """End-to-end per-step latency.

    Measures one (action selection + env.step) cycle.
    Notes:
    - This includes python overhead + env dynamics.
    - It does NOT include training/update.
    """

    # Warmup
    for _ in range(cfg.warmup):
        act = action_fn()
        step_fn(act)

    times = []
    for _ in range(cfg.iters):
        t0 = _now()
        act = action_fn()
        step_fn(act)
        t1 = _now()
        times.append((t1 - t0) * 1000.0)  # ms

    return {
        "method": name,
        "iters": cfg.iters,
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times)),
        "p50_ms": float(_percentile(times, 0.50)),
        "p95_ms": float(_percentile(times, 0.95)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "raw_ms": times,
    }


def _try_build_agents():
    """Build minimal agents for inference benchmarking.

    We do NOT train; we only instantiate networks with random weights.
    """

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch (torch) is required to benchmark inference speed. "
            "Please install torch in the current environment."
        ) from e

    # Each item is:
    # (
    #   name,
    #   select_fn: Callable[[], Any],
    #   optional_e2e: tuple[action_fn, step_fn] | Exception
    # )
    agents = []

    def _pack_pa(out: Any):
        """Convert select_action output to an env.step-compatible (p, a).

        Most env.step implementations in this repo expect action=(p, a).
        Some agents may return extra info (e.g., log_prob, value, etc.).
        """
        if isinstance(out, tuple):
            if len(out) >= 2:
                return (out[0], out[1])
        return out

    # ---- ST-SACA (SACA.py) ----
    try:
        import SACA as STSACA

        cfg = STSACA.Config()
        env = STSACA.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + len(env.buses)
        action_dim = 2 * cfg.num_destinations
        agent = STSACA.SAC(cfg, state_dim, action_dim, env.dispatcher, env.dest_coords)
        state = env.get_state()

        def _fn():
            agent.select_action(state, deterministic=True)

        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))

        def _step(action):
            # Most envs in this repo accept a tuple (p, a)
            try:
                return env.step(action)
            except TypeError:
                # Some envs might be step(p, a)
                if isinstance(action, tuple) and len(action) == 2:
                    return env.step(*action)
                raise

        agents.append(("ST-SACA", _fn, (_act, _step)))
    except Exception as e:
        agents.append(("ST-SACA", e, e))

    # ---- SACA baseline (SACA_baseline.py) ----
    try:
        import SACA_baseline as SACA

        cfg = SACA.Config()
        env = SACA.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + len(env.buses)
        action_dim = 2 * cfg.num_destinations
        agent = SACA.SAC(cfg, state_dim, action_dim, env.dispatcher)
        state = env.get_state()

        def _fn():
            agent.select_action(state, deterministic=True)

        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))

        def _step(action):
            try:
                return env.step(action)
            except TypeError:
                if isinstance(action, tuple) and len(action) == 2:
                    return env.step(*action)
                raise

        agents.append(("SACA", _fn, (_act, _step)))
    except Exception as e:
        agents.append(("SACA", e, e))

    # ---- GRC-ELG (GRC_ELG.py) ----
    try:
        import GRC_ELG as GRC

        cfg = GRC.Config()
        env = GRC.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + len(env.buses)
        action_dim = 2 * cfg.num_destinations
        agent = GRC.GRC_SAC(cfg, state_dim, action_dim, env.dispatcher, env.dest_coords)
        state = env.get_state()

        def _fn():
            agent.select_action(state, deterministic=True)

        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))

        def _step(action):
            try:
                return env.step(action)
            except TypeError:
                if isinstance(action, tuple) and len(action) == 2:
                    return env.step(*action)
                raise

        agents.append(("GRC-ELG", _fn, (_act, _step)))
    except Exception as e:
        agents.append(("GRC-ELG", e, e))

    # ---- JDRL-POMO (JDRL_POMO.py) ----
    try:
        import JDRL_POMO as JDRL

        cfg = JDRL.Config()
        env = JDRL.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + env.num_buses
        action_dim = 2 * cfg.num_destinations
        agent = JDRL.JDRLAgent(cfg, state_dim, action_dim)
        state = env.get_state()

        def _fn():
            # JDRL select_action returns 5 items; we only time the call
            agent.select_action(state, deterministic=True)

        # For E2E timing we need an action compatible with env.step.
        # JDRL's select_action may return a richer tuple; we attempt to extract (p, a).
        def _act():
            out = agent.select_action(state, deterministic=True)
            # Common patterns:
            # - (p, a)
            # - (p, a, ...)
            if isinstance(out, tuple) and len(out) >= 2:
                return (out[0], out[1])
            return out

        def _step(action):
            try:
                return env.step(action)
            except TypeError:
                if isinstance(action, tuple) and len(action) == 2:
                    return env.step(*action)
                raise

        agents.append(("JDRL-POMO", _fn, (_act, _step)))
    except Exception as e:
        agents.append(("JDRL-POMO", e, e))

    return agents


def _build_agent_with_num_buses(algorithm: str, num_buses: int):
    """Build a single agent with a specific num_buses setting.
    
    Also adjusts demand_amplitude = num_buses * 2 to scale demand with fleet size.
    
    Returns: (select_fn, action_fn, step_fn, env) or raises Exception
    - select_fn: pure inference timing
    - action_fn: returns (p, a) for env.step
    - step_fn: env.step wrapper
    """
    import torch
    
    # 需求振幅随车辆数调整
    demand_amplitude = num_buses * 2
    
    def _pack_pa(out):
        """Convert select_action output to env.step-compatible (p, a)."""
        if isinstance(out, tuple) and len(out) >= 2:
            return (out[0], out[1])
        return out
    
    def _make_step_fn(env):
        def _step(action):
            try:
                return env.step(action)
            except TypeError:
                if isinstance(action, tuple) and len(action) == 2:
                    return env.step(*action)
                raise
        return _step
    
    if algorithm == "ST-SACA":
        import SACA as STSACA
        cfg = STSACA.Config()
        cfg.num_buses = num_buses
        cfg.demand_amplitude = demand_amplitude
        env = STSACA.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + len(env.buses)
        action_dim = 2 * cfg.num_destinations
        agent = STSACA.SAC(cfg, state_dim, action_dim, env.dispatcher, env.dest_coords)
        state = env.get_state()
        def _fn():
            agent.select_action(state, deterministic=True)
        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))
        return _fn, _act, _make_step_fn(env), env
        
    elif algorithm == "SACA":
        import SACA_baseline as SACA
        cfg = SACA.Config()
        cfg.num_buses = num_buses
        cfg.demand_amplitude = demand_amplitude
        env = SACA.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + len(env.buses)
        action_dim = 2 * cfg.num_destinations
        agent = SACA.SAC(cfg, state_dim, action_dim, env.dispatcher)
        state = env.get_state()
        def _fn():
            agent.select_action(state, deterministic=True)
        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))
        return _fn, _act, _make_step_fn(env), env
        
    elif algorithm == "GRC-ELG":
        import GRC_ELG as GRC
        cfg = GRC.Config()
        cfg.num_buses = num_buses
        cfg.demand_amplitude = demand_amplitude
        env = GRC.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + len(env.buses)
        action_dim = 2 * cfg.num_destinations
        agent = GRC.GRC_SAC(cfg, state_dim, action_dim, env.dispatcher, env.dest_coords)
        state = env.get_state()
        def _fn():
            agent.select_action(state, deterministic=True)
        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))
        return _fn, _act, _make_step_fn(env), env
        
    elif algorithm == "JDRL-POMO":
        import JDRL_POMO as JDRL
        cfg = JDRL.Config()
        cfg.num_buses = num_buses
        cfg.demand_amplitude = demand_amplitude
        env = JDRL.BusBookingEnv(cfg)
        state_dim = cfg.num_destinations + env.num_buses
        action_dim = 2 * cfg.num_destinations
        agent = JDRL.JDRLAgent(cfg, state_dim, action_dim)
        state = env.get_state()
        def _fn():
            agent.select_action(state, deterministic=True)
        def _act():
            return _pack_pa(agent.select_action(state, deterministic=True))
        return _fn, _act, _make_step_fn(env), env
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    cfg = BenchConfig()
    np.random.seed(cfg.seed)

    # Environment hint (helps avoid running in a no-torch env)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX")
    print(f"Python: {sys.executable}")
    print(f"Conda env: {conda_env}")
    print("Tip: run this script after activating your (pytorch) env, e.g. `conda activate pytorch`.\n")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    # ====== 参数扫描模式：不同 num_buses ======
    algorithms = ["ST-SACA", "SACA", "GRC-ELG", "JDRL-POMO"]
    num_buses_list = cfg.num_buses_list
    
    print(f"Benchmarking with num_buses = {num_buses_list}")
    print("=" * 70)
    
    # 存储结果：{algorithm: {num_buses: mean_ms}}
    # 纯推理
    results_table = {alg: {} for alg in algorithms}
    # E2E (推理 + env.step)
    results_table_e2e = {alg: {} for alg in algorithms}
    all_results = []
    
    for num_buses in num_buses_list:
        print(f"\n--- num_buses = {num_buses} ---")
        for alg in algorithms:
            try:
                fn, act_fn, step_fn, env = _build_agent_with_num_buses(alg, num_buses)
                
                # 纯推理测试
                res = _bench(f"{alg}", fn, cfg)
                mean_ms = res["mean_ms"]
                results_table[alg][num_buses] = mean_ms
                all_results.append({
                    "algorithm": alg,
                    "type": "inference",
                    "num_buses": num_buses,
                    "mean_ms": mean_ms,
                    "std_ms": res["std_ms"],
                    "p50_ms": res["p50_ms"],
                    "p95_ms": res["p95_ms"],
                })
                
                # E2E 测试 (推理 + env.step)
                res_e2e = _bench_e2e(f"{alg}-E2E", act_fn, step_fn, cfg)
                mean_ms_e2e = res_e2e["mean_ms"]
                results_table_e2e[alg][num_buses] = mean_ms_e2e
                all_results.append({
                    "algorithm": alg,
                    "type": "e2e",
                    "num_buses": num_buses,
                    "mean_ms": mean_ms_e2e,
                    "std_ms": res_e2e["std_ms"],
                    "p50_ms": res_e2e["p50_ms"],
                    "p95_ms": res_e2e["p95_ms"],
                })
                
                print(f"  {alg}: inference={mean_ms:.2f} ms, E2E={mean_ms_e2e:.2f} ms")
            except Exception as e:
                print(f"  {alg}: FAILED - {e}")
                results_table[alg][num_buses] = float("nan")
                results_table_e2e[alg][num_buses] = float("nan")
    
    # ====== 生成论文格式的表格：纯推理 ======
    print("\n" + "=" * 70)
    print("Pure Inference Time (ms) - Algorithm × num_buses")
    print("=" * 70)
    
    table_data = []
    for alg in algorithms:
        row = {"Algorithm": alg}
        for nb in num_buses_list:
            val = results_table[alg].get(nb, float("nan"))
            row[f"{nb} buses"] = f"{val:.1f}" if not np.isnan(val) else "N/A"
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    print(table_df.to_string(index=False))
    
    # ====== 生成论文格式的表格：E2E ======
    print("\n" + "=" * 70)
    print("End-to-End Time (inference + env.step) (ms) - Algorithm × num_buses")
    print("=" * 70)
    
    table_data_e2e = []
    for alg in algorithms:
        row = {"Algorithm": alg}
        for nb in num_buses_list:
            val = results_table_e2e[alg].get(nb, float("nan"))
            row[f"{nb} buses"] = f"{val:.1f}" if not np.isnan(val) else "N/A"
        table_data_e2e.append(row)
    
    table_df_e2e = pd.DataFrame(table_data_e2e)
    print(table_df_e2e.to_string(index=False))
    
    # ====== 保存详细结果到 CSV ======
    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(log_dir, f"inference_speed_by_buses_{timestamp}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")
    
    # ====== 保存表格格式到 CSV ======
    table_csv = os.path.join(log_dir, f"inference_speed_table_{timestamp}.csv")
    table_df.to_csv(table_csv, index=False)
    print(f"Inference table saved to: {table_csv}")
    
    table_csv_e2e = os.path.join(log_dir, f"e2e_speed_table_{timestamp}.csv")
    table_df_e2e.to_csv(table_csv_e2e, index=False)
    print(f"E2E table saved to: {table_csv_e2e}")


if __name__ == "__main__":
    main()
