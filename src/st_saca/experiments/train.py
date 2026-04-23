"""Training entrypoint for ST-SACA experiments."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt


METHODS = ("st-saca", "saca", "grc-elg", "jdrl-pomo")


def _configure(config, *, A: float, w: float, episodes: int | None, time_slots: int | None):
    config.demand_fluctuation = A
    config.demand_frequency = w
    if episodes is not None:
        config.episodes = episodes
        config.max_episodes = episodes
        if hasattr(config, "min_episodes"):
            config.min_episodes = min(config.min_episodes, episodes)
    if time_slots is not None:
        config.time_slots_per_episode = time_slots
    return config


def run_method(
    method: str,
    *,
    A: float,
    w: float,
    episodes: int | None = None,
    time_slots: int | None = None,
):
    config_factory, train_fn = _load_method(method)
    config = _configure(config_factory(), A=A, w=w, episodes=episodes, time_slots=time_slots)
    return train_fn(config)


def _load_method(method: str):
    if method == "st-saca":
        from st_saca.agents import st_saca as module

        return module.Config, module.train_saca
    if method == "saca":
        from st_saca.agents import saca_baseline as module

        return module.Config, module.train_saca
    if method == "grc-elg":
        from st_saca.baselines import grc_elg as module

        return module.Config, module.train_grc_elg
    if method == "jdrl-pomo":
        from st_saca.baselines import jdrl_pomo as module

        return module.Config, module.train_jdrl
    raise ValueError(f"Unknown training method: {method}")


def main(
    A: float = 10,
    w: float = 1,
    method: str = "all",
    episodes: int | None = None,
    time_slots: int | None = None,
):
    """Run one method or the full comparison suite.

    The positional-like ``A`` and ``w`` arguments are kept for compatibility
    with the previous ``train.main(A=..., w=...)`` call pattern.
    """

    plt.show = lambda: None

    if method == "all":
        results = {}
        for name in ("jdrl-pomo", "st-saca", "grc-elg", "saca"):
            results[name] = run_method(name, A=A, w=w, episodes=episodes, time_slots=time_slots)
        return results
    return run_method(method, A=A, w=w, episodes=episodes, time_slots=time_slots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-SACA paper experiments.")
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument("--uncertainty-scale", "--A", dest="A", type=float, default=10.0)
    parser.add_argument("--uncertainty-frequency", "--w", dest="w", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--time-slots", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(A=args.A, w=args.w, method=args.method, episodes=args.episodes, time_slots=args.time_slots)
