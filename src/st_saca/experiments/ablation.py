"""Ablation-study entrypoint."""

from __future__ import annotations

import argparse


def run_variant(variant: str):
    if variant == "wo-route":
        from st_saca.experiments import ablation_wo_route

        return ablation_wo_route.train_ablation_wo_route()
    if variant == "wo-orr":
        from st_saca.experiments import ablation_wo_orr

        config = ablation_wo_orr.Config()
        return ablation_wo_orr.train_saca(config)
    if variant == "wo-spatial":
        from st_saca.agents import saca_baseline

        config = saca_baseline.Config()
        return saca_baseline.train_saca(config)
    raise ValueError(f"Unknown ablation variant: {variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-SACA ablation variants.")
    parser.add_argument("--variant", choices=["wo-route", "wo-orr", "wo-spatial"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_variant(args.variant)
