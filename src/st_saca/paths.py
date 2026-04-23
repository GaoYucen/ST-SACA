"""Shared filesystem paths for ST-SACA.

The environment variables here are intentionally small and explicit so
experiments can be run from any working directory without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ST_SACA_DATA_DIR", REPO_ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.environ.get("ST_SACA_OUTPUT_DIR", REPO_ROOT / "outputs")).resolve()
ROUTING_CKPT_DIR = Path(
    os.environ.get("ST_SACA_ROUTING_CKPT_DIR", REPO_ROOT / "checkpoints" / "routing")
).resolve()

STATIONS_FILE = DATA_DIR / "stations" / "chengdu_30_bus_stations.txt"


def ensure_output_dir(*parts: str) -> Path:
    """Return an output directory and create it if needed."""

    path = OUTPUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, description: str) -> Path:
    """Return an existing path or raise a clear setup error."""

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. "
            "Place the file there or set ST_SACA_DATA_DIR / "
            "ST_SACA_ROUTING_CKPT_DIR to the correct location."
        )
    return path
