"""Compatibility wrapper for comparison plotting."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from st_saca.analysis.plot_comparison import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    plot_comparison()
