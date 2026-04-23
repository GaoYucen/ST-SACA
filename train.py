"""Compatibility wrapper for the experiment training CLI."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from st_saca.experiments.train import main, parse_args, run_method  # noqa: E402,F401


if __name__ == "__main__":
    args = parse_args()
    main(A=args.A, w=args.w, method=args.method, episodes=args.episodes, time_slots=args.time_slots)
