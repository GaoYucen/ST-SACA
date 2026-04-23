"""Compatibility wrapper for the SACA baseline agent."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from st_saca.agents.saca_baseline import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    config = Config()
    train_saca(config)
