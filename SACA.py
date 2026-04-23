"""Compatibility wrapper for the ST-SACA agent.

Prefer ``python -m st_saca.experiments.train --method st-saca`` after
installing the package with ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from st_saca.agents.st_saca import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    config = Config()
    train_saca(config)
