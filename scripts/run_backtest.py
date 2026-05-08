"""CLI wrapper around src/trading/backtest.run_backtest."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.trading.backtest import run_backtest

if __name__ == "__main__":
    run_backtest()
