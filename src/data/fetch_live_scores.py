"""Live-match state ingestion.

Real Kalshi only. The bot's live monitor (``scripts/run_live_monitor.py``)
is the sole writer of ``data/raw/live_state.json``: each tick it pulls
KXATPMATCH + KXWTAMATCH markets from Kalshi via ``kalshi_markets.py``
and writes the canonical schema. This module just reads that file —
no demo fixture fallback. If the file doesn't exist or is empty, the
watchlist exporter renders an empty state and the dashboard shows
"no markets right now"; we never fabricate tickers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.live")


def _state_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["raw_dir"]) / "live_state.json"


def load_live_state() -> list[dict[str, Any]]:
    """Read the canonical live-state file written by the live monitor.
    Returns ``[]`` when the file is missing — the bot has not yet run
    its first Kalshi fetch."""
    fp = _state_path()
    if not fp.exists():
        log.info("live-state file %s missing — empty watchlist this tick", fp)
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("live-state file %s unreadable: %s", fp, exc)
        return []
    return data if isinstance(data, list) else []
