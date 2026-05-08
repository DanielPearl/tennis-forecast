"""Live-monitor loop.

One iteration per ``dashboard.refresh_seconds``:

  1. Advance the live-state file (synthetic match progression — only
     when no real provider is plumbed in via SOFASCORE_BASE_URL).
  2. Build the watchlist (model probabilities, live adjustment, signals).
  3. Tick the paper-trade simulator (open / mark / settle positions).

The dashboard reads ``data/outputs/watchlist.json`` and
``data/outputs/sim_state.json`` directly, so there's no IPC needed —
the monitor is the only writer, the dashboard is the only reader.

Why a single process: keeps the model run independent of the HTTP
server. If the model run errors mid-tick, the dashboard keeps serving
the last good snapshot rather than 500-ing.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.dashboard.export_watchlist import build_watchlist_records, export
from src.data import match_progression
from src.data.fetch_live_scores import fetch_provider_state
from src.trading.simulator import tick as simulator_tick
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

log = setup_logging("scripts.live_monitor",
                     log_path=str(_REPO / "data" / "live_monitor.log"))


def _provider_returns_data() -> bool:
    """Probe the real provider. We tick the synthetic engine if either
    the provider isn't configured OR if a configured provider fails its
    call (e.g. 403 on the SofaScore probe URL — which is the default
    behavior for callers without an authenticated API agreement).

    Without this probe, a misconfigured provider would silently freeze
    the simulation: ``load_live_state`` falls back to the fixture file
    on provider failure, and the fixture would never advance.
    """
    if not os.environ.get("SOFASCORE_BASE_URL", "").strip():
        return False
    try:
        rows = fetch_provider_state()
    except Exception:
        return False
    return rows is not None


def _one_tick() -> None:
    # 1) Match progression — runs whenever no real provider data is
    #    available, so the simulation has visible activity. Real
    #    providers, when plumbed in, write fresh state directly.
    if not _provider_returns_data():
        match_progression.tick()

    # 2) Build the watchlist. We grab the standardized records via the
    #    same path the export uses so the simulator sees the same
    #    label / probability values that the dashboard does.
    rows = build_watchlist_records()
    export(records=rows)

    # 3) Reload the live state for the simulator settlement step. We
    #    need the ``completed`` / ``winner_side`` flags that the
    #    progression engine sets — those don't end up in the watchlist
    #    (which only carries display-shaped rows).
    from src.data.fetch_live_scores import load_live_state
    live = load_live_state()
    state = simulator_tick(rows, live)

    log.info("tick — %d watchlist rows, %d open positions, %d closed (P&L %+.3f, ROI %s)",
             len(rows),
             state["stats"].get("open_count", 0),
             state["stats"].get("total_closed", 0),
             state["stats"].get("total_realized_pnl", 0.0),
             ("—" if state["stats"].get("roi") is None
              else f"{state['stats']['roi']*100:+.1f}%"))


def main() -> None:
    cfg = load_config()
    period = int(cfg["dashboard"]["refresh_seconds"])
    log.info("live monitor started — refresh every %ds (provider=%s)",
              period, "live" if _provider_returns_data() else "synthetic")
    while True:
        try:
            _one_tick()
        except Exception as exc:  # never let the loop die
            log.exception("tick error: %s", exc)
        time.sleep(period)


if __name__ == "__main__":
    main()
