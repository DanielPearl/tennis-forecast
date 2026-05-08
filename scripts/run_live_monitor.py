"""Live-monitor loop.

Re-runs the watchlist export every ``refresh_seconds`` seconds. This
is what you point a systemd unit at on the droplet (see
``deploy/baseline-break-monitor.service``). The dashboard never calls
the model directly — it just re-reads the cached JSON written here.

Why a separate process: keeps the model run independent of the HTTP
server. If the model run errors mid-tick, the dashboard keeps serving
the last good snapshot rather than 500-ing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.dashboard.export_watchlist import export
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

log = setup_logging("scripts.live_monitor", log_path=str(_REPO / "data" / "live_monitor.log"))


def main() -> None:
    cfg = load_config()
    period = int(cfg["dashboard"]["refresh_seconds"])
    log.info("live monitor started — refresh every %ds", period)
    while True:
        try:
            csv_path, json_path = export()
            log.info("watchlist refreshed (%s)", json_path.name)
        except Exception as exc:  # never let the loop die
            log.exception("export error: %s", exc)
        time.sleep(period)


if __name__ == "__main__":
    main()
