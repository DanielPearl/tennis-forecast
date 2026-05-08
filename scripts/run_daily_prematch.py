"""Daily pre-match cron entrypoint.

Run once per day (or on-demand). Steps:

  1. Optionally refresh Sackmann CSVs (--refresh-data flag)
  2. Re-train the pre-match model on the current data
  3. Pull odds (if THE_ODDS_API_KEY is set) — otherwise the watchlist
     uses the market_prob_a values already on the live-state fixture.
  4. Generate the watchlist (writes data/outputs/watchlist.{csv,json})

The dashboard re-reads the JSON file on every page load, so the
moment this script finishes the site shows the fresh data.

Usage:
  python scripts/run_daily_prematch.py            # full run
  python scripts/run_daily_prematch.py --skip-train  # just rebuild watchlist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import src.*` resolve when run as a script from the repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.dashboard.export_watchlist import export
from src.models.train_prematch_model import train_and_persist
from src.utils.logging_setup import setup_logging

log = setup_logging("scripts.daily")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="Skip the (slow) re-train step; just rebuild the watchlist.")
    args = p.parse_args()

    if not args.skip_train:
        log.info("training pre-match model…")
        metrics = train_and_persist()
        log.info("training done. accuracy=%.3f brier=%.3f",
                 metrics["blended"]["accuracy"], metrics["blended"]["brier"])

    log.info("building watchlist…")
    csv_path, json_path = export()
    log.info("watchlist ready: %s", json_path)


if __name__ == "__main__":
    main()
