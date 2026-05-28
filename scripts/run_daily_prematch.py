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
    p.add_argument("--retrain-inmatch", action="store_true",
                   help="Refresh slam PBP and retrain the in-match adjustment model. "
                        "Slam draws don't change daily, so this stays off by default.")
    args = p.parse_args()

    if not args.skip_train:
        log.info("training pre-match model…")
        metrics = train_and_persist()
        log.info("training done. accuracy=%.3f brier=%.3f",
                 metrics["blended"]["accuracy"], metrics["blended"]["brier"])

    if args.retrain_inmatch:
        # Local imports — these pull heavy deps (sklearn calibration)
        # and we only want them on the days we actually refresh PBP.
        from src.data.fetch_pbp import fetch_all as fetch_pbp_all
        from src.features.build_pbp_snapshots import build_snapshots
        from src.models.train_inmatch_model import train_and_eval as train_inmatch
        from src.utils.config import load_config, resolve_path

        log.info("refreshing slam PBP…")
        bundles = fetch_pbp_all()
        snaps = build_snapshots(bundles["matches"], bundles["points"])
        cfg = load_config()
        out = resolve_path(cfg["paths"]["processed_dir"]) / "pbp_snapshots.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        snaps.to_csv(out, index=False)
        log.info("retraining in-match model on %d snapshots…", len(snaps))
        inm = train_inmatch()
        log.info("in-match model: brier=%.4f vs rules baseline=%.4f",
                 inm["model"]["brier"], inm["rules_baseline"]["brier"])

    log.info("building watchlist…")
    csv_path, json_path = export()
    log.info("watchlist ready: %s", json_path)

    # Calibrate against our actual Kalshi bets — writes
    # ``data/processed/artifacts/kalshi_calibration.json`` with
    # Brier / log-loss / accuracy / calibration buckets over every
    # tennis settlement we've placed. The dashboard's Models tab
    # renders this alongside the held-out (Sackmann-based) metrics
    # so we always know whether the live model is staying accurate
    # on real money. Wrapped — any auth/network failure shouldn't
    # fail the retrain.
    try:
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        from src.data.kalshi_sync import (
            compute_calibration, fetch_enriched_bets,
        )
        from src.utils.config import load_config as _lc, resolve_path as _rp
        sim_state_paths = [
            _REPO / "data/outputs-live/sim_state.json",
            _REPO / "data/outputs/sim_state.json",
        ]
        rows = fetch_enriched_bets(sim_state_paths)
        metrics = compute_calibration(rows)
        metrics["generated_at"] = _dt.now(_tz.utc).isoformat(timespec="seconds")
        metrics["recent_bets"] = [
            {
                "ticker": r.get("ticker"),
                "settled_time": r.get("settled_time"),
                "side_player": r.get("side_player"),
                "winner_name": r.get("winner_name"),
                "tournament": r.get("tournament"),
                "won": bool(r.get("won")),
                "entry_model_prob": r.get("entry_model_prob"),
            }
            for r in rows
        ]
        artifacts_dir = _rp(_lc()["paths"]["artifacts_dir"])
        out = artifacts_dir / "kalshi_calibration.json"
        with out.open("w", encoding="utf-8") as f:
            _json.dump(metrics, f, indent=2)
        log.info(
            "kalshi calibration: n=%s brier=%s acc=%s win_rate=%s",
            metrics["n"],
            (f"{metrics['brier']:.4f}"
             if metrics["brier"] is not None else "—"),
            (f"{metrics['accuracy']:.3f}"
             if metrics["accuracy"] is not None else "—"),
            (f"{metrics['win_rate']:.3f}"
             if metrics["win_rate"] is not None else "—"),
        )
    except Exception as exc:  # noqa: BLE001 — never fail the retrain
        log.warning("kalshi calibration sync failed: %s", exc)


if __name__ == "__main__":
    main()
