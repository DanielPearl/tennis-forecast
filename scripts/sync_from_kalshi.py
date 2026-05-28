"""Sync our actual Kalshi tennis bets into the model-evaluation
pipeline. Writes ``data/processed/artifacts/kalshi_calibration.json``
with Brier / log-loss / accuracy / calibration buckets on every
settled bet we've placed — the honest "is the model still right on
real money?" signal the dashboard's Models tab surfaces.

Designed to be called from ``run_daily_prematch.py`` right after
the retrain so the metrics reflect the freshly-fitted model.

Standalone use::

  python scripts/sync_from_kalshi.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.data.kalshi_sync import compute_calibration, fetch_enriched_bets
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import setup_logging

log = setup_logging("scripts.sync_from_kalshi")


# The trading-dashboard process writes sim_state to two distinct
# locations depending on its mode. Both should be read by the
# calibration sync — most production bets live in outputs-live, but
# we keep outputs/ as a fallback so the sim-mode metrics aren't lost.
_DEFAULT_SIM_STATES = [
    "data/outputs-live/sim_state.json",
    "data/outputs/sim_state.json",
]


def main() -> None:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sim_paths = [_REPO / p for p in _DEFAULT_SIM_STATES]
    log.info("syncing Kalshi calibration from sim_states: %s",
             [str(p) for p in sim_paths])

    try:
        rows = fetch_enriched_bets(sim_paths)
    except Exception:  # noqa: BLE001 — surface any auth/network problem
        log.exception("fetch_enriched_bets failed; calibration not updated")
        return

    metrics = compute_calibration(rows)
    metrics["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Persist the per-bet rows alongside the metrics so the
    # dashboard's Models tab can show a small recent-bets table or a
    # per-bet scatter without re-fetching Kalshi on every render.
    # Keep only the fields actually needed downstream to stay small.
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

    out_path = artifacts_dir / "kalshi_calibration.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(
        "wrote %s — n=%s brier=%s acc=%s win_rate=%s calib_gap=%s",
        out_path,
        metrics["n"],
        f"{metrics['brier']:.4f}" if metrics["brier"] is not None else "—",
        f"{metrics['accuracy']:.3f}" if metrics["accuracy"] is not None else "—",
        f"{metrics['win_rate']:.3f}" if metrics["win_rate"] is not None else "—",
        f"{metrics['calibration_gap']:+.3f}"
        if metrics["calibration_gap"] is not None else "—",
    )


if __name__ == "__main__":
    main()
