"""Pull our actual Kalshi tennis bets + join with the model's
pre-bet predictions, so we can measure how the live model is
performing on real money (vs the held-out Sackmann eval).

Data flow:
  Kalshi /portfolio/settlements  -> outcome (yes / no resolved)
  Kalshi /portfolio/fills        -> the price we paid + side held
  sim_state.json closed_positions -> entry_model_prob + player names

For each settled tennis bet we end up with one row carrying
``entry_model_prob`` (what the model thought) and ``won`` (what
actually happened). That's everything ``compute_calibration`` needs
to compute Brier / log-loss / accuracy / calibration gap.

Phantom settlements (canceled orders where we held 0 contracts at
settle) are dropped — same logic as the dashboard's history view.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..utils.logging_setup import setup_logging

log = setup_logging("data.kalshi_sync")


def _client():
    """Build a Kalshi SDK client from the bot's env vars."""
    from kalshi_sdk import KalshiClient  # local import — heavy SDK
    api = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api or not pkey:
        raise RuntimeError(
            "KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset; "
            "can't pull Kalshi history"
        )
    return KalshiClient(api_key_id=api, private_key_path=pkey)


def _load_sim_states(paths: list[Path]) -> dict[str, dict]:
    """Merge any number of sim_state.json files into one
    ``{ticker -> closed_position_dict}`` lookup. Order matters: later
    paths override earlier ones for the same ticker — but in practice
    each ticker appears in exactly one of (live sim_state.json, paper
    sim_state.json)."""
    out: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("sim_state %s unreadable: %s", p, exc)
            continue
        for cpos in data.get("closed_positions") or []:
            t = cpos.get("ticker")
            if t:
                out[t] = cpos
    return out


def fetch_enriched_bets(sim_state_paths: list[Path]) -> list[dict[str, Any]]:
    """Pull every Kalshi tennis settlement, join with the local
    sim_state(s) for entry_model_prob + full player names, and
    return enriched rows.

    Each row carries:
      ticker, event_ticker, settled_time, won (bool),
      entry_model_prob (float | None — None when sim_state has no
      record of this ticker, e.g. a bet placed before the bot logged
      its sim_state schema), side_player (the player we bet on),
      winner_name + loser_name (derived from sim_state + Kalshi
      market_result), tournament.

    Phantom settlements (yes_count_fp == no_count_fp == 0) are skipped
    — Kalshi sometimes emits a settlement row for a ticker we
    interacted with but never actually held at settle (e.g. an order
    canceled before fill), and crediting those as wins/losses is
    misleading.
    """
    c = _client()
    settlements = (
        c.iter_settlements(ticker_prefix="KXATPMATCH") +
        c.iter_settlements(ticker_prefix="KXWTAMATCH")
    )
    sim_by_ticker = _load_sim_states(sim_state_paths)

    rows: list[dict[str, Any]] = []
    for s in settlements:
        ticker = s.get("ticker") or ""
        yes_n = float(s.get("yes_count_fp") or 0)
        no_n = float(s.get("no_count_fp") or 0)
        if yes_n <= 0 and no_n <= 0:
            continue
        side_held = "yes" if yes_n > 0 else "no"
        won = (s.get("market_result") == side_held)
        sim = sim_by_ticker.get(ticker, {})
        side_player = sim.get("side_player") or ""
        player_a = sim.get("player_a") or ""
        player_b = sim.get("player_b") or ""
        # Resolve winner/loser by name when we have both players locally.
        winner_name: str | None = None
        loser_name: str | None = None
        if side_player and player_a and player_b:
            if won:
                winner_name = side_player
                loser_name = (player_a if side_player == player_b else player_b)
            else:
                loser_name = side_player
                winner_name = (player_a if side_player == player_b else player_b)
        rows.append({
            "ticker": ticker,
            "event_ticker": s.get("event_ticker") or "",
            "settled_time": s.get("settled_time") or "",
            "won": won,
            "entry_model_prob": sim.get("entry_model_prob"),
            "side_player": side_player,
            "player_a": player_a,
            "player_b": player_b,
            "tournament": sim.get("tournament") or "",
            "surface": sim.get("surface") or "",
            "winner_name": winner_name,
            "loser_name": loser_name,
        })
    log.info("fetched %d enriched tennis bets from Kalshi (joined with %d sim_state tickers)",
             len(rows), len(sim_by_ticker))
    return rows


def compute_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute calibration metrics from the enriched bet rows.
    Returns: n, brier, log_loss, accuracy, win_rate,
    mean_predicted_prob, calibration_gap (mean_pred − win_rate).

    Reliability-diagram buckets are a list of {bin_lo, bin_hi, n,
    predicted_mean, actual_win_rate} for the dashboard's mini
    calibration chart.
    """
    import math
    scored = [r for r in rows if r.get("entry_model_prob") is not None]
    if not scored:
        return {
            "n": 0,
            "brier": None, "log_loss": None, "accuracy": None,
            "win_rate": None, "mean_predicted_prob": None,
            "calibration_gap": None, "buckets": [],
        }
    ys = [1 if r["won"] else 0 for r in scored]
    ps = [float(r["entry_model_prob"]) for r in scored]
    n = len(scored)
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    ll = -sum(
        (y * math.log(max(min(p, 1 - 1e-6), 1e-6))
         + (1 - y) * math.log(max(min(1 - p, 1 - 1e-6), 1e-6)))
        for p, y in zip(ps, ys)
    ) / n
    accuracy = sum(int((p >= 0.5) == bool(y)) for p, y in zip(ps, ys)) / n
    win_rate = sum(ys) / n
    mean_pred = sum(ps) / n
    # Reliability-diagram buckets.
    edges = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.01]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = [(p, y) for p, y in zip(ps, ys) if lo <= p < hi]
        if not in_bin:
            continue
        bin_ps = [p for p, _ in in_bin]
        bin_ys = [y for _, y in in_bin]
        buckets.append({
            "bin_lo": lo,
            "bin_hi": min(hi, 1.0),
            "n": len(in_bin),
            "predicted_mean": sum(bin_ps) / len(bin_ps),
            "actual_win_rate": sum(bin_ys) / len(bin_ys),
        })
    return {
        "n": n,
        "brier": brier,
        "log_loss": ll,
        "accuracy": accuracy,
        "win_rate": win_rate,
        "mean_predicted_prob": mean_pred,
        "calibration_gap": mean_pred - win_rate,
        "buckets": buckets,
    }
