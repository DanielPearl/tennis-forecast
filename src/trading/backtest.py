"""Backtest the pre-match model against held-out matches.

We replay the chronological match log: at each match, the model
predicts using only the Elo state at that moment (built up from
prior-match results), and the bet is sized as a unit stake. Without
a real historical odds feed for tennis we approximate the closing
market by the Pinnacle-style "fair" line — i.e. the Elo-only
probability with a small noise term — to demonstrate the framework.
For real ROI numbers, plug in your captured closing prices via the
``closing_market_prob`` column in the input parquet.

Reported metrics:
  - accuracy / log_loss / Brier
  - ROI on simulated trades that fire when |edge| >= small_edge_min
  - win rate / average EV / max drawdown
  - splits by surface / level / favorite-vs-dog / edge bucket
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ..features.build_prematch_features import (
    build_full_panel, build_player_a_panel, select_features,
)
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging
from ..data.fetch_matches import fetch_all
from .ev import ev as ev_calc
from .signals import label_match

log = setup_logging("trading.backtest")


def _favorite(prob_a: float) -> str:
    return "fav" if prob_a >= 0.5 else "dog"


def _edge_bucket(edge: float) -> str:
    a = abs(edge)
    if a < 0.04: return "<4pp"
    if a < 0.08: return "4-8pp"
    if a < 0.15: return "8-15pp"
    return ">15pp"


def run_backtest(use_synthetic_market: bool = True) -> dict:
    cfg = load_config()
    t = cfg["trading"]
    art = resolve_path(cfg["paths"]["artifacts_dir"])

    bundle = joblib.load(art / "prematch_model.joblib")

    log.info("re-fetching match log for backtest…")
    matches = fetch_all()
    panel, _, _, _ = build_full_panel(matches, elo_cfg=cfg["elo"])
    oriented = build_player_a_panel(panel)

    cutoff = oriented["tourney_date"].max() - pd.DateOffset(
        months=int(cfg["model"]["test_window_months"])
    )
    test = oriented[oriented["tourney_date"] >= cutoff].copy()
    log.info("backtest rows: %d (cutoff %s)", len(test), cutoff.date())

    X = select_features(test)
    p_ens = bundle["ensemble"].predict_proba(X)[:, 1]
    p_log = bundle["logistic"].predict_proba(X[bundle["elo_only_features"]])[:, 1]
    p_blend = (
        bundle["blend_weight_ensemble"] * p_ens
        + bundle["blend_weight_logistic"] * p_log
    )
    p_blend = np.clip(p_blend, 0.01, 0.99)

    test["model_prob_a"] = p_blend

    # Synthetic market: a noisy, slightly-biased version of the model.
    # If you have actual closing prices, drop them in a column named
    # ``closing_market_prob`` before calling this function.
    if use_synthetic_market or "closing_market_prob" not in test.columns:
        rng = np.random.default_rng(seed=42)
        # Centered on Elo-only (strict baseline) with σ=4pp — wider than
        # a real book but enough to expose where the GBT adds value.
        baseline = 1.0 / (1.0 + 10.0 ** (-test["diff_elo_pre"].values / 400.0))
        test["closing_market_prob"] = np.clip(
            baseline + rng.normal(0, 0.04, size=len(baseline)), 0.05, 0.95
        )

    # Trade simulation — the same logic as the live signal labeler,
    # but without the live-state inputs (volatility=0, no injury, no
    # overreaction). This reflects pre-match-only ROI.
    pnl_rows = []
    for _, r in test.iterrows():
        side_market = r["closing_market_prob"] if r["model_prob_a"] >= 0.5 else (
            1.0 - r["closing_market_prob"]
        )
        side_model = r["model_prob_a"] if r["model_prob_a"] >= 0.5 else (
            1.0 - r["model_prob_a"]
        )
        side_won = (r["y"] == 1 and r["model_prob_a"] >= 0.5) or (
            r["y"] == 0 and r["model_prob_a"] < 0.5
        )
        evr = ev_calc(side_model, side_market, slippage=t["slippage_pct"])
        sig = label_match(
            r["model_prob_a"], r["closing_market_prob"],
            volatility=0.0, injury_flag=False,
            market_overreaction=False, rules_fired=[],
        )
        traded = sig.label in ("SMALL_EDGE", "STRONG_EDGE", "MARKET_OVERREACTION")
        # PnL: stake = bet_size; if we win, we make (1 - side_market - slippage)
        # per dollar; if we lose, we lose (side_market + slippage) per dollar.
        if traded:
            stake = float(t["bet_size"])
            slip = float(t["slippage_pct"])
            if side_won:
                pnl = stake * (1.0 - side_market - slip)
            else:
                pnl = -stake * (side_market + slip)
        else:
            pnl = 0.0
        pnl_rows.append({
            "date": r["tourney_date"], "tourney_name": r.get("tourney_name", ""),
            "surface": r.get("surface", ""), "level_rank": int(r["level_rank"]),
            "model_prob_a": float(r["model_prob_a"]),
            "market_prob_a": float(r["closing_market_prob"]),
            "edge": float(evr.edge),
            "ev_per_contract": float(evr.ev_per_contract),
            "label": sig.label, "traded": traded, "pnl": pnl,
            "y": int(r["y"]), "favorite_vs_dog": _favorite(r["model_prob_a"]),
            "edge_bucket": _edge_bucket(evr.edge),
        })

    df = pd.DataFrame(pnl_rows).sort_values("date").reset_index(drop=True)

    # Aggregate metrics
    p = test["model_prob_a"].values
    y = test["y"].values
    metrics = {
        "rows": int(len(test)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, p)),
    }
    traded = df[df["traded"]].copy()
    if len(traded):
        wins = (traded["pnl"] > 0).sum()
        metrics.update({
            "trades": int(len(traded)),
            "win_rate": float(wins / len(traded)),
            "roi": float(traded["pnl"].sum() / max(1.0, len(traded))),
            "avg_ev_at_trade": float(traded["ev_per_contract"].mean()),
            "max_drawdown": float(_max_drawdown(traded["pnl"].cumsum())),
            "by_surface": traded.groupby("surface")["pnl"].agg(["count", "sum", "mean"]).reset_index().to_dict(orient="records"),
            "by_level_rank": traded.groupby("level_rank")["pnl"].agg(["count", "sum", "mean"]).reset_index().to_dict(orient="records"),
            "by_favorite_vs_dog": traded.groupby("favorite_vs_dog")["pnl"].agg(["count", "sum", "mean"]).reset_index().to_dict(orient="records"),
            "by_edge_bucket": traded.groupby("edge_bucket")["pnl"].agg(["count", "sum", "mean"]).reset_index().to_dict(orient="records"),
        })
    else:
        metrics.update({"trades": 0, "win_rate": None, "roi": None})

    out_csv = resolve_path(cfg["paths"]["backtest_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("wrote %s (%d trade rows)", out_csv, len(df))
    log.info("backtest metrics: %s", json.dumps(
        {k: v for k, v in metrics.items() if not isinstance(v, list)},
        indent=2, default=str,
    ))
    return metrics


def _max_drawdown(curve: pd.Series) -> float:
    if len(curve) == 0:
        return 0.0
    high = curve.cummax()
    dd = (curve - high)
    return float(dd.min())


if __name__ == "__main__":
    run_backtest()
