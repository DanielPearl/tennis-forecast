"""Train the pre-match win-probability model.

Pipeline:
  1. Fetch + clean Sackmann data (data/fetch_matches.py)
  2. Build full panel (features/build_prematch_features.py)
  3. Mirror to player_a orientation
  4. Hold out the last ``test_window_months`` for evaluation
  5. Train: Elo-only logistic baseline + GBT/XGB ensemble; calibrate
     on a holdout slice; report Brier/log-loss/accuracy
  6. Persist: model + Elo state + H2H table + last-match-date dict

We deliberately keep the model small. Tennis match data is shallow
(~50 main-draw matches per top-tour player per year), and tree models
overfit fast on derived features. A logistic over Elo + a calibrated
HGB ensemble gets within a few hundredths of a Brier score of much
fancier setups while training in seconds.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ..data.fetch_matches import fetch_all, save_clean
from ..features.build_prematch_features import (
    PREMATCH_FEATURES, build_full_panel, build_player_a_panel, select_features,
)
from ..features.elo import EloState
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.train_prematch")


def _try_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


def _split_by_date(df: pd.DataFrame, months: int):
    cutoff = df["tourney_date"].max() - pd.DateOffset(months=months)
    train = df[df["tourney_date"] < cutoff].copy()
    test = df[df["tourney_date"] >= cutoff].copy()
    return train, test, cutoff


def _calibrate(model, X, y, holdout_frac: float = 0.2):
    """Hold out the tail of training data for calibration. Sigmoid is
    a safe default for tree ensembles whose raw scores are pushed to
    the extremes.

    sklearn 1.8 dropped ``cv='prefit'`` and replaced it with
    ``FrozenEstimator``; older versions (<1.6) only know ``'prefit'``.
    Try the new API first, fall back to the old one — keeps the bot
    portable across droplet pip-pinned versions.
    """
    n = len(X)
    cut = int(n * (1 - holdout_frac))
    X_fit, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_fit, y_cal = y[:cut], y[cut:]
    model.fit(X_fit, y_fit)
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except Exception:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def train_and_persist() -> dict[str, Any]:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching match history…")
    matches = fetch_all()
    save_clean(matches)

    log.info("building feature panel (Elo + form + serve/return)…")
    panel, elo_state, h2h_table, last_match_date = build_full_panel(
        matches, elo_cfg=cfg["elo"]
    )
    oriented = build_player_a_panel(panel)
    oriented = oriented.sort_values("tourney_date").reset_index(drop=True)

    train, test, cutoff = _split_by_date(
        oriented, months=int(cfg["model"]["test_window_months"])
    )
    log.info("train rows %d / test rows %d (cutoff %s)",
             len(train), len(test), cutoff.date())

    X_train = select_features(train); y_train = train["y"].values
    X_test = select_features(test); y_test = test["y"].values

    # 1) Elo-only logistic baseline (interpretable, fast). Most of the
    #    signal in tennis is in Elo; keeping a logistic baseline lets
    #    us check the GBT actually adds value before shipping it.
    elo_only_features = ["diff_elo_pre", "diff_surface_elo_pre"]
    base = LogisticRegression(max_iter=400)
    base.fit(X_train[elo_only_features], y_train)
    base_p = base.predict_proba(X_test[elo_only_features])[:, 1]

    # 2) Boosted ensemble — XGBoost when available, sklearn HGB fallback.
    xgb = _try_xgb() if cfg["model"]["ensemble"]["use_xgboost_if_available"] else None
    if xgb is not None:
        clf = xgb.XGBClassifier(
            n_estimators=int(cfg["model"]["ensemble"]["n_estimators"]),
            learning_rate=float(cfg["model"]["ensemble"]["learning_rate"]),
            max_depth=int(cfg["model"]["ensemble"]["max_depth"]),
            tree_method="hist",
            random_state=int(cfg["model"]["random_state"]),
            eval_metric="logloss",
            n_jobs=2,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=int(cfg["model"]["ensemble"]["n_estimators"]),
            learning_rate=float(cfg["model"]["ensemble"]["learning_rate"]),
            max_depth=int(cfg["model"]["ensemble"]["max_depth"]),
            random_state=int(cfg["model"]["random_state"]),
        )

    cal_clf = _calibrate(clf, X_train, y_train, holdout_frac=0.2)
    ens_p = cal_clf.predict_proba(X_test)[:, 1]

    # 3) Blend: 70% calibrated GBT + 30% logistic baseline. The logistic
    #    is a stabilizer — when the GBT overfits a particular cohort,
    #    the logistic pulls the prediction back toward the Elo prior.
    blend_p = 0.70 * ens_p + 0.30 * base_p

    metrics = {
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "cutoff_date": str(cutoff.date()),
        "elo_only": _eval(y_test, base_p),
        "ensemble": _eval(y_test, ens_p),
        "blended": _eval(y_test, blend_p),
        "xgboost": xgb is not None,
    }
    log.info("metrics: %s", json.dumps(metrics, indent=2))

    bundle = {
        "ensemble": cal_clf,
        "logistic": base,
        "feature_list": PREMATCH_FEATURES,
        "elo_only_features": elo_only_features,
        "blend_weight_ensemble": 0.70,
        "blend_weight_logistic": 0.30,
        "metrics": metrics,
    }
    model_path = artifacts_dir / "prematch_model.joblib"
    joblib.dump(bundle, model_path)
    log.info("wrote model bundle → %s", model_path)

    # Persist Elo state + H2H + last-match-date so inference can build
    # features for any (player_a, player_b, surface) triple without
    # re-running the whole pipeline.
    state_path = artifacts_dir / "elo_state.joblib"
    joblib.dump(_elo_state_to_dict(elo_state), state_path)

    h2h_path = artifacts_dir / "h2h_table.joblib"
    joblib.dump(dict(h2h_table), h2h_path)

    rest_path = artifacts_dir / "last_match_date.joblib"
    joblib.dump({k: pd.Timestamp(v).isoformat() for k, v in last_match_date.items()},
                rest_path)

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _eval(y_true, y_prob) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _elo_state_to_dict(state: EloState) -> dict[str, Any]:
    return {
        "default_rating": state.default_rating,
        "k_base": state.k_base,
        "k_floor": state.k_floor,
        "k_decay_matches": state.k_decay_matches,
        "surface_k_multiplier": state.surface_k_multiplier,
        "surface_blend": state.surface_blend,
        "overall": dict(state.overall),
        "surface": {f"{k[0]}|{k[1]}": v for k, v in state.surface.items()},
        "matches_played": dict(state.matches_played),
        "surface_matches": {f"{k[0]}|{k[1]}": v for k, v in state.surface_matches.items()},
    }


def load_elo_state(d: dict[str, Any]) -> EloState:
    s = EloState(
        default_rating=d["default_rating"],
        k_base=d["k_base"], k_floor=d["k_floor"],
        k_decay_matches=d["k_decay_matches"],
        surface_k_multiplier=d["surface_k_multiplier"],
        surface_blend=d["surface_blend"],
    )
    s.overall = dict(d["overall"])
    s.surface = {tuple(k.split("|", 1)): v for k, v in d["surface"].items()}
    s.matches_played.update(d["matches_played"])
    s.surface_matches.update({tuple(k.split("|", 1)): v
                              for k, v in d["surface_matches"].items()})
    return s


if __name__ == "__main__":
    train_and_persist()
