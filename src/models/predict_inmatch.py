"""Inference wrapper for the in-match adjustment model.

Loads the calibrated classifier produced by
``train_inmatch_model.py`` and exposes a single function that
turns a standardized live_record into a P(player_a wins match)
prediction.

The mapping from live_record → feature vector mirrors what the
snapshot builder produced at training time. Fields not observed
in the live feed (Sackmann-style per-point flags the dashboard
provider may not surface) fall back to sensible neutrals so a
sparse record still yields a usable probability, just with the
model effectively defaulting toward 0.5 on those dimensions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..features.build_pbp_snapshots import FEATURE_COLUMNS
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.predict_inmatch")


def _artifact_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"]) / "inmatch_model.joblib"


@lru_cache(maxsize=1)
def _load_model():
    path = _artifact_path()
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 — surface any deserialize problem
        log.warning("failed to load %s: %s", path, exc)
        return None


def model_available() -> bool:
    return _load_model() is not None


def _live_to_features(rec: dict[str, Any]) -> np.ndarray:
    """Map a live_record dict (as produced by ``standardize`` in
    ``build_live_features``) into the model's feature vector. Fields
    that the provider didn't populate fall back to neutrals."""
    set_a = float(rec.get("set_score_a") or 0)
    set_b = float(rec.get("set_score_b") or 0)
    # Current set's game count isn't surfaced explicitly on the
    # canonical live_record — fall back to 0 if unavailable.
    g_a = float(rec.get("current_set_games_a") or 0)
    g_b = float(rec.get("current_set_games_b") or 0)
    current_set = float(rec.get("current_set") or (set_a + set_b + 1))
    best_of = float(rec.get("best_of") or 3)
    fs_a = float(rec.get("first_serve_pct_a") if rec.get("first_serve_pct_a") is not None else 0.6)
    fs_b = float(rec.get("first_serve_pct_b") if rec.get("first_serve_pct_b") is not None else 0.6)
    fs_won_a = float(rec.get("first_serve_won_pct_a") if rec.get("first_serve_won_pct_a") is not None else 0.7)
    fs_won_b = float(rec.get("first_serve_won_pct_b") if rec.get("first_serve_won_pct_b") is not None else 0.7)
    ss_won_a = float(rec.get("second_serve_won_pct_a") if rec.get("second_serve_won_pct_a") is not None else 0.55)
    ss_won_b = float(rec.get("second_serve_won_pct_b") if rec.get("second_serve_won_pct_b") is not None else 0.55)
    aces_a = float(rec.get("aces_a") or 0)
    aces_b = float(rec.get("aces_b") or 0)
    dfs_a = float(rec.get("double_faults_a") or 0)
    dfs_b = float(rec.get("double_faults_b") or 0)
    unf_a = float(rec.get("unforced_errors_a") or 0)
    unf_b = float(rec.get("unforced_errors_b") or 0)
    bp_c_a = float(rec.get("break_points_created_a") or 0)
    bp_c_b = float(rec.get("break_points_created_b") or 0)
    bp_w_a = float(rec.get("break_points_won_a") or 0)
    bp_w_b = float(rec.get("break_points_won_b") or 0)
    bp_conv_a = (bp_w_a / bp_c_a) if bp_c_a else 0.0
    bp_conv_b = (bp_w_b / bp_c_b) if bp_c_b else 0.0
    last10_a = float(rec.get("last10_share_a") if rec.get("last10_share_a") is not None else 0.5)
    g_last3_a = float(rec.get("games_won_last_3_a") or 0)
    g_last3_b = float(rec.get("games_won_last_3_b") or 0)
    serving_a = 1.0 if rec.get("serving_a") else 0.0
    is_decider = 1.0 if rec.get("is_decider") else 0.0
    is_tiebreak = 1.0 if rec.get("is_tiebreak") else 0.0
    set_just_ended = 1.0 if rec.get("set_just_ended") else 0.0
    progress = float(rec.get("progress") or
                     ((set_a + set_b) / float(best_of) if best_of else 0.0))

    values: dict[str, float] = {
        "set_score_a": set_a,
        "set_score_b": set_b,
        "sets_diff": set_a - set_b,
        "current_set_games_a": g_a,
        "current_set_games_b": g_b,
        "games_diff": g_a - g_b,
        "current_set": current_set,
        "best_of": best_of,
        "serving_a": serving_a,
        "is_decider": is_decider,
        "is_tiebreak": is_tiebreak,
        "set_just_ended": set_just_ended,
        "progress": progress,
        "first_serve_pct_a": fs_a,
        "first_serve_pct_b": fs_b,
        "first_serve_won_pct_a": fs_won_a,
        "first_serve_won_pct_b": fs_won_b,
        "second_serve_won_pct_a": ss_won_a,
        "second_serve_won_pct_b": ss_won_b,
        "aces_a": aces_a,
        "aces_b": aces_b,
        "double_faults_a": dfs_a,
        "double_faults_b": dfs_b,
        "unforced_errors_a": unf_a,
        "unforced_errors_b": unf_b,
        "break_points_created_a": bp_c_a,
        "break_points_created_b": bp_c_b,
        "break_points_won_a": bp_w_a,
        "break_points_won_b": bp_w_b,
        "bp_conversion_a": bp_conv_a,
        "bp_conversion_b": bp_conv_b,
        "last10_share_a": last10_a,
        "games_won_last_3_a": g_last3_a,
        "games_won_last_3_b": g_last3_b,
    }
    return np.array([values[c] for c in FEATURE_COLUMNS], dtype=float).reshape(1, -1)


def predict(live_record: dict[str, Any]) -> float | None:
    """Return the trained model's P(player_a wins match) for this
    snapshot, or ``None`` if the artifact isn't installed."""
    model = _load_model()
    if model is None:
        return None
    X = _live_to_features(live_record)
    p = model.predict_proba(X)[0, 1]
    return float(np.clip(p, 1e-3, 1 - 1e-3))
