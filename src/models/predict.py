"""Pre-match inference for any (player_a, player_b, surface, ...) tuple.

Loads the persisted model bundle and Elo state, builds the feature
vector for the matchup, and returns the calibrated win probability for
``player_a``. The dashboard and the live-monitor both call this.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..features.elo import EloState, lookup_pair_features
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging
from .train_prematch_model import load_elo_state

log = setup_logging("models.predict")


# Round-trip the heavyweight artefacts once per process. Loading the
# model bundle is ~50ms but the Elo dict can be ~5MB across all tours,
# and the dashboard re-renders many times per minute.
_BUNDLE = None
_ELO: EloState | None = None
_H2H: dict | None = None
_LAST_MATCH: dict[str, pd.Timestamp] | None = None


def _artifacts_dir() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"])


def _ensure_loaded() -> None:
    global _BUNDLE, _ELO, _H2H, _LAST_MATCH
    if _BUNDLE is not None:
        return
    art = _artifacts_dir()
    bundle = joblib.load(art / "prematch_model.joblib")
    elo = load_elo_state(joblib.load(art / "elo_state.joblib"))
    h2h = joblib.load(art / "h2h_table.joblib")
    rest = joblib.load(art / "last_match_date.joblib")
    rest = {k: pd.Timestamp(v) for k, v in rest.items()}
    _BUNDLE = bundle
    _ELO = elo
    _H2H = h2h
    _LAST_MATCH = rest


def _h2h_diff(player_a: str, player_b: str) -> int:
    """Net H2H wins for player_a vs player_b."""
    assert _H2H is not None
    key = tuple(sorted([player_a, player_b]))
    raw = _H2H.get(key, 0)
    # Stored as +1 for first-name win in sorted order; flip if needed.
    return raw if key[0] == player_a else -raw


def _days_rest(player: str, ref: pd.Timestamp) -> float:
    assert _LAST_MATCH is not None
    last = _LAST_MATCH.get(player)
    if last is None:
        return 7.0
    delta = (ref - last).days
    return float(min(60, max(0, delta)))


def _level_rank(level: str) -> int:
    table = {"G": 4, "M": 3, "A": 2, "F": 4, "D": 1, "C": 1, "S": 1}
    return table.get(level, 1)


def _round_rank(r: str) -> int:
    table = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 8}
    return table.get(r, 0)


def predict_match(
    player_a: str,
    player_b: str,
    surface: str = "Hard",
    level: str = "A",
    round_: str = "R32",
    rank_a: float | None = None,
    rank_b: float | None = None,
    match_date: datetime | date | None = None,
) -> dict[str, Any]:
    """Return ``{prob_a, prob_b, elo_winprob_a, features}`` for the matchup."""
    _ensure_loaded()
    assert _BUNDLE is not None and _ELO is not None
    bundle = _BUNDLE
    if match_date is None:
        match_date = datetime.utcnow()
    if isinstance(match_date, datetime):
        ref = pd.Timestamp(match_date.date())
    else:
        ref = pd.Timestamp(match_date)

    elo_feats = lookup_pair_features(_ELO, player_a, player_b, surface)
    feats = {
        "diff_elo_pre": elo_feats["elo_diff"],
        "diff_surface_elo_pre": elo_feats["surface_elo_diff"],
        # Rolling form: we don't keep per-player buffers in the
        # persisted bundle for MVP — the Elo signal absorbs most of
        # the form information anyway. Set to 0 (= no informative
        # difference between the two players).
        "diff_form_last5": 0.0,
        "diff_form_last10": 0.0,
        "diff_avg_serve_pts_won_10": 0.0,
        "diff_avg_return_pts_won_10": 0.0,
        "diff_avg_bp_saved_10": 0.0,
        "diff_days_rest": _days_rest(player_a, ref) - _days_rest(player_b, ref),
        "h2h_a_wins_minus_b_wins": float(_h2h_diff(player_a, player_b)),
        "rank_diff": float((rank_b or 500) - (rank_a or 500)),
        "level_rank": float(_level_rank(level)),
        "round_rank": float(_round_rank(round_)),
    }
    X = pd.DataFrame([feats])[bundle["feature_list"]].fillna(0.0)
    p_ens = float(bundle["ensemble"].predict_proba(X)[0, 1])
    p_log = float(bundle["logistic"].predict_proba(X[bundle["elo_only_features"]])[0, 1])
    blended = (
        bundle["blend_weight_ensemble"] * p_ens
        + bundle["blend_weight_logistic"] * p_log
    )
    blended = max(0.01, min(0.99, blended))
    return {
        "prob_a": blended,
        "prob_b": 1.0 - blended,
        "elo_winprob_a": elo_feats["elo_winprob_a"],
        "feats": feats,
        "elo": elo_feats,
    }


def predict_with_elo_only(player_a: str, player_b: str, surface: str = "Hard"
                           ) -> dict[str, float]:
    """Last-resort fallback used when the trained model isn't available
    (e.g. dashboard running on a fresh droplet pre-train). Returns the
    raw Elo win-prob in the same shape as ``predict_match``."""
    _ensure_loaded() if (_BUNDLE is not None) else None
    if _ELO is None:
        # Even the elo bundle is missing — return a literal 50/50.
        return {"prob_a": 0.5, "prob_b": 0.5, "elo_winprob_a": 0.5}
    f = lookup_pair_features(_ELO, player_a, player_b, surface)
    p = max(0.05, min(0.95, f["elo_winprob_a"]))
    return {"prob_a": p, "prob_b": 1.0 - p, "elo_winprob_a": f["elo_winprob_a"]}


def safe_predict(*args, **kwargs) -> dict[str, Any]:
    """Try the trained model, fall back to Elo-only if any artefact
    is missing. The dashboard prefers always rendering *something*
    over erroring."""
    try:
        return predict_match(*args, **kwargs)
    except Exception as exc:
        log.warning("predict_match failed (%s); falling back to Elo-only", exc)
        try:
            player_a, player_b = args[0], args[1]
            surface = kwargs.get("surface", args[2] if len(args) > 2 else "Hard")
            return predict_with_elo_only(player_a, player_b, surface)
        except Exception:
            return {"prob_a": 0.5, "prob_b": 0.5, "elo_winprob_a": 0.5}
