"""Pre-match inference for any (player_a, player_b, surface, ...) tuple.

Loads the persisted model bundle and Elo state, builds the feature
vector for the matchup, and returns the calibrated win probability for
``player_a``. The dashboard and the live-monitor both call this.
"""
from __future__ import annotations

import difflib
import unicodedata
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


# ---------------------------------------------------------------------
# Name normalization. Kalshi and Sackmann disagree on hyphens vs
# spaces (Felix Auger-Aliassime vs Felix Auger Aliassime), on
# capitalization for Mc/Mac names (McCartney vs Mccartney), and on
# diacritics (Muchová vs Muchova). Prior to this normalizer the
# inference path was resolving 12% of Kalshi players to the Elo
# default (rating 1500) — the audit found "unknown" hits on names
# that WERE in the training data under a slightly different spelling.
#
# Resolution order: exact hit → normalised hit (lowercase, strip
# diacritics, hyphens→spaces, strip punctuation) → fuzzy close-match
# above _FUZZY_CUTOFF. Anything that still can't resolve gets logged
# once so we can catch new mismatches early instead of silently
# defaulting to Elo 1500.
# ---------------------------------------------------------------------
_FUZZY_CUTOFF = 0.90
# Populated by _rebuild_name_index() on each artifact load.
_NAME_INDEX: dict[str, str] = {}
_UNRESOLVED_LOGGED: set[str] = set()


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                    if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    """Canonical normalization: lowercase, no diacritics, hyphens→
    spaces, no punctuation, collapsed whitespace."""
    n = _strip_diacritics(s).lower().replace("-", " ")
    n = "".join(c if c.isalnum() or c.isspace() else " " for c in n)
    return " ".join(n.split())


def _rebuild_name_index(elo_state: EloState) -> None:
    """Refresh the normalized→canonical name map. Called from
    _ensure_loaded whenever a fresh Elo state is loaded from disk.
    First entry per key wins so ties resolve deterministically."""
    global _NAME_INDEX, _UNRESOLVED_LOGGED
    idx: dict[str, str] = {}
    for canonical in elo_state.overall.keys():
        key = _norm(canonical)
        if key and key not in idx:
            idx[key] = canonical
    _NAME_INDEX = idx
    _UNRESOLVED_LOGGED = set()


def _resolve_name(name: str) -> str:
    """Return the training-set canonical name for ``name``, or the
    input verbatim when nothing resolves. Every downstream lookup
    (Elo, H2H, rolling, last-match-date) is keyed on the canonical
    form, so calling this once at match-time is enough."""
    if not name or _ELO is None:
        return name
    if name in _ELO.overall:
        return name  # already canonical — fast path
    n = _norm(name)
    hit = _NAME_INDEX.get(n)
    if hit is not None:
        return hit
    # Fuzzy fallback — kept conservative so we don't collapse two
    # different players into one. 0.90 catches Soonwoo Kwon → Soon Woo
    # Kwon but rejects looser matches.
    candidates = difflib.get_close_matches(
        n, _NAME_INDEX.keys(), n=1, cutoff=_FUZZY_CUTOFF,
    )
    if candidates:
        return _NAME_INDEX[candidates[0]]
    # Nothing resolved — log ONCE per name so we don't spam every
    # tick. Falling back to the input means downstream lookups miss
    # and the feature values default to their neutral priors.
    if name not in _UNRESOLVED_LOGGED:
        log.warning("name-normalizer: %r not found in Elo state "
                     "(no exact, normalized, or fuzzy match) — "
                     "prediction will use default features",
                     name)
        _UNRESOLVED_LOGGED.add(name)
    return name


# Round-trip the heavyweight artefacts once per process — loading the
# model bundle is ~50ms but the Elo dict can be ~5MB across all tours,
# and the dashboard re-renders many times per minute.
#
# Auto-reload: on every ``_ensure_loaded`` call we stat the bundle's
# joblib and reload all four artifacts when the file is newer than
# the cached load time. A stat() call is microseconds; this is the
# only sane way to make the daily retrain timer (which writes a
# fresh bundle at 05:00 UTC) actually take effect in the live
# dashboard process without requiring an operator-driven restart.
_BUNDLE = None
_BUNDLE_MTIME: float = 0.0
_ELO: EloState | None = None
_H2H: dict | None = None
_LAST_MATCH: dict[str, pd.Timestamp] | None = None
_ROLLING: dict[str, dict[str, float]] | None = None


def _artifacts_dir() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"])


def _ensure_loaded() -> None:
    global _BUNDLE, _BUNDLE_MTIME, _ELO, _H2H, _LAST_MATCH
    art = _artifacts_dir()
    model_path = art / "prematch_model.joblib"
    try:
        current_mtime = model_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0
    # Cache hit: bundle loaded AND artifact hasn't been rewritten.
    if _BUNDLE is not None and current_mtime == _BUNDLE_MTIME:
        return
    bundle = joblib.load(model_path)
    elo = load_elo_state(joblib.load(art / "elo_state.joblib"))
    h2h = joblib.load(art / "h2h_table.joblib")
    rest = joblib.load(art / "last_match_date.joblib")
    rest = {k: pd.Timestamp(v) for k, v in rest.items()}
    # Rolling form state — present from the Phase-2 retrain onward;
    # gracefully empty for bundles trained before the snapshot existed
    # (predict() falls back to the same defaults the trainer's
    # _avg helper uses).
    rolling_path = art / "rolling_form_state.joblib"
    rolling = joblib.load(rolling_path) if rolling_path.exists() else {}
    was_reload = _BUNDLE is not None
    _BUNDLE = bundle
    _BUNDLE_MTIME = current_mtime
    _ELO = elo
    _H2H = h2h
    _LAST_MATCH = rest
    global _ROLLING
    _ROLLING = rolling
    # Rebuild the canonical-name index whenever a fresh Elo state
    # loads — the training-set player pool changes between retrains.
    _rebuild_name_index(elo)
    if was_reload:
        log.info(
            "predict bundle reloaded from disk (mtime=%s) — daily "
            "retrain has been picked up without process restart",
            pd.Timestamp(current_mtime, unit="s")
            .strftime("%Y-%m-%d %H:%M:%S"),
        )


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

    # Resolve Kalshi-formatted names to their training-set canonical
    # form so downstream Elo / H2H / rolling / last-match-date lookups
    # actually hit. Before the 2026-07-08 normalizer landed, ~12% of
    # Kalshi players were resolving to the Elo default (rating 1500)
    # purely because of hyphen-vs-space or accent differences.
    player_a = _resolve_name(player_a)
    player_b = _resolve_name(player_b)

    elo_feats = lookup_pair_features(_ELO, player_a, player_b, surface)
    # Phase-2: look up each player's persisted rolling-form snapshot.
    # When the bundle predates the snapshot OR the player isn't in
    # the snapshot (debutant, name mismatch), fall back to the same
    # neutral defaults the trainer's _avg helper used so the trained
    # weights see a familiar input.
    ra = (_ROLLING or {}).get(player_a, {})
    rb = (_ROLLING or {}).get(player_b, {})
    def _diff(key: str, default: float) -> float:
        return float(ra.get(key, default)) - float(rb.get(key, default))
    # Populate every feature the trainer might have kept in its final
    # ``feature_list``. Real values where we have them (rolling snapshot
    # + Elo + H2H + last-match dates); neutral defaults for the columns
    # whose backing state isn't persisted per player yet — the trainer
    # imputed the same defaults during fit so the model sees a familiar
    # input distribution rather than crashing on a KeyError.
    #
    # This function was silently returning ``model_source: elo_only``
    # (via safe_predict's exception path) for every prediction before
    # the fix, because the trained bundle carries ~38 features and
    # this dict only listed 12. See the 2026-07-07 audit: 100% of
    # watchlist rows fell back to Elo-only, meaning the 695-trade
    # paper history was a graded run of the Elo-only baseline, not
    # the full ensemble.
    ranks_diff = float((rank_b or 500) - (rank_a or 500))
    log_rank_pts = 0.0  # rank points not tracked at inference — neutral
    feats = {
        # ── Elo core ─────────────────────────────────────────────────
        "diff_elo_pre": elo_feats["elo_diff"],
        "diff_surface_elo_pre": elo_feats["surface_elo_diff"],
        "diff_elo_delta_30d": 0.0,
        "diff_elo_delta_90d": 0.0,
        # ── Form / serve / return (from rolling snapshot) ───────────
        "diff_form_last5": _diff("form_last5", 0.5),
        "diff_form_last10": _diff("form_last10", 0.5),
        # surface_form_last5 not persisted per player — use overall
        # form as a proxy so the trained tree isn't fed a hard 0.
        "diff_surface_form_last5": _diff("form_last5", 0.5),
        "diff_avg_serve_pts_won_10": _diff("avg_serve_pts_won_10", 0.6),
        "diff_avg_return_pts_won_10": _diff("avg_return_pts_won_10", 0.4),
        "diff_avg_bp_saved_10": _diff("avg_bp_saved_10", 0.6),
        # ── Rest / fatigue / layoff ────────────────────────────────
        "diff_days_rest": _days_rest(player_a, ref) - _days_rest(player_b, ref),
        "diff_matches_last_7d": 0.0,
        "diff_matches_last_14d": 0.0,
        "diff_layoff_days": 0.0,
        # ── Head-to-head ───────────────────────────────────────────
        "h2h_a_wins_minus_b_wins": float(_h2h_diff(player_a, player_b)),
        # Recency/surface H2H not tracked separately; overall H2H is
        # the safest proxy.
        "h2h_a_wins_minus_b_wins_recency": float(_h2h_diff(player_a, player_b)),
        "h2h_a_wins_minus_b_wins_on_surface": float(_h2h_diff(player_a, player_b)),
        # ── Rank ────────────────────────────────────────────────────
        "rank_diff": ranks_diff,
        "log_rank_points_diff": log_rank_pts,
        # ── Tournament context ─────────────────────────────────────
        "level_rank": float(_level_rank(level)),
        "round_rank": float(_round_rank(round_)),
        "best_of": 3.0,  # Kalshi tour markets — bo3 except finals
        "draw_size": 32.0,  # median tour-level draw
        "diff_round_win_pct": 0.0,
        # ── Score-derived rates (rolling snapshot) ─────────────────
        "diff_retirement_rate_20": _diff("retirement_rate_20", 0.02),
        "diff_tiebreak_win_pct": _diff("tiebreak_win_pct", 0.5),
        "diff_comeback_rate": _diff("comeback_rate", 0.1),
        "diff_choke_rate": _diff("choke_rate", 0.1),
        "diff_set_win_pct": _diff("set_win_pct", 0.5),
        "diff_avg_opp_elo_10": _diff("avg_opp_elo_10", 1500.0),
        "diff_avg_match_min_10": _diff("avg_match_min_10", 90.0),
        # ── Physical attributes (not tracked per player yet) ───────
        "diff_ht": 0.0,
        "diff_age": 0.0,
        "diff_is_lefty": 0.0,
        "a_is_lefty": 0.0,
        "b_is_lefty": 0.0,
        "a_age": 28.0,  # median tour-level age
        "b_age": 28.0,
    }
    # Only feed the columns the bundle actually asked for — extra keys
    # are harmless (pandas ignores them) but missing keys would crash
    # the sklearn column-order check.
    missing = [c for c in bundle["feature_list"] if c not in feats]
    for c in missing:
        # Anything unrecognised gets a 0 default so the pipeline never
        # crashes even if the trainer adds new features between
        # bundle-write and predict-loader upgrade.
        feats[c] = 0.0
    X = pd.DataFrame([feats])[bundle["feature_list"]].fillna(0.0)
    p_ens = float(bundle["ensemble"].predict_proba(X)[0, 1])
    p_log = float(bundle["logistic"].predict_proba(X[bundle["elo_only_features"]])[0, 1])
    blended = (
        bundle["blend_weight_ensemble"] * p_ens
        + bundle["blend_weight_logistic"] * p_log
    )
    blended = max(0.01, min(0.99, blended))
    # Apply the live-bet calibration layer — rescales over-confident
    # predictions toward their empirical win rate using the Platt fit
    # over our own Kalshi outcomes. Pass-through when no calibration
    # JSON exists yet; ramps in linearly with sample size.
    raw_blended = blended
    try:
        from . import calibration_layer
        calibration_layer_path = _artifacts_dir() / "kalshi_calibration.json"
        blended = calibration_layer.recalibrate(
            blended, calibration_layer_path,
        )
    except Exception:  # noqa: BLE001 — never let calibration fail prediction
        log.warning("calibration_layer failed; serving raw model prob",
                    exc_info=True)
    return {
        "prob_a": blended,
        "prob_b": 1.0 - blended,
        "raw_model_prob_a": raw_blended,  # pre-recalibration view
        "elo_winprob_a": elo_feats["elo_winprob_a"],
        "feats": feats,
        "elo": elo_feats,
        # ``model_source`` propagates through the watchlist so the live
        # executor can refuse to place orders when the prob came from
        # a fallback path (Elo-only or literal 50/50). Anything other
        # than ``"trained"`` means "no real prediction; don't bet".
        "model_source": "trained",
    }


def predict_with_elo_only(player_a: str, player_b: str, surface: str = "Hard"
                           ) -> dict[str, float]:
    """Last-resort fallback used when the trained model isn't available
    (e.g. dashboard running on a fresh droplet pre-train). Returns the
    raw Elo win-prob in the same shape as ``predict_match``."""
    _ensure_loaded() if (_BUNDLE is not None) else None
    if _ELO is None:
        # Even the elo bundle is missing — return a literal 50/50.
        return {"prob_a": 0.5, "prob_b": 0.5, "elo_winprob_a": 0.5,
                "model_source": "default_50_50"}
    player_a = _resolve_name(player_a)
    player_b = _resolve_name(player_b)
    f = lookup_pair_features(_ELO, player_a, player_b, surface)
    p = max(0.05, min(0.95, f["elo_winprob_a"]))
    return {"prob_a": p, "prob_b": 1.0 - p,
            "elo_winprob_a": f["elo_winprob_a"],
            "model_source": "elo_only"}


def safe_predict(*args, **kwargs) -> dict[str, Any]:
    """Try the trained model, fall back to Elo-only if any artefact
    is missing. The dashboard prefers always rendering *something*
    over erroring. The returned dict's ``model_source`` field tells
    the caller which path produced the probability (``"trained"`` /
    ``"elo_only"`` / ``"default_50_50"``) so safety gates like the
    live executor's "no real prediction → no bets" rule can refuse
    fallback-derived rows."""
    try:
        return predict_match(*args, **kwargs)
    except Exception as exc:
        log.warning("predict_match failed (%s); falling back to Elo-only", exc)
        try:
            player_a, player_b = args[0], args[1]
            surface = kwargs.get("surface", args[2] if len(args) > 2 else "Hard")
            return predict_with_elo_only(player_a, player_b, surface)
        except Exception:
            return {"prob_a": 0.5, "prob_b": 0.5, "elo_winprob_a": 0.5,
                    "model_source": "default_50_50"}
