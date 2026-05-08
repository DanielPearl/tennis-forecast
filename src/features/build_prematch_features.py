"""Build the pre-match feature panel.

A row is one match. Columns include Elo features (added by
``elo.build_elo_features``), serve/return rolling stats, recent form,
H2H, days-of-rest, tournament level, round, and ranking-derived
features. We orient everything from "player_a" perspective and stamp
``y = 1`` if player_a wins, then mirror each row so the model sees
both orientations equally — this kills the orientation bias that the
raw Sackmann winner/loser layout otherwise injects.

Why a deliberately small panel: tennis match data is shallow (a
top-tour player has ~60 main-draw matches per year), and tree models
overfit quickly on derived features that correlate with Elo. We keep
the panel tight and let the boosted ensemble handle interactions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd

from .elo import build_elo_features, EloState


# Tournament-level encoding. Sackmann uses single-letter codes.
_LEVEL_RANK = {
    "G": 4,   # Grand Slam
    "M": 3,   # Masters / WTA-1000
    "A": 2,   # ATP-500 / WTA-500
    "F": 4,   # Tour Finals
    "D": 1,   # Davis Cup
    "C": 1,   # Challenger
    "S": 1,   # Satellite
    "T": 1,   # Other
}


def _round_rank(r: str) -> int:
    """Numeric ordering for round strings ("R128"=1 ... "F"=8)."""
    if not isinstance(r, str):
        return 0
    table = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6,
             "F": 8, "RR": 4, "BR": 6}
    return table.get(r, 0)


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.where(b > 0, a / np.maximum(b, 1e-9), 0.0)
    return out


def _serve_return_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row serve/return percentages computed from Sackmann's
    raw point counts. Output columns are oriented winner/loser; the
    caller re-orients to player_a/player_b later."""
    out = df.copy()

    # First-serve %
    out["w_first_serve_pct"] = _safe_div(out["w_1stIn"], out["w_svpt"])
    out["l_first_serve_pct"] = _safe_div(out["l_1stIn"], out["l_svpt"])

    # First-serve points won
    out["w_first_serve_won_pct"] = _safe_div(out["w_1stWon"], out["w_1stIn"])
    out["l_first_serve_won_pct"] = _safe_div(out["l_1stWon"], out["l_1stIn"])

    # Second-serve points won
    second_in_w = out["w_svpt"] - out["w_1stIn"] - out["w_df"]
    second_in_l = out["l_svpt"] - out["l_1stIn"] - out["l_df"]
    out["w_second_serve_won_pct"] = _safe_div(out["w_2ndWon"], second_in_w)
    out["l_second_serve_won_pct"] = _safe_div(out["l_2ndWon"], second_in_l)

    # Break-point save %
    out["w_bp_saved_pct"] = _safe_div(out["w_bpSaved"], out["w_bpFaced"])
    out["l_bp_saved_pct"] = _safe_div(out["l_bpSaved"], out["l_bpFaced"])

    # Total return-points-won % = 1 - serve points won % from opp side
    serve_pts_won_w = out["w_1stWon"] + out["w_2ndWon"]
    serve_pts_won_l = out["l_1stWon"] + out["l_2ndWon"]
    out["w_serve_pts_won_pct"] = _safe_div(serve_pts_won_w, out["w_svpt"])
    out["l_serve_pts_won_pct"] = _safe_div(serve_pts_won_l, out["l_svpt"])
    # Return points won = points contested on opponent's serve minus the
    # serve-points-won that opponent won.
    out["w_return_pts_won_pct"] = 1.0 - out["l_serve_pts_won_pct"]
    out["l_return_pts_won_pct"] = 1.0 - out["w_serve_pts_won_pct"]
    return out


def _rolling_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-player rolling form and serve/return averages.

    We walk the dataframe in chronological order and, for each row,
    look up the player's prior N matches before applying the row's
    update. This guarantees no in-row leakage — the very feature the
    model sees is the value it would have had at match time.

    We only keep the small set of windows the dashboard actually needs
    (5 / 10) so the column count stays tractable.
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Per-player rolling buffers
    last5_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    last10_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    serve_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    return_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    bp_save_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_match_date: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple[str, str], int] = defaultdict(int)

    cols = {
        "w_form_last5": [], "l_form_last5": [],
        "w_form_last10": [], "l_form_last10": [],
        "w_avg_serve_pts_won_10": [], "l_avg_serve_pts_won_10": [],
        "w_avg_return_pts_won_10": [], "l_avg_return_pts_won_10": [],
        "w_avg_bp_saved_10": [], "l_avg_bp_saved_10": [],
        "w_days_rest": [], "l_days_rest": [],
        "h2h_w_wins_minus_l_wins": [],
    }

    def _avg(buf: deque, default: float = 0.6) -> float:
        return float(np.mean(buf)) if len(buf) else default

    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        date = row["tourney_date"]

        # Form
        cols["w_form_last5"].append(_avg(last5_results[w], 0.5))
        cols["l_form_last5"].append(_avg(last5_results[l], 0.5))
        cols["w_form_last10"].append(_avg(last10_results[w], 0.5))
        cols["l_form_last10"].append(_avg(last10_results[l], 0.5))

        # Rolling serve/return
        cols["w_avg_serve_pts_won_10"].append(_avg(serve_buf[w], 0.6))
        cols["l_avg_serve_pts_won_10"].append(_avg(serve_buf[l], 0.6))
        cols["w_avg_return_pts_won_10"].append(_avg(return_buf[w], 0.4))
        cols["l_avg_return_pts_won_10"].append(_avg(return_buf[l], 0.4))
        cols["w_avg_bp_saved_10"].append(_avg(bp_save_buf[w], 0.6))
        cols["l_avg_bp_saved_10"].append(_avg(bp_save_buf[l], 0.6))

        # Days rest
        wd = (date - last_match_date[w]).days if w in last_match_date else 7
        ld = (date - last_match_date[l]).days if l in last_match_date else 7
        # Cap at 60 — beyond that it's "returning from injury" not "rest".
        cols["w_days_rest"].append(min(60, max(0, wd)))
        cols["l_days_rest"].append(min(60, max(0, ld)))

        # H2H (winner-side perspective)
        key = tuple(sorted([w, l]))
        sign = 1 if key[0] == w else -1
        cols["h2h_w_wins_minus_l_wins"].append(h2h[key] * sign)

        # Now update buffers (post-match). The features above used
        # only state from before this match.
        last5_results[w].append(1.0); last5_results[l].append(0.0)
        last10_results[w].append(1.0); last10_results[l].append(0.0)
        if not pd.isna(row.get("w_serve_pts_won_pct")):
            serve_buf[w].append(float(row["w_serve_pts_won_pct"]))
        if not pd.isna(row.get("l_serve_pts_won_pct")):
            serve_buf[l].append(float(row["l_serve_pts_won_pct"]))
        if not pd.isna(row.get("w_return_pts_won_pct")):
            return_buf[w].append(float(row["w_return_pts_won_pct"]))
        if not pd.isna(row.get("l_return_pts_won_pct")):
            return_buf[l].append(float(row["l_return_pts_won_pct"]))
        if not pd.isna(row.get("w_bp_saved_pct")):
            bp_save_buf[w].append(float(row["w_bp_saved_pct"]))
        if not pd.isna(row.get("l_bp_saved_pct")):
            bp_save_buf[l].append(float(row["l_bp_saved_pct"]))
        last_match_date[w] = date; last_match_date[l] = date
        h2h[key] += 1 if key[0] == w else -1

    for k, v in cols.items():
        df[k] = v
    return df, h2h, last_match_date


def build_full_panel(matches: pd.DataFrame, elo_cfg: dict | None = None
                     ) -> tuple[pd.DataFrame, EloState, dict, dict]:
    """End-to-end: enrich Sackmann match data with everything the
    pre-match model wants. Returns the wide panel + the trained Elo
    state and accumulated H2H / last-match-date dicts (used at
    inference time)."""
    df = _serve_return_panel(matches)
    df, elo_state = build_elo_features(df, elo_cfg)
    df, h2h_table, last_match_date = _rolling_form_features(df)

    # Tournament + round encodings
    df["level_rank"] = df["tourney_level"].map(_LEVEL_RANK).fillna(1).astype(int)
    df["round_rank"] = df["round"].map(_round_rank).astype(int)

    # Rank diff (Sackmann ships winner_rank / loser_rank, sometimes blank).
    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"] = pd.to_numeric(df["loser_rank"], errors="coerce")
    df["rank_diff"] = (df["loser_rank"] - df["winner_rank"]).fillna(0.0)

    return df, elo_state, h2h_table, last_match_date


def build_player_a_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Re-orient the winner/loser-encoded panel into player_a/player_b
    with a balanced ``y`` label. Each match contributes two rows: one
    where player_a = winner (y=1) and one where player_a = loser (y=0).
    This removes the orientation bias before training."""
    a_pos = panel.copy()
    a_pos["player_a"] = panel["winner_name"]
    a_pos["player_b"] = panel["loser_name"]
    a_pos["y"] = 1
    _attach_oriented(a_pos, side="w_to_a", panel=panel)

    a_neg = panel.copy()
    a_neg["player_a"] = panel["loser_name"]
    a_neg["player_b"] = panel["winner_name"]
    a_neg["y"] = 0
    _attach_oriented(a_neg, side="l_to_a", panel=panel)

    return pd.concat([a_pos, a_neg], ignore_index=True)


def _attach_oriented(out: pd.DataFrame, side: str, panel: pd.DataFrame) -> None:
    """Copy the winner/loser-prefixed columns into player_a / player_b
    columns, depending on which orientation this row is."""
    pairs = [
        ("elo_pre", "winner_elo_pre", "loser_elo_pre"),
        ("surface_elo_pre", "winner_surface_elo_pre", "loser_surface_elo_pre"),
        ("form_last5", "w_form_last5", "l_form_last5"),
        ("form_last10", "w_form_last10", "l_form_last10"),
        ("avg_serve_pts_won_10", "w_avg_serve_pts_won_10", "l_avg_serve_pts_won_10"),
        ("avg_return_pts_won_10", "w_avg_return_pts_won_10", "l_avg_return_pts_won_10"),
        ("avg_bp_saved_10", "w_avg_bp_saved_10", "l_avg_bp_saved_10"),
        ("days_rest", "w_days_rest", "l_days_rest"),
    ]
    for short, w_col, l_col in pairs:
        if side == "w_to_a":
            out[f"a_{short}"] = panel[w_col].values
            out[f"b_{short}"] = panel[l_col].values
        else:
            out[f"a_{short}"] = panel[l_col].values
            out[f"b_{short}"] = panel[w_col].values

    # Diffs (a - b)
    for short, _, _ in pairs:
        out[f"diff_{short}"] = out[f"a_{short}"] - out[f"b_{short}"]

    # Shared (orientation-invariant) features pass through
    out["level_rank"] = panel["level_rank"].values
    out["round_rank"] = panel["round_rank"].values
    if side == "w_to_a":
        out["a_rank"] = pd.to_numeric(panel["winner_rank"], errors="coerce").values
        out["b_rank"] = pd.to_numeric(panel["loser_rank"], errors="coerce").values
        out["h2h_a_wins_minus_b_wins"] = panel["h2h_w_wins_minus_l_wins"].values
    else:
        out["a_rank"] = pd.to_numeric(panel["loser_rank"], errors="coerce").values
        out["b_rank"] = pd.to_numeric(panel["winner_rank"], errors="coerce").values
        out["h2h_a_wins_minus_b_wins"] = -panel["h2h_w_wins_minus_l_wins"].values
    # Rank diff: lower numeric rank = stronger player, so we want
    # b_rank - a_rank as "a's edge".
    out["rank_diff"] = (out["b_rank"].fillna(500) - out["a_rank"].fillna(500))


# Final feature list used by the model. Kept here so train + inference
# stay in lockstep — change in one place, both code paths see it.
PREMATCH_FEATURES = [
    "diff_elo_pre",
    "diff_surface_elo_pre",
    "diff_form_last5",
    "diff_form_last10",
    "diff_avg_serve_pts_won_10",
    "diff_avg_return_pts_won_10",
    "diff_avg_bp_saved_10",
    "diff_days_rest",
    "h2h_a_wins_minus_b_wins",
    "rank_diff",
    "level_rank",
    "round_rank",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[PREMATCH_FEATURES].fillna(0.0)
