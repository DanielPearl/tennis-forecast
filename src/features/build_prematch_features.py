"""Build the pre-match feature panel.

A row is one match. Columns include Elo features (added by
``elo.build_elo_features``), serve/return rolling stats, recent form,
H2H, days-of-rest, tournament level, round, ranking-derived features,
plus an expanded panel of orthogonal-to-Elo signals introduced in the
2026-07 sweep: handedness, height, age, log rank-points, Elo momentum
(30d + 90d), fatigue (matches played in last 7 / 14 days), recency-
weighted H2H (2-year half-life), comeback / choke / tiebreak / set-win
% parsed from the score string, quality-of-recent-competition (avg
opponent Elo over last 10), retirement rate, layoff-since-last-
tournament, surface-specific form, round-specific win rate, and avg
match minutes.

We orient everything from "player_a" perspective and stamp ``y = 1``
if player_a wins, then mirror each row so the model sees both
orientations equally — this kills the orientation bias that the raw
Sackmann winner/loser layout otherwise injects.

Why the panel is now wide: with 350k matches across the full open era,
the HistGradientBoosting trainer has enough headroom that a broader
panel + permutation pruning outperforms the previous minimalist
approach. The training script writes feature_importance.csv, so any
new column that turns out to be pure noise gets flagged and can be
pruned in a follow-up without changing this file.
"""
from __future__ import annotations

import re
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


# Half-life for the recency-weighted H2H: two years. A H2H win five
# years ago carries ~17% of the weight of a H2H win today; a match
# from ten years ago carries ~3%.
_H2H_HALFLIFE_DAYS = 730.0


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


# ---------------------------------------------------------------------
# Score-string parsers. Sackmann's ``score`` column is compact ASCII
# like "6-3 5-7 7-6(4)", possibly suffixed with " RET" / " W/O" /
# " DEF" (walkover / injury / default). All three parsers below are
# tolerant of missing / malformed rows and fall through to safe
# defaults — the caller decides whether to record the observation.
# ---------------------------------------------------------------------
_SET_RE = re.compile(r"(\d+)-(\d+)(?:\((\d+)\))?")


def _score_meta(score: str) -> dict:
    """Return {sets_w, sets_l, tb_won_w, tb_played, first_set_w_won,
    retired}. Fields default to zero / False if unparseable."""
    out = {"sets_w": 0, "sets_l": 0, "tb_won_w": 0, "tb_played": 0,
           "first_set_w_won": None, "retired": False}
    if not isinstance(score, str):
        return out
    tail = score.strip()
    out["retired"] = " RET" in tail or " DEF" in tail
    # Walkovers have no set score to parse.
    if " W/O" in tail or " WO" in tail:
        return out
    for i, m in enumerate(_SET_RE.finditer(tail)):
        w, l, tb = m.groups()
        try:
            wi, li = int(w), int(l)
        except ValueError:
            continue
        # A "set" only counts if the score reached 6-x or 7-x — this
        # filters out garbled fragments and doubles-scored rows.
        if max(wi, li) < 5:
            continue
        if wi > li:
            out["sets_w"] += 1
            first_won = True
        else:
            out["sets_l"] += 1
            first_won = False
        if i == 0:
            out["first_set_w_won"] = first_won
        if tb is not None:
            out["tb_played"] += 1
            # Convention: the "(K)" is the losing player's TB score,
            # so whoever won the set won the tiebreak.
            if first_won:
                out["tb_won_w"] += 1
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
    """Add per-player rolling form, serve/return averages, and the
    expanded panel of orthogonal signals introduced in 2026-07.

    We walk the dataframe in chronological order and, for each row,
    look up the player's prior state before applying the row's
    update. This guarantees no in-row leakage — the very feature the
    model sees is the value it would have had at match time.

    2026-07-08 correctness fix: Sackmann stamps every match in a
    tournament with the tournament's START date, so a plain sort on
    ``tourney_date`` leaves ~250 matches for a Grand Slam sharing the
    same key and pandas' default sort is unstable within the tie.
    That let later rounds be processed BEFORE earlier ones — e.g.
    Vondrousova's Wimbledon 2023 final at row 174 while her R128 was
    at row 175. When R128 was reached her ``last_match_date`` had
    already been updated to today's date by the earlier-processed
    later rounds, so ``days_rest = 0`` for every match she'd play,
    while opponents she beat in R128 kept a 7-day-old
    ``last_match_date`` from the previous tournament → ``days_rest =
    7``. Net effect: ``diff_days_rest`` and ``diff_matches_last_7d``
    became a proxy for "who advances deeper in this tournament,"
    which is the target. Permutation importance ranked them #1 and
    #3 on the broken bundle. Sorting by (date, round_rank, match_num)
    makes R128 come before R64 before R32 ... before F, so each row
    is processed with only genuinely-prior state.
    """
    df = df.copy()
    df["_round_rank_sort"] = df["round"].apply(_round_rank)
    df["_match_num_sort"] = pd.to_numeric(
        df.get("match_num"), errors="coerce",
    ).fillna(0).astype(int)
    df = df.sort_values(
        ["tourney_date", "_round_rank_sort", "_match_num_sort"],
        kind="mergesort",  # stable
    ).reset_index(drop=True)
    df = df.drop(columns=["_round_rank_sort", "_match_num_sort"])

    # ---- Legacy per-player rolling buffers (kept intact) ------------
    last5_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    last10_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    serve_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    return_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    bp_save_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_match_date: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple[str, str], int] = defaultdict(int)

    # ---- New 2026-07 buffers ---------------------------------------
    # Chronological (date, elo) snapshots per player. We walk them
    # from newest → oldest looking for the first entry ≥ N days old to
    # compute Elo momentum. Bounded at 200 entries so a top-tour player
    # who plays ~80 tour-level matches a year still has ~2.5 years of
    # history available for the 90-day lookback.
    elo_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
    # Dates of recent matches — used for fatigue (matches in last 7/14
    # days) and the layoff-since-last-tournament signal.
    recent_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
    # Rolling deques for score-derived flags.
    surface_last5: dict[tuple[str, str], deque] = defaultdict(
        lambda: deque(maxlen=5))
    retirement_last20: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    tiebreak_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    comeback_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    choke_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    opp_elo_last10: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    set_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    match_min_last10: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    round_hist: dict[tuple[str, int], deque] = defaultdict(
        lambda: deque(maxlen=20))
    # Recency-weighted H2H — store the raw list of (date, sign_for_key0).
    h2h_dates: dict[tuple[str, str], list] = defaultdict(list)
    # Per-surface H2H — a signed integer count per (player_pair, surface).
    h2h_by_surface: dict[tuple[str, str, str], int] = defaultdict(int)
    # Track the player's last tournament id so we can compute
    # layoff = days since last match in a DIFFERENT tournament.
    last_tourney_by_player: dict[str, tuple[str, pd.Timestamp]] = {}

    cols = {
        # Legacy columns
        "w_form_last5": [], "l_form_last5": [],
        "w_form_last10": [], "l_form_last10": [],
        "w_avg_serve_pts_won_10": [], "l_avg_serve_pts_won_10": [],
        "w_avg_return_pts_won_10": [], "l_avg_return_pts_won_10": [],
        "w_avg_bp_saved_10": [], "l_avg_bp_saved_10": [],
        "w_days_rest": [], "l_days_rest": [],
        "h2h_w_wins_minus_l_wins": [],
        # New — Elo momentum
        "w_elo_delta_30d": [], "l_elo_delta_30d": [],
        "w_elo_delta_90d": [], "l_elo_delta_90d": [],
        # New — fatigue
        "w_matches_last_7d": [], "l_matches_last_7d": [],
        "w_matches_last_14d": [], "l_matches_last_14d": [],
        # New — surface-specific form
        "w_surface_form_last5": [], "l_surface_form_last5": [],
        # New — score-derived rates
        "w_retirement_rate_20": [], "l_retirement_rate_20": [],
        "w_tiebreak_win_pct": [], "l_tiebreak_win_pct": [],
        "w_comeback_rate": [], "l_comeback_rate": [],
        "w_choke_rate": [], "l_choke_rate": [],
        "w_set_win_pct": [], "l_set_win_pct": [],
        # New — competition quality
        "w_avg_opp_elo_10": [], "l_avg_opp_elo_10": [],
        # New — layoff (days since last DIFFERENT tournament)
        "w_layoff_days": [], "l_layoff_days": [],
        # New — fitness / match length
        "w_avg_match_min_10": [], "l_avg_match_min_10": [],
        # New — round-specific win rate at this round
        "w_round_win_pct": [], "l_round_win_pct": [],
        # New — recency-weighted H2H (a - b, so mirror handles sign)
        "h2h_w_recency_weighted": [],
        # New — H2H on current surface
        "h2h_w_on_surface": [],
    }

    def _avg(buf: deque, default: float = 0.6) -> float:
        return float(np.mean(buf)) if len(buf) else default

    def _rate(buf: deque, default: float = 0.0) -> float:
        return float(sum(buf)) / len(buf) if len(buf) else default

    def _lookup_elo_delta(hist: deque, current_elo: float,
                           now: pd.Timestamp, days: int) -> float:
        """Elo change over ``days``: current elo minus the player's most
        recent recorded elo at least ``days`` old. Returns 0.0 when the
        player has no history that far back — this treats "no data" as
        "no momentum," which is safer than extrapolating."""
        if not hist:
            return 0.0
        cutoff = now - pd.Timedelta(days=days)
        # Walk newest → oldest; the first entry with date ≤ cutoff is
        # the value at ``days`` days ago.
        for d, elo in reversed(hist):
            if d <= cutoff:
                return float(current_elo) - float(elo)
        return 0.0

    def _count_in_window(dates: deque, now: pd.Timestamp,
                          days: int) -> int:
        cutoff = now - pd.Timedelta(days=days)
        return sum(1 for d in dates if d >= cutoff)

    def _weighted_h2h(entries: list, now: pd.Timestamp) -> float:
        """Sum of exp(-Δdays / halflife) × sign_for_key0. Positive means
        the alphabetically-first player leads the H2H after decay."""
        if not entries:
            return 0.0
        total = 0.0
        for d, sign in entries:
            days_ago = max(0.0, (now - d).total_seconds() / 86400.0)
            total += sign * np.exp(-days_ago / _H2H_HALFLIFE_DAYS)
        return float(total)

    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        date = row["tourney_date"]
        surface = row.get("surface") if isinstance(row.get("surface"), str) else "Unknown"
        tourney_id = row.get("tourney_id", "")
        round_r = _round_rank(row.get("round", ""))

        # ---- legacy features (read BEFORE state update) -------------
        cols["w_form_last5"].append(_avg(last5_results[w], 0.5))
        cols["l_form_last5"].append(_avg(last5_results[l], 0.5))
        cols["w_form_last10"].append(_avg(last10_results[w], 0.5))
        cols["l_form_last10"].append(_avg(last10_results[l], 0.5))

        cols["w_avg_serve_pts_won_10"].append(_avg(serve_buf[w], 0.6))
        cols["l_avg_serve_pts_won_10"].append(_avg(serve_buf[l], 0.6))
        cols["w_avg_return_pts_won_10"].append(_avg(return_buf[w], 0.4))
        cols["l_avg_return_pts_won_10"].append(_avg(return_buf[l], 0.4))
        cols["w_avg_bp_saved_10"].append(_avg(bp_save_buf[w], 0.6))
        cols["l_avg_bp_saved_10"].append(_avg(bp_save_buf[l], 0.6))

        wd = (date - last_match_date[w]).days if w in last_match_date else 7
        ld = (date - last_match_date[l]).days if l in last_match_date else 7
        cols["w_days_rest"].append(min(60, max(0, wd)))
        cols["l_days_rest"].append(min(60, max(0, ld)))

        key = tuple(sorted([w, l]))
        sign = 1 if key[0] == w else -1
        cols["h2h_w_wins_minus_l_wins"].append(h2h[key] * sign)

        # ---- new features (read BEFORE state update) ----------------
        # Elo momentum
        w_elo_pre = float(row.get("winner_elo_pre", 1500.0) or 1500.0)
        l_elo_pre = float(row.get("loser_elo_pre", 1500.0) or 1500.0)
        cols["w_elo_delta_30d"].append(_lookup_elo_delta(
            elo_history[w], w_elo_pre, date, 30))
        cols["l_elo_delta_30d"].append(_lookup_elo_delta(
            elo_history[l], l_elo_pre, date, 30))
        cols["w_elo_delta_90d"].append(_lookup_elo_delta(
            elo_history[w], w_elo_pre, date, 90))
        cols["l_elo_delta_90d"].append(_lookup_elo_delta(
            elo_history[l], l_elo_pre, date, 90))

        # Fatigue — matches played in trailing window
        cols["w_matches_last_7d"].append(_count_in_window(recent_dates[w], date, 7))
        cols["l_matches_last_7d"].append(_count_in_window(recent_dates[l], date, 7))
        cols["w_matches_last_14d"].append(_count_in_window(recent_dates[w], date, 14))
        cols["l_matches_last_14d"].append(_count_in_window(recent_dates[l], date, 14))

        # Surface-specific form
        cols["w_surface_form_last5"].append(_avg(surface_last5[(w, surface)], 0.5))
        cols["l_surface_form_last5"].append(_avg(surface_last5[(l, surface)], 0.5))

        # Score-derived rates
        cols["w_retirement_rate_20"].append(_rate(retirement_last20[w], 0.02))
        cols["l_retirement_rate_20"].append(_rate(retirement_last20[l], 0.02))
        cols["w_tiebreak_win_pct"].append(_rate(tiebreak_hist[w], 0.5))
        cols["l_tiebreak_win_pct"].append(_rate(tiebreak_hist[l], 0.5))
        cols["w_comeback_rate"].append(_rate(comeback_hist[w], 0.15))
        cols["l_comeback_rate"].append(_rate(comeback_hist[l], 0.15))
        cols["w_choke_rate"].append(_rate(choke_hist[w], 0.15))
        cols["l_choke_rate"].append(_rate(choke_hist[l], 0.15))
        cols["w_set_win_pct"].append(_avg(set_hist[w], 0.5))
        cols["l_set_win_pct"].append(_avg(set_hist[l], 0.5))

        # Quality of recent competition (default 1500 = league avg)
        cols["w_avg_opp_elo_10"].append(_avg(opp_elo_last10[w], 1500.0))
        cols["l_avg_opp_elo_10"].append(_avg(opp_elo_last10[l], 1500.0))

        # Layoff — days since the player's last DIFFERENT tournament.
        # A player playing back-to-back matches within one tournament
        # is not "returning from a layoff"; the layoff should reflect
        # rust from a prolonged absence.
        def _layoff(pl: str) -> float:
            prev = last_tourney_by_player.get(pl)
            if prev is None:
                return 21.0  # neutral default — average tour off-week
            prev_tid, prev_date = prev
            if prev_tid == tourney_id:
                # still inside the same tournament — no new layoff info
                return 0.0
            return float(max(0, (date - prev_date).days))
        cols["w_layoff_days"].append(min(180, _layoff(w)))
        cols["l_layoff_days"].append(min(180, _layoff(l)))

        # Avg match minutes (fitness / grinder-vs-blitzer)
        cols["w_avg_match_min_10"].append(_avg(match_min_last10[w], 100.0))
        cols["l_avg_match_min_10"].append(_avg(match_min_last10[l], 100.0))

        # Round-specific historical win rate at this round
        cols["w_round_win_pct"].append(_avg(round_hist[(w, round_r)], 0.5))
        cols["l_round_win_pct"].append(_avg(round_hist[(l, round_r)], 0.5))

        # Recency-weighted H2H (from key[0]'s perspective, mirrored for w)
        rw = _weighted_h2h(h2h_dates[key], date)
        cols["h2h_w_recency_weighted"].append(rw * sign)

        # H2H on the current surface
        surf_key = (key[0], key[1], surface)
        cols["h2h_w_on_surface"].append(h2h_by_surface[surf_key] * sign)

        # ---- state updates (post-match) ----------------------------
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

        # New — Elo history snapshots (use the post-match Elo so the
        # NEXT match's momentum calc reflects this result).
        w_elo_post = float(row.get("winner_elo_post", w_elo_pre) or w_elo_pre)
        l_elo_post = float(row.get("loser_elo_post", l_elo_pre) or l_elo_pre)
        elo_history[w].append((date, w_elo_post))
        elo_history[l].append((date, l_elo_post))

        # Recent dates for fatigue
        recent_dates[w].append(date)
        recent_dates[l].append(date)

        # Surface form
        surface_last5[(w, surface)].append(1.0)
        surface_last5[(l, surface)].append(0.0)

        # Score-derived updates
        meta = _score_meta(row.get("score", ""))
        retirement_last20[w].append(1 if meta["retired"] else 0)
        retirement_last20[l].append(1 if meta["retired"] else 0)
        if meta["tb_played"] > 0 and not meta["retired"]:
            for _i in range(meta["tb_played"]):
                pass  # placeholder — we only care about winner-side outcome
            # tb_won_w wins go to the match winner; the loser gets
            # (tb_played - tb_won_w) losses.
            for _i in range(meta["tb_won_w"]):
                tiebreak_hist[w].append(1)
                tiebreak_hist[l].append(0)
            for _i in range(meta["tb_played"] - meta["tb_won_w"]):
                # tiebreaks the match-winner LOST → still played by both
                tiebreak_hist[w].append(0)
                tiebreak_hist[l].append(1)
        if meta["first_set_w_won"] is not None and not meta["retired"]:
            if meta["first_set_w_won"] is False:
                # winner came back from losing set 1 → winner's comeback,
                # loser's choke
                comeback_hist[w].append(1); comeback_hist[l].append(0)
                choke_hist[w].append(0);    choke_hist[l].append(1)
            else:
                comeback_hist[w].append(0); comeback_hist[l].append(0)
                choke_hist[w].append(0);    choke_hist[l].append(0)
        # Set-win % — count each set as a data point
        if meta["sets_w"] + meta["sets_l"] > 0:
            for _i in range(meta["sets_w"]):
                set_hist[w].append(1.0); set_hist[l].append(0.0)
            for _i in range(meta["sets_l"]):
                set_hist[w].append(0.0); set_hist[l].append(1.0)

        # Competition quality — record the opponent's Elo AT this match
        opp_elo_last10[w].append(l_elo_pre)
        opp_elo_last10[l].append(w_elo_pre)

        # Layoff — update tournament seen
        last_tourney_by_player[w] = (tourney_id, date)
        last_tourney_by_player[l] = (tourney_id, date)

        # Match minutes
        mins = row.get("minutes")
        if pd.notna(mins):
            match_min_last10[w].append(float(mins))
            match_min_last10[l].append(float(mins))

        # Round-specific record
        round_hist[(w, round_r)].append(1.0)
        round_hist[(l, round_r)].append(0.0)

        # Recency-weighted H2H — record dated entry
        h2h_dates[key].append((date, 1 if key[0] == w else -1))
        # Cap the H2H list at 40 entries — 20 years of tour play is
        # plenty at a 2-year half-life; beyond that the weight is <5%
        # and adding is pure memory pressure.
        if len(h2h_dates[key]) > 40:
            h2h_dates[key] = h2h_dates[key][-40:]

        # H2H on surface
        h2h_by_surface[surf_key] += 1 if key[0] == w else -1

    for k, v in cols.items():
        df[k] = v

    # Persist the final per-player rolling-buffer averages so the
    # inference path (predict.py) can populate the form/serve/return/
    # bp_saved features with real values instead of zeros.
    rolling_snapshot: dict[str, dict[str, float]] = {}
    all_players = (set(last5_results.keys()) | set(last10_results.keys())
                   | set(serve_buf.keys()) | set(return_buf.keys())
                   | set(bp_save_buf.keys()) | set(elo_history.keys())
                   | set(recent_dates.keys()))
    for pl in all_players:
        rolling_snapshot[pl] = {
            "form_last5": _avg(last5_results[pl], 0.5),
            "form_last10": _avg(last10_results[pl], 0.5),
            "avg_serve_pts_won_10": _avg(serve_buf[pl], 0.6),
            "avg_return_pts_won_10": _avg(return_buf[pl], 0.4),
            "avg_bp_saved_10": _avg(bp_save_buf[pl], 0.6),
            # Expanded snapshot — inference can pull these too. Each is
            # a scalar, no need to carry deques into the joblib.
            "retirement_rate_20": _rate(retirement_last20[pl], 0.02),
            "tiebreak_win_pct": _rate(tiebreak_hist[pl], 0.5),
            "comeback_rate": _rate(comeback_hist[pl], 0.15),
            "choke_rate": _rate(choke_hist[pl], 0.15),
            "set_win_pct": _avg(set_hist[pl], 0.5),
            "avg_opp_elo_10": _avg(opp_elo_last10[pl], 1500.0),
            "avg_match_min_10": _avg(match_min_last10[pl], 100.0),
        }
    return df, h2h, last_match_date, rolling_snapshot


def build_full_panel(matches: pd.DataFrame, elo_cfg: dict | None = None
                     ) -> tuple[pd.DataFrame, EloState, dict, dict, dict]:
    """End-to-end: enrich Sackmann match data with everything the
    pre-match model wants. Returns the wide panel + the trained Elo
    state and accumulated H2H / last-match-date dicts (used at
    inference time) + the per-player rolling-form snapshot for the
    same purpose."""
    df = _serve_return_panel(matches)
    df, elo_state = build_elo_features(df, elo_cfg)
    df, h2h_table, last_match_date, rolling_snapshot = (
        _rolling_form_features(df)
    )

    # Tournament + round encodings
    df["level_rank"] = df["tourney_level"].map(_LEVEL_RANK).fillna(1).astype(int)
    df["round_rank"] = df["round"].map(_round_rank).astype(int)

    # Rank diff (Sackmann ships winner_rank / loser_rank, sometimes blank).
    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"] = pd.to_numeric(df["loser_rank"], errors="coerce")
    df["rank_diff"] = (df["loser_rank"] - df["winner_rank"]).fillna(0.0)
    # Log rank-points diff — captures the logarithmic point structure
    # (#1 vs #10 is a bigger gap than #100 vs #110). We add 1 before
    # log so rank_points == 0 (unranked / missing) → log(1) = 0.
    df["winner_rank_points"] = pd.to_numeric(
        df["winner_rank_points"], errors="coerce").fillna(0.0)
    df["loser_rank_points"] = pd.to_numeric(
        df["loser_rank_points"], errors="coerce").fillna(0.0)
    df["log_rank_points_diff"] = (
        np.log1p(df["winner_rank_points"]) - np.log1p(df["loser_rank_points"])
    )

    # Handedness — 'L' = 1, 'R' = 0, 'U' or missing = 0 (assume RH is
    # the default when unknown; the panel is 88% RH so this is closer
    # to true than treating U as its own category).
    df["winner_is_lefty"] = (df.get("winner_hand") == "L").astype(int)
    df["loser_is_lefty"] = (df.get("loser_hand") == "L").astype(int)

    # Physical attrs — pass through, NaN preserved for HGB.
    for c in ("winner_ht", "loser_ht", "winner_age", "loser_age"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # best_of / draw_size pass through as-is; both are already ints.

    return df, elo_state, h2h_table, last_match_date, rolling_snapshot


def build_player_a_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Re-orient the winner/loser-encoded panel into player_a/player_b
    with a balanced ``y`` label. Each match contributes two rows: one
    where player_a = winner (y=1) and one where player_a = loser (y=0).
    This removes the orientation bias before training.
    """
    keep = ["tourney_date", "tourney_name", "surface", "tourney_level",
            "round", "tour"]
    base = panel[keep].copy()

    a_pos = base.copy()
    a_pos["player_a"] = panel["winner_name"].values
    a_pos["player_b"] = panel["loser_name"].values
    a_pos["y"] = 1
    _attach_oriented(a_pos, side="w_to_a", panel=panel)

    a_neg = base.copy()
    a_neg["player_a"] = panel["loser_name"].values
    a_neg["player_b"] = panel["winner_name"].values
    a_neg["y"] = 0
    _attach_oriented(a_neg, side="l_to_a", panel=panel)

    return pd.concat([a_pos, a_neg], ignore_index=True)


# Columns paired as (short_name, w_col, l_col) so both orientations
# can be built with one loop. Keeping this as data (not code) makes
# adding new features later a one-line change.
_ORIENTED_PAIRS = (
    ("elo_pre",              "winner_elo_pre",           "loser_elo_pre"),
    ("surface_elo_pre",      "winner_surface_elo_pre",   "loser_surface_elo_pre"),
    ("form_last5",           "w_form_last5",             "l_form_last5"),
    ("form_last10",          "w_form_last10",            "l_form_last10"),
    ("avg_serve_pts_won_10", "w_avg_serve_pts_won_10",   "l_avg_serve_pts_won_10"),
    ("avg_return_pts_won_10","w_avg_return_pts_won_10",  "l_avg_return_pts_won_10"),
    ("avg_bp_saved_10",      "w_avg_bp_saved_10",        "l_avg_bp_saved_10"),
    ("days_rest",            "w_days_rest",              "l_days_rest"),
    # New 2026-07 orthogonal signals
    ("elo_delta_30d",        "w_elo_delta_30d",          "l_elo_delta_30d"),
    ("elo_delta_90d",        "w_elo_delta_90d",          "l_elo_delta_90d"),
    ("matches_last_7d",      "w_matches_last_7d",        "l_matches_last_7d"),
    ("matches_last_14d",     "w_matches_last_14d",       "l_matches_last_14d"),
    ("surface_form_last5",   "w_surface_form_last5",     "l_surface_form_last5"),
    ("retirement_rate_20",   "w_retirement_rate_20",     "l_retirement_rate_20"),
    ("tiebreak_win_pct",     "w_tiebreak_win_pct",       "l_tiebreak_win_pct"),
    ("comeback_rate",        "w_comeback_rate",          "l_comeback_rate"),
    ("choke_rate",           "w_choke_rate",             "l_choke_rate"),
    ("set_win_pct",          "w_set_win_pct",            "l_set_win_pct"),
    ("avg_opp_elo_10",       "w_avg_opp_elo_10",         "l_avg_opp_elo_10"),
    ("layoff_days",          "w_layoff_days",            "l_layoff_days"),
    ("avg_match_min_10",     "w_avg_match_min_10",       "l_avg_match_min_10"),
    ("round_win_pct",        "w_round_win_pct",          "l_round_win_pct"),
    # Physical attrs (not diff'd — passed as both a_/b_ and the diff
    # is a separate column later so tree models can use either)
    ("ht",                   "winner_ht",                "loser_ht"),
    ("age",                  "winner_age",               "loser_age"),
    ("is_lefty",             "winner_is_lefty",          "loser_is_lefty"),
    ("rank_points",          "winner_rank_points",       "loser_rank_points"),
)


def _attach_oriented(out: pd.DataFrame, side: str, panel: pd.DataFrame) -> None:
    """Copy the winner/loser-prefixed columns into player_a / player_b
    columns, depending on which orientation this row is."""
    for short, w_col, l_col in _ORIENTED_PAIRS:
        if w_col not in panel.columns or l_col not in panel.columns:
            # Feature not present in this panel (missing raw column) —
            # leave the model to see NaN; HGB handles it.
            out[f"a_{short}"] = np.nan
            out[f"b_{short}"] = np.nan
            continue
        if side == "w_to_a":
            out[f"a_{short}"] = panel[w_col].values
            out[f"b_{short}"] = panel[l_col].values
        else:
            out[f"a_{short}"] = panel[l_col].values
            out[f"b_{short}"] = panel[w_col].values

    # Diffs (a - b)
    for short, _, _ in _ORIENTED_PAIRS:
        out[f"diff_{short}"] = out[f"a_{short}"] - out[f"b_{short}"]

    # Shared (orientation-invariant) features pass through
    out["level_rank"] = panel["level_rank"].values
    out["round_rank"] = panel["round_rank"].values
    out["best_of"] = pd.to_numeric(panel.get("best_of"), errors="coerce").fillna(3).astype(int).values
    out["draw_size"] = pd.to_numeric(panel.get("draw_size"), errors="coerce").fillna(32).astype(int).values

    if side == "w_to_a":
        out["a_rank"] = pd.to_numeric(panel["winner_rank"], errors="coerce").values
        out["b_rank"] = pd.to_numeric(panel["loser_rank"], errors="coerce").values
        out["h2h_a_wins_minus_b_wins"] = panel["h2h_w_wins_minus_l_wins"].values
        out["h2h_a_wins_minus_b_wins_recency"] = panel["h2h_w_recency_weighted"].values
        out["h2h_a_wins_minus_b_wins_on_surface"] = panel["h2h_w_on_surface"].values
        out["log_rank_points_diff"] = panel["log_rank_points_diff"].values
    else:
        out["a_rank"] = pd.to_numeric(panel["loser_rank"], errors="coerce").values
        out["b_rank"] = pd.to_numeric(panel["winner_rank"], errors="coerce").values
        out["h2h_a_wins_minus_b_wins"] = -panel["h2h_w_wins_minus_l_wins"].values
        out["h2h_a_wins_minus_b_wins_recency"] = -panel["h2h_w_recency_weighted"].values
        out["h2h_a_wins_minus_b_wins_on_surface"] = -panel["h2h_w_on_surface"].values
        out["log_rank_points_diff"] = -panel["log_rank_points_diff"].values

    # Rank diff: lower numeric rank = stronger player, so we want
    # b_rank - a_rank as "a's edge".
    out["rank_diff"] = (out["b_rank"].fillna(500) - out["a_rank"].fillna(500))


# ---------------------------------------------------------------------
# Final feature list used by the model. Train + inference must stay in
# lockstep — change here, both code paths see it.
#
# 2026-07 sweep: expanded from 12 to 38 columns. Feature-importance
# pruning happens in train_prematch_model (permutation-based) and is
# reported in feature_importance.csv. Columns that turn out to be pure
# noise can be pruned by trimming this list in a follow-up.
# ---------------------------------------------------------------------
PREMATCH_FEATURES = [
    # --- Elo core ----------------------------------------------------
    "diff_elo_pre",
    "diff_surface_elo_pre",
    "diff_elo_delta_30d",
    "diff_elo_delta_90d",
    # --- Form / serve / return --------------------------------------
    "diff_form_last5",
    "diff_form_last10",
    "diff_surface_form_last5",
    "diff_avg_serve_pts_won_10",
    "diff_avg_return_pts_won_10",
    "diff_avg_bp_saved_10",
    # --- Rest / fatigue / layoff ------------------------------------
    "diff_days_rest",
    "diff_matches_last_7d",
    "diff_matches_last_14d",
    "diff_layoff_days",
    # --- Head-to-head ------------------------------------------------
    "h2h_a_wins_minus_b_wins",
    "h2h_a_wins_minus_b_wins_recency",
    "h2h_a_wins_minus_b_wins_on_surface",
    # --- Rank --------------------------------------------------------
    "rank_diff",
    "log_rank_points_diff",
    # --- Tournament context -----------------------------------------
    "level_rank",
    "round_rank",
    "best_of",
    "draw_size",
    "diff_round_win_pct",
    # --- Score-derived rates ----------------------------------------
    "diff_retirement_rate_20",
    "diff_tiebreak_win_pct",
    "diff_comeback_rate",
    "diff_choke_rate",
    "diff_set_win_pct",
    "diff_avg_opp_elo_10",
    "diff_avg_match_min_10",
    # --- Physical attributes ----------------------------------------
    "diff_ht",
    "diff_age",
    "diff_is_lefty",       # a_is_lefty − b_is_lefty ∈ {-1, 0, 1}
    "a_is_lefty",
    "b_is_lefty",
    "a_age",
    "b_age",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[PREMATCH_FEATURES].fillna(0.0)
