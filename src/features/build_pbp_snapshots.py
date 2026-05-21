"""Build per-snapshot training rows from Grand Slam point-by-point.

For each match in the slam PBP archive we replay the points and emit
one feature row per *completed game boundary*. Each row reflects what
the live-monitor would have observed at that moment in the match
(sets won, games in current set, who's serving the next game, running
serve %, recent momentum, …) and is labelled with the eventual match
winner.

Why game boundaries and not every point: consecutive points are
strongly autocorrelated. One snapshot per game gives ~25–40 rows per
match — enough density to model trajectory without the model seeing
500 near-identical rows from one upset.

Real-data only: features are computed from the published Sackmann
slam PBP files; nothing is synthesized or fixture-derived.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.logging_setup import setup_logging

log = setup_logging("features.build_pbp_snapshots")


# Slam → playing surface (constant per tournament).
_SURFACE = {
    "ausopen": "Hard",
    "usopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
}


def _best_of(match_num: str | int) -> int:
    """Singles best-of by Sackmann match_num convention.

    Most years use ``1xxx`` for men's singles (bo5) and ``2xxx`` for
    women's (bo3); 2021 ausopen switched to ``M`` / ``W`` prefixes.
    Anything else falls back to bo3, which is the safer default for
    a feature meant only to set ``is_decider``.
    """
    s = str(match_num)
    if not s:
        return 3
    head = s[0].upper()
    if head in ("1", "M"):
        return 5
    if head in ("2", "W"):
        return 3
    return 3


def _match_winner(points: pd.DataFrame) -> int | None:
    """Return 1 if player1 won the match, 2 if player2, else None.

    Uses the SetWinner column: count distinct sets won by each side.
    """
    if points.empty:
        return None
    last_per_set = points.groupby("SetNo").tail(1)
    sw = last_per_set["SetWinner"].astype("Int64").dropna()
    p1 = int((sw == 1).sum())
    p2 = int((sw == 2).sum())
    if p1 == p2:
        # Tied (retirement mid-set, suspended match, malformed row) —
        # fall back to the last point's PointWinner so we don't drop
        # data for retirements where one side was clearly ahead.
        last = points.iloc[-1]
        lpw = last.get("PointWinner")
        try:
            return int(lpw) if int(lpw) in (1, 2) else None
        except (TypeError, ValueError):
            return None
    return 1 if p1 > p2 else 2


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


@dataclass
class _Running:
    """Running counters updated point-by-point.

    Cheaper than reconstructing them with groupby cumsums every
    snapshot. One instance per match.
    """
    p1_first_in: int = 0
    p2_first_in: int = 0
    p1_first_won: int = 0
    p2_first_won: int = 0
    p1_second_in: int = 0
    p2_second_in: int = 0
    p1_second_won: int = 0
    p2_second_won: int = 0
    p1_serve_pts: int = 0      # points where P1 served
    p2_serve_pts: int = 0
    p1_aces: int = 0
    p2_aces: int = 0
    p1_dfs: int = 0
    p2_dfs: int = 0
    p1_unf: int = 0
    p2_unf: int = 0
    p1_bp_won: int = 0
    p2_bp_won: int = 0
    p1_bp_faced: int = 0       # break points faced (returner had BP chance)
    p2_bp_faced: int = 0
    # Trailing windows
    last10_winners: list[int] = None      # most-recent first
    games_won_history: list[int] = None   # 1 if P1 won that game, 2 if P2
    # First-serve velocity series (km/h). Index 0 is the oldest first
    # serve in the match; appended in time order. We compute both the
    # full-match baseline and a trailing-10 mean for the
    # ``first_serve_speed_delta_a/b`` decline feature.
    p1_first_speeds: list[float] = None
    p2_first_speeds: list[float] = None

    def __post_init__(self) -> None:
        if self.last10_winners is None:
            self.last10_winners = []
        if self.games_won_history is None:
            self.games_won_history = []
        if self.p1_first_speeds is None:
            self.p1_first_speeds = []
        if self.p2_first_speeds is None:
            self.p2_first_speeds = []

    def update_point(self, row: pd.Series) -> None:
        # Update last-10 sliding window
        try:
            w = int(row["PointWinner"]) if pd.notna(row["PointWinner"]) else 0
        except (TypeError, ValueError):
            w = 0
        if w in (1, 2):
            self.last10_winners.append(w)
            if len(self.last10_winners) > 10:
                self.last10_winners.pop(0)
        # Serve metrics — gated on PointServer
        try:
            srv = int(row["PointServer"])
        except (TypeError, ValueError):
            srv = 0
        if srv == 1:
            self.p1_serve_pts += 1
        elif srv == 2:
            self.p2_serve_pts += 1
        # Sackmann's slam PBP leaves the legacy P1FirstSrvIn / P2FirstSrvIn
        # columns blank in every file we've sampled — the populated field
        # is ``ServeNumber`` (1 = first serve in, 2 = second serve in,
        # 0/NaN = neither, i.e. double-faulted out or pre-match row).
        try:
            serve_no = int(row["ServeNumber"])
        except (TypeError, ValueError):
            serve_no = -1
        # Pull serve speed (km/h) once — Speed_KMH is 0 / missing on
        # rows without a tracked serve (e.g. start-of-match placeholder).
        try:
            speed_kmh = float(row["Speed_KMH"])
        except (TypeError, ValueError, KeyError):
            speed_kmh = 0.0
        if srv == 1:
            if serve_no == 1:
                self.p1_first_in += 1
                if w == 1:
                    self.p1_first_won += 1
                if speed_kmh > 50.0:  # ignore tracker dropouts
                    self.p1_first_speeds.append(speed_kmh)
            elif serve_no == 2:
                self.p1_second_in += 1
                if w == 1:
                    self.p1_second_won += 1
        elif srv == 2:
            if serve_no == 1:
                self.p2_first_in += 1
                if w == 2:
                    self.p2_first_won += 1
                if speed_kmh > 50.0:
                    self.p2_first_speeds.append(speed_kmh)
            elif serve_no == 2:
                self.p2_second_in += 1
                if w == 2:
                    self.p2_second_won += 1
        # Aces / DFs / unforced errors / break points
        def _flag(col: str) -> bool:
            v = row.get(col)
            try:
                return int(v) == 1
            except (TypeError, ValueError):
                return False
        if _flag("P1Ace"):
            self.p1_aces += 1
        if _flag("P2Ace"):
            self.p2_aces += 1
        if _flag("P1DoubleFault"):
            self.p1_dfs += 1
        if _flag("P2DoubleFault"):
            self.p2_dfs += 1
        if _flag("P1UnfErr"):
            self.p1_unf += 1
        if _flag("P2UnfErr"):
            self.p2_unf += 1
        if _flag("P1BreakPoint"):
            # Break-point opportunity where P1 was returning
            self.p1_bp_faced += 1
            if _flag("P1BreakPointWon"):
                self.p1_bp_won += 1
        if _flag("P2BreakPoint"):
            self.p2_bp_faced += 1
            if _flag("P2BreakPointWon"):
                self.p2_bp_won += 1

    def _serve_speed_features(self, speeds: list[float]) -> tuple[float, float, float]:
        """Return (baseline_avg, recent_avg, delta) in km/h.

        ``baseline_avg`` is the mean of the player's first-serve speeds
        across the whole match so far; ``recent_avg`` is the mean of
        the trailing 10 first serves. ``delta = recent - baseline`` is
        the actual decline signal — negative values mean the player is
        serving slower than their match average, the classic tell for
        fatigue or a developing injury. We require ≥ 5 baseline serves
        before reporting, otherwise the noise floor is too high and
        we return zeros.
        """
        n = len(speeds)
        if n < 5:
            return 0.0, 0.0, 0.0
        baseline = sum(speeds) / n
        window = min(10, n)
        recent = sum(speeds[-window:]) / window
        return float(baseline), float(recent), float(recent - baseline)

    def snapshot(self) -> dict[str, float]:
        a_base, a_recent, a_delta = self._serve_speed_features(self.p1_first_speeds)
        b_base, b_recent, b_delta = self._serve_speed_features(self.p2_first_speeds)
        return {
            "first_serve_pct_a": _safe_div(self.p1_first_in, self.p1_serve_pts),
            "first_serve_pct_b": _safe_div(self.p2_first_in, self.p2_serve_pts),
            "first_serve_won_pct_a": _safe_div(self.p1_first_won, self.p1_first_in),
            "first_serve_won_pct_b": _safe_div(self.p2_first_won, self.p2_first_in),
            "second_serve_won_pct_a": _safe_div(self.p1_second_won, self.p1_second_in),
            "second_serve_won_pct_b": _safe_div(self.p2_second_won, self.p2_second_in),
            "aces_a": float(self.p1_aces),
            "aces_b": float(self.p2_aces),
            "double_faults_a": float(self.p1_dfs),
            "double_faults_b": float(self.p2_dfs),
            "unforced_errors_a": float(self.p1_unf),
            "unforced_errors_b": float(self.p2_unf),
            "break_points_created_a": float(self.p1_bp_faced),
            "break_points_created_b": float(self.p2_bp_faced),
            "break_points_won_a": float(self.p1_bp_won),
            "break_points_won_b": float(self.p2_bp_won),
            "bp_conversion_a": _safe_div(self.p1_bp_won, self.p1_bp_faced),
            "bp_conversion_b": _safe_div(self.p2_bp_won, self.p2_bp_faced),
            "first_serve_speed_baseline_a": a_base,
            "first_serve_speed_baseline_b": b_base,
            "first_serve_speed_recent_a": a_recent,
            "first_serve_speed_recent_b": b_recent,
            "first_serve_speed_delta_a": a_delta,
            "first_serve_speed_delta_b": b_delta,
            "last10_share_a": _safe_div(
                sum(1 for x in self.last10_winners if x == 1),
                len(self.last10_winners) or 1,
            ),
            "games_won_last_3_a": float(
                sum(1 for g in self.games_won_history[-3:] if g == 1)
            ),
            "games_won_last_3_b": float(
                sum(1 for g in self.games_won_history[-3:] if g == 2)
            ),
        }


def _iter_match_snapshots(match_id: str, slam: str, year: int,
                          match_num: str | int,
                          points: pd.DataFrame) -> Iterable[dict]:
    """Yield one snapshot per *completed game boundary* for one match."""
    bo = _best_of(match_num)
    surface = _SURFACE.get(slam, "Hard")
    winner = _match_winner(points)
    if winner is None:
        return
    # Sort just in case the file isn't strictly time-ordered.
    points = points.sort_values(["SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)
    running = _Running()
    sets_won_a = 0
    sets_won_b = 0
    last_set_no = 0
    last_game_key: tuple[int, int] | None = None  # (SetNo, GameNo)

    for idx, row in points.iterrows():
        running.update_point(row)
        # Track set completion via SetWinner transitions
        try:
            cur_set = int(row["SetNo"])
        except (TypeError, ValueError):
            continue
        try:
            cur_game = int(row["GameNo"])
        except (TypeError, ValueError):
            cur_game = 0
        try:
            cur_p1g = int(row["P1GamesWon"])
            cur_p2g = int(row["P2GamesWon"])
        except (TypeError, ValueError):
            continue

        game_key = (cur_set, cur_game)
        # Did a game complete on this row? Sackmann marks the last point
        # of a game with GameWinner != 0. Use that as the boundary so we
        # snapshot exactly when the score on the board changes.
        try:
            game_completed = int(row["GameWinner"]) in (1, 2)
        except (TypeError, ValueError):
            game_completed = False

        if game_completed:
            gw = int(row["GameWinner"])
            running.games_won_history.append(gw)
            # If the row also completed a set, advance the set-count
            try:
                sw = int(row["SetWinner"])
            except (TypeError, ValueError):
                sw = 0
            if sw == 1:
                sets_won_a += 1
            elif sw == 2:
                sets_won_b += 1
            last_set_no = cur_set
            last_game_key = game_key

            # Build the snapshot — features describe the state *after*
            # this game ended, which is the moment the model would be
            # asked "what's the prob the leader holds on?".
            running_stats = running.snapshot()
            # Who serves the next point? Same player keeps serving for a
            # full game; the next game flips the server. So the
            # *next* server is the opposite of whoever served this point.
            try:
                this_server = int(row["PointServer"])
            except (TypeError, ValueError):
                this_server = 0
            next_serving_a = 1 if this_server == 2 else 0
            games_in_set_a = cur_p1g
            games_in_set_b = cur_p2g
            # After a set finishes, P1GamesWon/P2GamesWon reset on the
            # next row, but the snapshot here still reflects the
            # set-final score, which is meaningful.
            is_decider = (sets_won_a == sets_won_b == (bo // 2)) and not (sw in (1, 2))
            # Tiebreak window: in non-deciding sets a tiebreak triggers
            # at 6-6. Use the "next game" perspective — if we just
            # completed game 12 at 6-6, the next game is the tiebreak.
            is_tiebreak_next = (games_in_set_a == 6 and games_in_set_b == 6
                                and sw not in (1, 2))
            snap = {
                "match_id": match_id,
                "year": year,
                "slam": slam,
                "surface": surface,
                "best_of": bo,
                "match_num": str(match_num),
                "point_idx": int(idx),
                "set_score_a": float(sets_won_a),
                "set_score_b": float(sets_won_b),
                "current_set": float(cur_set),
                "current_set_games_a": float(games_in_set_a),
                "current_set_games_b": float(games_in_set_b),
                "games_diff": float(games_in_set_a - games_in_set_b),
                "sets_diff": float(sets_won_a - sets_won_b),
                "serving_a": float(next_serving_a),
                "is_decider": float(is_decider),
                "is_tiebreak": float(is_tiebreak_next),
                "set_just_ended": float(sw in (1, 2)),
                # Progress as fraction of "typical" match length. ~150
                # points per bo3, ~250 per bo5 — keeps the feature in
                # ~[0, 1.5] for retired-late matches.
                "progress": float((idx + 1) / (250.0 if bo == 5 else 150.0)),
                **running_stats,
                "won_a": int(winner == 1),
            }
            yield snap


def build_snapshots(matches: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Return one DataFrame with all per-snapshot rows ready for training."""
    # Map match_id → match_num so we can derive best_of without merging
    # the whole matches table point-by-point.
    matchnum_lookup = matches.set_index("match_id")["match_num"].to_dict()
    rows: list[dict] = []
    for match_id, mp in points.groupby("match_id", sort=False):
        if len(mp) < 5:
            continue
        slam = mp["slam"].iloc[0]
        year = int(mp["year"].iloc[0])
        match_num = matchnum_lookup.get(match_id, "")
        rows.extend(_iter_match_snapshots(match_id, slam, year, match_num, mp))
    if not rows:
        raise RuntimeError("no snapshots built — check PBP inputs")
    df = pd.DataFrame(rows)
    log.info(
        "built %d snapshots across %d matches",
        len(df), df["match_id"].nunique(),
    )
    return df


# Feature columns the trained model consumes. Kept here as the
# single source of truth so the inference path stays in sync.
FEATURE_COLUMNS: tuple[str, ...] = (
    "set_score_a", "set_score_b", "sets_diff",
    "current_set_games_a", "current_set_games_b", "games_diff",
    "current_set", "best_of",
    "serving_a", "is_decider", "is_tiebreak", "set_just_ended",
    "progress",
    "first_serve_pct_a", "first_serve_pct_b",
    "first_serve_won_pct_a", "first_serve_won_pct_b",
    "second_serve_won_pct_a", "second_serve_won_pct_b",
    "aces_a", "aces_b",
    "double_faults_a", "double_faults_b",
    "unforced_errors_a", "unforced_errors_b",
    "break_points_created_a", "break_points_created_b",
    "break_points_won_a", "break_points_won_b",
    "bp_conversion_a", "bp_conversion_b",
    # Serve velocity decline — recent trailing-10 mean of first-serve
    # speed minus the player's match baseline. Negative values are the
    # signal we care about (fatigue / developing injury); we include
    # the raw baseline + recent so the model can use absolute speed
    # for cross-player adjustment too.
    "first_serve_speed_baseline_a", "first_serve_speed_baseline_b",
    "first_serve_speed_recent_a", "first_serve_speed_recent_b",
    "first_serve_speed_delta_a", "first_serve_speed_delta_b",
    "last10_share_a",
    "games_won_last_3_a", "games_won_last_3_b",
)


if __name__ == "__main__":
    from ..data.fetch_pbp import fetch_all
    from ..utils.config import load_config, resolve_path

    bundles = fetch_all()
    snaps = build_snapshots(bundles["matches"], bundles["points"])
    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pbp_snapshots.csv"
    snaps.to_csv(out, index=False)
    log.info("wrote %s (%d rows, %d features)", out, len(snaps),
             len(FEATURE_COLUMNS))
