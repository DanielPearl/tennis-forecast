"""Build live-match feature rows.

These don't go into the trained classifier — phase-1 keeps the live
adjustment as a transparent rules-based layer (see
``src/models/live_adjustment_model.py``). What this module does is
*standardize* the heterogeneous live-state shapes coming from
``fetch_live_scores`` into a clean dict per match so the rules-engine
and the dashboard both operate on the same record.

Key: we don't try to invent precision we don't have. If a provider
doesn't report unforced errors, the field stays None and the rules
engine ignores any rule that consumes it. This keeps the system
resilient as we swap providers.
"""
from __future__ import annotations

from typing import Any


_FIELDS_NUMERIC: list[str] = [
    "set_score_a", "set_score_b",
    "games_won_last_3_a", "games_won_last_3_b",
    "first_serve_pct_a", "first_serve_pct_b",
    "second_serve_pts_won_a", "second_serve_pts_won_b",
    "return_pts_won_a", "return_pts_won_b",
    "break_points_created_a", "break_points_created_b",
    "break_points_saved_pct_a", "break_points_saved_pct_b",
    "double_faults_a", "double_faults_b",
    "unforced_errors_a", "unforced_errors_b",
    "aces_a", "aces_b",
    "market_prob_a", "market_prob_a_prev",
    # Liquidity / spread signals plumbed in from kalshi_markets — used by
    # the live-PV layer to widen volatility when the book is thin or the
    # spread blows out (price is less informative → trust it less).
    "open_interest", "volume", "spread_cents",
]
_FIELDS_FLAGS: list[str] = [
    "is_tiebreak", "is_decider",
    "medical_timeout",
    "injury_news_flag", "retirement_risk_flag",
    "serving_a",
]


def standardize(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw live-state dict into the canonical schema.

    Missing numeric fields → None (downstream rules treat None as
    "not observed" and skip). Missing flags → False.
    """
    out: dict[str, Any] = {
        "match_id": str(record.get("match_id", "")),
        "tournament": record.get("tournament", "Unknown"),
        "surface": record.get("surface", "Hard"),
        "player_a": record.get("player_a", ""),
        "player_b": record.get("player_b", ""),
    }
    # ``kalshi_markets`` writes liquidity numbers with the ``_a`` suffix
    # (per-side from the YES side of the event); the rules engine looks
    # them up without the suffix because the field describes the whole
    # book on this match. Bridge here so both shapes work.
    src_open_int = record.get("open_interest", record.get("open_interest_a"))
    src_volume = record.get("volume", record.get("volume_a"))
    src_spread = record.get("spread_cents")
    for k in _FIELDS_NUMERIC:
        if k == "open_interest":
            v = src_open_int
        elif k == "volume":
            v = src_volume
        elif k == "spread_cents":
            v = src_spread
        else:
            v = record.get(k)
        try:
            out[k] = float(v) if v is not None else None
        except (TypeError, ValueError):
            out[k] = None
    for k in _FIELDS_FLAGS:
        out[k] = bool(record.get(k, False))
    return out


def momentum_score(rec: dict[str, Any]) -> float:
    """A −1..+1 scalar for "who's surging right now" from player_a's POV.

    Built from set score, games-won-last-3, and serve-strength deltas.
    Used by the rules engine to decide direction of the in-match nudge.
    """
    a_sets = rec.get("set_score_a") or 0
    b_sets = rec.get("set_score_b") or 0
    a_g3 = rec.get("games_won_last_3_a") or 0
    b_g3 = rec.get("games_won_last_3_b") or 0
    fs_a = rec.get("first_serve_pct_a") or 0.6
    fs_b = rec.get("first_serve_pct_b") or 0.6

    set_term = (a_sets - b_sets) * 0.5
    games_term = (a_g3 - b_g3) * 0.15
    serve_term = (fs_a - fs_b) * 1.0
    raw = set_term + games_term + serve_term
    # Soft-clamp into [-1, 1].
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return float(raw)


def market_move(rec: dict[str, Any]) -> float | None:
    """Signed market move on player_a since the last snapshot.
    None if either price is missing (no overreaction analysis possible)."""
    cur = rec.get("market_prob_a")
    prev = rec.get("market_prob_a_prev")
    if cur is None or prev is None:
        return None
    return float(cur) - float(prev)
