"""Synthetic match-progression engine.

When no real live tennis feed is plumbed in (i.e. ``SOFASCORE_BASE_URL``
isn't set), we still want the simulation to feel alive — the dashboard
should show evolving scores, drifting market prices, matches that
eventually complete with a winner. This module ticks the seeded
``data/raw/live_state.json`` forward each refresh.

This is *only* active when there is no real provider feeding live state.
The instant ``fetch_live_scores.fetch_provider_state()`` returns a
non-None list, the live monitor uses that and skips this module.

The engine is biased toward visible activity: each tick a match has a
moderate chance of advancing one game, market price drifts toward the
model's live probability, tiebreak / decider flags toggle when score
state demands it. When a match completes, it stays in the file with a
``completed: true`` flag and a ``winner_side`` tag so the simulator
can settle paper positions on it. After a configurable cooldown,
completed matches are replaced with fresh seed pairings so the demo
keeps producing new opportunities without manual intervention.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.match_progression")


# How aggressively the synthetic engine advances. Higher = faster matches.
# 0.40 means each tick has a 40% chance of advancing one game per match —
# at a 60-second refresh that resolves a typical best-of-3 in ~30-60min,
# which is roughly real wall-clock match length and keeps the demo feeling
# live without burning through fixtures.
_ADVANCE_PROB = 0.40
# Drift the market price toward the model's live probability. σ in pp.
_MARKET_DRIFT_FRACTION = 0.30
_MARKET_NOISE_SIGMA = 0.015
# Probability of toggling on a "medical timeout" event mid-match. Rare —
# this is the kind of thing the live model is supposed to react to.
_MEDICAL_TIMEOUT_PROB = 0.01
# How long completed matches sit before being recycled with fresh pairings.
_COMPLETED_HOLD_TICKS = 3


# Replacement pool — players the engine cycles in when a match completes.
# Mix of ATP + WTA + lower-ranked names so the watchlist stays varied.
_REPLACEMENT_POOL_ATP = [
    ("Novak Djokovic", 6, "Hard"),
    ("Alexander Zverev", 8, "Hard"),
    ("Andrey Rublev", 12, "Clay"),
    ("Hubert Hurkacz", 15, "Hard"),
    ("Karen Khachanov", 18, "Clay"),
    ("Frances Tiafoe", 19, "Hard"),
    ("Felix Auger-Aliassime", 24, "Hard"),
    ("Ben Shelton", 26, "Hard"),
    ("Tommy Paul", 16, "Hard"),
    ("Lorenzo Musetti", 25, "Clay"),
]
_REPLACEMENT_POOL_WTA = [
    ("Elena Rybakina", 4, "Hard"),
    ("Jessica Pegula", 5, "Hard"),
    ("Ons Jabeur", 8, "Clay"),
    ("Marketa Vondrousova", 10, "Clay"),
    ("Karolina Muchova", 12, "Clay"),
    ("Daria Kasatkina", 13, "Clay"),
    ("Beatriz Haddad Maia", 17, "Clay"),
    ("Liudmila Samsonova", 18, "Hard"),
    ("Emma Navarro", 20, "Hard"),
    ("Mirra Andreeva", 22, "Clay"),
]
_REPLACEMENT_TOURNAMENTS = [
    ("Madrid Open", "M", "Clay"),
    ("Italian Open", "M", "Clay"),
    ("French Open", "G", "Clay"),
    ("Estoril Open", "A", "Clay"),
    ("BMW Open", "A", "Clay"),
    ("Geneva Open", "A", "Clay"),
]
_REPLACEMENT_ROUNDS = ["R32", "R16", "QF", "SF"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _live_state_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["raw_dir"]) / "live_state.json"


def load_state() -> list[dict[str, Any]]:
    fp = _live_state_path()
    if not fp.exists():
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: list[dict[str, Any]]) -> None:
    fp = _live_state_path()
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _advance_match(rec: dict[str, Any], rng: random.Random,
                    model_live_prob_a: float | None = None) -> dict[str, Any]:
    """Apply one synthetic tick to a match record.

    ``model_live_prob_a`` is the model's adjusted probability that
    player_a wins the match — used both as the per-game weight and as
    the drift target for the market price. When None, fall back to a
    naive estimate from set score.
    """
    if rec.get("completed"):
        # Keep completed matches around for a few ticks so the simulator
        # has time to settle, then recycle.
        rec["completed_ticks"] = int(rec.get("completed_ticks", 0)) + 1
        return rec

    # Internal game-counter state — initialized lazily so existing
    # fixture rows that don't carry these stay backwards-compatible.
    rec.setdefault("game_score_a", 0)
    rec.setdefault("game_score_b", 0)
    rec["set_score_a"] = int(rec.get("set_score_a") or 0)
    rec["set_score_b"] = int(rec.get("set_score_b") or 0)

    p_a = model_live_prob_a
    if p_a is None:
        # Fallback: in-set probability scaled by current set score.
        sets_diff = rec["set_score_a"] - rec["set_score_b"]
        p_a = 0.5 + 0.10 * sets_diff
        p_a = max(0.10, min(0.90, p_a))

    advanced = False
    if rng.random() < _ADVANCE_PROB:
        advanced = True
        # Pick the winner of the next game weighted by p_a, with a
        # small "in-game randomness" factor so even a 70/30 favorite
        # drops a service game now and then.
        p_a_game = 0.40 + 0.20 * p_a  # bounds: [0.40, 0.60] per game
        winner_a = rng.random() < p_a_game
        if winner_a:
            rec["game_score_a"] += 1
        else:
            rec["game_score_b"] += 1

        # Set boundary: first to 6 with a 2-game lead, or 7 if 6-6.
        ga, gb = rec["game_score_a"], rec["game_score_b"]
        set_won_a = (ga >= 6 and ga - gb >= 2) or ga == 7
        set_won_b = (gb >= 6 and gb - ga >= 2) or gb == 7
        if set_won_a or set_won_b:
            if set_won_a:
                rec["set_score_a"] += 1
            else:
                rec["set_score_b"] += 1
            rec["game_score_a"] = 0
            rec["game_score_b"] = 0

        # Update games_won_last_3 buffers — used by the live model's
        # momentum scorer. We just bias toward the recent winner.
        if winner_a:
            rec["games_won_last_3_a"] = min(3, int(rec.get("games_won_last_3_a") or 0) + 1)
            rec["games_won_last_3_b"] = max(0, int(rec.get("games_won_last_3_b") or 0) - 1)
        else:
            rec["games_won_last_3_b"] = min(3, int(rec.get("games_won_last_3_b") or 0) + 1)
            rec["games_won_last_3_a"] = max(0, int(rec.get("games_won_last_3_a") or 0) - 1)

    # Match completion check — best-of-3, first to 2 sets. (We're not
    # modeling 5-set Grand Slam men's matches here for simplicity; the
    # added states would mostly clutter the demo without changing the
    # mechanics the simulator exercises.)
    if rec["set_score_a"] >= 2 or rec["set_score_b"] >= 2:
        rec["completed"] = True
        rec["completed_ticks"] = 0
        rec["completed_at"] = _now_iso()
        rec["winner_side"] = "PLAYER_A" if rec["set_score_a"] > rec["set_score_b"] else "PLAYER_B"
        # Pin market to outcome (1.0 / 0.0) — what a real book does at
        # match end. Overrides any drift further down.
        rec["market_prob_a_prev"] = rec.get("market_prob_a")
        rec["market_prob_a"] = 1.0 if rec["winner_side"] == "PLAYER_A" else 0.0
        return rec

    # Tiebreak / decider flag updates.
    rec["is_tiebreak"] = bool(rec.get("game_score_a", 0) == 6 and rec.get("game_score_b", 0) == 6)
    rec["is_decider"] = (rec["set_score_a"] == 1 and rec["set_score_b"] == 1)

    # Rare medical-timeout event — exercises the volatility / injury rules.
    if not rec.get("medical_timeout") and rng.random() < _MEDICAL_TIMEOUT_PROB:
        rec["medical_timeout"] = True
        rec["injury_news_flag"] = True

    # Market drift — price moves toward the model's live probability
    # but with noise. Real markets do this on average, with the size
    # of the move depending on book depth; a fixed 30% pull is a
    # tractable approximation that keeps the demo lively.
    cur = rec.get("market_prob_a")
    if cur is not None and p_a is not None:
        prev = float(cur)
        target = float(p_a)
        new = prev + _MARKET_DRIFT_FRACTION * (target - prev)
        new += rng.gauss(0.0, _MARKET_NOISE_SIGMA)
        new = max(0.02, min(0.98, new))
        rec["market_prob_a_prev"] = prev
        rec["market_prob_a"] = round(new, 4)

    # Serve % drift — small random walk so the live rules engine sees
    # variation but doesn't get yanked around match-to-match.
    for k in ("first_serve_pct_a", "first_serve_pct_b"):
        v = rec.get(k)
        if v is None:
            continue
        rec[k] = round(max(0.30, min(0.85, v + rng.gauss(0.0, 0.01))), 4)

    return rec


def _replace_completed(rec: dict[str, Any], rng: random.Random
                        ) -> dict[str, Any]:
    """Recycle a completed match slot with fresh players so the demo
    keeps producing watchlist activity. We preserve ``match_id`` so the
    simulator's per-match cooldown bookkeeping doesn't accidentally
    re-open a position on the same match in the same tick."""
    is_men = rng.random() < 0.5
    pool = _REPLACEMENT_POOL_ATP if is_men else _REPLACEMENT_POOL_WTA
    a, b = rng.sample(pool, 2)
    tournament, level, surface = rng.choice(_REPLACEMENT_TOURNAMENTS)
    round_ = rng.choice(_REPLACEMENT_ROUNDS)
    # Seed market with a small random offset from the rank-implied edge.
    rank_a, rank_b = a[1], b[1]
    rank_implied = 0.5 + 0.005 * (rank_b - rank_a)
    rank_implied = max(0.20, min(0.80, rank_implied))
    market = round(max(0.10, min(0.90, rank_implied + rng.gauss(0.0, 0.04))), 4)

    return {
        "match_id": rec.get("match_id", f"sim-{rng.randrange(10000):04d}"),
        "tournament": tournament,
        "surface": surface,
        "level": level,
        "round": round_,
        "player_a": a[0], "player_b": b[0],
        "rank_a": rank_a, "rank_b": rank_b,
        "set_score_a": 0, "set_score_b": 0,
        "game_score_a": 0, "game_score_b": 0,
        "games_won_last_3_a": 0, "games_won_last_3_b": 0,
        "first_serve_pct_a": round(0.60 + rng.gauss(0.0, 0.04), 4),
        "first_serve_pct_b": round(0.60 + rng.gauss(0.0, 0.04), 4),
        "is_tiebreak": False, "is_decider": False,
        "medical_timeout": False,
        "injury_news_flag": False,
        "retirement_risk_flag": False,
        "market_prob_a": market,
        "market_prob_a_prev": market,
    }


def tick(model_probs_by_match_id: dict[str, float] | None = None,
         seed: int | None = None) -> list[dict[str, Any]]:
    """Advance the live state by one tick. Returns the new list, also
    persists it to disk.

    ``model_probs_by_match_id`` lets the caller pass in the live model
    probabilities so the synthetic engine drifts the market toward the
    *model's* view (which is what real markets do on average). Without
    it, drift uses a naive set-score-based estimate.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    state = load_state()
    if not state:
        log.info("no live state to advance — nothing to do")
        return state

    new_state: list[dict[str, Any]] = []
    for rec in state:
        mid = rec.get("match_id", "")
        p = (model_probs_by_match_id or {}).get(mid)

        # Was this match already completed and held long enough? Recycle.
        if rec.get("completed") and rec.get("completed_ticks", 0) >= _COMPLETED_HOLD_TICKS:
            rec = _replace_completed(rec, rng)
        else:
            rec = _advance_match(rec, rng, model_live_prob_a=p)
        new_state.append(rec)

    save_state(new_state)
    return new_state
