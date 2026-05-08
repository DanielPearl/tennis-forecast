"""Tennis Elo (overall + surface-specific).

Why a custom Elo and not just sklearn classifier weights:

* Tennis Elo is the single most predictive feature in any reasonable
  pre-match model (Sipko/Knottenbelt 2015, Kovalchik 2016). Treating it
  as a derived feature lets us use it with logistic regression / GBT
  ensembles without contaminating training: the rating at match time
  uses *only* prior-match results, so there's no leakage.
* Surface-specific Elos (one each for hard / clay / grass / carpet)
  capture style matchups that overall Elo misses — a clay specialist
  beats a grass specialist on clay even when their overall ratings
  are equal.

K-factor: starts at ``k_base`` and decays toward ``k_floor`` as a
player accumulates matches. This stabilizes top-100 ratings while
still letting unproven up-and-comers move quickly.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


_SURFACES = ("Hard", "Clay", "Grass", "Carpet")


@dataclass
class EloState:
    default_rating: float = 1500.0
    k_base: float = 32.0
    k_floor: float = 16.0
    k_decay_matches: int = 30
    surface_k_multiplier: float = 1.15
    surface_blend: float = 0.60

    overall: dict = field(default_factory=dict)
    surface: dict = field(default_factory=dict)
    matches_played: dict = field(default_factory=lambda: defaultdict(int))
    surface_matches: dict = field(default_factory=lambda: defaultdict(int))

    def get_overall(self, player: str) -> float:
        return self.overall.get(player, self.default_rating)

    def get_surface(self, player: str, surface: str) -> float:
        return self.surface.get((player, surface), self.default_rating)

    def k_for(self, player: str) -> float:
        n = self.matches_played[player]
        # Linearly decay from k_base → k_floor over k_decay_matches matches.
        if n >= self.k_decay_matches:
            return self.k_floor
        frac = n / max(1, self.k_decay_matches)
        return self.k_base - (self.k_base - self.k_floor) * frac


def _expected(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_pair(state: EloState, winner: str, loser: str, surface: str | None
                 ) -> tuple[float, float, float, float]:
    """Apply one match update; return (winner_pre, loser_pre,
    winner_surface_pre, loser_surface_pre) — used to build features
    out of a single chronological pass."""
    surface = surface if surface in _SURFACES else "Hard"

    w_pre = state.get_overall(winner)
    l_pre = state.get_overall(loser)
    ws_pre = state.get_surface(winner, surface)
    ls_pre = state.get_surface(loser, surface)

    # Overall update
    e_w = _expected(w_pre, l_pre)
    k_w = state.k_for(winner)
    k_l = state.k_for(loser)
    state.overall[winner] = w_pre + k_w * (1.0 - e_w)
    state.overall[loser] = l_pre + k_l * (0.0 - (1.0 - e_w))

    # Surface update — slightly higher K because surface matches per
    # player are scarcer (fewer data points → faster convergence is fine).
    e_ws = _expected(ws_pre, ls_pre)
    k_ws = state.k_for(winner) * state.surface_k_multiplier
    k_ls = state.k_for(loser) * state.surface_k_multiplier
    state.surface[(winner, surface)] = ws_pre + k_ws * (1.0 - e_ws)
    state.surface[(loser, surface)] = ls_pre + k_ls * (0.0 - (1.0 - e_ws))

    state.matches_played[winner] += 1
    state.matches_played[loser] += 1
    state.surface_matches[(winner, surface)] += 1
    state.surface_matches[(loser, surface)] += 1

    return w_pre, l_pre, ws_pre, ls_pre


def build_elo_features(matches: pd.DataFrame, state_cfg: dict | None = None
                       ) -> tuple[pd.DataFrame, EloState]:
    """Add pre-match Elo columns to a matches dataframe.

    Inputs (Sackmann column names):
      - tourney_date, surface, winner_name, loser_name

    Output adds (one row per match, oriented from the winner's view —
    the training pipeline will balance this by mirroring rows):
      - winner_elo_pre / loser_elo_pre
      - winner_surface_elo_pre / loser_surface_elo_pre
      - elo_diff, surface_elo_diff
      - elo_winprob (sigmoid mapping of overall diff)
      - blended_elo_diff (weighted combination — surface_blend on surface)
    """
    state = EloState(**(state_cfg or {}))
    df = matches.sort_values("tourney_date").reset_index(drop=True).copy()
    cols = {
        "winner_elo_pre": [],
        "loser_elo_pre": [],
        "winner_surface_elo_pre": [],
        "loser_surface_elo_pre": [],
    }
    for _, row in df.iterrows():
        w_pre, l_pre, ws_pre, ls_pre = _update_pair(
            state, row["winner_name"], row["loser_name"], row.get("surface")
        )
        cols["winner_elo_pre"].append(w_pre)
        cols["loser_elo_pre"].append(l_pre)
        cols["winner_surface_elo_pre"].append(ws_pre)
        cols["loser_surface_elo_pre"].append(ls_pre)
    for k, v in cols.items():
        df[k] = v
    df["elo_diff"] = df["winner_elo_pre"] - df["loser_elo_pre"]
    df["surface_elo_diff"] = df["winner_surface_elo_pre"] - df["loser_surface_elo_pre"]
    df["blended_elo_diff"] = (
        state.surface_blend * df["surface_elo_diff"]
        + (1.0 - state.surface_blend) * df["elo_diff"]
    )
    df["elo_winprob"] = 1.0 / (1.0 + 10.0 ** (-df["elo_diff"] / 400.0))
    return df, state


def lookup_pair_features(state: EloState, player_a: str, player_b: str,
                          surface: str) -> dict:
    """Build Elo features for a hypothetical (a vs b) on ``surface``.

    Used at inference time after training. We orient features around
    player_a so the rest of the pipeline doesn't need to know who's
    "home" vs "away".
    """
    a = state.get_overall(player_a)
    b = state.get_overall(player_b)
    a_s = state.get_surface(player_a, surface if surface in _SURFACES else "Hard")
    b_s = state.get_surface(player_b, surface if surface in _SURFACES else "Hard")
    elo_diff = a - b
    surface_elo_diff = a_s - b_s
    blended = (state.surface_blend * surface_elo_diff
               + (1.0 - state.surface_blend) * elo_diff)
    elo_winprob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    return {
        "player_a_elo": a, "player_b_elo": b,
        "player_a_surface_elo": a_s, "player_b_surface_elo": b_s,
        "elo_diff": elo_diff, "surface_elo_diff": surface_elo_diff,
        "blended_elo_diff": blended, "elo_winprob_a": elo_winprob,
    }
