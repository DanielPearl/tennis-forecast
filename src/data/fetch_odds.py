"""Optional pre-match odds pull for tennis.

Two paths:

1. The Odds API (https://the-odds-api.com) — set ``THE_ODDS_API_KEY``
   to fetch real moneylines for tour-level matches. Free tier is ~500
   credits/month. Each tournament call is 1 credit.
2. No key set → returns an empty list. Callers must tolerate this; the
   forecast pipeline falls back to using the model probability as a
   placeholder for the market column so the system stays demo-able
   without a paid feed.

We avoid scraping bookmaker sites directly — terms of service are
restrictive and DOM markup churns every season.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_odds")

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
_TENNIS_KEYS = ["tennis_atp", "tennis_wta"]


def fetch_pre_match_odds() -> list[dict[str, Any]]:
    """Return list of upcoming-match odds. One row per match per book.

    Output schema (only the fields we use downstream):

    - sport_key: e.g. ``tennis_atp_indian_wells``
    - commence_time: ISO timestamp
    - home_team / away_team: player names (Sackmann uses "First Last")
    - bookmakers[].markets[].outcomes[]: prices in decimal odds
    """
    api_key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        log.info("THE_ODDS_API_KEY not set — skipping odds pull")
        return []
    out: list[dict[str, Any]] = []
    for sport_key in _TENNIS_KEYS:
        url = f"{_ODDS_API_BASE}/{sport_key}/odds/"
        try:
            r = requests.get(
                url,
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=20,
            )
        except requests.RequestException as exc:
            log.warning("odds fetch failed for %s: %s", sport_key, exc)
            continue
        if r.status_code != 200:
            log.warning("odds fetch non-200 for %s: %d", sport_key, r.status_code)
            continue
        out.extend(r.json() or [])
    log.info("fetched %d tennis odds rows", len(out))
    return out


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (no overround removed).

    For two-way moneyline you'll typically want to normalize the pair
    so they sum to 1.0; the caller does that when both sides are present.
    """
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0.0
    return 1.0 / float(decimal_odds)


def normalize_pair(p_a: float, p_b: float) -> tuple[float, float]:
    """Strip the bookmaker overround so the two implied probs sum to 1."""
    s = p_a + p_b
    if s <= 0:
        return 0.5, 0.5
    return p_a / s, p_b / s
