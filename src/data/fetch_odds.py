"""Optional pre-match odds pull for tennis via The Odds API.

Two use cases:

1. **Pinnacle probability lookup** (2026-07-08 onwards): fetches
   the sharp global reference line (Pinnacle) for each Wimbledon /
   ATP / WTA match Kalshi is currently listing. The devigged
   probability is stamped on each watchlist row so the operator can
   eyeball ``Kalshi % vs Pinnacle %`` next to the model's own view.

   As of 2026-07-08 the fetcher itself lives in
   ``kalshi_sdk.pinnacle`` so every sport bot (tennis, WNBA, ...)
   uses the same devigged-Pinnacle implementation. This module is
   now a thin tennis-specific wrapper: it discovers the currently
   listed tennis sport keys and forwards to the SDK helper.

2. **Legacy raw-odds pull** (kept for backward compatibility with
   any external tooling that still calls ``fetch_pre_match_odds``).

Public API
----------
  * ``pinnacle_probs_by_pair()`` → ``{frozenset({name_a, name_b}):
    {name_a: prob_a, name_b: prob_b}}``. Used by the watchlist
    exporter to look up a Pinnacle probability per match.
  * ``fetch_pre_match_odds()`` → the legacy raw-odds passthrough.
  * Both silently no-op when ``THE_ODDS_API_KEY`` isn't set.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from kalshi_sdk.pinnacle import (
    discover_sport_keys,
    pinnacle_probs_by_pair as _sdk_pinnacle_probs_by_pair,
)

from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_odds")

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
_ODDS_REGIONS = "eu,us,uk,au"


def _api_key() -> str:
    return os.environ.get("THE_ODDS_API_KEY", "").strip()


def _tennis_sport_keys() -> list[str]:
    """Discover every currently-active tennis sport key on The Odds
    API. Thin wrapper over ``kalshi_sdk.pinnacle.discover_sport_keys``
    that pins the ``tennis_`` prefix; kept for backward compat with
    other callers in the tennis repo."""
    return discover_sport_keys("tennis_")


def pinnacle_probs_by_pair() -> dict[frozenset, dict[str, float]]:
    """Return a lookup mapping ``frozenset({player_a_name, player_b_name})``
    → ``{player_a_name: fair_prob_a, player_b_name: fair_prob_b}``.

    Forwards to ``kalshi_sdk.pinnacle.pinnacle_probs_by_pair`` after
    resolving the currently-active tennis sport keys.
    """
    return _sdk_pinnacle_probs_by_pair(_tennis_sport_keys())


def fetch_pre_match_odds() -> list[dict[str, Any]]:
    """Legacy raw-odds pull. Kept for backward compatibility with any
    external tooling that imports it; the current watchlist pipeline
    uses ``pinnacle_probs_by_pair`` instead."""
    key = _api_key()
    if not key:
        log.info("THE_ODDS_API_KEY not set — skipping odds pull")
        return []
    out: list[dict[str, Any]] = []
    for sport in _tennis_sport_keys():
        events = _cached_get(
            f"odds_{sport}",
            f"{_ODDS_API_BASE}/{sport}/odds/",
            {
                "apiKey": key,
                "regions": _ODDS_REGIONS,
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )
        if isinstance(events, list):
            out.extend(events)
    log.info("fetched %d tennis odds rows across %d sport keys",
              len(out), len(_tennis_sport_keys()))
    return out


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (no overround removed)."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0.0
    return 1.0 / float(decimal_odds)


def normalize_pair(p_a: float, p_b: float) -> tuple[float, float]:
    """Strip the bookmaker overround so the two implied probs sum to 1."""
    s = p_a + p_b
    if s <= 0:
        return 0.5, 0.5
    return p_a / s, p_b / s
