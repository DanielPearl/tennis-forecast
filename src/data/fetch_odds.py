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
    benchmark_probs_by_pair_with_guest as _sdk_benchmark_with_guest,
    discover_sport_keys,
)

from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_odds")

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
_ODDS_REGIONS = "eu,us,uk,au"


def _api_key() -> str:
    return os.environ.get("THE_ODDS_API_KEY", "").strip()


def _tennis_sport_keys() -> list[str]:
    """Currently-active MAIN-TOUR singles keys on The Odds API.

    Each key costs one paid /odds call per cache window, and slam
    weeks fan discovery out to many keys at once (qualifying, doubles,
    both tours). The paid cascade only exists to add Betfair as a
    second source on top of the free Pinnacle guest feed — main-tour
    singles is where that adds value, so filter to it and bound the
    fan-out (2026-07 credit-preservation pass).
    """
    keys = [k for k in discover_sport_keys("tennis_")
            if k.startswith(("tennis_atp_", "tennis_wta_"))
            and not any(x in k for x in ("doubles", "challenger",
                                          "itf", "utr"))]
    return keys[:8]


def pinnacle_probs_by_pair() -> dict[frozenset, dict[str, float]]:
    """Return a lookup mapping ``frozenset({player_a_name, player_b_name})``
    → ``{player_a_name: fair_prob_a, player_b_name: fair_prob_b}``.

    Sources, in priority order:

      1. Pinnacle's own public guest API (~400 tennis matchups on a
         typical day — full ITF Futures + Challenger + tour coverage,
         which The Odds API's tennis feed never carries).
      2. The Odds API cascade (Pinnacle → Betfair Exchange UK / EU)
         restricted to whatever sport keys the API currently lists
         (usually only the active Slam).

      Guest-API entries win when both sources quote the pair, because
      the guest feed is Pinnacle's real-time line-shopper source and
      is roughly 60s fresher than The Odds API's redistribution.
    """
    return _sdk_benchmark_with_guest(_tennis_sport_keys(), guest_sport="tennis")


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
