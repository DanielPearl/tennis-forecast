"""Optional pre-match odds pull for tennis via The Odds API.

Two use cases:

1. **Pinnacle probability lookup** (2026-07-08 onwards): fetches
   the sharp global reference line (Pinnacle) for each Wimbledon /
   ATP / WTA match Kalshi is currently listing. The devigged
   probability is stamped on each watchlist row so the operator can
   eyeball ``Kalshi % vs Pinnacle %`` next to the model's own view.

2. **Legacy raw-odds pull** (kept for backward compatibility with
   any external tooling that still calls ``fetch_pre_match_odds``).

Quota management
----------------

The Odds API's 20K-requests/month tier maps out as follows:

  * 1 request  = the /sports discovery call (all sports, all keys)
  * 1 request  = odds for ONE sport key at ONE region set
  * Currently  = 2 tennis sport keys during Wimbledon
                 (tennis_atp_wimbledon + tennis_wta_wimbledon)

So a naive per-tick refresh would burn ~86k/mo — over budget. This
module caches every response for ``_CACHE_TTL_SECONDS`` (default 300s
= 5 minutes), which cuts us to ~5,760 requests/month at a 60s tick
cadence. Pinnacle pre-match lines don't move faster than that on
tennis, so the freshness cost is zero.

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
import time
from typing import Any, Optional

import requests

from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_odds")

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Region combo that reliably includes Pinnacle. Pinnacle appears in the
# EU region on The Odds API; adding UK/AU as well doesn't cost extra
# requests (each odds call already returns every book across regions
# in one shot).
_ODDS_REGIONS = "eu,us,uk,au"

# Module-level cache. Reset on process restart; refreshed when TTL
# expires. Keeping this at module level (not inside a class) so it
# is shared across every watchlist tick without threading a state
# argument through every call site.
_cache: dict[str, tuple[float, Any]] = {}


def _api_key() -> str:
    return os.environ.get("THE_ODDS_API_KEY", "").strip()


def _cached_get(cache_key: str, url: str,
                 params: dict[str, Any]) -> Optional[Any]:
    """GET with a 5-minute cache keyed on ``cache_key``. Returns None
    on any failure — caller handles the graceful no-op."""
    now = time.time()
    hit = _cache.get(cache_key)
    if hit is not None and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    try:
        r = requests.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        log.warning("odds api request failed for %s: %s", cache_key, exc)
        return None
    if r.status_code != 200:
        log.warning("odds api %s non-200: %d — %s",
                     cache_key, r.status_code, r.text[:200])
        return None
    try:
        data = r.json()
    except ValueError:
        log.warning("odds api %s: non-JSON body", cache_key)
        return None
    _cache[cache_key] = (now, data)
    return data


def _tennis_sport_keys() -> list[str]:
    """Discover every currently-active tennis sport key on The Odds
    API. Between tournaments some keys disappear (e.g. Wimbledon is
    only listed while it's running); calling /sports each cache
    window lets us pick up new tournaments as they open."""
    key = _api_key()
    if not key:
        return []
    data = _cached_get(
        "_sports", f"{_ODDS_API_BASE}/",
        {"apiKey": key},
    )
    if not isinstance(data, list):
        return []
    return [s.get("key", "") for s in data
            if isinstance(s, dict)
            and (s.get("key", "") or "").startswith("tennis_")]


def _devig(price_a: float, price_b: float) -> Optional[tuple[float, float]]:
    """Convert two decimal odds to devigged (fair) probabilities that
    sum to 1.0. Returns None if either odds value is invalid."""
    if price_a is None or price_b is None:
        return None
    try:
        pa, pb = 1.0 / float(price_a), 1.0 / float(price_b)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    total = pa + pb
    if total <= 0:
        return None
    return pa / total, pb / total


def pinnacle_probs_by_pair() -> dict[frozenset, dict[str, float]]:
    """Return a lookup mapping ``frozenset({player_a_name, player_b_name})``
    → ``{player_a_name: fair_prob_a, player_b_name: fair_prob_b}``.

    Uses Pinnacle's line specifically (the sharpest global reference).
    Silently returns an empty dict when the API key isn't set, no
    tennis keys are currently listed, or the API is down — every
    caller must tolerate the missing signal.
    """
    key = _api_key()
    if not key:
        return {}
    out: dict[frozenset, dict[str, float]] = {}
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
        if not isinstance(events, list):
            continue
        for ev in events:
            home = ev.get("home_team") or ""
            away = ev.get("away_team") or ""
            if not home or not away:
                continue
            for book in ev.get("bookmakers") or []:
                if book.get("key") != "pinnacle":
                    continue
                markets = book.get("markets") or []
                if not markets:
                    continue
                outcomes = markets[0].get("outcomes") or []
                if len(outcomes) != 2:
                    continue
                # Map outcome name → decimal price
                by_name = {o.get("name", ""): o.get("price")
                            for o in outcomes}
                pa = by_name.get(home)
                pb = by_name.get(away)
                devigged = _devig(pa, pb)
                if devigged is None:
                    continue
                fair_a, fair_b = devigged
                out[frozenset({home, away})] = {
                    home: fair_a,
                    away: fair_b,
                }
                break  # only take Pinnacle; skip other bookmakers
    return out


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
