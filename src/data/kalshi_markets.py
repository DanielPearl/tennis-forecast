"""Real Kalshi tennis market fetcher.

Replaces the synthetic-match generator with the live KXATPMATCH +
KXWTAMATCH markets. Each Kalshi event is a single match with TWO
markets (one ``YES`` side per player); we collapse them into one row
per match where ``player_a`` is whichever side currently has the
lower-cents YES (= the underdog's ticker) and ``player_b`` is the
favoured side. The watchlist still presents both YES sides in the
ticker table; the simulator opens on whichever has +EV.

Title parsing
  Kalshi titles read:
    "Will {Player Name} win the {LastA} vs {LastB}: {Round} match?"
  We pull the full player name (left of "win the"), the round string
  (between ":" and "match"), and the matchup last-names.

Pricing
  Each market exposes ``yes_ask`` / ``yes_bid`` / ``no_ask`` /
  ``no_bid`` in cents. We treat ``yes_ask`` as the implied probability
  (cents → dollars). When a side is one-quoted (e.g. only no_ask
  populated), we derive the YES price from the opposite side's ask
  (yes ≈ 100 − no_ask).

Caching + rate-limit
  Each tick calls ``iter_open_markets`` per series with a ~1s pause
  between series to avoid the API's 429 rate limit. Per-tick results
  are written to ``data/raw/live_state.json`` so the existing
  watchlist exporter keeps working.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Iterable

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.kalshi_markets")


# Two series: men's tour and women's tour. Both share the same ticker
# shape; the title parsing is identical.
_TENNIS_SERIES = ("KXATPMATCH", "KXWTAMATCH")

# "Will {Player Name} win the {LastA} vs {LastB}: {Round} match?"
_TITLE_RE = re.compile(
    r"^Will (?P<player>.+?) win the (?P<lastA>[^\s]+) vs (?P<lastB>[^\s]+):"
    r"\s*(?P<round>[^?]+) match\?\s*$"
)


def _client():
    """Build a Kalshi SDK client. Raises if creds aren't on the env."""
    try:
        from kalshi_sdk import KalshiClient
    except ImportError as exc:
        raise RuntimeError(
            "kalshi_sdk not installed in this venv — pip install -e the "
            "shared sdk under /root/kalshi_sdk (or set up the editable "
            "dep in your local checkout)"
        ) from exc
    api_key = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api_key or not pkey:
        raise RuntimeError(
            "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set "
            "in the env (see the bot's systemd EnvironmentFile)"
        )
    return KalshiClient(api_key_id=api_key, private_key_path=pkey)


def _parse_title(title: str) -> dict[str, str]:
    """Extract player + round + matchup last-names from the Kalshi title."""
    if not title:
        return {}
    m = _TITLE_RE.match(title.strip())
    if not m:
        return {}
    return {
        "player": m.group("player").strip(),
        "round": m.group("round").strip(),
        "lastA": m.group("lastA").strip(),
        "lastB": m.group("lastB").strip(),
    }


def _yes_price_dollars(market: dict) -> float | None:
    """Best-effort YES implied probability from Kalshi cents."""
    ya = market.get("yes_ask")
    if ya is not None:
        try:
            return max(0.01, min(0.99, float(ya) / 100.0))
        except (TypeError, ValueError):
            pass
    na = market.get("no_ask")
    if na is not None:
        try:
            return max(0.01, min(0.99, 1.0 - float(na) / 100.0))
        except (TypeError, ValueError):
            pass
    yb = market.get("yes_bid")
    if yb is not None:
        try:
            return max(0.01, min(0.99, float(yb) / 100.0))
        except (TypeError, ValueError):
            pass
    return None


def _surface_from_rules(rules: str) -> str:
    """Best-effort surface inference from rules_primary text. Tennis
    surface drives a major feature in our model (surface-specific Elo);
    Kalshi doesn't tag surface so we sniff for hard / clay / grass /
    carpet keywords in the rules paragraph."""
    if not rules:
        return "Hard"
    s = rules.lower()
    if "clay" in s or "roland" in s or "french open" in s:
        return "Clay"
    if "grass" in s or "wimbledon" in s:
        return "Grass"
    if "carpet" in s:
        return "Carpet"
    return "Hard"


def fetch_tennis_markets(
    series: Iterable[str] = _TENNIS_SERIES,
    inter_series_pause_s: float = 1.0,
) -> list[dict]:
    """Pull every active Kalshi tennis market (both tours).

    Returns a flat list of raw market dicts (one per side of each
    event). The caller groups by ``event_ticker`` to collapse them
    into one record per match.
    """
    c = _client()
    out: list[dict] = []
    for s in series:
        try:
            for m in c.iter_open_markets(series_ticker=s):
                out.append(m)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch %s failed: %s", s, exc)
        time.sleep(inter_series_pause_s)
    log.info("fetched %d tennis markets across %d series",
             len(out), len(list(series)))
    return out


def collapse_to_matches(markets: list[dict],
                        prev_markets_by_ticker: dict[str, dict] | None = None
                        ) -> list[dict]:
    """Group two-sided markets into one record per event_ticker.

    Each event has two markets: YES on player_a, YES on player_b. We
    pick the alphabetically-first ticker as ``player_a`` so the side
    is deterministic across ticks. Output schema matches the live-
    state file the rest of the bot already consumes.

    ``prev_markets_by_ticker`` carries the last-tick YES asks so we
    can compute ``market_prob_a_prev`` for the overreaction rule.
    """
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        ev = m.get("event_ticker") or ""
        if not ev:
            continue
        by_event.setdefault(ev, []).append(m)
    out: list[dict] = []
    for event_ticker, sides in by_event.items():
        if len(sides) < 1:
            continue
        sides.sort(key=lambda x: x.get("ticker") or "")
        a_market = sides[0]
        b_market = sides[1] if len(sides) >= 2 else None
        a_title = _parse_title(a_market.get("title", ""))
        b_title = (_parse_title(b_market.get("title", ""))
                    if b_market else {})
        player_a = a_title.get("player") or (a_market.get("ticker") or "").split("-")[-1]
        player_b = b_title.get("player") or ""
        if not player_b and b_market is None:
            # Single-sided event (shouldn't happen for tennis but handle
            # gracefully) — derive opponent from the matchup last-names.
            lastA, lastB = a_title.get("lastA", ""), a_title.get("lastB", "")
            player_b = lastB if a_title.get("player","").endswith(lastA) else lastA
        rules = a_market.get("rules_primary") or ""
        surface = _surface_from_rules(rules)
        # Tournament name — best-effort from rules text. Kalshi rules
        # mention the tournament; we extract the year + ATP/WTA event
        # name when possible, else fall back to the series.
        tournament = _tournament_from_rules(rules)
        # Round string parsed from the title (e.g. "Round Of 64").
        round_str = a_title.get("round") or "R32"
        round_code = _round_to_code(round_str)
        # Real-only — when Kalshi hasn't published a quote on either
        # side (most upcoming tennis markets sit unquoted until close
        # to tipoff), pass ``None`` through. Downstream the simulator
        # skips opening a position on an unquoted market and the
        # watchlist renders "—" rather than fabricating a 50% default.
        market_yes_a = _yes_price_dollars(a_market)
        prev = (prev_markets_by_ticker or {}).get(a_market.get("ticker") or "")
        market_yes_a_prev = (_yes_price_dollars(prev) if prev else None)
        # Kalshi marks markets ``closed`` once an event has settled.
        is_closed = (a_market.get("status") or "").lower() in ("closed", "settled", "finalized")
        # Determine the winner (when settled): whichever side has YES
        # final price ≈ 100c is the winner.
        winner_side = None
        if is_closed and b_market is not None:
            ya = a_market.get("yes_ask") or 0
            yb = b_market.get("yes_ask") or 0
            try:
                if int(ya) >= 99:
                    winner_side = "PLAYER_A"
                elif int(yb) >= 99:
                    winner_side = "PLAYER_B"
            except (TypeError, ValueError):
                pass
        out.append({
            # Use the event_ticker as the canonical match_id — stable
            # across both sides + across ticks. This is the real
            # Kalshi ticker that the user can bet on.
            "match_id": event_ticker,
            "ticker_a": a_market.get("ticker"),
            "ticker_b": (b_market.get("ticker") if b_market else None),
            "tournament": tournament,
            "surface": surface,
            "level": "M",  # Kalshi tennis events are mostly Masters / 500;
                            # we don't reliably distinguish from the rules
                            # text. The level feature contribution is
                            # small once Elo is in the model.
            "round": round_code,
            "player_a": player_a,
            "player_b": player_b,
            "set_score_a": 0, "set_score_b": 0,
            "games_won_last_3_a": 0, "games_won_last_3_b": 0,
            "first_serve_pct_a": 0.62,
            "first_serve_pct_b": 0.62,
            "is_tiebreak": False,
            "is_decider": False,
            "medical_timeout": False,
            "injury_news_flag": False,
            "retirement_risk_flag": False,
            "market_prob_a": market_yes_a,
            "market_prob_a_prev": market_yes_a_prev,
            # Settlement signals for the simulator.
            "completed": is_closed,
            "winner_side": winner_side,
            # Extra metadata the renderer surfaces.
            "expected_expiration_time": a_market.get("expected_expiration_time"),
            "rules_primary": rules,
            "yes_ask_cents_a": a_market.get("yes_ask"),
            "yes_ask_cents_b": (b_market.get("yes_ask") if b_market else None),
            "volume_a": a_market.get("volume"),
            "open_interest_a": a_market.get("open_interest"),
            # Kalshi-published market titles for both sides — surface
            # the favoured side's YES question as the watchlist's
            # "Title" column.
            "title_a": a_market.get("title"),
            "title_b": (b_market.get("title") if b_market else None),
        })
    return out


def _tournament_from_rules(rules: str) -> str:
    """Pull a tournament name out of the rules text. Kalshi's rules
    describe the event in a "the {YEAR} {Tournament} match" pattern."""
    if not rules:
        return "ATP/WTA"
    m = re.search(r"\d{4}\s+([A-Z][A-Za-z .'-]+?(?:Open|Cup|Masters|Championships|Final|Tour))",
                  rules)
    if m:
        return m.group(1).strip()
    return "ATP/WTA"


def _round_to_code(round_str: str) -> str:
    """Map Kalshi's "Round Of 64" / "Quarterfinals" / etc. to the same
    short codes ``build_prematch_features`` already understands."""
    if not round_str:
        return "R32"
    s = round_str.lower()
    if "round of 128" in s: return "R128"
    if "round of 64" in s: return "R64"
    if "round of 32" in s: return "R32"
    if "round of 16" in s: return "R16"
    if "quarter" in s: return "QF"
    if "semi" in s: return "SF"
    if "final" in s: return "F"
    return "R32"


def write_live_state(records: list[dict]) -> str:
    """Write the canonical live-state file the watchlist exporter reads."""
    import json
    cfg = load_config()
    fp = resolve_path(cfg["paths"]["raw_dir"]) / "live_state.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    return str(fp)
