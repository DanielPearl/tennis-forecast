"""Live ATP/WTA/ITF match state from api-tennis.com.

The bot's ``kalshi_markets.collapse_to_matches`` historically stamped
every in-match feature field with zero defaults (no live feed wired
in). This module supplies the real values, joined to the Kalshi
record by player name.

Endpoint
--------

``GET https://api.api-tennis.com/tennis/?method=get_livescore&APIkey=...``

Response shape (one match)::

    {
      "event_key": 12130665,
      "event_first_player": "S. Wei",          # abbreviated first name
      "event_second_player": "J. Zhang",
      "event_final_result": "1 - 1",            # SETS won so far
      "event_game_result": "A - 40",            # POINTS in current game
      "event_serve": "First Player" | "Second Player",
      "event_status": "Set 3" | "Tie Break" | "Finished" | ...,
      "event_live": "1",
      "event_type_type": "Atp Singles" | "Wta Singles" |
                          "Atp Grand Slam Singles" | "Itf Men Singles" | ...,
      "scores": [{"score_set": "1", "score_first": "6", "score_second": "1"},
                  ...],
      "pointbypoint": [...],
      "statistics": [...]    # serve%/aces/etc — empty on the free tier
    }

Name matching
-------------

api-tennis abbreviates first names ("S. Wei"); Kalshi has full
names ("Shuyue Wei"). Both reduce to the same ``(first-initial,
last-name)`` key — ``s.wei`` — so a one-shot lookup works.

Failure modes are silent: any HTTP / parse / env failure returns
``{}`` and the merger falls back to the existing zero-filled
behaviour. No regression vs the prior "no live feed" world.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from ..utils.logging_setup import setup_logging

log = setup_logging("data.api_tennis_live")


_ENDPOINT = "https://api.api-tennis.com/tennis/"
_TIMEOUT_S = 10


def _api_key() -> str:
    return os.environ.get("API_TENNIS_KEY", "").strip()


def fetch_livescore() -> list[dict[str, Any]]:
    """Return the current live-tennis match list, or ``[]`` on failure.

    Defensive against every failure mode (no key, network, non-JSON,
    success=0) — callers can treat the empty list as "no live feed
    this tick" and the merger falls through to zero defaults.
    """
    key = _api_key()
    if not key:
        return []
    try:
        r = requests.get(
            _ENDPOINT,
            params={"method": "get_livescore", "APIkey": key},
            timeout=_TIMEOUT_S,
        )
        r.raise_for_status()
        d = r.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("api-tennis livescore fetch failed: %s", exc)
        return []
    if not isinstance(d, dict) or d.get("success") != 1:
        return []
    res = d.get("result")
    return res if isinstance(res, list) else []


def _name_key(name: str) -> str:
    """Build the ``(first-initial).(last-name-joined)`` key used for
    cross-source matching. Lowercase, no spaces or periods,
    multi-word last names joined.

    Both "Alex de Minaur" and "A. de Minaur" reduce to ``a.deminaur``
    — same key, same match. Empty string when the input isn't parseable.
    """
    parts = name.replace(".", "").split()
    if not parts:
        return ""
    first = parts[0]
    rest = "".join(parts[1:]).lower()
    if not first or not rest:
        return ""
    return f"{first[0].lower()}.{rest}"


def _parse_pair(s: str) -> tuple[int, int] | None:
    """Parse api-tennis's "1 - 2" score format into ``(a, b)``.
    Returns ``None`` for any unparseable value (e.g. "A - 40" point
    scores where the deuce side reads "Advantage")."""
    if not isinstance(s, str) or "-" not in s:
        return None
    try:
        a, b = s.split("-", 1)
        return int(a.strip()), int(b.strip())
    except (TypeError, ValueError):
        return None


def _best_of(event_type_type: str) -> int:
    """Map ``event_type_type`` to best-of.

    Tennis convention: men's Grand Slam main draws are best-of-5,
    everything else (Masters, 500/250, Challengers, ITF, all women's
    tour-level + slams) is best-of-3. Davis Cup live rubbers are bo3.
    """
    et = (event_type_type or "").lower()
    if "grand slam" in et and ("men" in et or "atp" in et) and "women" not in et and "wta" not in et:
        return 5
    return 3


def _games_in_current_set(scores: list[dict[str, Any]],
                            current_set: int) -> tuple[int, int]:
    """Pull (player1_games, player2_games) for the in-progress set
    from the ``scores`` array. Returns ``(0, 0)`` when the set hasn't
    started or scores is malformed."""
    if not scores:
        return 0, 0
    for s in scores:
        try:
            if int(s.get("score_set") or 0) == current_set:
                f = int(s.get("score_first") or 0)
                sd = int(s.get("score_second") or 0)
                return f, sd
        except (TypeError, ValueError):
            continue
    # Set not found in scores yet — between sets / set just started.
    return 0, 0


def _games_won_last_3(pointbypoint: list[dict[str, Any]]) -> tuple[int, int]:
    """Count how many of the last 3 completed games each player won.
    ``pointbypoint`` entries have ``serve_winner`` set to "First Player"
    or "Second Player" for the side that won that game; entries
    without a serve_winner are still in progress."""
    if not pointbypoint:
        return 0, 0
    completed = [g for g in pointbypoint
                 if g.get("serve_winner") in ("First Player", "Second Player")]
    if not completed:
        return 0, 0
    last3 = completed[-3:]
    f = sum(1 for g in last3 if g.get("serve_winner") == "First Player")
    s = sum(1 for g in last3 if g.get("serve_winner") == "Second Player")
    return f, s


def _current_set_no(event_status: str, sets: tuple[int, int] | None) -> int:
    """Resolve the current set number from ``event_status`` ("Set 3",
    "Tie Break") falling back to ``sets_a + sets_b + 1``. Tiebreak
    counts as part of the current set, not a new one."""
    s = (event_status or "").strip().lower()
    if s.startswith("set "):
        try:
            return int(s.split()[1])
        except (IndexError, ValueError):
            pass
    if sets is not None:
        return sets[0] + sets[1] + 1
    return 1


def _normalize_match(m: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one api-tennis match dict to the canonical in-match
    state schema. Keys retain the api-tennis "first"/"second"
    orientation — the caller's lookup flips them to Kalshi's a/b
    assignment.

    Returns ``None`` for non-live or unparseable matches.
    """
    if m.get("event_live") != "1":
        return None
    p1 = (m.get("event_first_player") or "").strip()
    p2 = (m.get("event_second_player") or "").strip()
    if not p1 or not p2:
        return None
    sets = _parse_pair(m.get("event_final_result") or "") or (0, 0)
    best_of = _best_of(m.get("event_type_type") or "")
    current_set = _current_set_no(m.get("event_status") or "", sets)
    scores = m.get("scores") or []
    games = _games_in_current_set(scores, current_set)
    pbp = m.get("pointbypoint") or []
    g3 = _games_won_last_3(pbp)
    status = (m.get("event_status") or "").lower()
    is_tiebreak = "tie" in status or (games[0] == 6 and games[1] == 6)
    is_decider = (sets[0] + sets[1] == best_of - 1)
    progress = min(1.5, (sets[0] + sets[1]) / float(best_of)) if best_of else 0.0
    serving_first = m.get("event_serve") == "First Player"
    return {
        "set_score_first": sets[0],
        "set_score_second": sets[1],
        "current_set_games_first": games[0],
        "current_set_games_second": games[1],
        "current_set": current_set,
        "best_of": best_of,
        "is_tiebreak": is_tiebreak,
        "is_decider": is_decider,
        "progress": progress,
        "serving_first": serving_first,
        "games_won_last_3_first": g3[0],
        "games_won_last_3_second": g3[1],
        "_first_player": p1,
        "_second_player": p2,
        "_event_key": m.get("event_key"),
    }


def build_state_by_key(matches: list[dict[str, Any]]
                       ) -> dict[str, dict[str, Any]]:
    """Reduce a livescore response to a ``{name_key → state}`` dict.

    Each match is registered under BOTH players' name keys so a
    Kalshi-side lookup hits regardless of orientation. If two
    different live matches happen to collide on a name key (rare —
    "L. Williams" matching Lara vs Liz — and unlikely at tour level
    where Kalshi books trade), the later entry wins; the lookup
    layer rejects ambiguous matches via cross-player verification.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in matches:
        st = _normalize_match(m)
        if st is None:
            continue
        k1 = _name_key(st["_first_player"])
        k2 = _name_key(st["_second_player"])
        if k1:
            out[k1] = st
        if k2:
            out[k2] = st
    return out


def lookup_for_kalshi(state_by_key: dict[str, dict[str, Any]],
                       kalshi_player_a: str,
                       kalshi_player_b: str) -> dict[str, Any] | None:
    """Find the in-match state for ``(player_a, player_b)`` and
    orient it to Kalshi's a/b assignment. Returns ``None`` when no
    live match matches BOTH players (a single-name hit alone could
    be a collision so we don't take it).
    """
    ka = _name_key(kalshi_player_a)
    kb = _name_key(kalshi_player_b)
    if not ka or not kb:
        return None
    st = state_by_key.get(ka) or state_by_key.get(kb)
    if st is None:
        return None
    api_first_key = _name_key(st["_first_player"])
    api_second_key = _name_key(st["_second_player"])
    # Require BOTH Kalshi players to match the api-tennis players (in
    # either orientation). One-sided matches are likely name collisions
    # — a different match where one name happens to overlap — and we
    # refuse to merge them. Better to leave the record blank than
    # poison it with the wrong opponent's state.
    if {ka, kb} != {api_first_key, api_second_key}:
        return None
    a_is_first = (ka == api_first_key)
    if a_is_first:
        return {
            "set_score_a": st["set_score_first"],
            "set_score_b": st["set_score_second"],
            "current_set_games_a": st["current_set_games_first"],
            "current_set_games_b": st["current_set_games_second"],
            "games_won_last_3_a": st["games_won_last_3_first"],
            "games_won_last_3_b": st["games_won_last_3_second"],
            "current_set": st["current_set"],
            "best_of": st["best_of"],
            "is_tiebreak": st["is_tiebreak"],
            "is_decider": st["is_decider"],
            "progress": st["progress"],
            "serving_a": st["serving_first"],
        }
    return {
        "set_score_a": st["set_score_second"],
        "set_score_b": st["set_score_first"],
        "current_set_games_a": st["current_set_games_second"],
        "current_set_games_b": st["current_set_games_first"],
        "games_won_last_3_a": st["games_won_last_3_second"],
        "games_won_last_3_b": st["games_won_last_3_first"],
        "current_set": st["current_set"],
        "best_of": st["best_of"],
        "is_tiebreak": st["is_tiebreak"],
        "is_decider": st["is_decider"],
        "progress": st["progress"],
        "serving_a": not st["serving_first"],
    }
