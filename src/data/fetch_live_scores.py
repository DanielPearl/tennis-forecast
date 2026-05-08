"""Live-match state ingestion.

This module is the seam between the rest of the system and whichever
provider you want to use for in-match data. The MVP supports two paths:

1. A real provider (e.g. SofaScore's public-but-undocumented JSON API).
   Toggled by ``SOFASCORE_BASE_URL`` env var. Use at your own risk —
   not all providers permit programmatic access without an agreement.
2. A seeded JSON fixture at ``data/raw/live_state.json`` so the
   pipeline runs end-to-end without any third-party dependency. The
   shape mirrors the live-feature schema in ``build_live_features.py``.

The dashboard works on either path. For production, replace
``fetch_provider_state`` with whichever paid feed you license.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.live")


def _fixture_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["raw_dir"]) / "live_state.json"


def fetch_provider_state() -> list[dict[str, Any]] | None:
    """Pull live-match state from SofaScore (or whatever provider is
    plumbed in via env). Returns ``None`` to mean "no provider
    configured" — the caller then falls back to the fixture file.

    Replace the body of this function with your licensed feed's call.
    """
    base = os.environ.get("SOFASCORE_BASE_URL", "").strip()
    if not base:
        return None
    try:
        r = requests.get(f"{base}/sport/tennis/events/live", timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("provider live fetch failed: %s", exc)
        return None
    events = (r.json() or {}).get("events", [])
    out: list[dict[str, Any]] = []
    for ev in events:
        # Defensive parsing — provider payloads change without warning.
        try:
            home = ev["homeTeam"]["name"]
            away = ev["awayTeam"]["name"]
            score_h = ev.get("homeScore", {}).get("display", 0)
            score_a = ev.get("awayScore", {}).get("display", 0)
            out.append({
                "match_id": str(ev.get("id", f"{home}-{away}")),
                "tournament": ev.get("tournament", {}).get("name", "Unknown"),
                "surface": ev.get("groundType", "Hard"),
                "player_a": home,
                "player_b": away,
                "set_score_a": int(score_h),
                "set_score_b": int(score_a),
                # The remaining live fields aren't in the public list endpoint
                # — a full implementation would hit /event/{id}/statistics.
                "games_won_last_3_a": 0,
                "games_won_last_3_b": 0,
                "first_serve_pct_a": 0.62,
                "first_serve_pct_b": 0.62,
                "break_points_saved_pct_a": 0.65,
                "break_points_saved_pct_b": 0.65,
                "is_tiebreak": False,
                "is_decider": False,
                "medical_timeout": False,
                "market_prob_a": None,
                "market_prob_a_prev": None,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_live_state() -> list[dict[str, Any]]:
    """Public entry point. Provider first, fixture file fallback."""
    state = fetch_provider_state()
    if state is not None:
        return state
    fp = _fixture_path()
    if not fp.exists():
        log.info("no live fixture at %s — returning []", fp)
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def write_fixture(records: list[dict[str, Any]]) -> Path:
    """Write a fixture file used by tests / demo."""
    fp = _fixture_path()
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return fp
