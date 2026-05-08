"""End-to-end watchlist exporter.

Takes the live-state records (provider OR fixture), runs each through:
  pre-match model → live adjustment → EV/edge → signal label

…and writes both CSV and JSON outputs. The dashboard server reads the
JSON file directly; downstream tools (Sheets, Notion) can pull the CSV.

Output schema is exactly the one the user spec'd:

  match_id, tournament, surface, player_a, player_b,
  pre_match_prob_a, pre_match_prob_b,
  live_prob_a, live_prob_b,
  market_prob_a, market_prob_b,
  edge_a, edge_b, ev_a, ev_b,
  confidence_score, volatility_score,
  injury_news_flag, recommended_action, reason_for_signal,
  current_score, last_updated
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.fetch_live_scores import load_live_state
from ..features.build_live_features import standardize
from ..models.live_adjustment_model import adjust as live_adjust
from ..models.predict import safe_predict
from ..trading.ev import ev as ev_calc
from ..trading.signals import label_match
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("dashboard.export")


def _format_score(rec: dict[str, Any]) -> str:
    a = int(rec.get("set_score_a") or 0)
    b = int(rec.get("set_score_b") or 0)
    return f"{a}-{b}"


def _round_label(level: str, round_: str) -> str:
    return f"{level} / {round_}" if round_ else level


def build_watchlist_records(live_records: list[dict[str, Any]] | None = None
                             ) -> list[dict[str, Any]]:
    cfg = load_config()
    slip = float(cfg["trading"]["slippage_pct"])

    if live_records is None:
        live_records = load_live_state()

    out: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for raw in live_records:
        rec = standardize(raw)

        pre = safe_predict(
            rec["player_a"], rec["player_b"],
            surface=rec.get("surface", "Hard"),
            level=raw.get("level", "A"),
            round_=raw.get("round", "R32"),
            rank_a=raw.get("rank_a"), rank_b=raw.get("rank_b"),
        )
        pre_prob_a = pre["prob_a"]

        adj = live_adjust(pre_prob_a, rec)
        live_prob_a = adj.live_prob_a

        market_prob_a = rec.get("market_prob_a")
        market_prob_b = (1.0 - market_prob_a) if market_prob_a is not None else None

        edge_a = (live_prob_a - market_prob_a) if market_prob_a is not None else None
        edge_b = -edge_a if edge_a is not None else None

        ev_a = ev_calc(live_prob_a, market_prob_a, slip).ev_per_contract if market_prob_a is not None else None
        ev_b = ev_calc(1 - live_prob_a, 1 - market_prob_a, slip).ev_per_contract if market_prob_a is not None else None

        sig = label_match(
            live_prob_a, market_prob_a,
            volatility=adj.volatility_score,
            injury_flag=adj.injury_news_flag,
            market_overreaction=adj.market_overreaction,
            rules_fired=adj.rules_fired,
        )

        out.append({
            "match_id": rec["match_id"] or f"{rec['player_a']}-{rec['player_b']}",
            "tournament": rec["tournament"],
            "surface": rec["surface"],
            "player_a": rec["player_a"],
            "player_b": rec["player_b"],
            "current_score": _format_score(rec),
            "round_label": _round_label(raw.get("level", "A"), raw.get("round", "")),
            "pre_match_prob_a": round(pre_prob_a, 4),
            "pre_match_prob_b": round(1 - pre_prob_a, 4),
            "live_prob_a": round(live_prob_a, 4),
            "live_prob_b": round(1 - live_prob_a, 4),
            "market_prob_a": round(market_prob_a, 4) if market_prob_a is not None else None,
            "market_prob_b": round(market_prob_b, 4) if market_prob_b is not None else None,
            "edge_a": round(edge_a, 4) if edge_a is not None else None,
            "edge_b": round(edge_b, 4) if edge_b is not None else None,
            "ev_a": round(ev_a, 4) if ev_a is not None else None,
            "ev_b": round(ev_b, 4) if ev_b is not None else None,
            "confidence_score": round(sig.confidence_score, 4),
            "volatility_score": round(adj.volatility_score, 4),
            "injury_news_flag": bool(adj.injury_news_flag),
            "recommended_action": sig.label,
            "reason_for_signal": sig.reason,
            "last_updated": now,
            # Carry through Kalshi market metadata so the trading
            # dashboard can render the same NBA-style watchlist
            # columns (Contracts = open interest, Kalshi YES/NO from
            # raw cents, etc.) without inventing numbers.
            "open_interest": raw.get("open_interest_a"),
            "yes_ask_cents_a": raw.get("yes_ask_cents_a"),
            "yes_ask_cents_b": raw.get("yes_ask_cents_b"),
        })

    return out


def export(records: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    cfg = load_config()
    csv_path = resolve_path(cfg["paths"]["watchlist_csv"])
    json_path = resolve_path(cfg["paths"]["watchlist_json"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = records if records is not None else build_watchlist_records()
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "rows": rows}, f, indent=2, default=str)
    log.info("wrote %s + %s (%d rows)", csv_path, json_path, len(rows))
    return csv_path, json_path


if __name__ == "__main__":
    export()
