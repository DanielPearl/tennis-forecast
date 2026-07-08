"""End-to-end watchlist exporter.

Each tick takes the live-market records (Kalshi feed) and runs them
through:

  pre-match model → EV/edge → signal label → buy gate

…and writes both CSV and JSON outputs. The dashboard server reads the
JSON file directly; downstream tools (Sheets, Notion) can pull the CSV.

The in-match adjustment layer was removed on 2026-07-08 — the bot now
places bets based purely on the pre-match model's prob vs the Kalshi
market. ``live_prob_a`` is kept in the output schema as an alias of
``pre_match_prob_a`` so downstream consumers (dashboard renderers,
simulator, live executor) don't need schema updates.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.fetch_live_scores import load_live_state
from ..data.fetch_odds import pinnacle_probs_by_pair
from ..models.predict import safe_predict
from ..trading.buy_gate import evaluate as evaluate_buy
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

    # Pinnacle line lookup for the whole batch. One API call per
    # currently-active tennis sport key, cached 5 min inside
    # ``fetch_odds`` so per-tick cost stays inside the 20K/mo quota
    # of The Odds API's paid tier. Empty dict silently when the key
    # isn't set or the API is down — every downstream user tolerates
    # a missing pinnacle_prob field.
    pinnacle_lookup = pinnacle_probs_by_pair()

    out: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for raw in live_records:
        market_prob_a = raw.get("market_prob_a")
        try:
            market_prob_a = float(market_prob_a) if market_prob_a is not None else None
        except (TypeError, ValueError):
            market_prob_a = None
        rec = {
            "match_id": str(raw.get("match_id") or ""),
            "tournament": raw.get("tournament") or "Unknown",
            "surface": raw.get("surface") or "Hard",
            "player_a": raw.get("player_a") or "",
            "player_b": raw.get("player_b") or "",
            "market_prob_a": market_prob_a,
            "set_score_a": int(raw.get("set_score_a") or 0),
            "set_score_b": int(raw.get("set_score_b") or 0),
        }

        pre = safe_predict(
            rec["player_a"], rec["player_b"],
            surface=rec["surface"],
            level=raw.get("level", "A"),
            round_=raw.get("round", "R32"),
            rank_a=raw.get("rank_a"), rank_b=raw.get("rank_b"),
        )
        pre_prob_a = pre["prob_a"]
        # Surface which prediction path was used so the live executor
        # can refuse to place orders when the row's prob came from
        # the Elo-only or 50/50 fallback (i.e. the trained model
        # isn't loadable). Anything other than ``"trained"`` should
        # gate out new orders on the consuming side.
        model_source = pre.get("model_source", "trained")

        # Pre-match-only mode: live_prob is an alias for pre_match_prob
        # (no in-match adjustment layer). Kept in the output schema so
        # downstream code that reads ``live_prob_a`` keeps working.
        live_prob_a = pre_prob_a
        market_prob_b = (1.0 - market_prob_a) if market_prob_a is not None else None

        # Pinnacle probability lookup — sharp global-reference line.
        # Matched by frozenset of (player_a, player_b) so orientation
        # doesn't matter, then the per-side prob is picked by exact
        # name match (falls back to loose last-name match if the Odds
        # API spells a name slightly differently than Kalshi).
        pinnacle_prob_a = None
        pinnacle_prob_b = None
        pair_key = frozenset({rec["player_a"], rec["player_b"]})
        pinn_map = pinnacle_lookup.get(pair_key)
        if pinn_map is None and rec["player_a"] and rec["player_b"]:
            # Loose fallback — a hyphen / diacritic / initial difference
            # on ONE player would drop the exact match. Scan for a
            # frozenset that shares last-name tokens with both.
            la = rec["player_a"].split()[-1].lower()
            lb = rec["player_b"].split()[-1].lower()
            for key_set, probs in pinnacle_lookup.items():
                keyed_names = list(key_set)
                if len(keyed_names) != 2: continue
                lnames = [n.split()[-1].lower() for n in keyed_names]
                if la in lnames and lb in lnames:
                    pinn_map = probs
                    break
        if pinn_map is not None:
            # Now pick out which pinn_map entry is player_a's prob.
            # Try exact first, then loose last-name match.
            for name, prob in pinn_map.items():
                if name == rec["player_a"]:
                    pinnacle_prob_a = float(prob)
                    break
            if pinnacle_prob_a is None and rec["player_a"]:
                la = rec["player_a"].split()[-1].lower()
                for name, prob in pinn_map.items():
                    if la in name.lower():
                        pinnacle_prob_a = float(prob)
                        break
            if pinnacle_prob_a is not None:
                pinnacle_prob_b = 1.0 - pinnacle_prob_a

        edge_a = (live_prob_a - market_prob_a) if market_prob_a is not None else None
        edge_b = -edge_a if edge_a is not None else None

        ev_a = ev_calc(live_prob_a, market_prob_a, slip).ev_per_contract if market_prob_a is not None else None
        ev_b = ev_calc(1 - live_prob_a, 1 - market_prob_a, slip).ev_per_contract if market_prob_a is not None else None

        sig = label_match(
            live_prob_a, market_prob_a,
            volatility=0.0, injury_flag=False,
            market_overreaction=False, rules_fired=[],
        )

        # Stage the row, then evaluate the BUY gate against it so the
        # dashboard's "Top 10 buys" view and the simulator agree on
        # eligibility on every refresh.
        row = {
            "match_id": rec["match_id"] or f"{rec['player_a']}-{rec['player_b']}",
            "tournament": rec["tournament"],
            "surface": rec["surface"],
            "player_a": rec["player_a"],
            "player_b": rec["player_b"],
            "current_score": _format_score(rec),
            "round_label": _round_label(raw.get("level", "A"), raw.get("round", "")),
            "pre_match_prob_a": round(pre_prob_a, 4),
            "pre_match_prob_b": round(1 - pre_prob_a, 4),
            "model_source": model_source,
            "live_prob_a": round(live_prob_a, 4),
            "live_prob_b": round(1 - live_prob_a, 4),
            "market_prob_a": round(market_prob_a, 4) if market_prob_a is not None else None,
            "market_prob_b": round(market_prob_b, 4) if market_prob_b is not None else None,
            # Pinnacle devigged probability — sharp reference line pulled
            # from The Odds API. None when the API key isn't set, the
            # match isn't listed on The Odds API (e.g. Challenger/ITF
            # events), or the API's returning an empty book for it.
            "pinnacle_prob_a": (round(pinnacle_prob_a, 4)
                                 if pinnacle_prob_a is not None else None),
            "pinnacle_prob_b": (round(pinnacle_prob_b, 4)
                                 if pinnacle_prob_b is not None else None),
            "edge_a": round(edge_a, 4) if edge_a is not None else None,
            "edge_b": round(edge_b, 4) if edge_b is not None else None,
            "ev_a": round(ev_a, 4) if ev_a is not None else None,
            "ev_b": round(ev_b, 4) if ev_b is not None else None,
            "confidence_score": round(sig.confidence_score, 4),
            "volatility_score": 0.0,
            "injury_news_flag": False,
            "recommended_action": sig.label,
            "reason_for_signal": sig.reason,
            "last_updated": now,
            # Carry through Kalshi market metadata so the trading
            # dashboard can render the same NBA-style watchlist
            # columns (Contracts = open interest, Kalshi YES/NO from
            # raw cents, etc.) without inventing numbers.
            "open_interest": raw.get("open_interest_a"),
            "volume": raw.get("volume_a"),
            "spread_cents": raw.get("spread_cents"),
            "yes_ask_cents_a": raw.get("yes_ask_cents_a"),
            "yes_ask_cents_b": raw.get("yes_ask_cents_b"),
            # Kalshi-published contract titles — both sides carried
            # through so the simulator can stamp the right one on the
            # position record at open time, and the watchlist's Title
            # column shows whichever side the model favours.
            "title_a": raw.get("title_a"),
            "title_b": raw.get("title_b"),
            "title": (raw.get("title_a") if (edge_a or 0) >= 0
                       else raw.get("title_b")),
            # Kalshi event-page heading ("Choinski vs Herbert") — what
            # the user sees when they click the ticker. Pass it through
            # so the dashboard's Title column matches the click target.
            "event_title": raw.get("event_title"),
        }
        # BUY gate evaluation — sets buy_eligible, buy_score, buy_side,
        # buy_gates and buy_blockers using the shared evaluator.
        #
        # We feed Pinnacle's devigged probability (not our internal
        # model) as the ``live_prob_a`` the gate compares against
        # Kalshi. Pinnacle is the sharpest global reference, so a
        # Kalshi price that disagrees with Pinnacle is a real edge;
        # a Kalshi price that only disagrees with our model is just
        # our model being wrong. If Pinnacle isn't listing the match
        # (Challenger / ITF / between-tournaments), skip the buy —
        # trading without a sharp reference is what got us into the
        # miscalibration hole in the first place.
        if pinnacle_prob_a is not None:
            gate_row = dict(row)
            gate_row["live_prob_a"] = pinnacle_prob_a
            if market_prob_a is not None:
                gate_row["ev_a"] = ev_calc(
                    pinnacle_prob_a, market_prob_a, slip,
                ).ev_per_contract
                gate_row["ev_b"] = ev_calc(
                    1 - pinnacle_prob_a, 1 - market_prob_a, slip,
                ).ev_per_contract
            decision = evaluate_buy(gate_row, cfg.get("trading") or {})
        else:
            decision = evaluate_buy(row, cfg.get("trading") or {})
            decision.eligible = False
            decision.score = 0.0
            decision.blockers = list(decision.blockers) + ["no_pinnacle_reference"]
        row["buy_eligible"] = bool(decision.eligible)
        row["buy_score"] = round(float(decision.score), 6)
        row["buy_side"] = decision.side
        row["buy_side_edge"] = round(float(decision.side_edge), 4)
        row["buy_side_ev"] = (round(float(decision.side_ev), 4)
                                if decision.side_ev is not None else None)
        row["buy_gates"] = decision.gates
        row["buy_blockers"] = decision.blockers
        out.append(row)

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
