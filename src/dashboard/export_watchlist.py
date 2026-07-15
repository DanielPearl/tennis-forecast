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

        # Edge + EV columns are driven by Pinnacle (the sharp reference
        # the buy gate now uses) so the displayed numbers match the
        # decision the bot is actually making. When Pinnacle isn't
        # listing the match, fall back to the model view — better than
        # showing a blank cell.
        edge_prob_a = (pinnacle_prob_a if pinnacle_prob_a is not None
                        else live_prob_a)
        edge_a = ((edge_prob_a - market_prob_a)
                    if market_prob_a is not None else None)
        edge_b = -edge_a if edge_a is not None else None

        ev_a = (ev_calc(edge_prob_a, market_prob_a, slip).ev_per_contract
                  if market_prob_a is not None else None)
        ev_b = (ev_calc(1 - edge_prob_a, 1 - market_prob_a, slip).ev_per_contract
                  if market_prob_a is not None else None)

        # Signal label — use Pinnacle vs Kalshi (the same reference the
        # buy gate now uses) so ``recommended_action`` isn't stuck at
        # WATCH when Pinnacle disagrees strongly with Kalshi but our
        # own model happens to sit near the market price. Falls back to
        # the model view when Pinnacle doesn't list the match.
        sig_prob_a = (pinnacle_prob_a if pinnacle_prob_a is not None
                       else live_prob_a)
        sig = label_match(
            sig_prob_a, market_prob_a,
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
            "volume": ((raw.get("volume_a") or 0)
                       + (raw.get("volume_b") or 0)),
            "spread_cents": raw.get("spread_cents"),
            "yes_ask_cents_a": raw.get("yes_ask_cents_a"),
            "yes_ask_cents_b": raw.get("yes_ask_cents_b"),
            # Per-side Kalshi market tickers — the live executor uses
            # these to identify the exact market to place the order
            # against. Without them, ``_maybe_place`` silently returns
            # on every row (the ``if not ticker: return`` guard). Not
            # rendered by the dashboard directly but part of the
            # contract between this export and the trading loop.
            "ticker_a": raw.get("ticker_a"),
            "ticker_b": raw.get("ticker_b"),
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
            # Kalshi's ``rules_primary`` string for this contract —
            # the paragraph the dashboard's Kalshi-rules section
            # renders verbatim. Available on every market via the
            # Kalshi markets API; the tennis feed already reads it
            # into ``raw`` in ``collapse_to_matches``.
            "rules_primary": raw.get("rules_primary"),
            # Settlement flags — surfaced so the dashboard's Model-vs-
            # market table can drop rows whose Kalshi markets have
            # already settled (the contract is over, nothing left to
            # trade). ``completed`` is True whenever Kalshi's market
            # status is ``closed`` / ``settled`` / ``finalized``;
            # ``winner_side`` records which side prevailed (PLAYER_A /
            # PLAYER_B / None) for the dashboard's audit trail.
            # ``expected_expiration_time`` is Kalshi's own ISO stamp
            # for when the market is scheduled to expire; the
            # dashboard uses it as a secondary settled-detection
            # signal for the common case where Kalshi is slow to
            # flip the status flag on a match that's already ended
            # in real life.
            "completed": bool(raw.get("completed")),
            "winner_side": raw.get("winner_side"),
            "expected_expiration_time":
                raw.get("expected_expiration_time"),
        }
        # BUY gate evaluation — sets buy_eligible, buy_score, buy_side,
        # buy_gates and buy_blockers using the shared evaluator.
        #
        # Reference cascade for the buy-gate's edge computation:
        #   1. Pinnacle's devigged probability when the sharp-book
        #      cascade quotes the match (Pinnacle guest → Betfair
        #      Exchange UK → Betfair Exchange EU).
        #   2. Our trained model's ``live_prob_a`` when no professional
        #      book is quoting (deep ITF Futures, between-tournament
        #      weeks). This used to trigger a ``no_pinnacle_reference``
        #      blocker that hard-skipped the trade; 2026-07-10 the
        #      user asked us to drop the safety catch and trust the
        #      internal model on rows the sharp cascade misses, since
        #      the whole point of adding Betfair + Pinnacle guest was
        #      to extend coverage into the tier where Kalshi lists but
        #      no sharp reference wants to price. The trained model
        #      holds up on holdout (Brier 0.145) — good enough to be
        #      a reference when nothing else is quoting.
        if pinnacle_prob_a is not None:
            # The gate uses ``live_prob_a`` to compute its edge; swap
            # in Pinnacle so the gate compares Pinnacle-vs-Kalshi. The
            # row's own ev_a / ev_b are already Pinnacle-based (set
            # above) so the gate reads consistent numbers.
            gate_row = dict(row)
            gate_row["live_prob_a"] = pinnacle_prob_a
            decision = evaluate_buy(gate_row, cfg.get("trading") or {})
            # Model-disagreement veto (relaxed 2026-07-15 per user).
            # Original 07-14 gate required the internal model to
            # independently see edge >= 9pp on the same side — that
            # permanently disqualified matches where our internal
            # model has sparse data (ITF debutants, Challenger
            # prospects) even when the sharp book was fresh. New rule:
            # only block when the internal model ACTIVELY disagrees
            # by > 10pp on the same side. A silent internal model
            # (edge near zero) is treated as "no vote", not "no".
            _disagree_floor = 0.10  # pp of ACTIVE opposition
            if (decision.eligible and decision.side in ("A", "B")
                    and live_prob_a is not None):
                _ask_c = row.get("yes_ask_cents_a" if decision.side == "A"
                                  else "yes_ask_cents_b")
                _model_side = (float(live_prob_a) if decision.side == "A"
                                else 1.0 - float(live_prob_a))
                _model_edge = ((_model_side - float(_ask_c) / 100.0)
                                if _ask_c is not None else None)
                # Only block on ACTIVE disagreement (edge < -10pp).
                # None (no ask) is a separate broken-row problem, not
                # a model-disagreement problem — let it through to
                # the executor's own gates rather than blocking here.
                if _model_edge is not None and _model_edge < -_disagree_floor:
                    decision.eligible = False
                    decision.blockers = list(decision.blockers) + [
                        f"internal_model_disagrees_{_model_edge*100:+.1f}pp"
                        f"<-{_disagree_floor*100:.0f}pp"
                    ]
                    decision.gates = dict(decision.gates,
                                           model_confirms=False)
        else:
            # Fall through to the raw model prob. Every other gate
            # (min_ev, price_band, spread, open_interest, max_entry)
            # still applies; only the sharp-reference requirement
            # dropped.
            decision = evaluate_buy(row, cfg.get("trading") or {})
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


def _write_effective_config(cfg: dict[str, Any], json_path: Path) -> None:
    """Dump the tennis bot's *actual* buy-side thresholds to
    ``effective_config.json`` next to the watchlist. The trading
    dashboard reads this to render the "what does this bot need before
    it'll buy?" modal against real numbers instead of generic macro-bot
    defaults. Silently no-ops on any error — the dashboard already
    tolerates a missing file (falls back to display defaults).
    """
    try:
        t = cfg.get("trading") or {}
        pmin = float(t.get("min_market_prob", 0.0))
        pmax = float(t.get("max_market_prob", 1.0))
        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "edge": {
                "min_ev_per_contract": float(t.get("min_ev", 0.03)),
                "min_prob_edge_over_breakeven": float(t.get("small_edge_min", 0.05)),
                "min_raw_model_edge": float(t.get("small_edge_min", 0.05)),
                "max_entry_price_cents": t.get("max_entry_price_cents"),
                "min_model_confidence": None,
                "min_model_accuracy": None,
            },
            "validators": {
                "max_spread_cents": t.get("max_spread_cents"),
                "min_open_interest": t.get("min_open_interest"),
                "prob_bounds_cents": [int(round(pmin * 100)),
                                        int(round(pmax * 100))],
                "min_book_depth_contracts": None,
                "min_volume": None,
                "min_depth_at_best_ask": None,
                "min_minutes_to_close": None,
                "max_minutes_to_close": None,
                "basis_risk_strike_window_dollars": None,
                "basis_risk_max_hours_to_close": None,
            },
            "risk": {
                "bet_size_cents": int(round(float(t.get("bet_size", 1.0)) * 100)),
                "max_open_positions": t.get("max_open_positions"),
                "max_total_exposure_cents": None,
                "max_bets_per_day": None,
                "cooldown_seconds_same_market": None,
            },
            "hedge": {
                "enabled": True,
                "profit_lock_cents": None,
                "stop_loss_cents": None,
                "hedge_size_fraction": None,
                "profit_lock_market_prob": t.get("profit_lock_market_prob"),
            },
            "extra": {
                "kind": "tennis",
                "reference_book": "pinnacle_devigged",
                "notes": (
                    "This bot compares Kalshi to Pinnacle's devigged "
                    "probability (a sharp global reference), NOT the "
                    "internal model. Edge shown in every column is "
                    "Pinnacle − Kalshi. Rows Pinnacle doesn't list "
                    "(Challenger / ITF / between-tournaments) are "
                    "always skipped with a ``no_pinnacle_reference`` "
                    "blocker."
                ),
                "max_edge_skip": t.get("max_edge_skip"),
                "taper_edge_above": t.get("taper_edge_above"),
                "taper_min_stake_frac": t.get("taper_min_stake_frac"),
                "max_tradable_volatility": t.get("max_tradable_volatility"),
                "slippage_pct": t.get("slippage_pct"),
            },
        }
        eff_path = json_path.parent / "effective_config.json"
        tmp = eff_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(eff_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("effective_config.json write failed: %s", exc)


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
    _write_effective_config(cfg, json_path)
    log.info("wrote %s + %s (%d rows)", csv_path, json_path, len(rows))
    return csv_path, json_path


if __name__ == "__main__":
    export()
