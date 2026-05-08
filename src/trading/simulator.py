"""Paper-trade simulator for the tennis bot.

Mirrors the dry-run paper-trading the rest of the Kalshi suite uses:

  * Open a $1 paper position when a watchlist row's signal label is
    STRONG_EDGE / SMALL_EDGE / MARKET_OVERREACTION and there's no open
    position on that match yet.
  * Mark each open position to the current market price every tick;
    surfaces the unrealized P&L on the dashboard live.
  * When a match completes (the synthetic match-progression engine
    pins ``market_prob_a`` to 0/1 and sets ``winner_side``), settle
    the position: pay $1 if our side won, $0 otherwise. Realized P&L
    is the payout minus what we "spent" (entry market price + slippage).
  * Risk caps mirror the other bots: max-open positions, per-match
    cooldown after a settle so a wild market doesn't re-open us into
    the same losing edge twice in 60 seconds.

The state file (``data/outputs/sim_state.json``) is read by the
trading dashboard. It's a small JSON; rewriting it every tick is fine.
We don't use SQLite here for the same reason whale-watcher doesn't —
the schema is simple, append-mostly, and JSON-friendly.

Trade-cost assumptions live in ``config.trading``:
``slippage_pct`` is applied as a fixed cost on top of the entry price
(half-spread + book-walk approximation); ``bet_size`` is the unit
stake. Both are conservative defaults — real Kalshi tennis books are
typically tighter than this on liquid matches but wider on obscure
ones, so a fixed 2% is a reasonable middle.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("trading.simulator")


# Signal labels that trigger an open. Everything else is monitor-only.
_TRADEABLE_LABELS = {"STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"}
# After a settle, don't re-open on the same (recycled) match for this long.
_SAME_MATCH_COOLDOWN_SECONDS = 1800


@dataclass
class Position:
    position_id: str
    match_id: str
    tournament: str
    surface: str
    player_a: str
    player_b: str
    side: str                        # "PLAYER_A" | "PLAYER_B"
    side_player: str                 # convenience: name we're betting on
    entry_market_prob: float         # what we paid (implied prob)
    entry_model_prob: float
    label_at_open: str
    stake: float
    slippage: float
    opened_at: str
    current_market_prob: float
    current_model_prob: float
    unrealized_pnl: float
    reason_at_open: str = ""

    def __post_init__(self) -> None:
        # Recompute unrealized P&L from the current mark.
        self.unrealized_pnl = _mark_to_market_pnl(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["outputs_dir"]) / "sim_state.json"


def _empty_state() -> dict[str, Any]:
    return {
        "started_at": _now_iso(),
        "last_tick_at": _now_iso(),
        "open_positions": [],
        "closed_positions": [],
        "stats": _zero_stats(),
        "last_settled_at_by_match_id": {},
    }


def _zero_stats() -> dict[str, Any]:
    return {
        "total_opened": 0, "total_closed": 0, "open_count": 0,
        "wins": 0, "losses": 0, "win_rate": None,
        "total_realized_pnl": 0.0, "total_unrealized_pnl": 0.0,
        "total_staked": 0.0, "roi": None,
    }


def _load_state() -> dict[str, Any]:
    fp = _state_path()
    if not fp.exists():
        return _empty_state()
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        log.warning("sim state at %s unreadable — starting fresh", fp)
        return _empty_state()
    # Backfill any missing fields from older state files.
    for k, v in _empty_state().items():
        data.setdefault(k, v)
    return data


def _save_state(state: dict[str, Any]) -> None:
    fp = _state_path()
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _payout_if_win(market_prob_for_side: float, slippage: float, stake: float
                   ) -> float:
    """One-side payout: stake × (1 - market_prob - slippage)."""
    return stake * (1.0 - market_prob_for_side - slippage)


def _payout_if_loss(market_prob_for_side: float, slippage: float, stake: float
                     ) -> float:
    return -stake * (market_prob_for_side + slippage)


def _mark_to_market_pnl(p: Position) -> float:
    """Mark-to-market P&L assuming the current market price is fair.
    Convention: positive = we'd be in profit if the contract closed
    at the current market price."""
    # Mid-quote unrealized: we paid entry_market_prob; current price
    # is current_market_prob. Difference, times stake.
    delta = p.current_market_prob - p.entry_market_prob
    return round(p.stake * delta, 4)


def _aggregate_stats(state: dict[str, Any]) -> dict[str, Any]:
    open_positions = state.get("open_positions") or []
    closed = state.get("closed_positions") or []
    wins = sum(1 for c in closed if c.get("won"))
    losses = len(closed) - wins
    total_realized = sum(float(c.get("realized_pnl", 0)) for c in closed)
    total_unrealized = sum(float(p.get("unrealized_pnl", 0)) for p in open_positions)
    total_staked = sum(float(c.get("stake", 0)) for c in closed)
    win_rate = (wins / len(closed)) if closed else None
    roi = (total_realized / total_staked) if total_staked > 0 else None
    return {
        "total_opened": int(state["stats"].get("total_opened", 0)),
        "total_closed": len(closed),
        "open_count": len(open_positions),
        "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "total_realized_pnl": round(total_realized, 4),
        "total_unrealized_pnl": round(total_unrealized, 4),
        "total_staked": round(total_staked, 4),
        "roi": roi,
    }


def _pick_side(model_prob_a: float, market_prob_a: float | None) -> tuple[str, float, float]:
    """Pick whichever side has the bigger model edge.
    Returns (side_label, market_prob_for_side, model_prob_for_side)."""
    if market_prob_a is None:
        # Without a market we can't trade — caller should already
        # have filtered these out. Defensive default.
        return "PLAYER_A", 0.5, model_prob_a
    edge_a = model_prob_a - market_prob_a
    if edge_a >= 0:
        return "PLAYER_A", float(market_prob_a), float(model_prob_a)
    return "PLAYER_B", float(1.0 - market_prob_a), float(1.0 - model_prob_a)


def _within_cooldown(state: dict[str, Any], match_id: str) -> bool:
    last = state.get("last_settled_at_by_match_id", {}).get(match_id)
    if not last:
        return False
    try:
        ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - ts).total_seconds() < _SAME_MATCH_COOLDOWN_SECONDS


def _settle_position(p: dict[str, Any], live_record: dict[str, Any],
                      slippage: float) -> dict[str, Any]:
    """Close a position. ``live_record`` carries ``winner_side`` set by
    the match-progression engine (or by a real provider, when one is
    plumbed in). Returns the closed-position dict."""
    won = (p["side"] == live_record.get("winner_side"))
    stake = float(p.get("stake", 1.0))
    entry = float(p["entry_market_prob"])
    if won:
        realized = stake * (1.0 - entry - slippage)
    else:
        realized = -stake * (entry + slippage)
    return {
        **p,
        "closed_at": _now_iso(),
        "winner_side": live_record.get("winner_side"),
        "won": bool(won),
        "settle_market_prob": float(live_record.get("market_prob_a") if p["side"] == "PLAYER_A"
                                     else 1.0 - (live_record.get("market_prob_a") or 0.5)),
        "realized_pnl": round(realized, 4),
    }


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #

def tick(watchlist_rows: list[dict[str, Any]], live_records: list[dict[str, Any]]
         ) -> dict[str, Any]:
    """Run one simulator tick.

    Inputs:
      ``watchlist_rows`` — output of ``dashboard.export_watchlist.build_watchlist_records``.
        One row per match, with ``recommended_action`` etc.
      ``live_records`` — the standardized live-state list (same length).
        We use it to pick up the ``completed`` / ``winner_side`` flags
        the match-progression engine sets when a match finishes.

    Side effects: writes ``data/outputs/sim_state.json``.
    Returns the new state dict.
    """
    cfg = load_config()
    t = cfg["trading"]
    slippage = float(t["slippage_pct"])
    stake = float(t["bet_size"])

    state = _load_state()

    # Map match_id → live record, for fast lookups.
    live_by_id = {str(r.get("match_id") or ""): r for r in live_records}

    # 1) Settle any open positions whose match has completed.
    still_open: list[dict[str, Any]] = []
    newly_closed: list[dict[str, Any]] = []
    for p in (state.get("open_positions") or []):
        live = live_by_id.get(p.get("match_id", ""))
        if live and live.get("completed") and live.get("winner_side"):
            closed = _settle_position(p, live, slippage)
            newly_closed.append(closed)
            state.setdefault("last_settled_at_by_match_id", {})[p["match_id"]] = closed["closed_at"]
            log.info("settled %s on %s vs %s — won=%s, P&L=%+.3f",
                      p["side"], p["player_a"], p["player_b"],
                      closed["won"], closed["realized_pnl"])
        else:
            still_open.append(p)

    state["closed_positions"] = (state.get("closed_positions") or []) + newly_closed
    state["open_positions"] = still_open

    # 2) Mark-to-market the still-open positions using current watchlist data.
    wl_by_id = {str(r.get("match_id") or ""): r for r in watchlist_rows}
    for p in state["open_positions"]:
        wl = wl_by_id.get(p.get("match_id", ""))
        if not wl:
            continue
        market_a = wl.get("market_prob_a")
        live_a = wl.get("live_prob_a")
        if market_a is not None:
            mark_for_side = float(market_a if p["side"] == "PLAYER_A" else 1.0 - market_a)
            p["current_market_prob"] = round(mark_for_side, 4)
            p["unrealized_pnl"] = round(stake * (mark_for_side - p["entry_market_prob"]), 4)
        if live_a is not None:
            p["current_model_prob"] = round(
                live_a if p["side"] == "PLAYER_A" else 1.0 - live_a, 4
            )

    # 3) Open new positions from any tradeable signal that doesn't
    #    already have one open or in cooldown. Honor max_open_positions.
    open_match_ids = {p["match_id"] for p in state["open_positions"]}
    # Per-tick cap on simultaneous paper positions. We want to open one
    # on every tradeable edge in the watchlist, so set this high enough
    # that the watchlist size — not the cap — is the bottleneck.
    max_open = 64

    for r in watchlist_rows:
        if len(state["open_positions"]) >= max_open:
            break
        label = r.get("recommended_action", "")
        if label not in _TRADEABLE_LABELS:
            continue
        match_id = str(r.get("match_id") or "")
        if not match_id or match_id in open_match_ids:
            continue
        if _within_cooldown(state, match_id):
            continue
        market_a = r.get("market_prob_a")
        live_a = r.get("live_prob_a")
        if market_a is None or live_a is None:
            continue
        side, mkt_for_side, model_for_side = _pick_side(float(live_a), float(market_a))
        # Don't open a position that's already at the loser-extremes —
        # mirrors the trading.min_market_prob / max_market_prob band.
        if not (float(t["min_market_prob"]) <= mkt_for_side <= float(t["max_market_prob"])):
            continue
        # Edge size gate (rules engine signal already passed; this is
        # belt-and-braces against label drift).
        if abs(model_for_side - mkt_for_side) < float(t["small_edge_min"]):
            continue
        side_player = r["player_a"] if side == "PLAYER_A" else r["player_b"]
        position_id = f"{match_id}-{side}-{int(datetime.now(timezone.utc).timestamp())}"
        new_p = {
            "position_id": position_id,
            "match_id": match_id,
            "tournament": r.get("tournament", ""),
            "surface": r.get("surface", ""),
            "player_a": r.get("player_a", ""),
            "player_b": r.get("player_b", ""),
            "side": side,
            "side_player": side_player,
            "entry_market_prob": round(mkt_for_side, 4),
            "entry_model_prob": round(model_for_side, 4),
            "label_at_open": label,
            "stake": stake,
            "slippage": slippage,
            "opened_at": _now_iso(),
            "current_market_prob": round(mkt_for_side, 4),
            "current_model_prob": round(model_for_side, 4),
            "unrealized_pnl": 0.0,
            "reason_at_open": r.get("reason_for_signal", ""),
        }
        state["open_positions"].append(new_p)
        open_match_ids.add(match_id)
        state["stats"]["total_opened"] = int(state["stats"].get("total_opened", 0)) + 1
        log.info("opened %s on %s (%s vs %s) — entry %.2f, model %.2f, label %s",
                  side, side_player, r["player_a"], r["player_b"],
                  mkt_for_side, model_for_side, label)

    # 4) Recompute aggregates and persist.
    state["last_tick_at"] = _now_iso()
    state["stats"] = {**_aggregate_stats(state),
                      "total_opened": int(state["stats"].get("total_opened", 0))}
    _save_state(state)
    return state


def load_state() -> dict[str, Any]:
    """Public read-only accessor used by the dashboard."""
    return _load_state()
