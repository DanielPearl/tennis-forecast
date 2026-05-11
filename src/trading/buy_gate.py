"""Shared BUY-gate evaluator.

The trading simulator and the watchlist exporter both need to ask the
same question of a candidate row: *if I were to open a paper position
on the favoured side of this match right now, would all the buy gates
pass?* Keeping the evaluation in one place means a tightened threshold
in ``config.trading`` takes effect everywhere on the next refresh and
the dashboard's "Top 10 buys" view never disagrees with the simulator's
actual fills.

Output schema (per row):

  buy_eligible: bool      — every gate passes for the favoured side
  buy_score: float        — edge × ev for the favoured side; the
                            ranking key for "best N buys" displays.
                            Always 0 when ineligible.
  buy_side: "A" | "B" | None
  buy_gates: dict[str, bool]  — per-gate pass/fail. Empty when
                                the row has no quoted market.
  buy_blockers: list[str]     — human-readable list of which gates
                                stopped the row, in priority order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ev import ev as ev_calc


_TRADEABLE_LABELS = {"STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"}


@dataclass
class BuyDecision:
    eligible: bool
    score: float
    side: str | None
    side_edge: float
    side_ev: float | None
    side_market: float | None
    gates: dict[str, bool]
    blockers: list[str]


def evaluate(row: dict[str, Any], trading_cfg: dict[str, Any]) -> BuyDecision:
    """Apply the standard BUY gate to one watchlist row. Returns a
    BuyDecision regardless of whether the row is eligible; the caller
    inspects ``eligible`` and ``score``."""
    live_a = row.get("live_prob_a")
    market_a = row.get("market_prob_a")

    if live_a is None or market_a is None:
        return BuyDecision(
            eligible=False, score=0.0, side=None,
            side_edge=0.0, side_ev=None, side_market=None,
            gates={}, blockers=["no quoted market"],
        )

    live_a = float(live_a)
    market_a = float(market_a)
    edge_a = live_a - market_a
    side = "A" if edge_a >= 0 else "B"
    side_edge = abs(edge_a)
    side_market = market_a if side == "A" else (1.0 - market_a)
    side_model = live_a if side == "A" else (1.0 - live_a)
    side_ev = (row.get("ev_a") if side == "A" else row.get("ev_b"))
    if side_ev is None:
        # Fall back to a fresh EV calculation when the exporter didn't
        # stamp one — keeps callers that build rows by hand happy.
        slippage = float(trading_cfg.get("slippage_pct", 0.02))
        side_ev = ev_calc(side_model, side_market, slippage).ev_per_contract
    side_ev = float(side_ev)

    # ``require_strong_edge`` (config flag) tightens the gate so the
    # simulator only fires on STRONG_EDGE labels — drops SMALL_EDGE and
    # MARKET_OVERREACTION. Also forces the numeric edge floor to
    # ``strong_edge_min`` so a STRONG_EDGE label with a smaller
    # underlying edge (shouldn't happen, but belt-and-braces) also
    # bounces.
    require_strong = bool(trading_cfg.get("require_strong_edge", False))
    small_edge = float(trading_cfg.get("small_edge_min", 0.04))
    strong_edge = float(trading_cfg.get("strong_edge_min", small_edge))
    edge_floor = strong_edge if require_strong else small_edge
    tradeable_labels = ({"STRONG_EDGE"} if require_strong
                          else _TRADEABLE_LABELS)
    min_ev = float(trading_cfg.get("min_ev", 0.0))
    min_p = float(trading_cfg.get("min_market_prob", 0.0))
    max_p = float(trading_cfg.get("max_market_prob", 1.0))
    min_oi_cfg = trading_cfg.get("min_open_interest")
    max_spread_cfg = trading_cfg.get("max_spread_cents")
    max_vol = float(trading_cfg.get("max_tradable_volatility", 1.0))

    oi = row.get("open_interest")
    spread = row.get("spread_cents")
    vol_score = float(row.get("volatility_score") or 0.0)
    label = (row.get("recommended_action") or "").upper()

    gates = {
        "label": label in tradeable_labels,
        "edge": side_edge >= edge_floor,
        "ev": side_ev >= min_ev,
        "price_band": min_p <= side_market <= max_p,
        "volatility": vol_score < max_vol,
    }
    if min_oi_cfg is not None:
        gates["open_interest"] = (oi is not None
                                    and float(oi) >= float(min_oi_cfg))
    if max_spread_cfg is not None:
        # Missing spread doesn't block — many Kalshi markets ship without
        # a bid quote in slow books, and rejecting on a missing field
        # would be too strict.
        gates["spread"] = (spread is None
                            or float(spread) <= float(max_spread_cfg))

    eligible = all(gates.values())
    score = (side_edge * side_ev) if eligible else 0.0
    blockers = [k for k, v in gates.items() if not v]

    return BuyDecision(
        eligible=eligible,
        score=float(score),
        side=side,
        side_edge=float(side_edge),
        side_ev=side_ev,
        side_market=float(side_market),
        gates=gates,
        blockers=blockers,
    )
