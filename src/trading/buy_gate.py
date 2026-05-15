"""Shared BUY-gate evaluator (thin wrapper over kalshi_sdk).

The trading simulator and the watchlist exporter both ask the same
question of a candidate row. Gate logic lives in
``kalshi_sdk.validators.evaluate_row_gates`` so tennis, darts, and
table-tennis all evaluate the same way.

Output schema (per row, preserved for backward compat):
  buy_eligible: bool       — every gate passes for the favoured side
  buy_score: float         — edge × ev for the favoured side
  buy_side: "A" | "B" | None
  buy_gates: dict[str, bool]
  buy_blockers: list[str]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kalshi_sdk.validators import evaluate_row_gates


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
    """Apply the standard BUY gate to one watchlist row."""
    row_with_ev = dict(row)
    row_with_ev.setdefault("__ev_module", "trading.ev")
    result = evaluate_row_gates(
        row_with_ev,
        small_edge_min=float(trading_cfg.get("small_edge_min", 0.05)),
        strong_edge_min=float(trading_cfg.get("strong_edge_min",
                                              trading_cfg.get("small_edge_min", 0.05))),
        require_strong_edge=bool(trading_cfg.get("require_strong_edge", False)),
        min_ev=float(trading_cfg.get("min_ev", 0.03)),
        min_market_prob=float(trading_cfg.get("min_market_prob", 0.0)),
        max_market_prob=float(trading_cfg.get("max_market_prob", 1.0)),
        max_tradable_volatility=float(trading_cfg.get("max_tradable_volatility", 1.0)),
        min_open_interest=trading_cfg.get("min_open_interest"),
        max_spread_cents=trading_cfg.get("max_spread_cents"),
        max_entry_price_cents=trading_cfg.get("max_entry_price_cents"),
        slippage_pct=float(trading_cfg.get("slippage_pct", 0.02)),
    )
    return BuyDecision(
        eligible=result.eligible,
        score=result.score,
        side=result.side,
        side_edge=result.side_edge,
        side_ev=result.side_ev,
        side_market=result.side_market,
        gates=result.gates,
        blockers=result.blockers,
    )
