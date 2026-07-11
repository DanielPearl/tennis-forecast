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
    # strong_edge_min / require_strong_edge were retired 2026-07-11
    # when the two-tier taxonomy collapsed to a single EDGE label
    # (see ``signals.py``). Feed the SDK the same ``small_edge_min``
    # for both floors + hard-code ``require_strong_edge=False`` so the
    # strong branch never fires, and pass ``tradeable_labels={"EDGE"}``
    # explicitly — the SDK's default is still {STRONG_EDGE, SMALL_EDGE,
    # MARKET_OVERREACTION}, which the tennis signals no longer emit,
    # so without this override every row's ``label`` gate fails.
    _sm = float(trading_cfg.get("small_edge_min", 0.05))
    result = evaluate_row_gates(
        row_with_ev,
        small_edge_min=_sm,
        strong_edge_min=_sm,
        require_strong_edge=False,
        tradeable_labels={"EDGE"},
        min_ev=float(trading_cfg.get("min_ev", 0.03)),
        min_market_prob=float(trading_cfg.get("min_market_prob", 0.0)),
        max_market_prob=float(trading_cfg.get("max_market_prob", 1.0)),
        max_tradable_volatility=float(trading_cfg.get("max_tradable_volatility", 1.0)),
        min_open_interest=trading_cfg.get("min_open_interest"),
        max_spread_cents=trading_cfg.get("max_spread_cents"),
        max_entry_price_cents=trading_cfg.get("max_entry_price_cents"),
        slippage_pct=float(trading_cfg.get("slippage_pct", 0.02)),
    )
    eligible = result.eligible
    blockers = list(result.blockers)
    gates = dict(result.gates)
    # Extreme-edge guardrail. The SDK gate has only a min-edge floor.
    # An edge above ``max_edge_skip`` (default 20pp) is almost always
    # model overconfidence — sparse Elo on qualifiers, surface
    # miscalibration, or the market pricing real-world info the
    # historical match panel lacks. Skip the trade entirely.
    max_edge_skip = trading_cfg.get("max_edge_skip")
    if max_edge_skip is not None and result.side_edge is not None:
        if abs(float(result.side_edge)) > float(max_edge_skip):
            eligible = False
            blockers.append(
                f"edge_too_large_{abs(result.side_edge)*100:.0f}pp"
                f"_(>{float(max_edge_skip)*100:.0f}pp_cap)"
            )
            gates["max_edge_skip"] = False
        else:
            gates["max_edge_skip"] = True
    return BuyDecision(
        eligible=eligible,
        score=result.score,
        side=result.side,
        side_edge=result.side_edge,
        side_ev=result.side_ev,
        side_market=result.side_market,
        gates=gates,
        blockers=blockers,
    )


def stake_taper(edge_abs: float, trading_cfg: dict[str, Any]) -> float:
    """Return a stake-multiplier in [taper_min_stake_frac, 1.0] for the
    given absolute edge. Used by the simulator to scale ``bet_size`` on
    large-but-not-extreme edges (Kelly-style variance protection).

    Returns 1.0 when the taper config is absent or the edge is below
    ``taper_edge_above``. Tapers linearly from 1.0 at ``taper_edge_above``
    to ``taper_min_stake_frac`` at ``max_edge_skip``.
    """
    taper_above = trading_cfg.get("taper_edge_above")
    max_edge = trading_cfg.get("max_edge_skip")
    min_frac = trading_cfg.get("taper_min_stake_frac")
    if taper_above is None or max_edge is None or min_frac is None:
        return 1.0
    taper_above = float(taper_above)
    max_edge = float(max_edge)
    min_frac = float(min_frac)
    if edge_abs <= taper_above:
        return 1.0
    if edge_abs >= max_edge:
        # Skip-cap handles this; defensive floor in case taper runs
        # outside the buy_gate (e.g. when callers compute it directly).
        return min_frac
    span = max_edge - taper_above
    if span <= 0:
        return min_frac
    t = (edge_abs - taper_above) / span
    return 1.0 - t * (1.0 - min_frac)
