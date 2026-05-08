"""Signal labelling + recommendation logic.

Signals don't recommend trades based purely on which player the model
likes. The user's spec is explicit: a signal only fires when the
model's view materially differs from the market's, and never inside
high-volatility or injury-tagged matches.

Labels (in priority order — the first match wins):

  INJURY_RISK         injury_news_flag is True
  AVOID_VOLATILE      volatility_score above the configured cap
  MARKET_OVERREACTION market_overreaction flag set by the live model
  STRONG_EDGE         |edge| >= strong_edge_min
  SMALL_EDGE          |edge| >= small_edge_min
  WATCH               match is interesting (Elo within 100, named tournament)
                      but no actionable edge
  NO_TRADE            default

The watchlist row stamps both the label and a one-line ``reason``
string — which is just the joined live-rules ``fired`` list, possibly
with the edge amount appended.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.config import load_config


@dataclass
class SignalResult:
    label: str
    reason: str
    confidence_score: float


_PRIORITY = [
    "INJURY_RISK", "AVOID_VOLATILE", "MARKET_OVERREACTION",
    "STRONG_EDGE", "SMALL_EDGE", "WATCH", "NO_TRADE",
]


def _confidence(model_prob: float, volatility: float) -> float:
    """0-1 confidence score. Drops with volatility and at the
    extremes of the calibration tail (over 90% / under 10%)."""
    extremity_penalty = 0.0
    if model_prob > 0.92 or model_prob < 0.08:
        extremity_penalty = 0.20
    base = 0.85
    base -= 0.55 * volatility
    base -= extremity_penalty
    return max(0.0, min(1.0, base))


def label_match(model_prob_a: float, market_prob_a: float | None,
                volatility: float, injury_flag: bool,
                market_overreaction: bool,
                rules_fired: list[str] | None = None,
                ) -> SignalResult:
    cfg = load_config()
    t = cfg["trading"]

    rules_fired = rules_fired or []
    # Use the higher-edge side: model picks A → look at A's edge;
    # model picks B → look at B's edge (= -edge_a).
    if market_prob_a is None:
        edge_signed = 0.0
    else:
        edge_signed = float(model_prob_a) - float(market_prob_a)
    edge_abs = abs(edge_signed)
    side = "player_a" if edge_signed >= 0 else "player_b"
    side_market = market_prob_a if edge_signed >= 0 else (
        1.0 - market_prob_a if market_prob_a is not None else None
    )

    # 1) Injury risk dominates everything.
    if injury_flag:
        return SignalResult(
            label="INJURY_RISK",
            reason="injury / medical risk flagged — skip until resolved",
            confidence_score=_confidence(model_prob_a, volatility) * 0.5,
        )

    # 2) Market overreaction — note this BEFORE the volatility cap because
    #    overreaction is itself a tradeable thesis (fade the move). But
    #    only when edge is on the right side of the move and the price
    #    is in a sane band.
    if market_overreaction and side_market is not None and edge_abs >= t["small_edge_min"]:
        if t["min_market_prob"] <= side_market <= t["max_market_prob"]:
            reason = "; ".join(rules_fired) or "market move outpaces model adjustment"
            return SignalResult(
                label="MARKET_OVERREACTION",
                reason=reason,
                confidence_score=_confidence(model_prob_a, volatility),
            )

    # 3) Volatility cap.
    if volatility >= t["max_tradable_volatility"]:
        return SignalResult(
            label="AVOID_VOLATILE",
            reason="volatility above tradeable cap — wait for the set to settle",
            confidence_score=_confidence(model_prob_a, volatility),
        )

    # 4) Edge size, gated on a sane market price band.
    if market_prob_a is None:
        return SignalResult(
            label="WATCH",
            reason="no market price observed — model-only forecast",
            confidence_score=_confidence(model_prob_a, volatility) * 0.7,
        )

    if not (t["min_market_prob"] <= side_market <= t["max_market_prob"]):
        return SignalResult(
            label="NO_TRADE",
            reason=f"market price on {side} ({side_market:.0%}) outside tradeable band",
            confidence_score=_confidence(model_prob_a, volatility),
        )

    if edge_abs >= t["strong_edge_min"]:
        return SignalResult(
            label="STRONG_EDGE",
            reason=f"model {edge_signed*100:+.1f}pp vs market on {side}",
            confidence_score=_confidence(model_prob_a, volatility),
        )
    if edge_abs >= t["small_edge_min"]:
        return SignalResult(
            label="SMALL_EDGE",
            reason=f"model {edge_signed*100:+.1f}pp vs market on {side}",
            confidence_score=_confidence(model_prob_a, volatility),
        )

    # 5) Default: WATCH for matches where the model has a non-trivial
    #    view even when there's no edge.
    return SignalResult(
        label="WATCH",
        reason="model view aligned with market — no edge",
        confidence_score=_confidence(model_prob_a, volatility),
    )
