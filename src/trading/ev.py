"""Expected value + edge math.

We treat every "buy this side" as a binary outcome on a unit-stake
contract: pays $1 if the side wins, $0 otherwise. Market price is
the implied probability — Kalshi cents map directly. EV is the
model probability minus the price minus assumed slippage. This is
the same math the NBA/CPI bots use; the field names here just say
"market price" instead of "Kalshi YES ask".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EVResult:
    edge: float            # model_prob - market_prob
    ev_per_contract: float # expected $ per $1 staked, slippage-adjusted
    breakeven_market_prob: float


def edge(model_prob: float, market_prob: float) -> float:
    return float(model_prob) - float(market_prob)


def ev(model_prob: float, market_prob: float, slippage: float) -> EVResult:
    """Compute EV for buying the side with the given ``market_prob``.

    Slippage models half-spread + small order book-walk. Conservative
    (0.02 default in config) and applied as a fixed cost on top of
    the price; the breakeven probability becomes ``market_prob + slippage``.
    """
    raw_edge = float(model_prob) - float(market_prob)
    breakeven = float(market_prob) + float(slippage)
    # EV per $1 staked: model_p * (1 - market_p - slippage) - (1 - model_p) * (market_p + slippage)
    payout_if_win = 1.0 - market_prob - slippage
    payout_if_loss = -(market_prob + slippage)
    ev_per = model_prob * payout_if_win + (1.0 - model_prob) * payout_if_loss
    return EVResult(edge=raw_edge, ev_per_contract=ev_per,
                    breakeven_market_prob=breakeven)
