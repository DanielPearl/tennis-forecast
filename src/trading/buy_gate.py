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
    from kalshi_sdk import buy_criteria as _bc
    # Shared-criteria clamp (user 2026-07-21): config may only
    # tighten the canonical gates in kalshi_sdk.buy_criteria.
    _sm = max(float(trading_cfg.get("small_edge_min", 0.05)),
              _bc.MIN_EDGE)
    result = evaluate_row_gates(
        row_with_ev,
        small_edge_min=_sm,
        strong_edge_min=_sm,
        require_strong_edge=False,
        tradeable_labels={"EDGE"},
        min_ev=float(trading_cfg.get("min_ev", 0.03)),
        min_market_prob=max(
            float(trading_cfg.get("min_market_prob", 0.0)),
            _bc.MIN_ENTRY_PRICE),
        max_market_prob=min(
            float(trading_cfg.get("max_market_prob", 1.0)),
            _bc.MAX_ENTRY_PRICE),
        max_tradable_volatility=float(trading_cfg.get("max_tradable_volatility", 1.0)),
        min_open_interest=trading_cfg.get("min_open_interest"),
        max_spread_cents=trading_cfg.get("max_spread_cents"),
        max_entry_price_cents=min(
            int(trading_cfg.get("max_entry_price_cents") or 100),
            int(_bc.MAX_ENTRY_PRICE * 100)),
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
    # Uniform suspect-edge ceiling (user 2026-07-21: same criteria
    # for every bot) — the old "deliberately omitted" tennis exemption
    # is retired; a >15pp gap vs the sharp line is a data problem,
    # not a trade.
    max_edge_skip = trading_cfg.get("max_edge_skip")
    if max_edge_skip is None:
        max_edge_skip = _bc.MAX_EDGE
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
    # Spread-aware true-edge gate. The SDK computes side_edge as
    # ``|live_prob_a − market_prob_a|`` and uses ``(1 − market_a)`` as
    # the reference market for side==B. That assumes yes_ask_a +
    # yes_ask_b == 100¢, which Kalshi tennis books rarely satisfy —
    # 6-10¢ spreads are routine. When the two asks sum > 100¢, the
    # SDK's edge is inflated by the spread, and the executor buys at
    # a price that's actually near-neutral to our model.
    #
    # Example (2026-07-11 Bueno vs Marcondes):
    #   live_a=50.9%, market_a (Bueno YES ask)=61¢,
    #   yes_ask_cents_b (Marcondes YES ask)=48¢ → book totals 109¢.
    #   SDK: side_edge = |0.509 − 0.61| = 10.1pp → clears 9pp gate.
    #   True: model_b (49.1%) − market_b_ask (48¢) = +1.1pp.
    #
    # Recompute with the actual yes-ask on the side we'd buy. If it
    # falls below the small_edge floor, mark the row ineligible and
    # emit ``spread_inflated_edge`` as a blocker so the audit trail
    # names the reason.
    if eligible and result.side in ("A", "B"):
        _side_key = "yes_ask_cents_a" if result.side == "A" \
            else "yes_ask_cents_b"
        _side_ask_c = row_with_ev.get(_side_key)
        _live_a_p = row_with_ev.get("live_prob_a")
        if _side_ask_c is not None and _live_a_p is not None:
            _live_a_p = float(_live_a_p)
            _side_model = _live_a_p if result.side == "A" \
                else (1.0 - _live_a_p)
            _true_market = float(_side_ask_c) / 100.0
            _true_edge = _side_model - _true_market
            if _true_edge < _sm:
                eligible = False
                blockers.append(
                    f"spread_inflated_edge_true"
                    f"{_true_edge*100:+.1f}pp"
                    f"<{_sm*100:.0f}pp_floor"
                    f"_(sdk_saw_{float(result.side_edge)*100:+.1f}pp)"
                )
                gates["true_edge"] = False
            else:
                gates["true_edge"] = True
    # Pinnacle-required gate: refuse to flag a row eligible when the
    # sharp benchmark line isn't quoting the match. Without this, the
    # internal Sackmann-trained model's ~50% fallback for unlisted
    # fixtures would still clear the edge floor against deep-underdog
    # Kalshi asks and stamp buy_eligible=True. The dashboard executor
    # already enforces the same rule (SportLiveExecutor.require_pinn-
    # acle) so trading is safe today; this gate keeps the watchlist's
    # Verdict cell honest — no-Pinnacle rows now read SKIP with a
    # ``no_pinnacle_line`` blocker rather than eligible-but-untradable.
    # Defaults to True; set trading.require_pinnacle: false in config
    # to opt back into the internal-model fallback.
    if eligible and bool(trading_cfg.get("require_pinnacle", True)):
        if row_with_ev.get("pinnacle_prob_a") is None:
            eligible = False
            blockers.append("no_pinnacle_line")
            gates["pinnacle"] = False
        else:
            gates["pinnacle"] = True
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
