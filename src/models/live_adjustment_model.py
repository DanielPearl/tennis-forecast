"""Live-match adjustment layer.

Phase-1 is deliberately rules-based — every nudge is auditable and
the dashboard's "reason" column reads like a sentence instead of
"GBT shap value 0.0034". When we have closed-bet feedback we'll
swap in an ML adjustment model that learns the deltas, but the
rules version is correct enough to ship and lets us debug bad
trades without re-training.

Inputs: pre-match probability + the standardized live record from
``features/build_live_features.py``.

Outputs:
  - live_prob_a: adjusted win-probability after applying rules
  - volatility_score: 0-1; high values mean "don't trust this"
  - injury_news_flag: bool
  - rules_fired: list of human-readable reasons (used by the
    dashboard's signal-reason column)
  - market_overreaction: bool — set when the market move outpaces
    the model's adjustment, suggesting the price has run past the
    underlying state
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..features.build_live_features import market_move, momentum_score
from ..models import predict_inmatch
from ..utils.config import load_config


@dataclass
class LiveAdjustment:
    live_prob_a: float
    pre_match_prob_a: float
    volatility_score: float
    injury_news_flag: bool
    market_overreaction: bool
    rules_fired: list[str]
    # ``model_prob_a`` is the raw trained in-match model prediction
    # (None when the artifact isn't installed or the model gate is
    # off). ``rules_prob_a`` is what the rules layer alone would have
    # said. Both are kept for the dashboard's "why?" column so the
    # operator can see when the two disagree.
    model_prob_a: float | None = None
    rules_prob_a: float | None = None


def _clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def adjust(pre_match_prob_a: float, live_record: dict[str, Any]) -> LiveAdjustment:
    cfg = load_config()
    rules = cfg["live_rules"]

    fired: list[str] = []
    delta = 0.0
    volatility = 0.0
    injury = bool(live_record.get("injury_news_flag", False)
                  or live_record.get("retirement_risk_flag", False))

    # ---- 1) Score-state nudge -----------------------------------------
    # Each completed game tilts the prob toward whoever just won it.
    # Capped so even a 6-0 set doesn't move us more than max_in_match_shift.
    a_sets = live_record.get("set_score_a") or 0
    b_sets = live_record.get("set_score_b") or 0
    a_g3 = live_record.get("games_won_last_3_a") or 0
    b_g3 = live_record.get("games_won_last_3_b") or 0
    momentum = momentum_score(live_record)
    score_delta = max(
        -rules["max_in_match_shift"],
        min(
            rules["max_in_match_shift"],
            momentum * rules["max_in_match_shift"],
        ),
    )
    if abs(score_delta) > 0.01:
        delta += score_delta
        fired.append(
            f"score-state momentum {momentum:+.2f} → "
            f"{score_delta*100:+.1f}pp on player_a"
        )

    # ---- 2) Serve-strength nudge --------------------------------------
    # If A's first-serve % is materially above their match average and
    # B's is below, lean toward A even when the score doesn't show it.
    fs_a = live_record.get("first_serve_pct_a")
    fs_b = live_record.get("first_serve_pct_b")
    if fs_a is not None and fs_b is not None:
        diff = fs_a - fs_b
        if abs(diff) > 0.05:
            serve_delta = diff * 0.10  # at most ~3pp
            delta += serve_delta
            if serve_delta > 0.005:
                fired.append(
                    f"first-serve % differential {diff*100:+.1f}pp "
                    f"→ {serve_delta*100:+.1f}pp on player_a"
                )

    # ---- 3) "Lost a set but stats are strong" rebound ------------------
    # If A is down a set yet has comparable or better serve numbers,
    # boost slightly — markets typically overreact to set-state.
    if (a_sets < b_sets) and (fs_a is not None) and (fs_b is not None):
        if fs_a >= fs_b - 0.01:
            rebound = 0.02
            delta += rebound
            fired.append("trailing player has comparable serve stats — partial rebound")

    # ---- 4) Volatility scoring ----------------------------------------
    if live_record.get("is_tiebreak"):
        volatility += rules["tiebreak_volatility_bump"]
        fired.append("tiebreak in progress → volatility")
    if live_record.get("is_decider"):
        volatility += rules["decider_volatility_bump"]
        fired.append("deciding set → volatility")
    if live_record.get("medical_timeout"):
        volatility += rules["medical_timeout_volatility_bump"]
        fired.append("medical timeout → volatility")
        injury = True
    # Liquidity / spread bump — when the book is thin or the spread blows
    # out, the market quote is noisy and the live-PV layer should trust
    # it less. Wide-spread tennis books (≥ 8c) cluster around match-
    # progression jumps, where the price tends to overshoot.
    spread = live_record.get("spread_cents")
    if spread is not None and float(spread) >= 8.0:
        volatility += 0.10
        fired.append(f"wide spread {float(spread):.0f}c → market quote noisy")
    oi = live_record.get("open_interest")
    if oi is not None and float(oi) < 100:
        volatility += 0.06
        fired.append(f"thin book OI={int(float(oi))} → market quote noisy")
    # Tail volatility floor: even calm matches have some uncertainty.
    volatility = min(1.0, max(0.05, volatility))

    # ---- 5) Injury risk from serve % collapse -------------------------
    if fs_a is not None and fs_a < 0.45:
        fired.append(f"player_a first-serve % {fs_a*100:.0f}% — possible injury")
        injury = True
        delta -= rules["injury_risk_drop_threshold"]
    if fs_b is not None and fs_b < 0.45:
        fired.append(f"player_b first-serve % {fs_b*100:.0f}% — possible injury")
        injury = True
        delta += rules["injury_risk_drop_threshold"]

    # ---- 5b) Serve velocity decline → fatigue / injury ---------------
    # When a player's trailing-10 first-serve speed drops materially
    # below their match baseline (≥ 8 km/h is the threshold the
    # commentary uses), nudge the other way and flag injury volatility.
    # Read the live_record fields populated by the live feed; absent
    # signals leave the rule silent.
    fs_speed_delta_a = live_record.get("first_serve_speed_delta_a")
    fs_speed_delta_b = live_record.get("first_serve_speed_delta_b")
    if fs_speed_delta_a is not None and float(fs_speed_delta_a) <= -8.0:
        d = float(fs_speed_delta_a)
        fired.append(
            f"player_a 1st-serve speed dropped {d:.1f} km/h vs match avg — "
            "fatigue / injury risk"
        )
        injury = True
        delta -= rules.get("serve_speed_decline_pp", 0.04)
        volatility = min(1.0, volatility + 0.10)
    if fs_speed_delta_b is not None and float(fs_speed_delta_b) <= -8.0:
        d = float(fs_speed_delta_b)
        fired.append(
            f"player_b 1st-serve speed dropped {d:.1f} km/h vs match avg — "
            "fatigue / injury risk"
        )
        injury = True
        delta += rules.get("serve_speed_decline_pp", 0.04)
        volatility = min(1.0, volatility + 0.10)

    # ---- 5c) Break-point conversion differential ---------------------
    # Players who convert their break opportunities reliably tend to
    # close out matches; a 25pp gap in conversion rate over a match
    # is a real differentiator (literature: Klaassen/Magnus tennis
    # production models). Conservative ±3pp nudge, only when both
    # sides have faced enough BPs for the rate to be meaningful. We
    # derive conversion from created + won so the rule fires even when
    # the live feed doesn't carry a pre-computed ratio.
    bp_created_a = live_record.get("break_points_created_a")
    bp_created_b = live_record.get("break_points_created_b")
    bp_won_a = live_record.get("break_points_won_a")
    bp_won_b = live_record.get("break_points_won_b")
    if (bp_created_a is not None and bp_created_b is not None
            and bp_won_a is not None and bp_won_b is not None
            and float(bp_created_a) >= 3 and float(bp_created_b) >= 3):
        conv_a = float(bp_won_a) / float(bp_created_a)
        conv_b = float(bp_won_b) / float(bp_created_b)
        bp_diff = conv_a - conv_b
        if abs(bp_diff) >= 0.25:
            bp_delta = max(-0.04, min(0.04, bp_diff * 0.10))
            delta += bp_delta
            fired.append(
                f"BP conversion differential {bp_diff*100:+.0f}pp "
                f"(A={conv_a*100:.0f}% B={conv_b*100:.0f}%) "
                f"→ {bp_delta*100:+.1f}pp on player_a"
            )

    rules_prob_a = _clamp(pre_match_prob_a + delta)

    # ---- 6) Trained in-match model (replaces rules nudge if enabled) ---
    # Gate behind config so we can flip back to the rules layer if the
    # model misbehaves on a new tournament. The rules layer still
    # supplies volatility/injury flags and the audit string — only the
    # numerical live_prob_a comes from the trained model when active.
    model_prob_a: float | None = None
    use_model = bool(rules.get("use_trained_inmatch_model", False))
    if use_model:
        # Catch + ignore model failures (e.g. when the artifact was
        # trained on an older feature set after we added new fields).
        # The rules-only path is always a safe fallback.
        try:
            model_prob_a = predict_inmatch.predict(live_record)
        except Exception:  # noqa: BLE001
            model_prob_a = None
    if model_prob_a is not None:
        # Progress-weighted blend with the pre-match prior. Early in
        # the match the trained model has little signal (Brier improves
        # only modestly over rules in the first 25% of points); late
        # in the match the live features dominate. The schedule below
        # mirrors the bucket-wise gain we measured at training time.
        prog = float(live_record.get("progress") or 0.0)
        w_model = max(0.0, min(1.0, 0.2 + 1.2 * prog))
        live_prob_a = _clamp(w_model * model_prob_a +
                              (1.0 - w_model) * pre_match_prob_a)
        fired.append(
            f"in-match model {model_prob_a*100:.1f}% blended w={w_model:.2f} "
            f"with pre-match {pre_match_prob_a*100:.1f}%"
        )
        # Use the model's adjustment magnitude for overreaction logic so
        # the gate stays consistent when the model is on.
        effective_delta = live_prob_a - pre_match_prob_a
    else:
        live_prob_a = rules_prob_a
        effective_delta = delta

    # ---- 7) Market overreaction detection -----------------------------
    move = market_move(live_record)
    overreaction = False
    if move is not None:
        if (abs(move) >= rules["overreaction_market_move"]
                and abs(effective_delta) < rules["overreaction_model_move"]):
            overreaction = True
            fired.append(
                f"market moved {move*100:+.1f}pp but model only "
                f"{effective_delta*100:+.1f}pp — possible overreaction"
            )

    return LiveAdjustment(
        live_prob_a=live_prob_a,
        pre_match_prob_a=pre_match_prob_a,
        volatility_score=volatility,
        injury_news_flag=injury,
        market_overreaction=overreaction,
        rules_fired=fired,
        model_prob_a=model_prob_a,
        rules_prob_a=rules_prob_a,
    )
