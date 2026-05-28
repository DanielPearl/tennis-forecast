"""Recalibrate the prematch model's probabilities using our actual
Kalshi bet outcomes.

The model claims 74.5% accuracy on the Sackmann holdout but the
real bets settled at 42.3% on real money — a +25.6pp over-confidence
gap on the 26 settled bets. Instead of waiting for the bot to lose
more money proving this, we recalibrate at inference: fit a Platt-
scaling sigmoid on the (model_predicted_prob, won) pairs from
``data/processed/artifacts/kalshi_calibration.json`` and apply it
to every prediction the live trader will use to compute edge.

Blend weight grows with sample size so the layer starts cautiously
(small n → mostly trust the raw model) and converges to the empirical
mapping (large n → mostly trust the live data). At n=0 the layer is
a no-op and the raw model probability passes through unchanged.

Fit: logistic regression P(won | x) = sigmoid(a + b * x) over the
recent_bets list in the calibration JSON. Closed-form scipy
optimization (no sklearn dependency — keeps the inference path
lean).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..utils.logging_setup import setup_logging

log = setup_logging("models.calibration_layer")


# Sample-size at which we fully trust the empirical fit. Below this,
# we blend the raw model prediction with the recalibrated one in
# linear proportion to (n / _N_FULL_WEIGHT). Picked to give half-
# weight at 25 bets (≈ where we are now) and full weight at 50.
_N_FULL_WEIGHT = 50


# Module-level cache. Reloads when the artifact's mtime changes —
# same pattern as predict.py so the daily retrain's freshly-written
# calibration is picked up without a process restart.
_LAYER_CACHE: dict[str, Any] = {"mtime": 0.0, "params": None, "n": 0}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _fit_platt(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    """Fit ``P(y=1 | p) = sigmoid(a + b * p)`` via gradient descent.
    Returns ``(a, b)``. With a small n this converges in a few
    hundred iterations.

    Pure stdlib — no scipy / sklearn. The loss is binary cross-
    entropy. Gradient: dL/da = mean(sigmoid(a+b*p) - y),
    dL/db = mean((sigmoid(a+b*p) - y) * p).
    """
    if not pairs:
        # No data → identity transform (a=0, b=4 gives sigmoid(4p)
        # which is close to p in the [0,1] range we care about).
        return 0.0, 4.0
    a, b = 0.0, 4.0
    lr = 0.5
    n = len(pairs)
    for _ in range(800):
        ga, gb, loss = 0.0, 0.0, 0.0
        for p, y in pairs:
            mu = _sigmoid(a + b * p)
            d = mu - y
            ga += d
            gb += d * p
            # Numerically-safe log loss (purely for monitoring; the
            # update doesn't use it).
            mu_clip = max(min(mu, 1 - 1e-9), 1e-9)
            loss += -(y * math.log(mu_clip)
                      + (1 - y) * math.log(1 - mu_clip))
        ga /= n
        gb /= n
        a -= lr * ga
        b -= lr * gb
    return a, b


def _load_layer(calibration_path: Path) -> dict[str, Any]:
    """Load the calibration JSON, refit when mtime changes, cache."""
    if not calibration_path.exists():
        return {"params": None, "n": 0, "blend_w": 0.0}
    try:
        current_mtime = calibration_path.stat().st_mtime
    except OSError:
        return {"params": None, "n": 0, "blend_w": 0.0}
    if (_LAYER_CACHE["params"] is not None
            and current_mtime == _LAYER_CACHE["mtime"]):
        n = _LAYER_CACHE["n"]
        blend_w = min(1.0, n / float(_N_FULL_WEIGHT))
        return {
            "params": _LAYER_CACHE["params"],
            "n": n,
            "blend_w": blend_w,
        }
    try:
        with calibration_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"params": None, "n": 0, "blend_w": 0.0}
    pairs: list[tuple[float, int]] = []
    for b in data.get("recent_bets", []) or []:
        p = b.get("entry_model_prob")
        won = b.get("won")
        if p is None or won is None:
            continue
        try:
            pairs.append((float(p), 1 if bool(won) else 0))
        except (TypeError, ValueError):
            continue
    if not pairs:
        _LAYER_CACHE.update({"mtime": current_mtime, "params": None, "n": 0})
        return {"params": None, "n": 0, "blend_w": 0.0}
    a, b = _fit_platt(pairs)
    n = len(pairs)
    blend_w = min(1.0, n / float(_N_FULL_WEIGHT))
    _LAYER_CACHE.update({"mtime": current_mtime, "params": (a, b), "n": n})
    log.info(
        "calibration layer refit on n=%d Kalshi bets — Platt (a=%.3f, "
        "b=%.3f), blend_weight=%.2f", n, a, b, blend_w,
    )
    return {"params": (a, b), "n": n, "blend_w": blend_w}


def recalibrate(model_prob: float, calibration_path: Path) -> float:
    """Apply the calibration layer to one model probability.

    Returns the blended estimate:
        blend_w * platt_recalibrated + (1 - blend_w) * model_prob
    where ``blend_w`` ramps from 0 (no Kalshi data) to 1 (n >= 50).

    Clamps to [0.01, 0.99] so downstream divisions never blow up.
    """
    p = float(model_prob)
    layer = _load_layer(calibration_path)
    if layer["params"] is None or layer["blend_w"] <= 0:
        return max(0.01, min(0.99, p))
    a, b = layer["params"]
    blended = (layer["blend_w"] * _sigmoid(a + b * p)
                + (1.0 - layer["blend_w"]) * p)
    return max(0.01, min(0.99, blended))


def layer_info(calibration_path: Path) -> dict[str, Any]:
    """Diagnostics for the dashboard's Models tab: sample size, blend
    weight, fitted (a, b), and a few illustrative remap points so the
    user can see what the layer is doing at common probabilities."""
    layer = _load_layer(calibration_path)
    if layer["params"] is None:
        return {"active": False, "n": 0, "blend_w": 0.0,
                "params": None, "remap": None}
    a, b = layer["params"]
    blend_w = layer["blend_w"]
    remap = [
        (p, blend_w * _sigmoid(a + b * p) + (1 - blend_w) * p)
        for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ]
    return {
        "active": True,
        "n": layer["n"],
        "blend_w": blend_w,
        "params": {"a": a, "b": b},
        "remap": remap,
    }
