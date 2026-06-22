"""Train the pre-match win-probability model.

Pipeline:
  1. Fetch + clean Sackmann data (data/fetch_matches.py)
  2. Build full panel (features/build_prematch_features.py)
  3. Mirror to player_a orientation
  4. Three-way time-series split: train / validation / holdout
  5. Train MULTIPLE candidate ensembles in parallel (HGB / GBM / RF /
     ExtraTrees / full-feature LR / XGB if installed), calibrate each
     via sigmoid on its own tail, then find non-negative blend
     weights that minimise log-loss on the validation slice.
  6. Elo-only logistic baseline is kept as a separate component;
     the final ``blended`` probability is 70% mixed-ensemble + 30%
     Elo-only logistic (the logistic acts as a regularising prior
     against tree overfit to a cohort).
  7. Persist: bundled model (WeightedEnsemble + Elo-only LR) + Elo
     state + H2H + last-match-date + rolling-form snapshot.

The constrained-weight search is the centerpiece of the 2026-06-02
revamp — before the rewrite the bundle was a single calibrated tree
ensemble with a fixed 70/30 blend; we found the live trader was
making systematically over-confident calls on the 55-65pp bucket.
Letting the data pick a non-negative weight per base model on a
held-out 20% slice gives a measurable Brier improvement and an
interpretable "which model does the work" breakdown for the
dashboard's model card.

In-game model: temporarily disabled in ``config.yaml`` while we
rebuild the pre-match. Once we trust the new ensemble, we can
retrain the inmatch adjustment on top of it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                HistGradientBoostingClassifier,
                                RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                              log_loss, precision_score, recall_score,
                              roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.fetch_matches import fetch_all, save_clean
from ..features.build_prematch_features import (
    PREMATCH_FEATURES, build_full_panel, build_player_a_panel, select_features,
)
from ..features.elo import EloState
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.train_prematch")


def _try_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


class WeightedEnsemble:
    """Convex blend of calibrated probability classifiers.

    Implements just the surface area ``predict.py`` reads on the
    bundle's ``ensemble`` slot — ``predict_proba(X)`` returning shape
    ``(n_samples, 2)``. Weights are stored as ``dict[name, float]``
    so the dashboard model card can render which base model is
    pulling the load; missing keys default to 0.

    Defined at module level so joblib can re-pickle the instance on
    bundle reload (the predict_path watcher needs the class to be
    importable, not nested inside ``train_and_persist``).
    """

    def __init__(self, models: dict, weights: dict,
                 model_order: list[str]):
        self.models = models  # name -> calibrated classifier
        self.weights = weights  # name -> non-negative float
        self.model_order = list(model_order)

    def predict_proba(self, X):
        out = None
        wsum = 0.0
        for name in self.model_order:
            w = float(self.weights.get(name, 0.0))
            if w <= 0:
                continue
            p = self.models[name].predict_proba(X)
            out = (w * p) if out is None else (out + w * p)
            wsum += w
        if out is None:
            # All-zero weights — defensive 50/50.
            return np.full((len(X), 2), 0.5)
        if wsum > 0 and abs(wsum - 1.0) > 1e-9:
            out = out / wsum
        return out


def _build_candidates(cfg: dict) -> dict:
    """Candidate (unfit) classifiers for the mixed ensemble search.

    Each model is calibrated separately via sigmoid; the constrained
    weight optimisation handles "which one to trust" downstream. We
    favour sklearn-native estimators so the droplet doesn't need
    extra binaries (xgboost / lightgbm are optional bumps).

    Notes on the candidate set:
      * ``hgb``  — sklearn's histogram-binned GBT. The strongest
        single model in our backtests; fast, low-memory.
      * ``gbm``  — classical sklearn GBT. Slower, but different
        regularisation profile than ``hgb`` — useful for blending.
      * ``rf``   — bagged trees. Lower variance, captures wider
        interaction patterns the boosters miss when overfit.
      * ``et``   — extremely randomised trees. Bias trade-off
        complement to ``rf``.
      * ``lr_full`` — logistic regression over the full feature
        set (NOT just Elo). Tennis is linear-ish at the macro
        level; this catches signal the trees discount.
      * ``mlp`` is intentionally NOT included — small tabular MLPs
        don't beat the tree set on shallow tennis features and add
        training time without payoff.
      * ``xgb``  — only added when xgboost is importable.
    """
    rs = int(cfg["model"]["random_state"])
    ne = int(cfg["model"]["ensemble"]["n_estimators"])
    lr = float(cfg["model"]["ensemble"]["learning_rate"])
    md = int(cfg["model"]["ensemble"]["max_depth"])
    candidates: dict = {
        "hgb": HistGradientBoostingClassifier(
            max_iter=ne, learning_rate=lr, max_depth=md,
            random_state=rs,
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=min(ne, 200), learning_rate=lr,
            max_depth=md, random_state=rs,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            n_jobs=2, random_state=rs,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            n_jobs=2, random_state=rs,
        ),
        # Standardise the linear model — its decision boundary is
        # scale-sensitive (rank_diff lives on a different magnitude
        # axis than the [-1,1]-ish Elo / form diffs).
        "lr_full": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=600, C=1.0)),
        ]),
    }
    xgb = _try_xgb() if cfg["model"]["ensemble"].get(
        "use_xgboost_if_available") else None
    if xgb is not None:
        candidates["xgb"] = xgb.XGBClassifier(
            n_estimators=ne, learning_rate=lr, max_depth=md,
            tree_method="hist", random_state=rs,
            eval_metric="logloss", n_jobs=2,
        )
    return candidates


def _optimize_weights(val_preds: dict[str, np.ndarray],
                       y_val: np.ndarray) -> dict[str, float]:
    """Constrained search for weights on the validation slice.

    Minimises log-loss subject to non-negative weights summing to 1.
    Returns a dict keyed by model name so the bundle can persist
    "which model got which share" for the dashboard.

    Falls back gracefully to uniform weights if scipy isn't
    installed or SLSQP can't converge — in either case the
    ensemble still works, just without the optimisation step.
    """
    names = list(val_preds.keys())
    n = len(names)
    if n == 0:
        return {}
    preds = np.stack([val_preds[k] for k in names], axis=1)
    y = np.asarray(y_val).astype(int)

    def loss(w: np.ndarray) -> float:
        blend = np.clip(preds @ w, 1e-6, 1 - 1e-6)
        return float(log_loss(y, blend))

    init = np.full(n, 1.0 / n)
    try:
        from scipy.optimize import minimize  # type: ignore
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq",
                          "fun": lambda w: float(np.sum(w) - 1.0)}]
        res = minimize(loss, init, bounds=bounds,
                        constraints=constraints, method="SLSQP",
                        options={"maxiter": 200, "ftol": 1e-8})
        w = res.x if res.success else init
    except Exception:  # noqa: BLE001
        w = init
    # Clamp negatives to zero, renormalise — defensive against
    # numerical drift even on a successful solve.
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        w = init
        s = 1.0
    w = w / s
    return {names[i]: float(w[i]) for i in range(n)}


def _split_by_date(df: pd.DataFrame, months: int):
    cutoff = df["tourney_date"].max() - pd.DateOffset(months=months)
    train = df[df["tourney_date"] < cutoff].copy()
    test = df[df["tourney_date"] >= cutoff].copy()
    return train, test, cutoff


def _calibrate(model, X, y, holdout_frac: float = 0.2):
    """Hold out the tail of training data for calibration. Sigmoid is
    a safe default for tree ensembles whose raw scores are pushed to
    the extremes.

    sklearn 1.8 dropped ``cv='prefit'`` and replaced it with
    ``FrozenEstimator``; older versions (<1.6) only know ``'prefit'``.
    Try the new API first, fall back to the old one — keeps the bot
    portable across droplet pip-pinned versions.
    """
    n = len(X)
    cut = int(n * (1 - holdout_frac))
    X_fit, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_fit, y_cal = y[:cut], y[cut:]
    model.fit(X_fit, y_fit)
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except Exception:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def train_and_persist() -> dict[str, Any]:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching match history…")
    matches = fetch_all()
    save_clean(matches)

    log.info("building feature panel (Elo + form + serve/return)…")
    panel, elo_state, h2h_table, last_match_date, rolling_snapshot = (
        build_full_panel(matches, elo_cfg=cfg["elo"])
    )
    oriented = build_player_a_panel(panel)
    oriented = oriented.sort_values("tourney_date").reset_index(drop=True)

    train_full, test, cutoff = _split_by_date(
        oriented, months=int(cfg["model"]["test_window_months"])
    )
    # Within the train block, hold the most recent 20% out as
    # validation for the ensemble weight search. The standard
    # holdout slice each candidate's _calibrate() carves off is a
    # SEPARATE 20% tail of its train_part — they don't overlap with
    # the val block, and val isn't ever shown to a candidate at fit
    # time. This keeps the weight optimisation honest.
    val_cut = int(len(train_full) * 0.80)
    train_part = train_full.iloc[:val_cut].copy()
    val_part = train_full.iloc[val_cut:].copy()
    log.info("train %d / val %d / test %d (cutoff %s)",
             len(train_part), len(val_part), len(test), cutoff.date())

    X_train = select_features(train_part); y_train = train_part["y"].values
    X_val = select_features(val_part); y_val = val_part["y"].values
    X_test = select_features(test); y_test = test["y"].values

    # 1) Elo-only logistic baseline (kept as a separate component
    #    of the final blend, NOT inside the ensemble search). It
    #    serves as a regularising prior pulling predictions toward
    #    the Elo prior when the tree mixture overfits a cohort.
    elo_only_features = ["diff_elo_pre", "diff_surface_elo_pre"]
    base = LogisticRegression(max_iter=400)
    base.fit(X_train[elo_only_features], y_train)
    base_p_test = base.predict_proba(X_test[elo_only_features])[:, 1]

    # 2) Multi-model search. Calibrate each candidate via sigmoid
    #    on its own training-tail slice, then collect validation
    #    predictions for the constrained weight optimisation.
    candidates = _build_candidates(cfg)
    fitted: dict = {}
    val_preds: dict[str, np.ndarray] = {}
    per_model_metrics: dict = {}
    for name, clf in candidates.items():
        try:
            cal = _calibrate(clf, X_train, y_train, holdout_frac=0.2)
        except Exception as exc:  # noqa: BLE001
            log.warning("candidate %s failed to train: %s — skipping",
                        name, exc)
            continue
        fitted[name] = cal
        val_preds[name] = cal.predict_proba(X_val)[:, 1]
        test_p = cal.predict_proba(X_test)[:, 1]
        per_model_metrics[name] = _eval(y_test, test_p)
        log.info("trained %-7s — brier=%.4f logloss=%.4f acc=%.3f",
                 name, per_model_metrics[name]["brier"],
                 per_model_metrics[name]["log_loss"],
                 per_model_metrics[name]["accuracy"])

    # 3) Constrained weight search on validation log-loss.
    weights = _optimize_weights(val_preds, y_val)
    log.info("ensemble weights: %s",
             {k: round(v, 3) for k, v in weights.items()})

    ensemble = WeightedEnsemble(
        models=fitted, weights=weights,
        model_order=list(fitted.keys()),
    )
    ens_p_test = ensemble.predict_proba(X_test)[:, 1]

    # 4) Final blend: optimised ensemble + Elo-only logistic (70/30).
    #    The split is kept at the previous default — the dashboard
    #    surfaces both components so the operator can see the
    #    per-trade decomposition; the data-driven optimisation lives
    #    INSIDE the ensemble component, not at this outer layer.
    blend_w_ens = 0.70
    blend_w_log = 0.30
    blend_p = blend_w_ens * ens_p_test + blend_w_log * base_p_test

    metrics = {
        "rows_train": int(len(train_part)),
        "rows_val": int(len(val_part)),
        "rows_test": int(len(test)),
        "cutoff_date": str(cutoff.date()),
        "elo_only": _eval(y_test, base_p_test),
        "per_model": per_model_metrics,
        "ensemble": _eval(y_test, ens_p_test),
        "blended": _eval(y_test, blend_p),
        "ensemble_weights": weights,
        "xgboost": "xgb" in fitted,
    }
    log.info("metrics: %s", json.dumps(metrics, indent=2))

    bundle = {
        "ensemble": ensemble,
        "logistic": base,
        "feature_list": PREMATCH_FEATURES,
        "elo_only_features": elo_only_features,
        "blend_weight_ensemble": blend_w_ens,
        "blend_weight_logistic": blend_w_log,
        "metrics": metrics,
    }
    model_path = artifacts_dir / "prematch_model.joblib"
    joblib.dump(bundle, model_path)
    log.info("wrote model bundle → %s", model_path)

    # Persist Elo state + H2H + last-match-date so inference can build
    # features for any (player_a, player_b, surface) triple without
    # re-running the whole pipeline.
    state_path = artifacts_dir / "elo_state.joblib"
    joblib.dump(_elo_state_to_dict(elo_state), state_path)

    h2h_path = artifacts_dir / "h2h_table.joblib"
    joblib.dump(dict(h2h_table), h2h_path)

    rest_path = artifacts_dir / "last_match_date.joblib"
    joblib.dump({k: pd.Timestamp(v).isoformat() for k, v in last_match_date.items()},
                rest_path)

    # Per-player rolling-form snapshot — Phase 2. Without this the
    # inference path in predict.py would have to hardcode form/serve/
    # return/bp_saved features to zero, which is the bug we just
    # fixed. ~250k players × 5 floats × 8 bytes ≈ 10MB.
    rolling_path = artifacts_dir / "rolling_form_state.joblib"
    joblib.dump(rolling_snapshot, rolling_path)
    log.info("wrote rolling form state → %s (%d players)",
             rolling_path, len(rolling_snapshot))

    # Persist the full panel (with engineered features + label) to the
    # training_history.db SQLite store so the dashboard's Training Data
    # page can render the exact rows the model trained on. Idempotent
    # — uniqueness on (tourney_date, player_a, player_b, round) means
    # rerunning the trainer overwrites prior records rather than
    # duplicating. Val cutoff: the trainer split was train_part / val
    # at index ``val_cut`` within ``train_full`` (chronological), so
    # the val cutoff is the FIRST date in ``val_part``.
    try:
        from ..data.training_db import (upsert_training_panel,
                                          backfill_extra_attrs)
        db_path = artifacts_dir.parent.parent / "training_history.db"
        val_cutoff = (val_part["tourney_date"].min()
                      if len(val_part) else cutoff)
        n_db = upsert_training_panel(
            db_path, oriented,
            split_cutoff_train=val_cutoff,
            split_cutoff_val=cutoff,
        )
        log.info("wrote training panel to db → %s (%d rows)",
                  db_path, n_db)
        # Pre-prune candidate features (age, height, hand, country,
        # rank, rank_points, seed, entry, plus derived diffs) come
        # from matches_clean.csv since build_player_a_panel doesn't
        # carry the raw attribute columns through. Run the backfill
        # so the dashboard's Training Data page can render every
        # candidate column, not just the 12 selected ones.
        matches_csv = (artifacts_dir.parent / "matches_clean.csv")
        if matches_csv.exists():
            n_back = backfill_extra_attrs(db_path, matches_csv)
            log.info("backfilled extra player attributes (%d rows)",
                      n_back)
    except Exception:  # noqa: BLE001
        log.exception("training_db updates failed (non-fatal — joblib "
                       "bundle is the source of truth for inference)")

    # Dump logistic regression coefficients next to metrics so the
    # downstream dashboard can show "model coefficients" on its card.
    # The coefficients here are interpretable per-feature weights from
    # the Elo-only baseline (intercept + diff_elo_pre + diff_surface_elo_pre);
    # the GBT contribution is opaque so we don't pretend to render it.
    try:
        import numpy as _np
        log_coef = list(map(float, base.coef_.ravel().tolist()))
        log_intercept = float(_np.array(base.intercept_).ravel()[0])
    except Exception:
        log_coef, log_intercept = [], 0.0

    coefficients = {
        "logistic": {
            "intercept": log_intercept,
            "features": elo_only_features,
            "coefficients": log_coef,
        },
        # Per-base-model summary of the mixed ensemble — the dashboard
        # model card renders one row per entry showing name, weight,
        # and its standalone Brier/log-loss on the test slice so the
        # operator can see "which base model is doing the work".
        "ensemble_components": [
            {
                "name": name,
                "weight": float(weights.get(name, 0.0)),
                "metrics": per_model_metrics.get(name, {}),
            }
            for name in fitted.keys()
        ],
        # Backward-compat: the previous bundle exposed
        # ``ensemble_top_features`` (gain importances from the single
        # GBT). With a mixed ensemble there is no single set of gain
        # importances; we surface the weight-sorted component list
        # instead and leave this field empty so older render code
        # silently falls back.
        "ensemble_top_features": [],
        "blend": {
            "ensemble_weight": bundle["blend_weight_ensemble"],
            "logistic_weight": bundle["blend_weight_logistic"],
        },
        "elo": {
            "k_base": cfg["elo"]["k_base"],
            "k_floor": cfg["elo"]["k_floor"],
            "surface_blend": cfg["elo"]["surface_blend"],
        },
    }
    with open(artifacts_dir / "model_coefficients.json", "w") as f:
        json.dump(coefficients, f, indent=2)

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Per-prediction holdout dump for the dashboard's Models tab ──
    # The dashboard reads holdout_predictions.csv to render the ROC
    # curve, confusion matrix, and decile-calibration chart from the
    # trainer's actual evaluation against historical match outcomes —
    # the same way it does for the macro bots. We dump the BLENDED
    # probability since that's the prob the live trader actually uses.
    try:
        holdout_path = artifacts_dir / "holdout_predictions.csv"
        with open(holdout_path, "w") as f:
            f.write("predicted_prob,actual_label\n")
            for p, y in zip(blend_p, y_test):
                f.write(f"{float(p):.6f},{int(y)}\n")
        log.info("wrote holdout predictions (%d rows) → %s",
                 len(y_test), holdout_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not dump holdout_predictions.csv: %s", exc)

    # ── Feature importance audit (historical-data-driven) ─────────────
    # Permutation importance on the BLENDED model, scored against the
    # held-out historical match outcomes. For each feature we shuffle
    # its values and measure how much the holdout log-loss degrades —
    # this is the same convention the trading-dashboard's standard model
    # page uses for sim.db bots, so the rendering carries verbatim.
    # We also dump the feature's univariate correlation with y, which
    # gives a sign for the dashboard's "all features" bar chart.
    try:
        perm = _permutation_importance(
            ensemble, base, X_test, y_test,
            elo_only_features, PREMATCH_FEATURES,
            blend_ens=blend_w_ens, blend_log=blend_w_log, n_repeats=5,
            random_state=int(cfg["model"]["random_state"]),
        )
        # Univariate correlations → signs for the flattened coef chart.
        corrs: dict[str, float] = {}
        for name in PREMATCH_FEATURES:
            try:
                col = X_test[name].astype(float).values
                if float(np.std(col)) > 1e-12:
                    corrs[name] = float(np.corrcoef(col, y_test)[0, 1])
                else:
                    corrs[name] = 0.0
            except Exception:
                corrs[name] = 0.0
        fi_path = artifacts_dir / "feature_importance.csv"
        with open(fi_path, "w") as f:
            f.write("feature,mean_importance,positive_folds,selected\n")
            perm_sorted = sorted(perm, key=lambda r: -r["mean_importance"])
            for r in perm_sorted:
                f.write(
                    f"{r['feature']},{float(r['mean_importance']):.6f},"
                    f"{int(r['positive_folds'])},True\n"
                )
        log.info("wrote permutation feature importance (%d features) → %s",
                 len(perm_sorted), fi_path)

        # Augment the coefficients JSON with the permutation results +
        # a signed-magnitude ``coefficients`` map so the dashboard's
        # "all features the model uses to make decisions" bar chart
        # renders every feature, not just the Elo-only-logistic pair.
        max_perm = max((abs(float(r["mean_importance"])) for r in perm),
                        default=1.0) or 1.0
        max_log = max((abs(float(c)) for c in log_coef), default=1.0) or 1.0
        flat: dict[str, float] = {}
        for n, c in zip(elo_only_features, log_coef):
            flat[n] = float(c)
        for r in perm:
            name = r["feature"]
            if name in flat:
                continue
            magnitude = float(r["mean_importance"])
            sign = 1.0 if corrs.get(name, 0.0) >= 0 else -1.0
            flat[name] = sign * (magnitude / max_perm) * max_log
        flat["(intercept)"] = float(log_intercept)
        coefficients["permutation_importance"] = perm_sorted
        coefficients["coefficients"] = flat
        with open(artifacts_dir / "model_coefficients.json", "w") as f:
            json.dump(coefficients, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not dump permutation importance: %s", exc)

    return metrics


def _permutation_importance(ens, base, X_test: pd.DataFrame, y_test,
                              elo_only_features: list[str],
                              feature_list: list[str],
                              blend_ens: float = 0.70,
                              blend_log: float = 0.30,
                              n_repeats: int = 5,
                              random_state: int = 42
                              ) -> list[dict[str, float]]:
    """Walk-forward permutation importance on the blended model. Same
    method as the sklearn helper but operates on the blended (ensemble
    + logistic) probability so the reported importances match the
    probabilities the live trader actually uses.

    Accepts the :class:`WeightedEnsemble` for ``ens``; the call shape
    is identical to a single sklearn classifier so the rest of the
    routine doesn't care that the ensemble is a weighted blend
    under the hood.
    """
    rng = np.random.default_rng(random_state)

    def _blended_log_loss(X: pd.DataFrame) -> float:
        p_ens = ens.predict_proba(X[feature_list])[:, 1]
        p_log = base.predict_proba(X[elo_only_features])[:, 1]
        p = np.clip(blend_ens * p_ens + blend_log * p_log, 1e-6, 1 - 1e-6)
        return float(log_loss(y_test, p))

    base_score = _blended_log_loss(X_test)
    out: list[dict[str, float]] = []
    for name in feature_list:
        diffs = []
        for _ in range(n_repeats):
            X_shuf = X_test.copy()
            X_shuf[name] = rng.permutation(X_shuf[name].values)
            diffs.append(_blended_log_loss(X_shuf) - base_score)
        out.append({
            "feature": name,
            "mean_importance": float(np.mean(diffs)),
            "std_importance": float(np.std(diffs)),
            "positive_folds": int(sum(1 for d in diffs if d > 0)),
        })
    return out


def _ensemble_top_features(clf, feature_names: list[str], top_n: int = 6
                            ) -> list[dict[str, float]]:
    """Return the top-``top_n`` features by gain importance, when the
    underlying classifier exposes them. We pull from the *uncalibrated*
    estimator because CalibratedClassifierCV wraps it."""
    try:
        # sklearn 1.8 wraps the calibrated estimator in `.estimator`;
        # older sklearn used `.base_estimator`. The training code keeps
        # ``clf`` (unfitted) as a separate variable, so we fall back to
        # importing from there.
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: -x[1],
        )
        return [{"name": n, "importance": float(v)} for n, v in pairs[:top_n]]
    except Exception:
        return []


def _eval(y_true, y_prob) -> dict[str, float]:
    """Standard probability-forecast metrics. Brier + log-loss are the
    proper-scoring ones; accuracy / F1 / precision / recall / ROC AUC
    are included so the trading dashboard can show the same eight
    fields it shows for the Kalshi bots without a special case."""
    y_prob_arr = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y_pred = (y_prob_arr >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob_arr)),
        "brier": float(brier_score_loss(y_true, y_prob_arr)),
        # zero_division=0: protects against the rare case where the
        # model never predicts class 1 in the holdout window (would
        # happen on a degenerate degenerate eval slice).
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_arr)),
    }


def _elo_state_to_dict(state: EloState) -> dict[str, Any]:
    return {
        "default_rating": state.default_rating,
        "k_base": state.k_base,
        "k_floor": state.k_floor,
        "k_decay_matches": state.k_decay_matches,
        "surface_k_multiplier": state.surface_k_multiplier,
        "surface_blend": state.surface_blend,
        "overall": dict(state.overall),
        "surface": {f"{k[0]}|{k[1]}": v for k, v in state.surface.items()},
        "matches_played": dict(state.matches_played),
        "surface_matches": {f"{k[0]}|{k[1]}": v for k, v in state.surface_matches.items()},
    }


def load_elo_state(d: dict[str, Any]) -> EloState:
    s = EloState(
        default_rating=d["default_rating"],
        k_base=d["k_base"], k_floor=d["k_floor"],
        k_decay_matches=d["k_decay_matches"],
        surface_k_multiplier=d["surface_k_multiplier"],
        surface_blend=d["surface_blend"],
    )
    s.overall = dict(d["overall"])
    s.surface = {tuple(k.split("|", 1)): v for k, v in d["surface"].items()}
    s.matches_played.update(d["matches_played"])
    s.surface_matches.update({tuple(k.split("|", 1)): v
                              for k, v in d["surface_matches"].items()})
    return s


if __name__ == "__main__":
    train_and_persist()
