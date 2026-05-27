"""Train the in-match adjustment model.

Replaces the rules-based layer in ``live_adjustment_model.py`` with a
calibrated classifier: live snapshot features → P(player_a wins
match). The rules layer remains as an auditable fallback and as the
source of the dashboard's ``rules_fired`` string; only the numerical
probability is taken from the trained model.

Training data: ``data/processed/pbp_snapshots.csv`` (built by
``features/build_pbp_snapshots.py`` from real Grand Slam PBP).

Validation: temporal holdout — most-recent ``test_window_years``
years are the test set, everything earlier is the train set. We
report Brier score, log loss, and accuracy bucketed by
match-progress so we can see whether the model is sharp at
early-match (where rules are weakest) and stays calibrated late.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from ..features.build_live_features import standardize
from ..features.build_pbp_snapshots import FEATURE_COLUMNS
from ..models.live_adjustment_model import adjust as rules_adjust
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.train_inmatch_model")


# How many of the most-recent slam years go to the held-out test set.
TEST_WINDOW_YEARS = 1


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_year = int(df["year"].max())
    cutoff = max_year - TEST_WINDOW_YEARS + 1
    train = df[df["year"] < cutoff].reset_index(drop=True)
    test = df[df["year"] >= cutoff].reset_index(drop=True)
    return train, test


def _snapshot_to_live_record(row: pd.Series) -> dict:
    """Map a snapshot row to the canonical live_record schema the
    rules layer expects, so we can score the rules baseline on the
    same rows the model is evaluated on."""
    rec = {
        "match_id": row.get("match_id", ""),
        "tournament": row.get("slam", ""),
        "surface": row.get("surface", "Hard"),
        "player_a": "P1",
        "player_b": "P2",
        "set_score_a": float(row.get("set_score_a", 0)),
        "set_score_b": float(row.get("set_score_b", 0)),
        "games_won_last_3_a": float(row.get("games_won_last_3_a", 0)),
        "games_won_last_3_b": float(row.get("games_won_last_3_b", 0)),
        "first_serve_pct_a": float(row.get("first_serve_pct_a", 0.6)),
        "first_serve_pct_b": float(row.get("first_serve_pct_b", 0.6)),
        "aces_a": float(row.get("aces_a", 0)),
        "aces_b": float(row.get("aces_b", 0)),
        "double_faults_a": float(row.get("double_faults_a", 0)),
        "double_faults_b": float(row.get("double_faults_b", 0)),
        "unforced_errors_a": float(row.get("unforced_errors_a", 0)),
        "unforced_errors_b": float(row.get("unforced_errors_b", 0)),
        "is_tiebreak": bool(row.get("is_tiebreak", 0)),
        "is_decider": bool(row.get("is_decider", 0)),
        "serving_a": bool(row.get("serving_a", 0)),
    }
    return standardize(rec)


def _rules_baseline_probs(test: pd.DataFrame) -> np.ndarray:
    """Apply the existing rules layer to each test row with a flat 50/50
    pre-match prior, return the live_prob_a predictions."""
    out = np.empty(len(test), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        s = pd.Series(row._asdict())
        live = _snapshot_to_live_record(s)
        adj = rules_adjust(0.5, live)
        out[i] = float(adj.live_prob_a)
    return out


def _metrics_by_bucket(p_pred: np.ndarray, y: np.ndarray,
                       bucket_key: np.ndarray,
                       buckets: list[tuple[float, float, str]]) -> list[dict]:
    rows = []
    for lo, hi, name in buckets:
        mask = (bucket_key >= lo) & (bucket_key < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bucket": name,
            "n": n,
            "brier": float(brier_score_loss(y[mask], p_pred[mask])),
            "logloss": float(log_loss(y[mask], np.clip(p_pred[mask], 1e-6, 1 - 1e-6),
                                       labels=[0, 1])),
            "acc": float(((p_pred[mask] >= 0.5) == y[mask].astype(bool)).mean()),
        })
    return rows


def train_and_eval() -> dict:
    cfg = load_config()
    proc_dir = resolve_path(cfg["paths"]["processed_dir"])
    snaps_path = proc_dir / "pbp_snapshots.csv"
    if not snaps_path.exists():
        raise FileNotFoundError(
            f"missing {snaps_path} — run features.build_pbp_snapshots first"
        )
    df = pd.read_csv(snaps_path)
    log.info("loaded %d snapshots (%d matches)", len(df), df["match_id"].nunique())

    train, test = _split(df)
    log.info("split: train %d (years %d-%d), test %d (years %d-%d)",
             len(train), int(train["year"].min()), int(train["year"].max()),
             len(test), int(test["year"].min()), int(test["year"].max()))

    X_train = train[list(FEATURE_COLUMNS)].astype(float).values
    y_train = train["won_a"].astype(int).values
    X_test = test[list(FEATURE_COLUMNS)].astype(float).values
    y_test = test["won_a"].astype(int).values

    # Base learner: HistGradientBoosting handles the non-linear
    # interactions (sets_diff × is_decider, serve % × is_tiebreak,
    # progress × games_diff) without manual feature engineering.
    # Wrap in a sigmoid-calibrator on the train split so the output
    # behaves like a probability rather than a classifier score.
    base = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.5,
        random_state=int(cfg["model"]["random_state"]),
    )
    # Group the CV folds by match_id. Each match contributes ~25-40
    # snapshots that all share the same ``won_a`` label; random k-fold
    # scatters them across folds, so the inner base estimator can
    # memorize match M's outcome from one snapshot and "predict" it
    # on a held-out snapshot from the same match. The OUTER test
    # split (different year) is honest, but the sigmoid calibrator
    # would be fit on optimistic out-of-fold probabilities, producing
    # mildly over-confident outputs on test. Switching to GroupKFold
    # by match_id makes the inner CV honest.
    gkf = GroupKFold(n_splits=5)
    groups = train["match_id"].values
    cv_splits = list(gkf.split(X_train, y_train, groups=groups))
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv_splits)
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]

    # Rules baseline at the same rows
    p_rules = _rules_baseline_probs(test)

    # Top-line metrics
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "model": {
            "brier": float(brier_score_loss(y_test, p_test)),
            "logloss": float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6),
                                       labels=[0, 1])),
            "acc": float(((p_test >= 0.5) == y_test.astype(bool)).mean()),
        },
        "rules_baseline": {
            "brier": float(brier_score_loss(y_test, p_rules)),
            "logloss": float(log_loss(y_test, np.clip(p_rules, 1e-6, 1 - 1e-6),
                                       labels=[0, 1])),
            "acc": float(((p_rules >= 0.5) == y_test.astype(bool)).mean()),
        },
        "by_progress_model": _metrics_by_bucket(
            p_test, y_test, test["progress"].values,
            [(0.0, 0.25, "early"), (0.25, 0.5, "mid"),
             (0.5, 0.75, "late"), (0.75, 5.0, "endgame")]),
        "by_progress_rules": _metrics_by_bucket(
            p_rules, y_test, test["progress"].values,
            [(0.0, 0.25, "early"), (0.25, 0.5, "mid"),
             (0.5, 0.75, "late"), (0.75, 5.0, "endgame")]),
        "by_setsdiff_model": _metrics_by_bucket(
            p_test, y_test, test["sets_diff"].values,
            [(-5, -1.5, "down 2+"), (-1.5, -0.5, "down 1"),
             (-0.5, 0.5, "even"), (0.5, 1.5, "up 1"),
             (1.5, 5, "up 2+")]),
        "feature_columns": list(FEATURE_COLUMNS),
    }

    # Persist artifacts
    art_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, art_dir / "inmatch_model.joblib")
    with open(art_dir / "inmatch_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    m = metrics["model"]
    r = metrics["rules_baseline"]
    log.info("OVERALL  model:  brier=%.4f  logloss=%.4f  acc=%.3f",
             m["brier"], m["logloss"], m["acc"])
    log.info("OVERALL  rules:  brier=%.4f  logloss=%.4f  acc=%.3f",
             r["brier"], r["logloss"], r["acc"])
    log.info("Brier improvement: %.4f (%.1f%% relative)",
             r["brier"] - m["brier"],
             100 * (r["brier"] - m["brier"]) / r["brier"])
    log.info("By progress (model):")
    for b in metrics["by_progress_model"]:
        log.info("  %-8s n=%-5d  brier=%.4f  logloss=%.4f  acc=%.3f",
                 b["bucket"], b["n"], b["brier"], b["logloss"], b["acc"])
    log.info("By progress (rules):")
    for b in metrics["by_progress_rules"]:
        log.info("  %-8s n=%-5d  brier=%.4f  logloss=%.4f  acc=%.3f",
                 b["bucket"], b["n"], b["brier"], b["logloss"], b["acc"])

    return metrics


if __name__ == "__main__":
    train_and_eval()
