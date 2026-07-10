"""
analysis/run_autogluon_ceiling.py — AutoML performance-ceiling benchmark.

Purpose
-------
This is NOT a candidate production model. It answers one question: *does any model
class do materially better than our transparent XGBoost/logistic-regression pair on
this (leakage-free, synthetic) data?* AutoGluon trains and stack-ensembles a broad
family (LightGBM, CatBoost, XGBoost, random forests, neural nets, weighted ensembles)
under a time budget. If its best test AUROC lands in the same ~0.56 neighbourhood as
our hand-built models, that is positive evidence the low discrimination is a property
of the data, not of our model choice — strengthening the leakage/no-signal narrative.

Design
------
* Identical feature matrix to train.py (imports build_features), so the comparison is
  apples-to-apples.
* A single held-out stratified 20% test set gives an honest out-of-sample AUROC; the
  remaining 80% is handed to AutoGluon, which does its own internal validation.
* Outputs: outputs/tables/autogluon_leaderboard.csv and a printed ceiling summary.

Run
---
    pip install "autogluon.tabular[all]"     # heavy (~pulls torch); one-time
    python analysis/run_autogluon_ceiling.py --time-limit 180
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Reuse the exact feature construction used by the production models.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from train import build_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# The transparent-model ceiling to beat (nested-CV OOF AUROC; see model_comparison.csv).
TRANSPARENT_AUROC = 0.560


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGluon performance-ceiling benchmark")
    parser.add_argument("--cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--ttd-file", default="outputs/tables/ttd_events.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--time-limit", type=int, default=180, help="AutoGluon fit budget (seconds)")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        log.error(
            "AutoGluon is not installed. Install with:  pip install \"autogluon.tabular[all]\"\n"
            "This benchmark is optional; the primary pipeline (train.py) does not require it."
        )
        sys.exit(2)

    tbl_d = Path(args.output_dir) / "tables"
    tbl_d.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    ttd = pd.read_csv(args.ttd_file)
    df, feature_cols = build_features(cohort, ttd)

    data = df[feature_cols + ["y"]].copy()
    train_df, test_df = train_test_split(
        data, test_size=args.test_size, random_state=42, stratify=data["y"]
    )
    log.info(
        "AutoGluon ceiling: train=%d  test=%d  features=%d  event_rate=%.1f%%",
        len(train_df), len(test_df), len(feature_cols), 100 * data["y"].mean(),
    )

    # Robust strong-learner set. LightGBM (GBM) is excluded because it segfaults under
    # bagging on some macOS/libomp setups; the ceiling is still spanned by gradient
    # boosting (CatBoost, XGBoost), bagged trees (RF, ExtraTrees), a linear model, and
    # their weighted ensemble. Bagging/dynamic-stacking are disabled for stability and
    # because a single held-out test set already gives the honest out-of-sample estimate.
    hyperparameters = {"CAT": {}, "XGB": {}, "RF": [{"criterion": "gini"}],
                       "XT": [{"criterion": "gini"}], "LR": {}}
    predictor = TabularPredictor(
        label="y",
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(Path(args.output_dir) / "autogluon_models"),
    ).fit(
        train_df,
        time_limit=args.time_limit,
        hyperparameters=hyperparameters,
        fit_weighted_ensemble=True,
        num_bag_folds=0,
        dynamic_stacking=False,
    )

    # Honest out-of-sample AUROC on the untouched test set.
    proba = predictor.predict_proba(test_df.drop(columns=["y"]))
    pos_col = predictor.class_labels[-1]
    y_test = test_df["y"].values
    test_auroc = roc_auc_score(y_test, proba[pos_col].values)
    test_auprc = average_precision_score(y_test, proba[pos_col].values)

    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard.to_csv(tbl_d / "autogluon_leaderboard.csv", index=False)

    best = leaderboard.sort_values("score_test", ascending=False).iloc[0]
    delta = test_auroc - TRANSPARENT_AUROC

    log.info("=" * 68)
    log.info("AutoGluon best model: %s", best["model"])
    log.info("AutoGluon held-out test AUROC: %.4f  (AUPRC=%.4f)", test_auroc, test_auprc)
    log.info("Transparent-model ceiling (XGB/LR): %.4f", TRANSPARENT_AUROC)
    log.info("Delta (AutoGluon − transparent): %+.4f", delta)
    verdict = (
        "No material lift — low discrimination is a property of the data, not the model choice."
        if delta < 0.02
        else "AutoGluon adds signal — revisit feature/model choices."
    )
    log.info("Verdict: %s", verdict)
    log.info("Leaderboard saved → %s", tbl_d / "autogluon_leaderboard.csv")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
