# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
ml/train.py — XGBoost 1-year discontinuation predictor.

Cohort restriction
------------------
Only patients with ≥365 days of follow-up are included. This is a prerequisite
for the binary classification framing: a patient cannot be classified as
"persisted for 12 months" unless 12 months were actually observed. Including
shorter-followup patients would mean some patients are administratively
incapable of being positive, creating an informative-censoring artefact that
the model would exploit rather than learn patient biology.
    Sperrin M et al. BMJ Open. 2017;7:e017271.

Feature set (28 features after removing leakage variables)
----------------------------------------------------------
Removed:
  followup_days   — encodes the observation window length, which is a
                    deterministic function of whether the 365-day outcome
                    is observable. The top SHAP predictor in prior runs was
                    followup_days, an immediate signal of leakage.
  drug_class_num  — ordinal-encoded duplicate of the drug_class one-hot
                    triplet (drug_metformin / drug_glp1 / drug_sglt2).
                    Redundant columns inflate apparent feature importance.
  sex_male        — perfectly collinear with sex_female in a binary-coded
                    sex variable. Including both inflates VIF and provides
                    zero additional information to the model.

Model: XGBClassifier, 5-fold stratified CV, final train on full data.

Outputs
-------
  outputs/models/xgb_discontinuation.pkl
  outputs/models/xgb_model.ubj
  outputs/tables/ml_metrics.csv          — AUC ± CI, AUPRC, Brier, ECE per fold
  outputs/tables/model_comparison.csv    — XGBoost vs. logistic regression vs. baselines
  outputs/tables/feature_importance.csv  — XGBoost gain importance (ranked)
  outputs/tables/fairness_report.csv     — per-subgroup AUC (sex, age band, drug class)
  outputs/figures/roc_curve.png          — OOF ROC with bootstrap 95% CI
  outputs/figures/calibration_curve.png  — reliability diagram with ECE annotation
  outputs/figures/confusion_matrix.png
  outputs/figures/shap_summary.png
  outputs/figures/shap_beeswarm.png
  outputs/figures/shap_force_examples.png
  outputs/figures/umap_phenotypes.png    — leaf embeddings coloured by drug class
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from io import BytesIO
from pathlib import Path

# Allow `from evaluation import ...` when run as python ml/train.py from repo root
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import umap
import xgboost as xgb
from evaluation import (
    build_evaluation_report,
    compute_e_values_for_cox_results,
    evaluate_fairness_subgroups,
    run_baseline_cv,
    save_calibration_curve,
    save_fairness_report,
    save_roc_with_ci,
    select_operating_threshold,
    summarise_baseline_results,
)
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# MLflow is an optional dependency. When absent the training pipeline runs
# without experiment tracking; when present, every run is automatically
# logged to the configured tracking server.
try:
    import mlflow
    import mlflow.xgboost
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    log.debug("mlflow not installed — experiment tracking disabled.")

# Minimum observation window required for the 365-day binary outcome to be defined.
# Patients with fewer than this many follow-up days are excluded.
MIN_FOLLOWUP_DAYS: int = 365

COMORBIDITY_COLS = [
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.5,
    reg_alpha=1.0,
    reg_lambda=5.0,
    eval_metric="auc",
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)


# ── Feature engineering ───────────────────────────────────────────────────────


def build_features(
    cohort: pd.DataFrame,
    ttd: pd.DataFrame,
    min_followup_days: int = MIN_FOLLOWUP_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Join cohort and TTD event files, derive features, and define the outcome.

    The 365-day binary outcome is only well-defined for patients who were
    under observation for at least 365 days. Patients with shorter follow-up
    are removed before feature construction, not after, to avoid any implicit
    leakage of observation-window length into the feature matrix.

    Args:
        cohort: Cohort baseline DataFrame (one row per patient).
        ttd: Time-to-discontinuation events DataFrame.
        min_followup_days: Minimum days of follow-up required for inclusion.

    Returns:
        Tuple of (feature DataFrame, list of feature column names).
    """
    df = cohort.merge(
        ttd[["person_id", "ttd_days", "discontinued"]],
        on="person_id", how="left",
    )
    df["discontinued"] = df["discontinued"].fillna(0).astype(int)
    df["ttd_days"]     = df["ttd_days"].fillna(df["followup_days"])
    df["followup_days"] = df["followup_days"].fillna(365).clip(lower=90)

    # Restrict to patients with ≥365 days of follow-up.
    # This is the at-risk window required for the 1-year binary outcome.
    n_before = len(df)
    df = df[df["followup_days"] >= min_followup_days].copy()
    n_after = len(df)
    log.info(
        "Followup filter (≥%d days): %d → %d patients retained (%.1f%% excluded)",
        min_followup_days, n_before, n_after, 100.0 * (n_before - n_after) / max(n_before, 1),
    )
    if n_after < 500:
        log.warning(
            "Only %d patients pass the ≥%d-day followup filter. "
            "Consider extending the observation window or relaxing the cutoff.",
            n_after, min_followup_days,
        )

    df["y"] = ((df["discontinued"] == 1) & (df["ttd_days"] <= 365)).astype(int)

    # Sex (one-hot, single column — sex_male is perfectly collinear with sex_female)
    df["sex_female"] = (df["gender_concept_id"] == 8532).astype(int)

    # Drug class — one-hot encoding only (ordinal drug_class_num excluded;
    # it encodes the same information redundantly and inflates feature importance)
    df["drug_metformin"] = (df["drug_class"] == "metformin").astype(int)
    df["drug_glp1"]      = (df["drug_class"] == "glp1").astype(int)
    df["drug_sglt2"]     = (df["drug_class"] == "sglt2").astype(int)

    # Comorbidities
    for c in COMORBIDITY_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)

    df["comorbidity_count"] = df[COMORBIDITY_COLS].sum(axis=1)
    df["age_at_index"]      = df["age_at_index"].fillna(60.0)
    df["cci"]               = df["cci"].fillna(0)

    # Time since T2DM diagnosis to index date (longer disease duration → different
    # treatment patterns and persistence behaviour)
    try:
        df["days_since_t2dm_dx"] = (
            pd.to_datetime(df["index_date"]) - pd.to_datetime(df["t2dm_date"])
        ).dt.days.clip(lower=0).fillna(0)
    except Exception:
        df["days_since_t2dm_dx"] = 0

    # Age binary flag
    df["age_over65"] = (df["age_at_index"] >= 65).astype(int)

    # Drug-class × comorbidity interaction terms (differential persistence by burden)
    df["glp1_x_codx"]  = df["drug_glp1"]  * df["comorbidity_count"]
    df["sglt2_x_codx"] = df["drug_sglt2"] * df["comorbidity_count"]
    df["glp1_x_cci"]   = df["drug_glp1"]  * df["cci"]
    df["sglt2_x_cci"]  = df["drug_sglt2"] * df["cci"]

    feature_cols = (
        [
            "age_at_index", "age_over65", "sex_female",
            "drug_metformin", "drug_glp1", "drug_sglt2",
            "cci", "comorbidity_count", "days_since_t2dm_dx",
            "glp1_x_codx", "sglt2_x_codx", "glp1_x_cci", "sglt2_x_cci",
        ]
        + COMORBIDITY_COLS
    )  # 28 features total

    df = df.dropna(subset=feature_cols + ["y"])
    log.info("Feature matrix: %d patients × %d features", len(df), len(feature_cols))
    return df, feature_cols


# ── Cross-validation ──────────────────────────────────────────────────────────


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    """
    Execute 5-fold stratified cross-validation for the XGBoost model.

    Early stopping is applied within each fold using the validation fold as
    the eval set. The final model (trained on all data) uses the mean
    best_iteration across folds as n_estimators to avoid overfitting.

    Args:
        X: Feature matrix.
        y: Binary outcome vector.

    Returns:
        Tuple of (summary dict, per-fold metrics list, OOF probabilities,
        OOF labels).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    all_probs, all_labels = [], []
    best_iters: list[int] = []

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=30)
        model.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)

        best_iters.append(model.best_iteration or XGB_PARAMS["n_estimators"])
        probs = model.predict_proba(X[val])[:, 1]
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_labels.extend(y[val])

        yt = y[val]
        fold_metrics.append({
            "fold":      fold + 1,
            "auc":       roc_auc_score(yt, probs),
            "accuracy":  accuracy_score(yt, preds),
            "precision": precision_score(yt, preds, zero_division=0),
            "recall":    recall_score(yt, preds, zero_division=0),
            "f1":        f1_score(yt, preds, zero_division=0),
            "brier":     brier_score_loss(yt, probs),
            "auprc":     average_precision_score(yt, probs),
        })
        log.info(
            "  Fold %d: AUC=%.3f  F1=%.3f  Brier=%.4f  best_iter=%d",
            fold + 1, fold_metrics[-1]["auc"], fold_metrics[-1]["f1"],
            fold_metrics[-1]["brier"], best_iters[-1],
        )

    fm = pd.DataFrame(fold_metrics)
    summary = {
        "mean_auc":       fm["auc"].mean(),       "std_auc":       fm["auc"].std(),
        "mean_accuracy":  fm["accuracy"].mean(),   "std_accuracy":  fm["accuracy"].std(),
        "mean_precision": fm["precision"].mean(),  "std_precision": fm["precision"].std(),
        "mean_recall":    fm["recall"].mean(),     "std_recall":    fm["recall"].std(),
        "mean_f1":        fm["f1"].mean(),         "std_f1":        fm["f1"].std(),
        "mean_brier":     fm["brier"].mean(),      "std_brier":     fm["brier"].std(),
        "mean_auprc":     fm["auprc"].mean(),      "std_auprc":     fm["auprc"].std(),
        "mean_best_iter": int(np.mean(best_iters)),
    }
    log.info(
        "CV summary: AUC=%.3f±%.3f  F1=%.3f±%.3f  Brier=%.4f±%.4f  mean_best_iter=%d",
        summary["mean_auc"], summary["std_auc"],
        summary["mean_f1"],  summary["std_f1"],
        summary["mean_brier"], summary["std_brier"],
        summary["mean_best_iter"],
    )
    return summary, fold_metrics, np.array(all_probs), np.array(all_labels)


# ── Nested cross-validation ────────────────────────────────────────────────────


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> tuple[dict, list[dict], np.ndarray]:
    """
    Nested stratified CV that yields an UNBIASED out-of-fold estimate.

    The flat CV previously used here selected XGBoost's early-stopping
    best_iteration on the *same* validation fold whose probabilities were then
    reported as OOF — a mild but real optimism leak. Nested CV removes it:

      * Inner loop (on the outer-train split only) selects best_iteration via
        early stopping — the outer-test fold is never seen during tuning.
      * The model is refit on the full outer-train with that fixed n_estimators
        (no early stopping) and predicts the untouched outer-test fold.

    OOF probabilities are returned in ORIGINAL row order (each sample is scored
    exactly once, on the fold where it is held out), so they align with the
    feature frame for downstream fairness/subgroup analysis.

    Returns:
        (summary dict, per-outer-fold metrics, OOF probabilities in original order).
    """
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.full(len(y), np.nan)
    fold_metrics: list[dict] = []
    best_iters: list[int] = []

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]

        # Inner CV: pick best_iteration without ever touching the outer-test fold.
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold + 1)
        inner_best: list[int] = []
        for itr, ival in inner.split(X_tr, y_tr):
            m = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=30)
            m.fit(X_tr[itr], y_tr[itr], eval_set=[(X_tr[ival], y_tr[ival])], verbose=False)
            inner_best.append(m.best_iteration or XGB_PARAMS["n_estimators"])
        best_n = max(int(np.mean(inner_best)), 1)
        best_iters.append(best_n)

        # Refit on full outer-train with fixed n_estimators; score untouched outer-test.
        final_params = {**XGB_PARAMS, "n_estimators": best_n}
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X[te])[:, 1]
        oof_probs[te] = probs

        yt = y[te]
        fold_metrics.append({
            "fold":  fold + 1,
            "auc":   roc_auc_score(yt, probs),
            "brier": brier_score_loss(yt, probs),
            "auprc": average_precision_score(yt, probs),
            "best_iter": best_n,
        })
        log.info(
            "  Outer fold %d: AUC=%.3f  Brier=%.4f  best_iter=%d (inner-selected)",
            fold + 1, fold_metrics[-1]["auc"], fold_metrics[-1]["brier"], best_n,
        )

    fm = pd.DataFrame(fold_metrics)
    summary = {
        "mean_auc":   fm["auc"].mean(),   "std_auc":   fm["auc"].std(),
        "mean_brier": fm["brier"].mean(), "std_brier": fm["brier"].std(),
        "mean_auprc": fm["auprc"].mean(), "std_auprc": fm["auprc"].std(),
        "mean_best_iter": int(np.mean(best_iters)),
    }
    log.info(
        "Nested CV: AUC=%.3f±%.3f  Brier=%.4f±%.4f  mean_best_iter=%d",
        summary["mean_auc"], summary["std_auc"], summary["mean_brier"],
        summary["std_brier"], summary["mean_best_iter"],
    )
    return summary, fold_metrics, oof_probs


# ── Figures ───────────────────────────────────────────────────────────────────


def save_confusion(probs_oof: np.ndarray, labels_oof: np.ndarray, fig_dir: Path, threshold: float = 0.5) -> None:
    preds = (probs_oof >= threshold).astype(int)
    cm = confusion_matrix(labels_oof, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Persistent", "Discontinued"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (tuned threshold = {threshold:.2f})\n1-Year Discontinuation (OOF)")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    log.info("Confusion matrix saved")


def save_shap(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
    fig_dir: Path,
) -> None:
    n_bg = min(500, len(X))
    X_bg = X[:n_bg]

    explainer  = shap.TreeExplainer(model)
    explanation = explainer(X_bg)
    sv = explanation.values

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_bg, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_bg, feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP Beeswarm — 1-Year Discontinuation", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    buffers = []
    for i in range(min(3, n_bg)):
        fig = plt.figure(figsize=(10, 3))
        shap.plots.waterfall(explanation[i], max_display=12, show=False)
        plt.title(f"Patient {i+1} — SHAP Waterfall", fontsize=9)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        buffers.append(Image.open(buf).copy())
        plt.close(fig)

    if buffers:
        total_h = sum(b.height for b in buffers)
        max_w   = max(b.width  for b in buffers)
        canvas  = Image.new("RGB", (max_w, total_h), "white")
        y_off   = 0
        for b in buffers:
            canvas.paste(b, (0, y_off))
            y_off += b.height
        canvas.save(str(fig_dir / "shap_force_examples.png"))

    log.info("SHAP plots saved")


def save_umap(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    drug_class_series: np.ndarray,
    fig_dir: Path,
) -> None:
    n = min(2000, len(X))
    booster  = model.get_booster()
    leaf_ids = booster.predict(xgb.DMatrix(X[:n]), pred_leaf=True).astype(float)

    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(leaf_ids)

    drug_labels = {0: "Metformin", 1: "GLP-1 RA", 2: "SGLT-2i"}
    colors      = {0: "#3498DB",   1: "#E74C3C",  2: "#2ECC71"}
    dc          = drug_class_series[:n]

    plt.figure(figsize=(7, 5))
    for dc_val, label in drug_labels.items():
        mask = dc == dc_val
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=colors[dc_val], label=label, alpha=0.5, s=6,
        )
    plt.legend(title="Drug Class", fontsize=9)
    plt.title("UMAP of XGBoost Leaf Embeddings\n(coloured by drug class)", fontsize=11)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(fig_dir / "umap_phenotypes.png", dpi=150)
    plt.close()
    log.info("UMAP saved")


# ── Main ──────────────────────────────────────────────────────────────────────


def _mlflow_setup() -> None:
    """Configure MLflow tracking URI and experiment, if available."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import MLFLOW as MLFLOW_CFG
        mlflow.set_tracking_uri(MLFLOW_CFG.tracking_uri)
        mlflow.set_experiment(MLFLOW_CFG.experiment_name)
        log.info("MLflow tracking: %s  experiment: %s",
                 MLFLOW_CFG.tracking_uri, MLFLOW_CFG.experiment_name)
    except Exception as exc:
        log.warning("MLflow setup failed (%s) — continuing without tracking.", exc)


def _mlflow_log(xgb_report, comparison_df, feature_cols, final, mod_d: Path,
                lr_report=None, primary_threshold=None) -> None:
    """
    Log hyperparameters, metrics, and the trained model artefact to MLflow.

    Runs inside an active mlflow.start_run() context. Raises no exceptions —
    any MLflow failure is caught and logged as a warning so the pipeline
    is not interrupted by a tracking-server outage.

    Logged parameters
    -----------------
    All XGB_PARAMS entries + min_followup_days + n_features.

    Logged metrics
    --------------
    oof_auroc, oof_auroc_ci_low, oof_auroc_ci_high, oof_auprc,
    oof_brier, oof_ece, oof_mce.
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        # Parameters
        for k, v in XGB_PARAMS.items():
            mlflow.log_param(k, v)
        mlflow.log_param("min_followup_days", MIN_FOLLOWUP_DAYS)
        mlflow.log_param("n_features",        len(feature_cols))
        mlflow.log_param("feature_set",       ",".join(feature_cols))

        # Metrics — OOF XGBoost
        d  = xgb_report.discrimination
        ca = xgb_report.calibration
        mlflow.log_metric("oof_auroc",        d.auroc)
        mlflow.log_metric("oof_auroc_ci_low",  d.auroc_ci_low)
        mlflow.log_metric("oof_auroc_ci_high", d.auroc_ci_high)
        mlflow.log_metric("oof_auprc",        d.auprc)
        mlflow.log_metric("oof_brier",        d.brier_score)
        mlflow.log_metric("oof_ece",          ca.expected_calibration_error)
        mlflow.log_metric("oof_mce",          ca.max_calibration_error)

        # Baseline deltas (XGBoost AUROC lift over logistic regression)
        lr_row = comparison_df[comparison_df["Model"].str.contains("Logistic Regression", case=False, na=False)]
        if not lr_row.empty:
            lr_auroc = lr_row["Mean AUROC"].values[0]
            mlflow.log_metric("auroc_lift_vs_lr", round(d.auroc - float(lr_auroc), 4))

        # Primary model (logistic regression) metrics + operating threshold
        if lr_report is not None:
            ld, lca = lr_report.discrimination, lr_report.calibration
            mlflow.log_param("primary_model", "logistic_regression")
            mlflow.log_metric("primary_oof_auroc", ld.auroc)
            mlflow.log_metric("primary_oof_auprc", ld.auprc)
            mlflow.log_metric("primary_oof_brier", ld.brier_score)
            mlflow.log_metric("primary_oof_ece",   lca.expected_calibration_error)
        if primary_threshold is not None:
            mlflow.log_param("primary_operating_threshold", primary_threshold)

        # Model artefact
        mlflow.xgboost.log_model(final, artifact_path="xgb_discontinuation")
        log.info("MLflow run logged successfully.")
    except Exception as exc:
        log.warning("MLflow logging failed (%s) — run artefacts saved locally.", exc)


def run_training(cohort_path: str, ttd_path: str, output_dir: str) -> None:
    _mlflow_setup()

    out   = Path(output_dir)
    fig_d = out / "figures"
    tbl_d = out / "tables"
    mod_d = out / "models"
    for d in (fig_d, tbl_d, mod_d):
        d.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(cohort_path)
    ttd    = pd.read_csv(ttd_path)

    df, feature_cols = build_features(cohort, ttd)
    X  = df[feature_cols].values.astype(float)
    y  = df["y"].values
    dc = df["drug_class_num"].values if "drug_class_num" in df.columns else np.zeros(len(df))

    pos_rate = y.mean()
    log.info(
        "Training set: n=%d  features=%d  1-year_discontinuation_rate=%.1f%%",
        len(df), len(feature_cols), 100 * pos_rate,
    )

    if len(np.unique(y)) < 2:
        log.warning(
            "Outcome has no variation after ≥%d-day followup filter (n=%d). "
            "Verify that the TTD file is correctly joined and the cohort contains "
            "a mix of persistent and discontinued patients.",
            MIN_FOLLOWUP_DAYS, len(df),
        )
        return

    if pos_rate < 0.05 or pos_rate > 0.95:
        log.warning(
            "Extreme class imbalance detected (event_rate=%.1f%%). "
            "Consider scale_pos_weight in XGBoost or oversampling if recall is low.",
            100 * pos_rate,
        )

    # ── Nested CV for XGBoost (unbiased OOF, no early-stopping leak) ───────────
    log.info("Running nested stratified CV for XGBoost (5 outer × 3 inner) …")
    summary, xgb_fold_metrics, oof_probs = run_nested_cv(X, y, n_splits=5, inner_splits=3)
    oof_labels = y  # nested-CV OOF is in original row order → aligns with the frame

    if summary["mean_auc"] < 0.60:
        log.warning(
            "Nested-CV AUC=%.3f is below 0.60 — genuine signal is weak; a parsimonious "
            "linear model is preferred as primary (see model_comparison.csv).",
            summary["mean_auc"],
        )

    # ── Baselines (Majority, Stratified, Logistic Regression) — same folds ─────
    log.info("Running baseline cross-validation (majority class + logistic regression) …")
    baseline_df, baseline_oof = run_baseline_cv(X, y, n_splits=5, random_state=42)
    lr_oof = baseline_oof["Logistic Regression"]

    # ── Evaluation reports: LR = PRIMARY, XGBoost = SENSITIVITY ────────────────
    # Logistic regression is promoted to primary because XGBoost shows no
    # meaningful AUROC lift over it; the parsimonious, natively-calibrated model
    # is preferred for clinical reporting (Steyerberg et al. 2010).
    lr_report  = build_evaluation_report(oof_labels, lr_oof, "Logistic Regression", "oof")
    xgb_report = build_evaluation_report(oof_labels, oof_probs, "XGBoost", "oof")

    lift = xgb_report.discrimination.auroc - lr_report.discrimination.auroc
    log.info(
        "XGBoost AUROC lift over logistic regression: %+.4f — %s.",
        lift, "keeping LR as primary" if lift < 0.02 else "XGBoost adds signal",
    )

    # ── Operating-threshold selection (replaces the naive 0.5) ─────────────────
    log.info("Selecting operating thresholds (Youden's J) …")
    lr_thr  = select_operating_threshold(oof_labels, lr_oof, method="youden")
    xgb_thr = select_operating_threshold(oof_labels, oof_probs, method="youden")
    # For reference, also record the prevalence-matched threshold for the primary model.
    lr_prev_thr = select_operating_threshold(oof_labels, lr_oof, method="prevalence")

    thresholds_out = {
        "primary_model": "logistic_regression",
        "logistic_regression_youden": lr_thr,
        "logistic_regression_prevalence": lr_prev_thr,
        "xgboost_youden": xgb_thr,
        "event_prevalence": round(float(oof_labels.mean()), 4),
    }
    (tbl_d / "operating_thresholds.json").write_text(json.dumps(thresholds_out, indent=2))

    # ── Model comparison table (primary/sensitivity labelled) ──────────────────
    comparison_df = summarise_baseline_results(
        baseline_df, baseline_oof, oof_labels, xgb_report,
        xgb_label="XGBoost (sensitivity)", primary_baseline="Logistic Regression",
    )
    comparison_df.to_csv(tbl_d / "model_comparison.csv", index=False)
    log.info("Model comparison table:\n%s", comparison_df.to_string(index=False))

    # ── Metrics CSV: both models, threshold-free + tuned-threshold metrics ─────
    def _metric_row(name: str, rep, thr: dict) -> dict:
        d, ca = rep.discrimination, rep.calibration
        return {
            "model":         name,
            "split":         "oof_nested" if name == "XGBoost" else "oof",
            "auroc":         round(d.auroc, 4),
            "auroc_ci_low":  round(d.auroc_ci_low, 4),
            "auroc_ci_high": round(d.auroc_ci_high, 4),
            "auprc":         round(d.auprc, 4),
            "brier":         round(d.brier_score, 4),
            "ece":           round(ca.expected_calibration_error, 4),
            "mce":           round(ca.max_calibration_error, 4),
            "op_threshold":  thr["threshold"],
            "accuracy":      thr["accuracy"],
            "precision":     thr["precision"],
            "recall":        thr["recall"],
            "specificity":   thr["specificity"],
            "f1":            thr["f1"],
        }

    metric_rows = [
        _metric_row("Logistic Regression (primary)", lr_report, lr_thr),
        _metric_row("XGBoost", xgb_report, xgb_thr),
    ]
    # Per-outer-fold XGBoost discrimination for transparency
    for m in xgb_fold_metrics:
        metric_rows.append({
            "model": "XGBoost", "split": f"outer_fold_{m['fold']}",
            "auroc": round(m["auc"], 4), "auprc": round(m["auprc"], 4),
            "brier": round(m["brier"], 4),
        })
    pd.DataFrame(metric_rows).to_csv(tbl_d / "ml_metrics.csv", index=False)

    # ── Final PRIMARY model: scaled logistic regression on full data ───────────
    log.info("Fitting final primary model (scaled logistic regression) on full data …")
    primary = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)),
    ])
    primary.fit(X, y)
    with open(mod_d / "logreg_discontinuation.pkl", "wb") as f:
        pickle.dump({"model": primary, "feature_cols": feature_cols,
                     "operating_threshold": lr_thr["threshold"]}, f)
    log.info("Primary model saved: logreg_discontinuation.pkl (threshold=%.4f)", lr_thr["threshold"])

    lr_coef = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": primary.named_steps["lr"].coef_[0],
    }).reindex(primary.named_steps["lr"].coef_[0].argsort()[::-1]).reset_index(drop=True)
    lr_coef.to_csv(tbl_d / "logreg_coefficients.csv", index=False)

    # ── Final SENSITIVITY model: XGBoost (retained for SHAP/UMAP) ──────────────
    best_n = summary.get("mean_best_iter", XGB_PARAMS["n_estimators"])
    final_params = {**XGB_PARAMS, "n_estimators": best_n}
    log.info("Fitting sensitivity model (XGBoost, n_estimators=%d from nested-CV mean) …", best_n)
    final = xgb.XGBClassifier(**final_params)
    final.fit(X, y)
    with open(mod_d / "xgb_discontinuation.pkl", "wb") as f:
        pickle.dump(final, f)
    final.save_model(str(mod_d / "xgb_model.ubj"))
    log.info("Sensitivity model saved: xgb_discontinuation.pkl + xgb_model.ubj")

    # ── Feature importance (XGBoost gain) ──────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": final.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(tbl_d / "feature_importance.csv", index=False)
    log.info("Top 5 XGBoost features: %s", fi.head(5)["feature"].tolist())

    # ── Figures ────────────────────────────────────────────────────────────────
    # Headline ROC/calibration/confusion use the PRIMARY (logistic) model.
    save_roc_with_ci(oof_labels, lr_oof, fig_d / "roc_curve.png", model_name="Logistic Regression (OOF)")
    save_calibration_curve(oof_labels, lr_oof, fig_d / "calibration_curve.png", model_name="Logistic Regression")
    save_confusion(lr_oof, oof_labels, fig_d, threshold=lr_thr["threshold"])
    # XGBoost sensitivity curves kept alongside.
    save_roc_with_ci(oof_labels, oof_probs, fig_d / "roc_curve_xgb.png", model_name="XGBoost (OOF, nested)")
    save_calibration_curve(oof_labels, oof_probs, fig_d / "calibration_curve_xgb.png", model_name="XGBoost")
    save_shap(final, X, feature_cols, fig_d)

    try:
        save_umap(final, X, dc, fig_d)
    except Exception as e:
        log.warning("UMAP failed: %s", e)

    # ── Fairness evaluation (on the PRIMARY model's OOF predictions) ───────────
    log.info("Running subgroup fairness evaluation on the primary model …")
    fairness_reports = []

    sex_series = pd.Series(df["sex_female"].values).map({1: "Female", 0: "Male"})
    fairness_reports.append(
        evaluate_fairness_subgroups(oof_labels, lr_oof, sex_series, axis_name="sex")
    )

    age_bins  = pd.cut(df["age_at_index"], bins=[0, 44, 64, 74, 120],
                       labels=["18–44", "45–64", "65–74", "75+"])
    age_series = pd.Series(age_bins.values.astype(str))
    fairness_reports.append(
        evaluate_fairness_subgroups(oof_labels, lr_oof, age_series, axis_name="age_band")
    )

    dc_series = pd.Series(df["drug_class"].values if "drug_class" in df.columns else np.full(len(df), "unknown"))
    fairness_reports.append(
        evaluate_fairness_subgroups(oof_labels, lr_oof, dc_series, axis_name="drug_class")
    )

    save_fairness_report(fairness_reports, tbl_d / "fairness_report.csv")

    # ── E-values for Cox PH results ────────────────────────────────────────────
    cox_csv = tbl_d / "cox_ttd_results_r.csv"
    if cox_csv.exists():
        try:
            compute_e_values_for_cox_results(cox_csv, tbl_d / "cox_ttd_results_evalues.csv")
        except Exception as exc:
            log.warning("E-value computation failed: %s", exc)
    else:
        log.debug("cox_ttd_results_r.csv not found — E-value step skipped.")

    # ── MLflow logging ─────────────────────────────────────────────────────────
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run():
            _mlflow_log(xgb_report, comparison_df, feature_cols, final, mod_d,
                        lr_report=lr_report, primary_threshold=lr_thr["threshold"])

    log.info("Training complete — all outputs written to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost 1-year discontinuation predictor")
    parser.add_argument("--cohort",     default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--ttd-file",   default="outputs/tables/ttd_events.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_training(args.cohort, args.ttd_file, args.output_dir)


if __name__ == "__main__":
    main()
