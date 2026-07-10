# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
ml/evaluation.py — Clinical prediction model evaluation suite.

Provides threshold-independent discrimination metrics, calibration diagnostics,
bootstrap confidence intervals, subgroup fairness reporting, and baseline model
benchmarking for the XGBoost 1-year discontinuation predictor.

Statistical foundations
-----------------------
Discrimination
    AUROC: area under the receiver operating characteristic curve.
        Hanley JA, McNeil BJ. Radiology. 1982;143(1):29–36.
    AUPRC: area under the precision-recall curve; preferred when positive
        events are rare relative to the cohort (Davis J, Goadrich M.
        Proc ICML. 2006:233–240).
    Bootstrap CI: non-parametric percentile method on 1,000 resamples.
        Efron B, Hastie T. Computer Age Statistical Inference. 2016:Ch. 11.

Calibration
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE):
        Naeini MP et al. Proc AAAI. 2015:2901–2907.
    In clinical risk prediction, miscalibrated probabilities directly harm
    patients via inappropriate risk stratification thresholds. Perfect
    calibration is defined as P(Y=1 | score=p) = p for all p in [0,1].
        Van Calster B et al. Stat Med. 2019;38(13):2290–2303.

Baseline anchoring
    A clinical ML model's complexity is justified only when it meaningfully
    exceeds both a majority-class classifier and a logistic regression on
    the same CV folds. Failure to report these anchors is a standard
    reason for desk-rejection in clinical informatics journals.
        Steyerberg EW et al. Ann Intern Med. 2010;152(8):491–495.

Fairness
    Per-subgroup AUROC and demographic parity difference surface
    disparate model performance across age, sex, and comorbidity burden.
        Hardt M et al. Equality of Opportunity in Supervised Learning.
        Proc NeurIPS. 2016.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ── Structured output types ───────────────────────────────────────────────────


@dataclass(frozen=True)
class DiscriminationMetrics:
    """Threshold-independent metrics quantifying ranking ability."""
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    auprc: float
    brier_score: float


@dataclass(frozen=True)
class CalibrationMetrics:
    """Agreement between predicted probabilities and observed event rates."""
    expected_calibration_error: float
    max_calibration_error: float
    mean_predicted_proba: float
    outcome_prevalence: float


@dataclass
class EvaluationReport:
    """Consolidated evaluation record for one model on one data split."""
    model_name: str
    split_name: str
    n_samples: int
    discrimination: DiscriminationMetrics
    calibration: CalibrationMetrics

    def to_series(self) -> pd.Series:
        return pd.Series({
            "model":          self.model_name,
            "split":          self.split_name,
            "n":              self.n_samples,
            "auroc":          round(self.discrimination.auroc, 4),
            "auroc_ci_low":   round(self.discrimination.auroc_ci_low, 4),
            "auroc_ci_high":  round(self.discrimination.auroc_ci_high, 4),
            "auprc":          round(self.discrimination.auprc, 4),
            "brier":          round(self.discrimination.brier_score, 4),
            "ece":            round(self.calibration.expected_calibration_error, 4),
            "mce":            round(self.calibration.max_calibration_error, 4),
            "mean_pred_prob": round(self.calibration.mean_predicted_proba, 4),
            "prevalence":     round(self.calibration.outcome_prevalence, 4),
        })


@dataclass
class FairnessReport:
    """Per-subgroup discrimination metrics and aggregate parity statistics."""
    subgroup_axis: str
    per_group_auc: dict[str, float]
    overall_auc: float
    max_auc_gap: float
    demographic_parity_diff: float
    flagged_groups: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "group": grp,
                "auc": round(auc, 4),
                "auc_gap": round(auc - self.overall_auc, 4),
                "flagged": grp in self.flagged_groups,
            }
            for grp, auc in self.per_group_auc.items()
        ]
        return pd.DataFrame(rows).sort_values("auc_gap")


# ── Core metric functions ─────────────────────────────────────────────────────


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Compute a non-parametric bootstrap confidence interval for AUROC.

    Uses the percentile method (Efron & Hastie 2016, Ch. 11). This approach
    is preferred over DeLong's parametric estimator when class imbalance is
    present or sample sizes are below 500, as it makes no distributional
    assumptions about the score.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted probabilities or scores for the positive class.
        n_bootstrap: Number of resamples. 1,000 is the minimum recommended
            for publication-quality intervals (Efron & Tibshirani 1994, Ch. 13).
        ci_level: Nominal coverage probability. Default 0.95.
        random_state: Seed for reproducibility.

    Returns:
        Tuple (lower_bound, upper_bound) for the CI at the requested level.

    References:
        Efron B, Hastie T. Computer Age Statistical Inference. 2016:Ch. 11.
        Hanley JA, McNeil BJ. Radiology. 1982;143(1):29–36.
    """
    rng = np.random.default_rng(random_state)
    boot_aucs = np.full(n_bootstrap, np.nan)

    for i in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs[i] = roc_auc_score(yt, ys)

    alpha = (1.0 - ci_level) / 2.0
    return (
        float(np.nanpercentile(boot_aucs, 100.0 * alpha)),
        float(np.nanpercentile(boot_aucs, 100.0 * (1.0 - alpha))),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE is the bin-size-weighted absolute difference between the mean predicted
    probability and the observed event rate in each bin. MCE identifies the
    worst-case bin — critical for clinical settings where a specific risk
    stratum (e.g., 60–70% predicted probability) may drive high-stakes decisions.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of equal-width probability bins.

    Returns:
        Tuple (ECE, MCE): both in [0, 1]; lower values indicate better calibration.

    References:
        Naeini MP et al. Proc AAAI. 2015:2901–2907.
        Van Calster B et al. Stat Med. 2019;38(13):2290–2303.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    mce = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)

    return ece, mce


def compute_discrimination(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
) -> DiscriminationMetrics:
    """Compute AUROC (with bootstrap CI), AUPRC, and Brier score."""
    auroc = roc_auc_score(y_true, y_prob)
    ci_low, ci_high = bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=n_bootstrap)
    return DiscriminationMetrics(
        auroc=auroc,
        auroc_ci_low=ci_low,
        auroc_ci_high=ci_high,
        auprc=average_precision_score(y_true, y_prob),
        brier_score=brier_score_loss(y_true, y_prob),
    )


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute ECE, MCE, mean predicted probability, and event prevalence."""
    ece, mce = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return CalibrationMetrics(
        expected_calibration_error=ece,
        max_calibration_error=mce,
        mean_predicted_proba=float(y_prob.mean()),
        outcome_prevalence=float(y_true.mean()),
    )


# ── Operating-threshold selection ─────────────────────────────────────────────


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute threshold-dependent classification metrics at a given cut-point."""
    preds = (y_prob >= threshold).astype(int)
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold":   round(float(threshold), 4),
        "accuracy":    round(accuracy_score(y_true, preds), 4),
        "precision":   round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":      round(recall_score(y_true, preds, zero_division=0), 4),
        "specificity": round(specificity, 4),
        "f1":          round(f1_score(y_true, preds, zero_division=0), 4),
    }


def select_operating_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> dict:
    """
    Choose a decision threshold instead of the naive 0.5, which is wrong whenever
    the event rate differs from 50% or the score is poorly separated.

    Methods
    -------
    youden      : maximise Youden's J = sensitivity + specificity − 1 (the point
                  farthest from the chance diagonal on the ROC). Balances
                  sensitivity and specificity; standard for clinical cut-points.
    prevalence  : set the threshold to the outcome prevalence, so the predicted
                  positive rate matches the base rate under a calibrated model.
    f1          : maximise F1 over a grid of candidate thresholds.

    A default 0.5 threshold on a ~42% event-rate, low-separation problem drives
    recall/F1 to ~0 as an artefact; this corrects that.

    References
    ----------
    Youden WJ. Cancer. 1950;3(1):32–35.
    Van Calster B et al. Stat Med. 2019;38(13):2290–2303.
    """
    if method == "prevalence":
        thr = float(y_true.mean())
    elif method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j = tpr - fpr
        thr = float(thresholds[int(np.argmax(j))])
        # roc_curve can emit an inf sentinel as the first threshold; guard it.
        if not np.isfinite(thr):
            thr = float(y_true.mean())
    elif method == "f1":
        grid = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
        f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
    else:
        raise ValueError(f"unknown threshold method {method!r}")

    result = threshold_metrics(y_true, y_prob, thr)
    result["method"] = method
    log.info(
        "Operating threshold (%s) = %.4f  →  recall=%.3f  precision=%.3f  specificity=%.3f  F1=%.3f",
        method, result["threshold"], result["recall"], result["precision"],
        result["specificity"], result["f1"],
    )
    return result


# ── Report builder ────────────────────────────────────────────────────────────


def build_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    split_name: str = "oof",
    n_bootstrap: int = 1000,
) -> EvaluationReport:
    """
    Assemble a complete evaluation report from raw labels and predicted probabilities.

    This is the primary entry point for post-training evaluation. It computes
    all discrimination and calibration metrics in a single call and logs a
    concise summary at INFO level.

    Args:
        y_true: Binary ground-truth array (0 = persistent, 1 = discontinued).
        y_prob: Predicted probability of 1-year discontinuation.
        model_name: Identifier for this model (e.g., "XGBoost", "Logistic Regression").
        split_name: Identifier for the data split context (e.g., "oof", "test").
        n_bootstrap: Bootstrap resamples for AUROC CI.

    Returns:
        Populated EvaluationReport dataclass.

    Raises:
        ValueError: If y_true is constant (single class). This indicates a
            malformed cohort or outcome definition that must be corrected
            before any model evaluation is meaningful.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            f"Split '{split_name}' for model '{model_name}' contains only one outcome class. "
            "Check cohort construction — the binary target may be constant."
        )

    log.info(
        "Evaluating  model=%-24s  split=%s  n=%d  event_rate=%.1f%%",
        f"'{model_name}'", split_name, len(y_true), 100.0 * y_true.mean(),
    )

    disc  = compute_discrimination(y_true, y_prob, n_bootstrap=n_bootstrap)
    calib = compute_calibration_metrics(y_true, y_prob)

    log.info(
        "  AUROC=%.3f (95%% CI %.3f–%.3f)  AUPRC=%.3f  Brier=%.4f  ECE=%.4f  MCE=%.4f",
        disc.auroc, disc.auroc_ci_low, disc.auroc_ci_high,
        disc.auprc, disc.brier_score,
        calib.expected_calibration_error, calib.max_calibration_error,
    )

    return EvaluationReport(
        model_name=model_name,
        split_name=split_name,
        n_samples=len(y_true),
        discrimination=disc,
        calibration=calib,
    )


# ── Baseline benchmarking ─────────────────────────────────────────────────────


def run_baseline_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Run 5-fold stratified cross-validation for naive baseline models.

    Evaluates a majority-class classifier and regularised logistic regression
    using the same fold partitions used by XGBoost, ensuring a valid apples-
    to-apples comparison. All reported XGBoost metrics must exceed the logistic
    regression baseline to justify model complexity (Steyerberg et al. 2010).

    Logistic regression uses z-scored features to prevent scale sensitivity
    from artificially disadvantaging it relative to XGBoost.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Binary outcome vector.
        n_splits: Number of CV folds. Must match the XGBoost CV configuration.
        random_state: Seed shared with XGBoost CV for identical fold assignments.

    Returns:
        Tuple of:
            - DataFrame with per-fold metrics for each baseline model.
            - Dict mapping model name to OOF predicted probabilities.

    References:
        Steyerberg EW et al. Ann Intern Med. 2010;152(8):491–495.
        Hosmer DW, Lemeshow S. Applied Logistic Regression. 2000:Ch. 5.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records: list[dict] = []

    baseline_specs: dict[str, object] = {
        "Majority Class": DummyClassifier(strategy="most_frequent"),
        "Stratified Random": DummyClassifier(strategy="stratified", random_state=random_state),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state
        ),
    }

    oof_probs: dict[str, np.ndarray] = {name: np.zeros(len(y)) for name in baseline_specs}

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X[tr])
        X_va_sc = scaler.transform(X[val])

        for name, spec in baseline_specs.items():
            clf = clone(spec)
            uses_scaling = "Logistic" in name

            if uses_scaling:
                clf.fit(X_tr_sc, y[tr])
                probs = clf.predict_proba(X_va_sc)[:, 1]
            else:
                clf.fit(X[tr], y[tr])
                probs = clf.predict_proba(X[val])[:, 1]

            oof_probs[name][val] = probs
            yt = y[val]

            if len(np.unique(yt)) < 2:
                log.warning("Fold %d: single class in validation — skipping %s", fold + 1, name)
                continue

            records.append({
                "model":  name,
                "fold":   fold + 1,
                "auc":    roc_auc_score(yt, probs),
                "brier":  brier_score_loss(yt, probs),
                "auprc":  average_precision_score(yt, probs),
            })

    return pd.DataFrame(records), oof_probs


def summarise_baseline_results(
    baseline_df: pd.DataFrame,
    baseline_oof: dict[str, np.ndarray],
    y_true: np.ndarray,
    xgb_report: EvaluationReport,
    xgb_label: str = "XGBoost (primary)",
    primary_baseline: str | None = None,
) -> pd.DataFrame:
    """
    Produce a side-by-side comparison table: XGBoost vs. all baseline models.

    Each model is summarised by its mean OOF AUROC (±SD), AUPRC, and Brier score.
    The XGBoost row is annotated with its bootstrap CI on AUROC.

    Args:
        baseline_df: Per-fold baseline metrics from run_baseline_cv().
        baseline_oof: OOF probability arrays keyed by model name.
        y_true: Full ground-truth label array.
        xgb_report: EvaluationReport for the XGBoost model.

    Returns:
        DataFrame sorted by AUROC descending, suitable for publication tables.
    """
    rows: list[dict] = []

    grp = baseline_df.groupby("model")
    for name in baseline_oof:
        if name not in grp.groups:
            continue
        sub = grp.get_group(name)
        label = f"{name} (primary)" if name == primary_baseline else name
        rows.append({
            "Model":          label,
            "Mean AUROC":     round(sub["auc"].mean(), 3),
            "SD AUROC":       round(sub["auc"].std(), 3),
            "95% CI":         "—",
            "Mean AUPRC":     round(sub["auprc"].mean(), 3),
            "Mean Brier":     round(sub["brier"].mean(), 4),
        })

    d = xgb_report.discrimination
    rows.append({
        "Model":          xgb_label,
        "Mean AUROC":     round(d.auroc, 3),
        "SD AUROC":       "—",
        "95% CI":         f"{d.auroc_ci_low:.3f}–{d.auroc_ci_high:.3f}",
        "Mean AUPRC":     round(d.auprc, 3),
        "Mean Brier":     round(d.brier_score, 4),
    })

    df = pd.DataFrame(rows)
    df["_sort"] = pd.to_numeric(df["Mean AUROC"], errors="coerce")
    df = df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return df


# ── Fairness evaluation ───────────────────────────────────────────────────────


def evaluate_fairness_subgroups(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_features: pd.Series,
    axis_name: str,
    min_group_size: int = 50,
    disparate_threshold: float = 0.05,
) -> FairnessReport:
    """
    Compute per-subgroup AUROC and demographic parity difference.

    Surfaces disparate model performance across a categorical sensitive
    attribute (e.g., sex, age band, comorbidity burden tier). Groups with
    fewer than min_group_size observations are excluded from discrimination
    metrics to avoid unreliable estimates from small cells.

    The demographic parity difference measures the maximum absolute gap in
    predicted positive rates across groups. A value exceeding 0.10 warrants
    further investigation under FDA guidance on AI/ML-based SaMD (2021).

    Args:
        y_true: Binary ground-truth array.
        y_prob: Predicted probabilities for the positive class.
        sensitive_features: Categorical series aligned with y_true.
        axis_name: Human-readable name for this attribute (e.g., "sex").
        min_group_size: Minimum group membership for AUC computation.
        disparate_threshold: AUC gap threshold above which a group is flagged.

    Returns:
        FairnessReport containing per-group AUC and parity statistics.

    References:
        Hardt M et al. Proc NeurIPS. 2016.
        FDA. Artificial Intelligence/Machine Learning Action Plan. 2021.
    """
    overall_auc = roc_auc_score(y_true, y_prob)
    per_group_auc: dict[str, float] = {}
    flagged: list[str] = []
    positive_rates: list[float] = []

    for grp in sorted(sensitive_features.unique()):
        mask = sensitive_features == grp
        if mask.sum() < min_group_size:
            log.debug("Skipping group '%s' (n=%d < min_group_size=%d)", grp, mask.sum(), min_group_size)
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        if len(np.unique(yt)) < 2:
            continue

        grp_auc = roc_auc_score(yt, yp)
        per_group_auc[str(grp)] = grp_auc
        positive_rates.append(float((yp >= 0.5).mean()))

        if abs(grp_auc - overall_auc) > disparate_threshold:
            flagged.append(str(grp))
            log.warning(
                "Fairness flag — axis='%s' group='%s': AUC=%.3f (gap=%.3f from overall=%.3f)",
                axis_name, grp, grp_auc, grp_auc - overall_auc, overall_auc,
            )

    dp_diff = (max(positive_rates) - min(positive_rates)) if len(positive_rates) >= 2 else 0.0
    max_gap  = max((abs(v - overall_auc) for v in per_group_auc.values()), default=0.0)

    return FairnessReport(
        subgroup_axis=axis_name,
        per_group_auc=per_group_auc,
        overall_auc=overall_auc,
        max_auc_gap=max_gap,
        demographic_parity_diff=dp_diff,
        flagged_groups=flagged,
    )


# ── Figure generation ─────────────────────────────────────────────────────────


def save_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    model_name: str = "XGBoost",
    n_bins: int = 10,
) -> None:
    """
    Save a reliability diagram (calibration curve) with ECE annotation.

    A perfectly calibrated model traces the diagonal. Deviation above the
    diagonal (under-confidence) and below (over-confidence) are both
    clinically meaningful: over-confidence leads to unnecessary interventions,
    while under-confidence can delay action on high-risk patients.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities.
        output_path: Destination file path (.png).
        model_name: Label for the curve in the legend.
        n_bins: Calibration bins for both the curve and ECE computation.

    References:
        Van Calster B et al. Stat Med. 2019;38(13):2290–2303.
        Steyerberg EW et al. Ann Intern Med. 2010;152(8):491–495.
    """
    ece, mce = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(
        y_true, y_prob, n_bins=n_bins, ax=ax, name=model_name,
    )
    ax.set_title(
        "Calibration Curve — 1-Year Discontinuation\n"
        "(a perfectly calibrated model traces the diagonal)",
        fontsize=11,
    )
    ax.annotate(
        f"ECE = {ece:.3f}   MCE = {mce:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=9, color="#555555",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F5F5F5", "edgecolor": "#CCCCCC"},
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Calibration curve saved → %s  (ECE=%.4f  MCE=%.4f)", output_path, ece, mce)


def save_roc_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    model_name: str = "XGBoost (OOF)",
) -> None:
    """
    Save an ROC curve figure with bootstrap AUROC CI in the legend.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        output_path: Destination file path (.png).
        model_name: Legend label.
    """
    auroc = roc_auc_score(y_true, y_prob)
    ci_low, ci_high = bootstrap_auroc_ci(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true, y_prob, ax=ax,
        name=f"{model_name}  (AUC = {auroc:.3f}; 95% CI {ci_low:.3f}–{ci_high:.3f})",
    )
    ax.set_title("ROC Curve — 1-Year Discontinuation", fontsize=11)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve saved → %s  (AUROC=%.3f  CI %.3f–%.3f)", output_path, auroc, ci_low, ci_high)


def e_value(hr: float, se_log_hr: Optional[float] = None) -> dict:
    """
    Compute the E-value for a hazard ratio and, optionally, its confidence limit.

    The E-value is the minimum strength of association that an unmeasured
    confounder would need to have with both the treatment and the outcome
    — on the risk ratio scale — to fully explain away an observed effect.
    A large E-value means the confounding required to nullify the result is
    implausibly strong; a small E-value indicates greater vulnerability.

    Formula (risk ratio scale, HR ≈ RR for rare outcomes):
        E = HR + sqrt(HR * (HR - 1))    for HR ≥ 1
        E = 1/HR + sqrt((1/HR) * (1/HR - 1))   for HR < 1

    For confidence intervals, apply the formula to the CI limit closest to
    the null (i.e., the lower limit if HR > 1, the upper limit if HR < 1)
    to obtain a conservative E-value for the confidence bound.

    Args:
        hr: Point-estimate hazard ratio. Must be > 0.
        se_log_hr: Standard error of log(HR). If provided, the E-value for
                   the 95% CI limit closest to the null is also returned.

    Returns:
        Dict with keys:
            hr          — original HR
            e_value     — E-value for the point estimate
            ci_limit    — CI limit closest to null (None if se_log_hr not given)
            e_value_ci  — E-value for the CI limit (None if se_log_hr not given)

    References
    ----------
    VanderWeele TJ, Ding P. Sensitivity analysis in observational research:
        introducing the E-value. Ann Intern Med. 2017;167(4):268–274.
    VanderWeele TJ et al. Ann Intern Med. 2019;171(4):297–298.  [correction]
    """
    if hr <= 0:
        raise ValueError(f"Hazard ratio must be positive, got {hr}")

    def _ev(rr: float) -> float:
        rr = max(rr, 1.0 / rr)   # work on the side ≥ 1
        return rr + np.sqrt(rr * (rr - 1))

    ev_point = _ev(hr)

    ci_limit:   Optional[float] = None
    ev_ci:      Optional[float] = None

    if se_log_hr is not None and se_log_hr > 0:
        log_hr = np.log(hr)
        # CI limit closest to null (least extreme)
        if hr >= 1.0:
            ci_limit = np.exp(log_hr - 1.96 * se_log_hr)   # lower CI
        else:
            ci_limit = np.exp(log_hr + 1.96 * se_log_hr)   # upper CI

        if ci_limit <= 1.0:
            # CI crosses the null — E-value for the CI limit is 1 (no confounding
            # needed because the CI already includes no effect)
            ev_ci = 1.0
        else:
            ev_ci = _ev(ci_limit)

    return {
        "hr":         round(hr, 4),
        "e_value":    round(ev_point, 4),
        "ci_limit":   round(ci_limit, 4) if ci_limit is not None else None,
        "e_value_ci": round(ev_ci, 4)    if ev_ci is not None else None,
    }


def compute_e_values_for_cox_results(cox_csv_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Compute E-values for all Cox PH hazard ratios in a results CSV.

    Reads the cox_ttd_results_r.csv produced by analysis/survival_analysis.R
    and appends E-value columns for each row where exp(coef) is available.

    Args:
        cox_csv_path: Path to the Cox results CSV (broom::tidy() format).
        output_path: Destination path for the annotated CSV.

    Returns:
        DataFrame with appended e_value and e_value_ci columns.
    """
    df = pd.read_csv(cox_csv_path)

    # Column-name compatibility: broom::tidy() uses 'estimate' for exp(coef)
    # when exponentiate=TRUE; the R script may also export 'exp(coef)' directly.
    hr_col = "estimate" if "estimate" in df.columns else "exp(coef)"
    se_col = None
    for candidate in ("std.error", "se(coef)", "se"):
        if candidate in df.columns:
            se_col = candidate
            break

    e_vals   = []
    e_ci_vals = []

    for _, row in df.iterrows():
        hr = row.get(hr_col)
        if pd.isna(hr) or hr <= 0:
            e_vals.append(None)
            e_ci_vals.append(None)
            continue

        se_log = None
        if se_col:
            raw_se = row.get(se_col)
            # If the SE column is on the log-HR scale (survival output convention)
            # it can be used directly; otherwise approximate from exp(coef)
            se_log = float(raw_se) if not pd.isna(raw_se) else None

        result    = e_value(float(hr), se_log)
        e_vals.append(result["e_value"])
        e_ci_vals.append(result["e_value_ci"])

    df["e_value"]    = e_vals
    df["e_value_ci"] = e_ci_vals

    df.to_csv(output_path, index=False)
    log.info("Cox results with E-values saved → %s", output_path)
    return df


def save_fairness_report(
    reports: list[FairnessReport],
    output_path: Path,
) -> None:
    """
    Save per-subgroup fairness metrics to CSV and log flagged groups.

    Args:
        reports: List of FairnessReport objects, one per sensitive axis.
        output_path: Destination CSV path.
    """
    frames: list[pd.DataFrame] = []
    for rpt in reports:
        df = rpt.to_dataframe()
        df.insert(0, "axis", rpt.subgroup_axis)
        df["demographic_parity_diff"] = round(rpt.demographic_parity_diff, 4)
        df["overall_auc"] = round(rpt.overall_auc, 4)
        frames.append(df)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    log.info("Fairness report saved → %s  (%d axes)", output_path, len(reports))

    flagged_total = sum(len(r.flagged_groups) for r in reports)
    if flagged_total:
        log.warning(
            "%d subgroup(s) flagged for AUC disparity >%.0f%% — review fairness report.",
            flagged_total, 0.05 * 100,
        )
