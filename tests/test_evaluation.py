# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
tests/test_evaluation.py — Unit tests for ml/evaluation.py.

Tests cover the core statistical functions: bootstrap CI coverage, calibration
error computation, fairness subgroup reporting, and baseline model benchmarking.
Each test is anchored to the methodological reference in evaluation.py's docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation import (
    bootstrap_auroc_ci,
    expected_calibration_error,
    run_baseline_cv,
    evaluate_fairness_subgroups,
    build_evaluation_report,
)


# ── bootstrap_auroc_ci ────────────────────────────────────────────────────────

class TestBootstrapAUROC:
    """bootstrap_auroc_ci returns (lo, hi) — a 2-tuple of CI bounds."""

    def test_ci_width_positive(self, binary_predictions):
        y_true, y_prob = binary_predictions
        lo, hi = bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=200)
        assert hi > lo, "CI width must be positive"

    def test_ci_lower_bound_positive(self, binary_predictions):
        y_true, y_prob = binary_predictions
        lo, hi = bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=200)
        assert lo >= 0.0

    def test_ci_upper_bound_at_most_one(self, binary_predictions):
        y_true, y_prob = binary_predictions
        lo, hi = bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=200)
        assert hi <= 1.0

    def test_perfect_classifier_ci_above_threshold(self):
        y_true = np.array([0] * 500 + [1] * 500)
        y_prob = np.where(y_true == 1, 0.95, 0.05).astype(float)
        lo, hi = bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=200)
        assert lo > 0.90, f"Lower CI for near-perfect classifier should be >0.90, got {lo:.4f}"


# ── expected_calibration_error ────────────────────────────────────────────────

class TestCalibrationError:

    def test_ece_perfect_calibration_near_zero(self):
        """A perfectly calibrated model has ECE ≈ 0."""
        rng = np.random.default_rng(7)
        n = 2_000
        probs = rng.uniform(0, 1, n)
        y_true = rng.binomial(1, probs)
        ece, mce = expected_calibration_error(y_true, probs, n_bins=10)
        assert ece < 0.05, f"ECE should be near 0 for a well-calibrated model, got {ece:.4f}"

    def test_ece_miscalibrated_model(self):
        """Uniform high predictions (0.9) on a low-prevalence outcome → high ECE.

        All 1,000 patients are predicted at 0.9 but only 10% actually have the
        event. Every bin's predicted mean is 0.9 while the observed rate is 0.1,
        yielding ECE = 0.8.
        """
        rng = np.random.default_rng(0)
        n = 1_000
        y_true = rng.binomial(1, 0.10, n)       # 10% prevalence
        y_prob  = np.full(n, 0.9, dtype=float)  # predict 90% for everyone
        ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece > 0.50, f"Uniformly-overconfident model must yield high ECE, got {ece:.4f}"

    def test_mce_geq_ece(self, binary_predictions):
        y_true, y_prob = binary_predictions
        ece, mce = expected_calibration_error(y_true, y_prob)
        assert mce >= ece - 1e-9, "Maximum calibration error must be ≥ ECE"

    def test_returns_floats(self, binary_predictions):
        y_true, y_prob = binary_predictions
        ece, mce = expected_calibration_error(y_true, y_prob)
        assert isinstance(ece, float)
        assert isinstance(mce, float)


# ── run_baseline_cv ────────────────────────────────────────────────────────────

class TestBaselineCV:

    @pytest.fixture(scope="class")
    def baseline_output(self, binary_predictions):
        y_true, y_prob = binary_predictions
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (len(y_true), 5))
        baseline_df, baseline_oof = run_baseline_cv(X, y_true, n_splits=3)
        return baseline_df, baseline_oof, y_true

    def test_returns_exactly_three_model_names(self, baseline_output):
        """run_baseline_cv must produce rows for exactly three model types."""
        baseline_df, _, _ = baseline_output
        model_names = baseline_df["model"].unique()
        assert len(model_names) == 3, (
            f"Expected 3 distinct model names, got {len(model_names)}: {model_names}"
        )

    def test_logistic_regression_present(self, baseline_output):
        baseline_df, _, _ = baseline_output
        assert any("Logistic" in m for m in baseline_df["model"].unique()), (
            "Logistic Regression must be one of the baseline models"
        )

    def test_majority_class_present(self, baseline_output):
        baseline_df, _, _ = baseline_output
        assert any("Majority" in m for m in baseline_df["model"].unique()), (
            "Majority Class must be one of the baseline models"
        )

    def test_oof_dict_length_matches_input(self, baseline_output, binary_predictions):
        """OOF probabilities dict must contain arrays of length n_samples."""
        _, baseline_oof, y_true = baseline_output
        assert isinstance(baseline_oof, dict), "oof_probs should be a dict"
        for name, arr in baseline_oof.items():
            assert len(arr) == len(y_true), (
                f"OOF array for '{name}' has length {len(arr)}, expected {len(y_true)}"
            )


# ── evaluate_fairness_subgroups ────────────────────────────────────────────────

class TestFairnessSubgroups:

    def test_reports_all_subgroups(self, binary_predictions):
        y_true, y_prob = binary_predictions
        rng = np.random.default_rng(5)
        sensitive = pd.Series(rng.choice(["F", "M"], len(y_true)))
        report = evaluate_fairness_subgroups(y_true, y_prob, sensitive, axis_name="sex")
        subgroups = list(report.per_group_auc.keys())
        assert "F" in subgroups and "M" in subgroups

    def test_auroc_in_valid_range(self, binary_predictions):
        y_true, y_prob = binary_predictions
        rng = np.random.default_rng(6)
        sensitive = pd.Series(rng.choice(["18-44", "45-64", "65+"], len(y_true)))
        report = evaluate_fairness_subgroups(y_true, y_prob, sensitive, axis_name="age_band")
        for grp, auc in report.per_group_auc.items():
            assert 0.0 <= auc <= 1.0, f"AUROC out of range for subgroup {grp}: {auc}"

    def test_overall_auroc_matches(self, binary_predictions):
        from sklearn.metrics import roc_auc_score
        y_true, y_prob = binary_predictions
        rng = np.random.default_rng(7)
        sensitive = pd.Series(rng.choice(["A", "B"], len(y_true)))
        report = evaluate_fairness_subgroups(y_true, y_prob, sensitive, axis_name="group")
        expected = roc_auc_score(y_true, y_prob)
        assert abs(report.overall_auc - expected) < 1e-6


# ── build_evaluation_report ────────────────────────────────────────────────────

class TestBuildEvaluationReport:

    def test_report_has_required_fields(self, binary_predictions):
        y_true, y_prob = binary_predictions
        report = build_evaluation_report(y_true, y_prob, "xgboost", "test")
        assert report.model_name == "xgboost"
        assert report.split_name == "test"
        assert report.n_samples == len(y_true)
        assert 0.0 < report.discrimination.auroc < 1.0

    def test_to_series_has_expected_keys(self, binary_predictions):
        y_true, y_prob = binary_predictions
        report = build_evaluation_report(y_true, y_prob, "xgboost", "oof")
        s = report.to_series()
        for key in ("auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "brier", "ece", "mce"):
            assert key in s.index, f"Missing key '{key}' in EvaluationReport.to_series()"
