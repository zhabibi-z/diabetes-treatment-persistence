"""
tests/conftest.py — Shared pytest fixtures for the T2DM pipeline test suite.

All fixtures produce synthetic DataFrames or minimal file artefacts sufficient
to exercise the pipeline logic without requiring real OMOP data or trained
model files. Fixtures are scoped to the session where construction is expensive
and to the function where state isolation is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ and src/ml/ available to all tests so imports resolve correctly
# after the source tree was moved into src/.
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "ml"))


# ── Reproducibility ───────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


# ── Cohort fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def matched_cohort() -> pd.DataFrame:
    """Minimal matched cohort DataFrame mirroring cohort_matched.csv schema.

    Three equal-sized drug-class arms (600 patients each) with realistic
    distributions of age, sex, and comorbidity flags. Patients are assigned
    follow-up days drawn from a truncated normal centred at 500 days so that
    the ≥365-day filter retains most but not all patients.
    """
    n = 1_800
    drug_classes = ["metformin"] * 600 + ["glp1"] * 600 + ["sglt2"] * 600

    df = pd.DataFrame({
        "person_id":           np.arange(1, n + 1),
        "drug_class":          drug_classes,
        "age_at_index":        RNG.integers(30, 80, n).astype(float),
        "sex_female":          RNG.integers(0, 2, n).astype(float),
        "cci":                 RNG.integers(0, 6, n).astype(float),
        "days_since_t2dm_dx":  RNG.integers(0, 3650, n).astype(float),
        "followup_days":       np.clip(RNG.normal(500, 150, n), 90, 1500).astype(float),
        # Comorbidity flags
        "hypertension":        RNG.integers(0, 2, n).astype(float),
        "obesity":             RNG.integers(0, 2, n).astype(float),
        "ckd":                 RNG.integers(0, 2, n).astype(float),
        "heart_failure":       RNG.integers(0, 2, n).astype(float),
        "hyperlipidemia":      RNG.integers(0, 2, n).astype(float),
        "nash":                RNG.integers(0, 2, n).astype(float),
        "neuropathy":          RNG.integers(0, 2, n).astype(float),
        "retinopathy":         RNG.integers(0, 2, n).astype(float),
        "depression":          RNG.integers(0, 2, n).astype(float),
        "atrial_fibrillation": RNG.integers(0, 2, n).astype(float),
        "sleep_apnea":         RNG.integers(0, 2, n).astype(float),
        "nafld":               RNG.integers(0, 2, n).astype(float),
        "pvd":                 RNG.integers(0, 2, n).astype(float),
        "stroke":              RNG.integers(0, 2, n).astype(float),
        "mi":                  RNG.integers(0, 2, n).astype(float),
    })
    df["age_over65"] = (df["age_at_index"] >= 65).astype(float)
    df["sex_male"] = 1.0 - df["sex_female"]
    df["drug_metformin"] = (df["drug_class"] == "metformin").astype(float)
    df["drug_glp1"]      = (df["drug_class"] == "glp1").astype(float)
    df["drug_sglt2"]     = (df["drug_class"] == "sglt2").astype(float)
    df["comorbidity_count"] = df[["hypertension","obesity","ckd","heart_failure",
                                   "hyperlipidemia","nash","neuropathy","retinopathy",
                                   "depression","atrial_fibrillation","sleep_apnea",
                                   "nafld","pvd","stroke","mi"]].sum(axis=1)
    return df


@pytest.fixture(scope="session")
def ttd_events(matched_cohort: pd.DataFrame) -> pd.DataFrame:
    """Minimal TTD events DataFrame aligned to matched_cohort person_ids.

    discontinuation probability is 0.45; event times drawn from
    Exponential(1/365) to give realistic right-skewed follow-up.
    """
    n = len(matched_cohort)
    discontinued = RNG.binomial(1, 0.45, n)
    ttd_days = np.where(
        discontinued,
        np.clip(RNG.exponential(365, n), 1, None),
        matched_cohort["followup_days"].values,
    ).astype(float)
    return pd.DataFrame({
        "person_id":    matched_cohort["person_id"].values,
        "ttd_days":     ttd_days,
        "discontinued": discontinued.astype(int),
        "drug_class":   matched_cohort["drug_class"].values,
    })


# ── Prediction fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Perfect-calibration binary classification output for calibration tests."""
    y_true = RNG.binomial(1, 0.3, 500)
    # Deliberately well-calibrated: draw scores from Beta shaped by true labels
    y_prob = np.where(
        y_true == 1,
        RNG.beta(3, 2, 500),
        RNG.beta(2, 5, 500),
    ).clip(0.01, 0.99)
    return y_true.astype(int), y_prob.astype(float)


@pytest.fixture(scope="session")
def single_class_predictions() -> tuple[np.ndarray, np.ndarray]:
    """All-negative labels — should trigger ValueError in AUROC computation."""
    return np.zeros(100, dtype=int), RNG.uniform(0, 1, 100)
