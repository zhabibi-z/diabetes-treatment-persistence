"""
tests/test_feature_engineering.py — Unit tests for ml/train.py::build_features().

These tests verify the integrity of the feature engineering pipeline, with
emphasis on the data-leakage guards introduced in v2. Every assertion here
corresponds to a methodological requirement documented in ml/train.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Resolve import path so tests can run from the repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))
from train import build_features, MIN_FOLLOWUP_DAYS, COMORBIDITY_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

DRUG_CLASSES = ["metformin", "glp1", "sglt2"]


def _make_cohort(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    drug_cycle = (DRUG_CLASSES * ((n // 3) + 1))[:n]
    return pd.DataFrame({
        "person_id":           np.arange(1, n + 1),
        "drug_class":          drug_cycle,
        "age_at_index":        rng.integers(30, 80, n).astype(float),
        "gender_concept_id":   rng.choice([8532, 8507], n),
        "cci":                 rng.integers(0, 5, n).astype(float),
        "index_date":          "2020-01-01",
        "t2dm_date":           "2018-06-01",
        "followup_days":       rng.choice([200, 400, 600, 800], n).astype(float),
        **{c: rng.integers(0, 2, n).astype(float) for c in COMORBIDITY_COLS},
    })


def _make_ttd(cohort: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(cohort)
    return pd.DataFrame({
        "person_id":    cohort["person_id"].values,
        "ttd_days":     rng.integers(90, 730, n).astype(float),
        "discontinued": rng.integers(0, 2, n),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFollowupFilter:
    """Patients below MIN_FOLLOWUP_DAYS must be excluded before feature creation."""

    def test_short_followup_excluded(self):
        cohort = _make_cohort(300)
        ttd = _make_ttd(cohort)
        # Force half the cohort below the threshold
        cohort.loc[:149, "followup_days"] = 180.0
        cohort.loc[150:, "followup_days"] = 500.0

        df, _ = build_features(cohort, ttd)
        assert (df["followup_days"] >= MIN_FOLLOWUP_DAYS).all(), (
            "build_features() returned rows with followup_days < MIN_FOLLOWUP_DAYS"
        )

    def test_all_short_returns_empty(self):
        cohort = _make_cohort(100)
        cohort["followup_days"] = 100.0
        ttd = _make_ttd(cohort)
        df, _ = build_features(cohort, ttd)
        assert len(df) == 0

    def test_threshold_boundary(self):
        """Patients with exactly MIN_FOLLOWUP_DAYS must be retained."""
        cohort = _make_cohort(30)
        cohort["followup_days"] = float(MIN_FOLLOWUP_DAYS)
        ttd = _make_ttd(cohort)
        df, _ = build_features(cohort, ttd)
        assert len(df) == 30


class TestFeatureCount:
    """The feature matrix must have exactly 28 columns."""

    def test_exactly_28_features(self):
        cohort = _make_cohort(300)
        ttd = _make_ttd(cohort)
        _, feature_cols = build_features(cohort, ttd)
        assert len(feature_cols) == 28, (
            f"Expected 28 features, got {len(feature_cols)}: {feature_cols}"
        )


class TestLeakageGuards:
    """Leakage variables must not appear in the returned feature list."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        cohort = _make_cohort(300)
        cohort["followup_days"] = 500.0
        ttd = _make_ttd(cohort)
        _, self.feature_cols = build_features(cohort, ttd)

    def test_followup_days_excluded(self):
        assert "followup_days" not in self.feature_cols, (
            "followup_days encodes observation window length and must not be a feature"
        )

    def test_drug_class_num_excluded(self):
        assert "drug_class_num" not in self.feature_cols, (
            "drug_class_num is an ordinal duplicate of the one-hot triplet and must be excluded"
        )

    def test_sex_male_excluded(self):
        assert "sex_male" not in self.feature_cols, (
            "sex_male is perfectly collinear with sex_female and must be excluded"
        )


class TestOutcomeLabel:
    """Binary outcome y must be defined correctly relative to the TTD event."""

    def test_y_is_binary(self):
        cohort = _make_cohort(300)
        cohort["followup_days"] = 500.0
        ttd = _make_ttd(cohort)
        df, _ = build_features(cohort, ttd)
        assert set(df["y"].unique()).issubset({0, 1})

    def test_y_positive_requires_event_within_365(self):
        """y=1 requires discontinued==1 AND ttd_days≤365."""
        cohort = _make_cohort(60)
        cohort["followup_days"] = 500.0
        ttd = _make_ttd(cohort)
        # Force first 20 patients: event after 365 days → should be y=0
        ttd.loc[:19, "ttd_days"] = 400.0
        ttd.loc[:19, "discontinued"] = 1
        df, _ = build_features(cohort, ttd)
        late_events = df[df["person_id"].isin(range(1, 21))]
        assert (late_events["y"] == 0).all(), (
            "Discontinuation after 365 days must not be classified as y=1"
        )


class TestOneHotDrugClass:
    """Drug-class one-hot columns must be mutually exclusive and exhaustive."""

    def test_onehot_sums_to_one(self):
        cohort = _make_cohort(300)
        cohort["followup_days"] = 500.0
        ttd = _make_ttd(cohort)
        df, _ = build_features(cohort, ttd)
        total = df["drug_metformin"] + df["drug_glp1"] + df["drug_sglt2"]
        assert (total == 1).all(), "Each patient must belong to exactly one drug class"
