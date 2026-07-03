# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
tests/test_api.py — Integration tests for the FastAPI inference service.

Tests exercise request validation, response schema correctness, and error
handling without requiring a trained model file. The model loading step is
patched to return a minimal stub that outputs fixed probabilities, isolating
the API contract from the ML training artefact.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Minimal model stub ────────────────────────────────────────────────────────

class _StubModel:
    """Minimal XGBoost-compatible model stub for API testing."""
    feature_names_in_ = None

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])

    def __call__(self, X):
        return self.predict_proba(X)


# ── Valid base payload ────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "age_at_index":        55.0,
    "sex_female":          1,
    "drug_metformin":      1,
    "drug_glp1":           0,
    "drug_sglt2":          0,
    "cci":                 2.0,
    "comorbidity_count":   3,
    "days_since_t2dm_dx":  730.0,
    "age_over65":          0,
    "glp1_x_codx":         0.0,
    "sglt2_x_codx":        0.0,
    "glp1_x_cci":          0.0,
    "sglt2_x_cci":         0.0,
    "hypertension":        1,
    "obesity":             1,
    "ckd":                 0,
    "heart_failure":       0,
    "hyperlipidemia":      1,
    "nash":                0,
    "neuropathy":          0,
    "retinopathy":         0,
    "depression":          0,
    "atrial_fibrillation": 0,
    "sleep_apnea":         0,
    "nafld":               0,
    "pvd":                 0,
    "stroke":              0,
    "mi":                  0,
}


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    TestClient with the AppState singleton pre-populated to bypass filesystem loading.

    api.main uses a module-level `state = AppState()` singleton; the lifespan
    context manager loads from disk on startup. In tests we patch the state
    attributes directly so no model file or CSV is required.
    """
    import pandas as pd
    import api.main as api_main

    stub_ttd = pd.DataFrame({
        "person_id": [1, 2, 3],
        "ttd_days": [365.0, 200.0, 500.0],
        "discontinued": [1, 0, 1],
        "drug_class": ["metformin", "glp1", "sglt2"],
    })
    stub_cox = pd.DataFrame({
        "term": ["drug_classGLP-1 RA", "drug_classSGLT-2i"],
        "estimate": [0.85, 0.80],
        "conf.low": [0.70, 0.65],
        "conf.high": [1.02, 0.97],
        "p.value": [0.08, 0.04],
    })

    # Patch the module-level state before constructing the TestClient.
    # TestClient does NOT run the lifespan by default unless configured, so
    # setting state attributes here ensures routes see the stub model.
    api_main.state.model  = _StubModel()
    api_main.state.ttd_df = stub_ttd
    api_main.state.cox_df = stub_cox

    yield TestClient(api_main.app)

    # Restore to None after tests so other test modules start clean
    api_main.state.model  = None
    api_main.state.ttd_df = None
    api_main.state.cox_df = None


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200

    def test_status_field_ok(self, client):
        r = client.get("/v1/health")
        data = r.json()
        assert "status" in data


# ── Prediction endpoint ───────────────────────────────────────────────────────

class TestPredictionEndpoint:

    def test_valid_payload_returns_200(self, client):
        r = client.post("/v1/predict/discontinuation", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_risk_score(self, client):
        r = client.post("/v1/predict/discontinuation", json=VALID_PAYLOAD)
        data = r.json()
        # Field name as defined in api/schemas.py PredictionResponse
        assert "discontinuation_probability" in data
        assert 0.0 <= data["discontinuation_probability"] <= 1.0

    def test_drug_class_sum_not_one_raises_422(self, client):
        bad = {**VALID_PAYLOAD, "drug_metformin": 1, "drug_glp1": 1, "drug_sglt2": 0}
        r = client.post("/v1/predict/discontinuation", json=bad)
        assert r.status_code == 422

    def test_all_drug_class_zero_raises_422(self, client):
        bad = {**VALID_PAYLOAD, "drug_metformin": 0, "drug_glp1": 0, "drug_sglt2": 0}
        r = client.post("/v1/predict/discontinuation", json=bad)
        assert r.status_code == 422

    def test_age_below_minimum_raises_422(self, client):
        bad = {**VALID_PAYLOAD, "age_at_index": 10.0}
        r = client.post("/v1/predict/discontinuation", json=bad)
        assert r.status_code == 422

    def test_feature_array_length_28(self, client):
        """PatientFeatures.to_feature_array() must return exactly 28 elements."""
        from api.schemas import PatientFeatures
        pf = PatientFeatures(**VALID_PAYLOAD)
        arr = pf.to_feature_array()
        assert len(arr) == 28, f"to_feature_array() returned {len(arr)} elements, expected 28"


# ── Graph endpoint ────────────────────────────────────────────────────────────

class TestGraphEndpoint:

    def test_valid_drug_class_returns_200(self, client):
        for drug in ("metformin", "glp1", "sglt2"):
            r = client.get(f"/v1/graph/drug/{drug}")
            assert r.status_code == 200, f"Expected 200 for drug={drug}, got {r.status_code}"

    def test_invalid_drug_class_returns_404(self, client):
        r = client.get("/v1/graph/drug/insulin")
        assert r.status_code == 404
