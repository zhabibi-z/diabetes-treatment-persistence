# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
api/main.py — FastAPI inference service for the T2DM Persistence RWE study.

Endpoints
---------
POST /v1/predict/discontinuation
    Accept a PatientFeatures payload and return a 1-year discontinuation risk.

GET  /v1/survival/km-data
    Return pre-computed Kaplan-Meier persistence curves for all drug classes.

GET  /v1/graph/drug/{drug_class}
    Return knowledge graph context (mechanisms, comorbidities, Cox HR) for one drug.

GET  /v1/health
    Liveness check confirming model and data availability.

Running locally
---------------
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

The service expects to be launched from the repository root so that relative
paths to outputs/ and data/ resolve correctly. The XGBoost model is loaded
once at startup via the lifespan context manager to avoid per-request I/O.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# config.py lives in src/; add src/ to path so all subpackages can import it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import (
    DrugGraphContext,
    HealthResponse,
    KMDataPoint,
    PatientFeatures,
    PredictionResponse,
    SurvivalResponse,
)
from config import ML, PATHS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH    = PATHS.xgb_model_pkl
COX_TTD_PATH  = PATHS.cox_results_r
TTD_PATH      = PATHS.ttd_events
MODEL_VERSION = "xgb-v2.0-28feat"

_FEATURE_NAMES: list[str] = list(ML.FEATURE_COLS)

_ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if o.strip()
]

DRUG_DISPLAY = {"metformin": "Metformin", "glp1": "GLP-1 RA", "sglt2": "SGLT-2i"}

DRUG_MECHANISMS = {
    "metformin": (
        "Hepatic gluconeogenesis inhibition via AMPK activation and Complex I inhibition. "
        "First-line agent per ADA 2024 Standards of Care for most T2DM patients."
    ),
    "glp1": (
        "GLP-1 receptor agonism promoting glucose-dependent insulin secretion, "
        "gastric emptying delay, and central appetite suppression. "
        "Demonstrated CV and renal benefits in LEADER, SUSTAIN-6, and REWIND trials."
    ),
    "sglt2": (
        "Sodium-glucose cotransporter-2 inhibition in the proximal nephron, "
        "reducing renal glucose reabsorption and delivering natriuresis. "
        "Demonstrated HF hospitalisation and CKD progression reduction in "
        "EMPA-REG OUTCOME, CANVAS, and DAPA-HF trials."
    ),
}

DRUG_BENEFITS = {
    "metformin": ["obesity", "hyperlipidemia"],
    "glp1":  ["heart_failure", "ckd", "stroke", "mi", "obesity"],
    "sglt2": ["heart_failure", "ckd", "pvd", "mi"],
}


# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    model: xgb.XGBClassifier | None = None
    ttd_df: pd.DataFrame | None = None
    cox_df: pd.DataFrame | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Load the XGBoost model and cached data tables at startup."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            state.model = pickle.load(f)
        log.info("XGBoost model loaded from %s", MODEL_PATH)
    else:
        log.warning("Model not found at %s — /predict will return 503.", MODEL_PATH)

    if TTD_PATH.exists():
        state.ttd_df = pd.read_csv(TTD_PATH)
        log.info("TTD events loaded: %d rows", len(state.ttd_df))

    if COX_TTD_PATH.exists():
        state.cox_df = pd.read_csv(COX_TTD_PATH)
        log.info("Cox results loaded")

    yield

    state.model  = None
    state.ttd_df = None
    state.cox_df = None


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="T2DM Persistence RWE — Inference API",
    description=(
        "REST API exposing the XGBoost 1-year discontinuation predictor, "
        "pre-computed Kaplan-Meier survival curves, and knowledge graph "
        "context for the T2DM Persistence RWE study."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/v1/health", response_model=HealthResponse, tags=["Infrastructure"])
def health() -> HealthResponse:
    """Liveness and readiness check."""
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        km_data_available=state.ttd_df is not None,
    )


@app.post("/v1/predict/discontinuation", response_model=PredictionResponse, tags=["Prediction"])
def predict_discontinuation(patient: PatientFeatures) -> PredictionResponse:
    """
    Return the 1-year treatment discontinuation risk for one patient.

    The feature vector is constructed from the request payload in the exact
    column order used during training (28 features, no followup_days).
    A SHAP-based driver list is included when the model supports it.

    Raises:
        503: If the XGBoost model has not been loaded (run ml/train.py first).
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "XGBoost model is not available. "
                "Run `python ml/train.py` to train and serialise the model."
            ),
        )

    features = np.array(patient.to_feature_array(), dtype=float).reshape(1, -1)
    proba    = float(state.model.predict_proba(features)[0, 1])

    risk_tier: str
    if proba < 0.40:
        risk_tier = "low"
    elif proba < 0.60:
        risk_tier = "moderate"
    else:
        risk_tier = "high"

    top_drivers = _get_top_shap_drivers(features)

    return PredictionResponse(
        discontinuation_probability=round(proba, 4),
        risk_tier=risk_tier,
        model_version=MODEL_VERSION,
        top_risk_drivers=top_drivers,
    )


@app.get("/v1/survival/km-data", response_model=SurvivalResponse, tags=["Survival"])
def get_km_data() -> SurvivalResponse:
    """
    Return pre-computed Kaplan-Meier persistence curves for all three drug classes.

    Curves are derived from the matched cohort TTD events file. Time points are
    sampled at 30-day intervals up to 730 days (2 years). Confidence intervals
    are Greenwood-based (lifelines default).

    Raises:
        503: If the TTD events file has not been generated.
    """
    if state.ttd_df is None:
        raise HTTPException(
            status_code=503,
            detail="TTD events data is not available. Run analysis/run_ttd.py first.",
        )

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        raise HTTPException(status_code=500, detail="lifelines package is required for KM computation.")

    result: dict[str, list[KMDataPoint]] = {}
    medians: dict[str, float | None] = {}
    drug_map = {"metformin": "metformin", "glp1": "glp1", "sglt2": "sglt2"}

    for key, dc in drug_map.items():
        sub = state.ttd_df[state.ttd_df["drug_class"] == dc].dropna(subset=["ttd_days", "discontinued"])
        if sub.empty:
            result[key] = []
            medians[key] = None
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(sub["ttd_days"], sub["discontinued"], label=dc)

        timeline = np.arange(0, min(sub["ttd_days"].max(), 730) + 30, 30)
        sf = kmf.survival_function_at_times(timeline)
        ci = kmf.confidence_interval_survival_function_at_times(timeline)

        points: list[KMDataPoint] = []
        for t, p, ci_lo, ci_hi in zip(
            timeline,
            sf.values,
            ci.iloc[:, 0].values,
            ci.iloc[:, 1].values,
        ):
            n_risk = int((sub["ttd_days"] >= t).sum())
            points.append(KMDataPoint(
                time_days=float(t),
                persistence_probability=round(float(p), 4),
                ci_lower=round(float(ci_lo), 4),
                ci_upper=round(float(ci_hi), 4),
                n_at_risk=n_risk,
            ))

        result[key] = points
        med = kmf.median_survival_time_
        medians[key] = None if np.isinf(med) else round(float(med), 1)

    log_rank_p = _compute_logrank_p(state.ttd_df)

    return SurvivalResponse(
        metformin=result.get("metformin", []),
        glp1=result.get("glp1", []),
        sglt2=result.get("sglt2", []),
        log_rank_p_value=log_rank_p,
        median_ttd_days=medians,
    )


@app.get("/v1/graph/drug/{drug_class}", response_model=DrugGraphContext, tags=["Knowledge Graph"])
def get_drug_context(drug_class: str) -> DrugGraphContext:
    """
    Return knowledge graph context for one drug class.

    Includes trial-evidenced cardiorenal benefits, study-derived comorbidity
    associations, and the empirical Cox HR from the matched cohort analysis.

    Args:
        drug_class: One of "metformin", "glp1", "sglt2".

    Raises:
        404: If drug_class is not a recognised study drug.
    """
    if drug_class not in DRUG_DISPLAY:
        raise HTTPException(
            status_code=404,
            detail=f"Drug class '{drug_class}' not recognised. Valid options: {list(DRUG_DISPLAY)}.",
        )

    cox_hr, cox_ci = _lookup_cox_hr(drug_class)
    comorbidities  = _get_associated_comorbidities(drug_class)

    return DrugGraphContext(
        drug_class=drug_class,
        display_name=DRUG_DISPLAY[drug_class],
        mechanism_summary=DRUG_MECHANISMS[drug_class],
        cardiorenal_benefits=DRUG_BENEFITS.get(drug_class, []),
        associated_comorbidities=comorbidities,
        cox_hr_vs_metformin=cox_hr,
        cox_hr_ci=cox_ci,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_top_shap_drivers(features: np.ndarray, top_n: int = 3) -> list[str]:
    """Return the names of the top_n SHAP contributors for this patient."""
    try:
        import shap
        explainer = shap.TreeExplainer(state.model)
        sv = explainer.shap_values(features)[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        return [_FEATURE_NAMES[i] for i in top_idx if i < len(_FEATURE_NAMES)]
    except Exception:
        return []


def _lookup_cox_hr(drug_class: str) -> tuple[float | None, tuple[float, float] | None]:
    """Look up Cox HR from the pre-computed results CSV."""
    if state.cox_df is None or drug_class == "metformin":
        return None, None
    try:
        col_cov = state.cox_df.columns[0]
        row = state.cox_df[state.cox_df[col_cov].str.contains(drug_class, case=False, na=False)]
        if row.empty:
            return None, None
        hr  = float(row["exp(coef)"].iloc[0])
        lo  = float(row["exp(coef) lower 95%"].iloc[0])
        hi  = float(row["exp(coef) upper 95%"].iloc[0])
        return round(hr, 3), (round(lo, 3), round(hi, 3))
    except Exception:
        return None, None


def _get_associated_comorbidities(drug_class: str) -> list[str]:
    """Return comorbidities significantly associated with discontinuation for this drug."""
    try:
        corr = pd.read_csv("outputs/tables/correlations.csv")
        top = (
            corr[corr["pearson_r"].abs() > 0.01]
            .sort_values("pearson_r", key=abs, ascending=False)
            .head(5)["comorbidity"]
            .tolist()
        )
        return top
    except Exception:
        return []


def _compute_logrank_p(ttd_df: pd.DataFrame) -> float | None:
    """Compute the multivariate log-rank p-value across all drug classes."""
    try:
        from lifelines.statistics import multivariate_logrank_test
        clean = ttd_df.dropna(subset=["ttd_days", "discontinued", "drug_class"])
        if clean["drug_class"].nunique() < 2:
            return None
        result = multivariate_logrank_test(
            clean["ttd_days"], clean["drug_class"], clean["discontinued"]
        )
        return round(float(result.p_value), 6)
    except Exception:
        return None
