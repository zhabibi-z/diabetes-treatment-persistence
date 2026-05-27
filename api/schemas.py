"""
api/schemas.py — Pydantic request/response models for the T2DM Persistence API.

All field descriptions are written at the level of detail expected by a
clinical informaticist reviewing the API contract. Units and valid ranges
are explicit to prevent integration errors downstream.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PatientFeatures(BaseModel):
    """
    Feature vector for a single patient's 1-year discontinuation risk prediction.

    All fields correspond exactly to the 28-feature set used during XGBoost
    training (see ml/train.py). Any field absent from the request body is
    imputed with its training-set median/mode value.
    """

    # Demographics
    age_at_index: Annotated[float, Field(ge=18, le=120, description="Age in years at the cohort index date.")]
    sex_female: Annotated[int, Field(ge=0, le=1, description="1 = female, 0 = male (OMOP gender_concept_id mapping).")]

    # Drug class (mutually exclusive one-hot; exactly one must equal 1)
    drug_metformin: Annotated[int, Field(ge=0, le=1, description="1 if index drug is metformin.")]
    drug_glp1: Annotated[int, Field(ge=0, le=1, description="1 if index drug is a GLP-1 receptor agonist.")]
    drug_sglt2: Annotated[int, Field(ge=0, le=1, description="1 if index drug is an SGLT-2 inhibitor.")]

    # Comorbidity burden
    cci: Annotated[float, Field(ge=0, description="Charlson Comorbidity Index score at index date.")] = 0.0
    comorbidity_count: Annotated[int, Field(ge=0, le=15, description="Count of active comorbidities (max 15 tracked).")] = 0

    # Disease history
    days_since_t2dm_dx: Annotated[float, Field(ge=0, description="Days between first T2DM diagnosis and index date.")] = 365.0

    # Derived features
    age_over65: Annotated[int, Field(ge=0, le=1, description="1 if age_at_index >= 65.")] = 0

    # Interaction terms (drug class × comorbidity burden)
    glp1_x_codx: Annotated[float, Field(ge=0, description="drug_glp1 × comorbidity_count.")] = 0.0
    sglt2_x_codx: Annotated[float, Field(ge=0, description="drug_sglt2 × comorbidity_count.")] = 0.0
    glp1_x_cci: Annotated[float, Field(ge=0, description="drug_glp1 × cci.")] = 0.0
    sglt2_x_cci: Annotated[float, Field(ge=0, description="drug_sglt2 × cci.")] = 0.0

    # Comorbidity flags (SNOMED-mapped binary indicators)
    hypertension: Annotated[int, Field(ge=0, le=1)] = 0
    obesity: Annotated[int, Field(ge=0, le=1)] = 0
    ckd: Annotated[int, Field(ge=0, le=1)] = 0
    heart_failure: Annotated[int, Field(ge=0, le=1)] = 0
    hyperlipidemia: Annotated[int, Field(ge=0, le=1)] = 0
    nash: Annotated[int, Field(ge=0, le=1)] = 0
    neuropathy: Annotated[int, Field(ge=0, le=1)] = 0
    retinopathy: Annotated[int, Field(ge=0, le=1)] = 0
    depression: Annotated[int, Field(ge=0, le=1)] = 0
    atrial_fibrillation: Annotated[int, Field(ge=0, le=1)] = 0
    sleep_apnea: Annotated[int, Field(ge=0, le=1)] = 0
    nafld: Annotated[int, Field(ge=0, le=1)] = 0
    pvd: Annotated[int, Field(ge=0, le=1)] = 0
    stroke: Annotated[int, Field(ge=0, le=1)] = 0
    mi: Annotated[int, Field(ge=0, le=1)] = 0

    @model_validator(mode="after")
    def check_drug_class_exclusivity(self) -> "PatientFeatures":
        """Exactly one drug class flag must be set to 1."""
        total = self.drug_metformin + self.drug_glp1 + self.drug_sglt2
        if total != 1:
            raise ValueError(
                f"Exactly one of drug_metformin / drug_glp1 / drug_sglt2 must be 1 "
                f"(got sum={total}). These are mutually exclusive one-hot indicators."
            )
        return self

    @model_validator(mode="after")
    def align_age_flag(self) -> "PatientFeatures":
        """Ensure age_over65 is consistent with age_at_index."""
        expected = int(self.age_at_index >= 65)
        if self.age_over65 != expected:
            self.age_over65 = expected
        return self

    def to_feature_array(self) -> list[float]:
        """Return features in the exact column order expected by the XGBoost model."""
        return [
            self.age_at_index,      self.age_over65,          self.sex_female,
            self.drug_metformin,    self.drug_glp1,            self.drug_sglt2,
            self.cci,               self.comorbidity_count,    self.days_since_t2dm_dx,
            self.glp1_x_codx,       self.sglt2_x_codx,         self.glp1_x_cci,
            self.sglt2_x_cci,
            self.hypertension,      self.obesity,              self.ckd,
            self.heart_failure,     self.hyperlipidemia,       self.nash,
            self.neuropathy,        self.retinopathy,          self.depression,
            self.atrial_fibrillation, self.sleep_apnea,        self.nafld,
            self.pvd,               self.stroke,               self.mi,
        ]


class PredictionResponse(BaseModel):
    """Predicted 1-year discontinuation risk with clinical context."""
    discontinuation_probability: float = Field(
        description="Model-estimated probability of treatment discontinuation within 365 days.",
        ge=0.0, le=1.0,
    )
    risk_tier: Literal["low", "moderate", "high"] = Field(
        description="Clinical risk tier: low (<40%), moderate (40–60%), high (>60%)."
    )
    model_version: str = Field(description="Identifier for the XGBoost model version used.")
    top_risk_drivers: list[str] = Field(
        description="Feature names with the largest positive SHAP contributions for this patient.",
        default_factory=list,
    )
    calibration_note: str = Field(
        description="Contextual note on model calibration and population applicability.",
        default=(
            "This model was trained and validated on synthetic Synthea-generated data "
            "following OMOP CDM v5.4. Predicted probabilities should not be used for "
            "individual clinical decisions without validation on real-world data."
        ),
    )


class KMDataPoint(BaseModel):
    """Single time-point on a Kaplan-Meier persistence curve."""
    time_days: float
    persistence_probability: float
    ci_lower: float
    ci_upper: float
    n_at_risk: int


class SurvivalResponse(BaseModel):
    """Kaplan-Meier persistence curves for all three drug classes."""
    metformin: list[KMDataPoint]
    glp1: list[KMDataPoint]
    sglt2: list[KMDataPoint]
    log_rank_p_value: Optional[float] = None
    median_ttd_days: dict[str, Optional[float]] = Field(default_factory=dict)


class DrugGraphContext(BaseModel):
    """Knowledge graph context for a single drug class."""
    drug_class: str
    display_name: str
    mechanism_summary: str
    cardiorenal_benefits: list[str]
    associated_comorbidities: list[str]
    cox_hr_vs_metformin: Optional[float] = Field(
        description="Hazard ratio for discontinuation vs. metformin (from study Cox model).",
        default=None,
    )
    cox_hr_ci: Optional[tuple[float, float]] = None


class HealthResponse(BaseModel):
    """API health check response."""
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    km_data_available: bool
    version: str = "2.0.0"
    study: str = "T2DM Persistence RWE — Synthetic Cohort"
