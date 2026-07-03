# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
config.py — Study-wide configuration for the T2DM Persistence RWE pipeline.

All scripts import from this module rather than hardcoding paths, constants,
or hyperparameters. Environment variables override path defaults, which allows
the pipeline to run in containerised or HPC environments without source edits.

Design principle: constants are grouped by concern (paths, cohort parameters,
ML hyperparameters). Adding a new parameter here should be the only change
needed to propagate it across the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()


# ── Path configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Paths:
    """Filesystem layout for data, outputs, and model artefacts.

    All paths are absolute. Override OMOP_DB_PATH via the environment variable
    of the same name (useful for containerised deployments or HPC scratch dirs).
    """
    root:            Path = ROOT
    data:            Path = ROOT / "data"
    synthea_output:  Path = ROOT / "data" / "synthea_output"
    omop_db:         Path = field(default_factory=lambda: Path(
                         os.environ.get("OMOP_DB_PATH", str(ROOT / "data" / "omop" / "omop.duckdb"))
                     ))
    outputs:         Path = ROOT / "outputs"
    tables:          Path = ROOT / "outputs" / "tables"
    figures:         Path = ROOT / "outputs" / "figures"
    models:          Path = ROOT / "outputs" / "models"
    logs:            Path = ROOT / "logs"
    graph_export:    Path = ROOT / "src" / "graph" / "cypher_export"

    # Derived outputs — cohort
    cohort_baseline: Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "cohort_baseline.csv")
    cohort_matched:  Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "cohort_matched.csv")

    # Derived outputs — analysis
    ttd_events:      Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "ttd_events.csv")
    cox_results_r:   Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "cox_ttd_results_r.csv")
    ml_metrics:      Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "ml_metrics.csv")
    model_comparison:Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "model_comparison.csv")
    fairness_report: Path = field(default_factory=lambda: ROOT / "outputs" / "tables" / "fairness_report.csv")
    xgb_model_pkl:   Path = field(default_factory=lambda: ROOT / "outputs" / "models" / "xgb_discontinuation.pkl")

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in (self.tables, self.figures, self.models, self.logs, self.graph_export):
            d.mkdir(parents=True, exist_ok=True)


# ── Cohort definition parameters ─────────────────────────────────────────────

@dataclass(frozen=True)
class CohortParams:
    """Parameters governing cohort construction and pharmacoepidemiology design.

    References
    ----------
    New-user, active-comparator design:
        Schneeweiss S et al. Epidemiology. 2007;18(2):161–168.
    Grace-period definition for discontinuation:
        Lim SH et al. Pharmacoepidemiol Drug Saf. 2025 (in press).
    Washout (new-user criterion):
        Ray WA. Pharmacoepidemiol Drug Saf. 2003;12(6):539–545.
    """
    # Treatment episodes
    grace_period_days:   int = 90    # allowable gap before calling discontinuation
    washout_days:        int = 365   # look-back period defining new-user status
    min_followup_days:   int = 90    # minimum observation after index date for cohort inclusion

    # Synthea generation
    default_patient_count: int = 30_000
    default_seed:          int = 42

    # Drug classes (keep in sync with OMOP concept sets in cohort/build_cohort.py)
    drug_classes: tuple[str, ...] = ("metformin", "glp1", "sglt2")


# ── Machine-learning parameters ───────────────────────────────────────────────

@dataclass(frozen=True)
class MLParams:
    """Hyperparameters and evaluation settings for the XGBoost predictor.

    XGB_PARAMS are intentionally conservative (shallow trees, strong
    regularisation) to mitigate overfitting on a synthetic cohort and to
    produce calibrated probabilities without post-hoc isotonic regression.

    References
    ----------
    Early stopping and CV alignment:
        Chen T, Guestrin C. KDD. 2016:785–794.
    Bootstrap AUROC CI:
        Efron B, Hastie T. Computer Age Statistical Inference. 2016:Ch. 11.
    Calibration assessment:
        Van Calster B et al. Stat Med. 2019;38(13):2290–2303.
    """
    # Follow-up filter — must match CohortParams for the ML binary label to be valid
    min_followup_days: int = 365

    # Cross-validation
    n_cv_folds:    int = 5
    random_state:  int = 42

    # Bootstrap CI resamples
    n_bootstrap_ci: int = 1_000

    # XGBoost hyperparameters
    XGB_PARAMS: dict = field(default_factory=lambda: {
        "n_estimators":      500,
        "max_depth":         4,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  10,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "scale_pos_weight":  1,    # adjusted dynamically in train.py if class imbalance >3:1
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      42,
    })

    # Feature columns — in exact order expected by the trained model
    # Any change here must be mirrored in ml/train.py::build_features()
    COMORBIDITY_COLS: tuple[str, ...] = (
        "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
        "nash", "neuropathy", "retinopathy", "depression",
        "atrial_fibrillation", "sleep_apnea", "nafld", "pvd", "stroke", "mi",
    )

    FEATURE_COLS: tuple[str, ...] = field(default_factory=lambda: (
        "age_at_index", "age_over65", "sex_female",
        "drug_metformin", "drug_glp1", "drug_sglt2",
        "cci", "comorbidity_count", "days_since_t2dm_dx",
        "glp1_x_codx", "sglt2_x_codx", "glp1_x_cci", "sglt2_x_cci",
        "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
        "nash", "neuropathy", "retinopathy", "depression",
        "atrial_fibrillation", "sleep_apnea", "nafld", "pvd", "stroke", "mi",
    ))


# ── MLflow tracking ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MLflowConfig:
    """MLflow experiment tracking settings.

    Set MLFLOW_TRACKING_URI to a remote tracking server URI (e.g., a Databricks
    workspace or self-hosted MLflow server) to persist runs across environments.
    Defaults to a local ./mlruns directory.
    """
    tracking_uri:    str = os.environ.get("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
    experiment_name: str = "t2dm-discontinuation-predictor"
    model_artifact:  str = "xgb_discontinuation"


# ── API configuration ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class APIConfig:
    """FastAPI inference service configuration."""
    host:    str = os.environ.get("API_HOST", "0.0.0.0")
    port:    int = int(os.environ.get("API_PORT", "8000"))
    workers: int = int(os.environ.get("API_WORKERS", "1"))
    title:   str = "T2DM Persistence — Discontinuation Risk API"
    version: str = "1.0.0"


# ── Singleton accessors ───────────────────────────────────────────────────────

PATHS   = Paths()
COHORT  = CohortParams()
ML      = MLParams()
MLFLOW  = MLflowConfig()
API_CFG = APIConfig()

__all__ = [
    "PATHS", "COHORT", "ML", "MLFLOW", "API_CFG",
    "Paths", "CohortParams", "MLParams", "MLflowConfig", "APIConfig",
    "ROOT",
]
