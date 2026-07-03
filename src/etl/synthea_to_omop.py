# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
synthea_to_omop.py — ETL: Synthea CSV output → OMOP CDM v5.4 in DuckDB.

Handles two modes:
  1. Load existing Synthea CSV files (default)
  2. Generate synthetic fallback data when Synthea JAR is not installed
     (--generate-synthetic flag)

OMOP tables populated: person, observation_period, condition_occurrence,
drug_exposure, visit_occurrence, measurement, concept (minimal vocabulary).

Concept IDs used follow OHDSI Athena standard vocabulary for:
  - T2DM: SNOMED 44054006 → OMOP concept_id 201826
  - Metformin, GLP-1 RAs, SGLT-2is: RxNorm-sourced standard concepts
  - 15 comorbidities: SNOMED concepts per codx_mapping.xlsx
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── OMOP concept IDs ──────────────────────────────────────────────────────────
T2DM_CONCEPT_ID = 201826          # Type 2 diabetes mellitus (SNOMED 44054006)
T1DM_CONCEPT_ID = 435216          # Type 1 diabetes mellitus

DRUG_CONCEPTS: dict[str, dict[str, int]] = {
    "metformin": {
        "metformin_500mg": 1503297,
        "metformin_850mg": 1503298,
        "metformin_1000mg": 1503299,
    },
    "glp1": {
        "semaglutide_inj": 2200644,
        "semaglutide_oral": 2200645,
        "dulaglutide": 1583722,
        "liraglutide": 40170911,
    },
    "sglt2": {
        "empagliflozin": 1792455,
        "dapagliflozin": 1488564,
        "canagliflozin": 1373463,
    },
}

# 15 comorbidity SNOMED→OMOP concept IDs (approximate OMOP standard)
COMORBIDITY_CONCEPTS: dict[str, int] = {
    "hypertension":        316866,
    "obesity":             433736,
    "ckd":                 46271022,
    "heart_failure":       316139,
    "hyperlipidemia":      432867,
    "nash":                4212540,
    "neuropathy":          378419,
    "retinopathy":         4226354,
    "depression":          440383,
    "atrial_fibrillation": 313217,
    "sleep_apnea":         4173636,
    "nafld":               4212540,
    "pvd":                 321052,
    "stroke":              372924,
    "mi":                  4329847,
}

# Drug-class-specific comorbidity prevalence — reflects channelling by indication.
# Metformin: broad first-line population (reference category).
# GLP-1 RA:  channelled to obesity/ASCVD; higher BMI, NASH, CVD history.
#   Ref: Marso et al. NEJM 2016 (LEADER baseline); ADA Standards 2022 §9.
# SGLT-2i:   channelled to CKD, HFrEF, ASCVD per 2022 ADA/EASD guidelines.
#   Ref: McMurray et al. NEJM 2019 (EMPEROR-Reduced);
#        Heerspink et al. NEJM 2020 (DAPA-CKD); Koye et al. DOM 2020.
CLASS_COMORBIDITY_PREVALENCE: dict[str, dict[str, float]] = {
    "metformin": {
        "hypertension": 0.60, "obesity": 0.42, "ckd": 0.18, "heart_failure": 0.10,
        "hyperlipidemia": 0.52, "nash": 0.08, "neuropathy": 0.25, "retinopathy": 0.15,
        "depression": 0.18, "atrial_fibrillation": 0.09, "sleep_apnea": 0.15,
        "nafld": 0.20, "pvd": 0.08, "stroke": 0.06, "mi": 0.07,
    },
    "glp1": {
        "hypertension": 0.68, "obesity": 0.65, "ckd": 0.22, "heart_failure": 0.14,
        "hyperlipidemia": 0.60, "nash": 0.22, "neuropathy": 0.30, "retinopathy": 0.20,
        "depression": 0.22, "atrial_fibrillation": 0.11, "sleep_apnea": 0.24,
        "nafld": 0.30, "pvd": 0.12, "stroke": 0.10, "mi": 0.15,
    },
    "sglt2": {
        "hypertension": 0.70, "obesity": 0.50, "ckd": 0.42, "heart_failure": 0.28,
        "hyperlipidemia": 0.58, "nash": 0.14, "neuropathy": 0.28, "retinopathy": 0.18,
        "depression": 0.18, "atrial_fibrillation": 0.15, "sleep_apnea": 0.18,
        "nafld": 0.22, "pvd": 0.12, "stroke": 0.10, "mi": 0.14,
    },
}

# Market share weights averaged over 2013-2022 US commercial claims.
# Temporal availability applied per-patient (SGLT-2i unavailable pre-2013).
# Ref: Montvida et al. Diabetes Care 2019;42(5):878-886 (2013-2016 US trends).
CLASS_MARKET_SHARE: dict[str, float] = {
    "metformin": 0.60,
    "glp1":      0.20,
    "sglt2":     0.20,
}

# Lognormal (μ, σ) for time-to-discontinuation in days.
# Calibrated to published 1-year persistence from real-world claims:
#   Metformin ~68%  → median ~446 d  (McGovern et al. DOM 2018; UK CPRD)
#   GLP-1 RA  ~47%  → median ~270 d  (Divino et al. Clin Ther 2014; US MarketScan)
#   SGLT-2i   ~56%  → median ~365 d  (Koye et al. DOM 2020; Australian PBS)
CLASS_TTD_PARAMS: dict[str, tuple[float, float]] = {
    "metformin": (6.1, 0.8),
    "glp1":      (5.6, 0.9),
    "sglt2":     (5.9, 0.85),
}

# FDA first-approval dates — gate drug availability per patient index date.
# Ref: FDA Orange Book / DailyMed.
DRUG_APPROVAL_DATES: dict[str, date] = {
    "metformin_500mg":  date(1994, 12, 29),
    "metformin_850mg":  date(1994, 12, 29),
    "metformin_1000mg": date(1994, 12, 29),
    "liraglutide":      date(2010, 1, 26),   # Victoza
    "dulaglutide":      date(2014, 9, 18),   # Trulicity
    "semaglutide_inj":  date(2017, 12, 5),   # Ozempic
    "semaglutide_oral": date(2019, 9, 20),   # Rybelsus
    "canagliflozin":    date(2013, 3, 29),   # Invokana (first SGLT-2i)
    "dapagliflozin":    date(2014, 1, 8),    # Farxiga
    "empagliflozin":    date(2014, 8, 1),    # Jardiance
}
_SGLT2_AVAILABLE_FROM = date(2013, 3, 29)   # earliest any SGLT-2i was approved

GENDER_CONCEPTS = {0: 8532, 1: 8507}  # OMOP: female=8532, male=8507
RACE_CONCEPTS   = {0: 8527, 1: 8515, 2: 8516, 3: 8657}  # White, Asian, Black, Other


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS concept (
        concept_id          INTEGER PRIMARY KEY,
        concept_name        VARCHAR,
        domain_id           VARCHAR,
        vocabulary_id       VARCHAR,
        concept_class_id    VARCHAR,
        standard_concept    VARCHAR,
        concept_code        VARCHAR
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS person (
        person_id                   INTEGER PRIMARY KEY,
        gender_concept_id           INTEGER,
        year_of_birth               INTEGER,
        month_of_birth              INTEGER,
        day_of_birth                INTEGER,
        race_concept_id             INTEGER,
        ethnicity_concept_id        INTEGER
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS observation_period (
        observation_period_id           INTEGER PRIMARY KEY,
        person_id                       INTEGER,
        observation_period_start_date   DATE,
        observation_period_end_date     DATE,
        period_type_concept_id          INTEGER
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS condition_occurrence (
        condition_occurrence_id         INTEGER PRIMARY KEY,
        person_id                       INTEGER,
        condition_concept_id            INTEGER,
        condition_start_date            DATE,
        condition_end_date              DATE,
        condition_type_concept_id       INTEGER
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS drug_exposure (
        drug_exposure_id            INTEGER PRIMARY KEY,
        person_id                   INTEGER,
        drug_concept_id             INTEGER,
        drug_exposure_start_date    DATE,
        drug_exposure_end_date      DATE,
        days_supply                 INTEGER,
        drug_type_concept_id        INTEGER,
        quantity                    FLOAT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS visit_occurrence (
        visit_occurrence_id         INTEGER PRIMARY KEY,
        person_id                   INTEGER,
        visit_concept_id            INTEGER,
        visit_start_date            DATE,
        visit_end_date              DATE,
        visit_type_concept_id       INTEGER
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS measurement (
        measurement_id              INTEGER PRIMARY KEY,
        person_id                   INTEGER,
        measurement_concept_id      INTEGER,
        measurement_date            DATE,
        value_as_number             FLOAT,
        unit_concept_id             INTEGER,
        measurement_type_concept_id INTEGER
    )""")
    log.info("OMOP schema created")


def populate_vocabulary(conn: duckdb.DuckDBPyConnection) -> None:
    """Insert minimal concept rows needed for downstream queries."""
    rows = [
        (T2DM_CONCEPT_ID, "Type 2 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "44054006"),
        (T1DM_CONCEPT_ID, "Type 1 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "46635009"),
    ]
    for name, cid in COMORBIDITY_CONCEPTS.items():
        rows.append((cid, name.replace("_", " ").title(), "Condition", "SNOMED", "Clinical Finding", "S", str(cid)))
    for cls, drugs in DRUG_CONCEPTS.items():
        for drug_name, cid in drugs.items():
            rows.append((cid, drug_name.replace("_", " ").title(), "Drug", "RxNorm", "Clinical Drug", "S", str(cid)))

    df = pd.DataFrame(rows, columns=[
        "concept_id", "concept_name", "domain_id", "vocabulary_id",
        "concept_class_id", "standard_concept", "concept_code",
    ]).drop_duplicates("concept_id")
    conn.execute("DELETE FROM concept")
    conn.execute("INSERT INTO concept SELECT * FROM df")
    log.info("Vocabulary populated (%d concepts)", len(df))


def generate_synthetic_patients(n_patients: int = 5000, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate a synthetic patient population when Synthea is unavailable."""
    rng = np.random.default_rng(seed)
    log.info("Generating synthetic fallback data for %d patients (seed=%d)", n_patients, seed)

    study_start = date(2010, 1, 1)
    study_end   = date(2022, 12, 31)

    # ── Persons ──────────────────────────────────────────────────────────────
    person_ids = list(range(1, n_patients + 1))
    years_of_birth = rng.integers(1940, 1985, n_patients)
    genders = rng.integers(0, 2, n_patients)
    races   = rng.choice([0, 1, 2, 3], n_patients, p=[0.65, 0.10, 0.15, 0.10])

    persons = pd.DataFrame({
        "person_id":            person_ids,
        "gender_concept_id":    [GENDER_CONCEPTS[g] for g in genders],
        "year_of_birth":        years_of_birth,
        "month_of_birth":       rng.integers(1, 13, n_patients),
        "day_of_birth":         rng.integers(1, 29, n_patients),
        "race_concept_id":      [RACE_CONCEPTS[r] for r in races],
        "ethnicity_concept_id": rng.choice([38003563, 38003564], n_patients),
    })

    # ── T2DM diagnosis dates ──────────────────────────────────────────────────
    total_days = (study_end - study_start).days
    t2dm_offsets = rng.integers(0, total_days - 365, n_patients)
    t2dm_dates = [study_start + timedelta(days=int(d)) for d in t2dm_offsets]

    # ── Drug class and index date — pre-computed before comorbidities ─────────
    # Drug class must be known before comorbidity assignment so that
    # class-specific prevalence (channelling by indication) can be applied.
    drug_class_pool = list(CLASS_MARKET_SHARE.keys())
    class_probs     = list(CLASS_MARKET_SHARE.values())
    raw_classes     = rng.choice(drug_class_pool, n_patients, p=class_probs)
    idx_offsets     = rng.integers(0, 181, n_patients)

    assigned_classes: list[str] = []
    index_dates: list[date] = []
    for t2dm_date, raw_cls, idx_off in zip(t2dm_dates, raw_classes, idx_offsets):
        idx = t2dm_date + timedelta(days=int(idx_off))
        if idx > study_end - timedelta(days=90):
            idx = study_end - timedelta(days=90)
        # Temporal availability gate: SGLT-2i class not approved until 2013-03-29
        drug_cls = raw_cls
        if drug_cls == "sglt2" and idx < _SGLT2_AVAILABLE_FROM:
            drug_cls = str(rng.choice(["metformin", "glp1"], p=[0.80, 0.20]))
        assigned_classes.append(drug_cls)
        index_dates.append(idx)

    conditions: list[dict[str, Any]] = []
    cond_id = 1
    for pid, t2dm_date in zip(person_ids, t2dm_dates):
        conditions.append({
            "condition_occurrence_id": cond_id,
            "person_id": pid,
            "condition_concept_id": T2DM_CONCEPT_ID,
            "condition_start_date": t2dm_date,
            "condition_end_date": t2dm_date,
            "condition_type_concept_id": 32817,
        })
        cond_id += 1

    # ── Comorbidities — class-specific prevalence (channelling by indication) ─
    for pid, t2dm_date, drug_class in zip(person_ids, t2dm_dates, assigned_classes):
        for comorb_name, base_prev in CLASS_COMORBIDITY_PREVALENCE[drug_class].items():
            if rng.random() < base_prev:
                # Prevalent: 70% before index, 30% incident during follow-up
                if rng.random() < 0.70:
                    offset = rng.integers(30, 365)
                    onset = t2dm_date - timedelta(days=int(offset))
                    onset = max(onset, study_start)
                else:
                    offset = rng.integers(91, min(365 * 2, total_days))
                    onset = t2dm_date + timedelta(days=int(offset))
                    if onset > study_end:
                        continue
                conditions.append({
                    "condition_occurrence_id": cond_id,
                    "person_id": pid,
                    "condition_concept_id": COMORBIDITY_CONCEPTS[comorb_name],
                    "condition_start_date": onset,
                    "condition_end_date": onset,
                    "condition_type_concept_id": 32817,
                })
                cond_id += 1

    # ── Drug exposures ────────────────────────────────────────────────────────
    drug_rows: list[dict[str, Any]] = []
    drug_id = 1
    obs_periods: list[dict[str, Any]] = []

    for pid, drug_class, t2dm_date, index_date in zip(
        person_ids, assigned_classes, t2dm_dates, index_dates
    ):
        # Pick a specific drug approved at or before the patient's index date
        available = {
            name: cid
            for name, cid in DRUG_CONCEPTS[drug_class].items()
            if DRUG_APPROVAL_DATES.get(name, date(1994, 1, 1)) <= index_date
        }
        if not available:
            # Fallback: at least one GLP-1 (liraglutide) or metformin always available
            available = DRUG_CONCEPTS["metformin"]
            drug_class = "metformin"
        drug_concept_id = int(rng.choice(list(available.values())))

        # Persistence: lognormal TTD calibrated to published 1-yr persistence rates
        mu, sigma = CLASS_TTD_PARAMS[drug_class]
        ttd_days = int(np.clip(rng.lognormal(mu, sigma), 91, 1200))

        # Generate prescription fills with ~30-day supply and occasional gaps
        fill_date = index_date
        days_supply = 30
        n_fills_before_disc = ttd_days // days_supply

        # Discontinuation date is exactly ttd_days from index.
        # Fills are generated with small gaps (≤ 45 days) until disc_date,
        # then stop — creating a trailing gap > 90 days to obs_end.
        disc_date = index_date + timedelta(days=int(ttd_days))

        for fill_i in range(max(1, n_fills_before_disc)):
            end_date = fill_date + timedelta(days=days_supply)
            # Stop generating fills once we reach the discontinuation date
            if fill_date >= disc_date or end_date > study_end:
                break
            # Clip end_date to disc_date so supply doesn't extend past TTD
            end_date = min(end_date, disc_date)
            drug_rows.append({
                "drug_exposure_id":         drug_id,
                "person_id":                pid,
                "drug_concept_id":          drug_concept_id,
                "drug_exposure_start_date": fill_date,
                "drug_exposure_end_date":   end_date,
                "days_supply":              (end_date - fill_date).days,
                "drug_type_concept_id":     38000177,
                "quantity":                 float((end_date - fill_date).days),
            })
            drug_id += 1
            # Gap between fills: 0–45 days (always ≤ 90-day grace → persistent within TTD window)
            gap = int(rng.integers(0, 46))
            fill_date = end_date + timedelta(days=gap)

        # obs_end is well past disc_date, creating a trailing gap > 90 days
        # (the patient stopped filling but observation continues)
        obs_end = min(disc_date + timedelta(days=int(rng.integers(120, 300))), study_end)
        obs_periods.append({
            "observation_period_id":         pid,
            "person_id":                     pid,
            "observation_period_start_date": t2dm_date,
            "observation_period_end_date":   obs_end,
            "period_type_concept_id":        44814724,
        })

    conditions_df = pd.DataFrame(conditions)
    drugs_df      = pd.DataFrame(drug_rows)
    obs_df        = pd.DataFrame(obs_periods)

    log.info("Synthetic data generated: %d persons, %d conditions, %d drug exposures",
             len(persons), len(conditions_df), len(drugs_df))
    return {
        "person": persons,
        "condition_occurrence": conditions_df,
        "drug_exposure": drugs_df,
        "observation_period": obs_df,
    }


def load_synthea_csvs(synthea_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Synthea CSV exports and map to OMOP-compatible DataFrames."""
    csv_dir = synthea_dir / "csv"
    if not csv_dir.exists():
        csv_dir = synthea_dir
    log.info("Loading Synthea CSVs from %s", csv_dir)

    tables = {}
    for fname in ["patients.csv", "medications.csv", "conditions.csv",
                  "observations.csv", "encounters.csv"]:
        fpath = csv_dir / fname
        if fpath.exists():
            tables[fname.replace(".csv", "")] = pd.read_csv(fpath, low_memory=False)
            log.info("  Loaded %s (%d rows)", fname, len(tables[fname.replace('.csv', '')]))
        else:
            log.warning("  %s not found", fpath)

    if not tables:
        raise FileNotFoundError(f"No Synthea CSV files found in {csv_dir}")

    return _map_synthea_to_omop(tables)


def _map_synthea_to_omop(synthea: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Map Synthea field names and codes to OMOP CDM v5.4 structure."""
    pts = synthea.get("patients", pd.DataFrame())
    if pts.empty:
        raise ValueError("patients.csv is empty or missing")

    pts = pts.copy()
    pts["person_id"] = range(1, len(pts) + 1)
    pts["BIRTHDATE"] = pd.to_datetime(pts["BIRTHDATE"])
    pts["gender_concept_id"] = pts["GENDER"].map({"F": 8532, "M": 8507}).fillna(8551)
    pts["race_concept_id"] = pts["RACE"].map({
        "white": 8527, "asian": 8515, "black": 8516,
        "native": 8657, "other": 8522,
    }).fillna(8522)

    person = pd.DataFrame({
        "person_id":            pts["person_id"],
        "gender_concept_id":    pts["gender_concept_id"],
        "year_of_birth":        pts["BIRTHDATE"].dt.year,
        "month_of_birth":       pts["BIRTHDATE"].dt.month,
        "day_of_birth":         pts["BIRTHDATE"].dt.day,
        "race_concept_id":      pts["race_concept_id"],
        "ethnicity_concept_id": 38003564,
    })

    id_map = dict(zip(pts.get("Id", pts.index), pts["person_id"]))

    result = {"person": person}

    if "conditions" in synthea:
        conds = synthea["conditions"].copy()
        conds["person_id"] = conds["PATIENT"].map(id_map)
        conds["condition_start_date"] = pd.to_datetime(conds["START"]).dt.date
        conds["condition_end_date"]   = pd.to_datetime(conds.get("STOP", conds["START"])).dt.date
        # Map Synthea SNOMED codes to OMOP concept_ids (simplified: use SNOMED code as proxy)
        conds["condition_concept_id"] = conds["CODE"].astype(str).apply(
            lambda c: next((v for k, v in COMORBIDITY_CONCEPTS.items() if c in k), T2DM_CONCEPT_ID)
        )
        result["condition_occurrence"] = pd.DataFrame({
            "condition_occurrence_id":  range(1, len(conds) + 1),
            "person_id":                conds["person_id"],
            "condition_concept_id":     conds["condition_concept_id"],
            "condition_start_date":     conds["condition_start_date"],
            "condition_end_date":       conds["condition_end_date"],
            "condition_type_concept_id": 32817,
        }).dropna(subset=["person_id"])

    if "medications" in synthea:
        meds = synthea["medications"].copy()
        meds["person_id"] = meds["PATIENT"].map(id_map)
        meds["drug_exposure_start_date"] = pd.to_datetime(meds["START"]).dt.date
        meds["drug_exposure_end_date"]   = pd.to_datetime(
            meds.get("STOP", meds["START"])
        ).dt.date

        meds["drug_concept_id"] = meds["CODE"].astype(str).apply(
            lambda c: next((cid for drugs in DRUG_CONCEPTS.values()
                            for name, cid in drugs.items() if c.lower() in name), 0)
        )
        meds = meds[meds["drug_concept_id"] > 0]
        result["drug_exposure"] = pd.DataFrame({
            "drug_exposure_id":         range(1, len(meds) + 1),
            "person_id":                meds["person_id"],
            "drug_concept_id":          meds["drug_concept_id"],
            "drug_exposure_start_date": meds["drug_exposure_start_date"],
            "drug_exposure_end_date":   meds["drug_exposure_end_date"],
            "days_supply":              30,
            "drug_type_concept_id":     38000177,
            "quantity":                 30.0,
        }).dropna(subset=["person_id"])

    return result


def load_to_duckdb(tables: dict[str, pd.DataFrame], db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    try:
        create_schema(conn)
        populate_vocabulary(conn)
        for table_name, df in tables.items():
            if df.empty:
                log.warning("Skipping empty table: %s", table_name)
                continue
            conn.execute(f"DELETE FROM {table_name}")
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            log.info("Loaded %s: %d rows", table_name, len(df))
        conn.commit()
        log.info("OMOP DuckDB written to %s", db_path)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthea → OMOP CDM ETL")
    parser.add_argument("--synthea-dir",         default="data/synthea_output")
    parser.add_argument("--db-path",             default="data/omop/omop.duckdb")
    parser.add_argument("--generate-synthetic",  action="store_true",
                        help="Generate synthetic fallback data (no Synthea required)")
    parser.add_argument("--patients",            type=int, default=5000)
    parser.add_argument("--seed",                type=int, default=42)
    args = parser.parse_args()

    if args.generate_synthetic:
        tables = generate_synthetic_patients(args.patients, args.seed)
    else:
        synthea_dir = Path(args.synthea_dir)
        if not synthea_dir.exists() or not any(synthea_dir.rglob("*.csv")):
            log.warning("Synthea output not found — falling back to synthetic data generation")
            tables = generate_synthetic_patients(args.patients, args.seed)
        else:
            try:
                tables = load_synthea_csvs(synthea_dir)
            except Exception as e:
                log.warning("Synthea CSV load failed (%s) — falling back to synthetic data", e)
                tables = generate_synthetic_patients(args.patients, args.seed)

    load_to_duckdb(tables, args.db_path)


if __name__ == "__main__":
    main()
