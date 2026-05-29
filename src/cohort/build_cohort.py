"""
build_cohort.py — Constructs three mutually exclusive new-user T2DM cohorts
from the OMOP CDM DuckDB database.

Cohorts:
  A — Metformin monotherapy initiators
  B — GLP-1 RA initiators (semaglutide, dulaglutide, liraglutide)
  C — SGLT-2i initiators (empagliflozin, dapagliflozin, canagliflozin)

Inclusion/exclusion logic follows the OHDSI LegendT2dm protocol and
Marcellusi 2019 cohort design (see PROTOCOL.md §3).

Outputs:
  outputs/tables/cohort_baseline.csv   — one row per patient, all covariates
  outputs/tables/comorbidity_prevalence.csv
  outputs/tables/cohort_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── OMOP concept ID sets (RxNorm-sourced, per OHDSI Athena) ──────────────────
METFORMIN_CONCEPTS  = {1503297, 1503298, 1503299, 1503300, 1503301}
GLP1_CONCEPTS       = {2200644, 2200645, 1583722, 40170911, 1583723, 40239491}
SGLT2_CONCEPTS      = {1792455, 1488564, 1373463, 1488565, 1373464, 1792456}
ALL_STUDY_CONCEPTS  = METFORMIN_CONCEPTS | GLP1_CONCEPTS | SGLT2_CONCEPTS

T2DM_CONCEPT_ID  = 201826
T1DM_CONCEPT_ID  = 435216
WASHOUT_DAYS     = 365
MIN_FOLLOWUP_DAYS = 90

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
COMORBIDITY_NAMES = list(COMORBIDITY_CONCEPTS.keys())


def assign_drug_class(concept_id: int) -> str | None:
    if concept_id in METFORMIN_CONCEPTS:
        return "metformin"
    if concept_id in GLP1_CONCEPTS:
        return "glp1"
    if concept_id in SGLT2_CONCEPTS:
        return "sglt2"
    return None


def build_cohort(db_path: str, output_dir: str) -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path, read_only=True)

    log.info("Loading drug exposures for study drugs")
    # All study-drug exposures
    drug_cids = tuple(ALL_STUDY_CONCEPTS)
    drugs = conn.execute(f"""
        SELECT person_id, drug_concept_id,
               drug_exposure_start_date AS rx_date,
               COALESCE(days_supply, 30) AS days_supply
        FROM drug_exposure
        WHERE drug_concept_id IN {drug_cids}
        ORDER BY person_id, rx_date
    """).df()
    drugs["drug_class"] = drugs["drug_concept_id"].map(assign_drug_class)

    log.info("Loading T2DM diagnoses")
    t2dm = conn.execute(f"""
        SELECT person_id, MIN(condition_start_date) AS t2dm_date
        FROM condition_occurrence
        WHERE condition_concept_id = {T2DM_CONCEPT_ID}
        GROUP BY person_id
    """).df()

    log.info("Loading persons")
    persons = conn.execute("""
        SELECT person_id, gender_concept_id, year_of_birth,
               race_concept_id, ethnicity_concept_id
        FROM person
    """).df()

    log.info("Loading observation periods")
    obs = conn.execute("""
        SELECT person_id,
               observation_period_start_date AS obs_start,
               observation_period_end_date   AS obs_end
        FROM observation_period
    """).df()

    log.info("Loading comorbidity conditions")
    comorb_cids = tuple(set(COMORBIDITY_CONCEPTS.values()))
    comorbs = conn.execute(f"""
        SELECT person_id, condition_concept_id, condition_start_date AS cond_date
        FROM condition_occurrence
        WHERE condition_concept_id IN {comorb_cids}
    """).df()

    # Map concept_id back to comorbidity name (many-to-one possible)
    cid_to_name = {v: k for k, v in COMORBIDITY_CONCEPTS.items()}
    comorbs["comorb_name"] = comorbs["condition_concept_id"].map(cid_to_name)
    comorbs = comorbs.dropna(subset=["comorb_name"])

    conn.close()

    # ── Derive index dates ────────────────────────────────────────────────────
    # Index date = first dispensing of that drug class after T2DM diagnosis
    drugs = drugs.merge(t2dm, on="person_id", how="inner")
    drugs["rx_date"]   = pd.to_datetime(drugs["rx_date"]).dt.date
    drugs["t2dm_date"] = pd.to_datetime(drugs["t2dm_date"]).dt.date

    # Must start on or after T2DM diagnosis
    drugs = drugs[drugs["rx_date"] >= drugs["t2dm_date"]]

    # First prescription per person per drug class
    first_rx = (
        drugs.sort_values("rx_date")
             .groupby(["person_id", "drug_class"])
             .first()
             .reset_index()
             [["person_id", "drug_class", "rx_date", "t2dm_date"]]
    )
    first_rx = first_rx.rename(columns={"rx_date": "index_date"})

    # ── Washout: exclude if prior use of any study class in washout window ────
    washout_prior = drugs.copy()
    washout_prior["prior_date"] = washout_prior["rx_date"]
    washout_prior["is_prior"] = True

    # For each candidate index event, check for prior study-class use
    candidate = first_rx.merge(persons, on="person_id", how="inner")
    candidate["index_date"]  = pd.to_datetime(candidate["index_date"]).dt.date
    candidate["t2dm_date"]   = pd.to_datetime(candidate["t2dm_date"]).dt.date
    candidate["age_at_index"] = (
        pd.to_datetime(candidate["index_date"]).dt.year - candidate["year_of_birth"]
    )

    # Merge obs period for follow-up check
    obs["obs_end"]   = pd.to_datetime(obs["obs_end"]).dt.date
    obs["obs_start"] = pd.to_datetime(obs["obs_start"]).dt.date
    candidate = candidate.merge(obs, on="person_id", how="inner")

    # Identify any prior study-drug dispensing in washout window
    prior_any = (
        drugs[drugs["drug_class"].isin(["metformin", "glp1", "sglt2"])]
        .groupby("person_id")["rx_date"]
        .agg(list)
        .reset_index()
        .rename(columns={"rx_date": "all_rx_dates"})
    )
    candidate = candidate.merge(prior_any, on="person_id", how="left")
    candidate["all_rx_dates"] = candidate["all_rx_dates"].apply(
        lambda x: [pd.to_datetime(d).date() for d in x] if isinstance(x, list) else []
    )

    def has_prior_use(row: pd.Series) -> bool:
        idx_date = row["index_date"]
        if hasattr(idx_date, "date"):
            idx_date = idx_date.date()
        from datetime import timedelta
        washout_start = idx_date - timedelta(days=WASHOUT_DAYS)
        return any(washout_start <= d < idx_date for d in row["all_rx_dates"])

    candidate["prior_use"] = candidate.apply(has_prior_use, axis=1)
    candidate = candidate[~candidate["prior_use"]]

    # ── Apply remaining inclusion/exclusion criteria ──────────────────────────
    # Age ≥ 18
    candidate = candidate[candidate["age_at_index"] >= 18]

    # Minimum 90-day follow-up
    from datetime import timedelta
    candidate["followup_days"] = (
        pd.to_datetime(candidate["obs_end"]).dt.date.apply(
            lambda x: (x - pd.Timestamp.today().date()).days
        )
    )
    candidate["followup_days"] = (
        (pd.to_datetime(candidate["obs_end"]) - pd.to_datetime(candidate["index_date"])).dt.days
    )
    candidate = candidate[candidate["followup_days"] >= MIN_FOLLOWUP_DAYS]

    # Mutually exclusive: keep only one row per person (first index date overall)
    candidate = candidate.sort_values("index_date").groupby("person_id").first().reset_index()

    log.info("Candidates after inclusion/exclusion: %d", len(candidate))

    # ── Assign baseline comorbidities ─────────────────────────────────────────
    comorbs["cond_date"] = pd.to_datetime(comorbs["cond_date"]).dt.date
    comorb_pivot = comorbs.merge(
        candidate[["person_id", "index_date"]], on="person_id", how="inner"
    )
    comorb_pivot["index_date"] = pd.to_datetime(comorb_pivot["index_date"]).dt.date
    # Prevalent: condition before or on index date
    comorb_baseline = comorb_pivot[comorb_pivot["cond_date"] <= comorb_pivot["index_date"]]
    comorb_wide = (
        comorb_baseline
        .groupby(["person_id", "comorb_name"])
        .size()
        .unstack(fill_value=0)
        .clip(0, 1)
        .reset_index()
    )
    # Ensure all 15 comorbidity columns exist
    for name in COMORBIDITY_NAMES:
        if name not in comorb_wide.columns:
            comorb_wide[name] = 0

    cohort = candidate.merge(comorb_wide, on="person_id", how="left")
    for name in COMORBIDITY_NAMES:
        if name not in cohort.columns:
            cohort[name] = 0
    cohort[COMORBIDITY_NAMES] = cohort[COMORBIDITY_NAMES].fillna(0).astype(int)

    # ── Charlson Comorbidity Index (simplified) ───────────────────────────────
    cci_weights = {
        "mi": 1, "heart_failure": 1, "pvd": 1, "stroke": 1,
        "ckd": 2, "depression": 0, "atrial_fibrillation": 0,
        "hypertension": 0, "obesity": 0, "nafld": 0, "nash": 0,
        "neuropathy": 1, "retinopathy": 0, "sleep_apnea": 0, "hyperlipidemia": 0,
    }
    cohort["cci"] = sum(cohort[c] * w for c, w in cci_weights.items() if c in cohort.columns)

    # ── Drug class label encoding ─────────────────────────────────────────────
    cohort["drug_class_num"] = cohort["drug_class"].map({"metformin": 0, "glp1": 1, "sglt2": 2})

    # ── Save outputs ──────────────────────────────────────────────────────────
    keep_cols = (
        ["person_id", "drug_class", "drug_class_num", "index_date", "t2dm_date",
         "obs_end", "followup_days", "age_at_index", "gender_concept_id",
         "race_concept_id", "cci"]
        + COMORBIDITY_NAMES
    )
    keep_cols = [c for c in keep_cols if c in cohort.columns]
    cohort_out = cohort[keep_cols].copy()

    cohort_out.to_csv(f"{output_dir}/cohort_baseline.csv", index=False)
    log.info("Wrote cohort_baseline.csv (%d rows)", len(cohort_out))

    # Cohort summary
    summary = (
        cohort_out.groupby("drug_class")
        .agg(
            n=("person_id", "count"),
            age_mean=("age_at_index", "mean"),
            age_sd=("age_at_index", "std"),
            pct_female=("gender_concept_id", lambda x: (x == 8532).mean() * 100),
            cci_mean=("cci", "mean"),
            followup_median=("followup_days", "median"),
        )
        .reset_index()
    )
    summary.to_csv(f"{output_dir}/cohort_summary.csv", index=False)
    log.info("Wrote cohort_summary.csv")

    # Comorbidity prevalence by drug class
    prev_rows = []
    for comorb in COMORBIDITY_NAMES:
        for dc in ["metformin", "glp1", "sglt2"]:
            subset = cohort_out[cohort_out["drug_class"] == dc]
            if len(subset) == 0:
                continue
            prev = subset[comorb].mean() * 100 if comorb in subset.columns else 0.0
            prev_rows.append({"comorbidity": comorb, "drug_class": dc, "prevalence_pct": round(prev, 2)})
    prev_df = pd.DataFrame(prev_rows)
    prev_df.to_csv(f"{output_dir}/comorbidity_prevalence.csv", index=False)
    log.info("Wrote comorbidity_prevalence.csv")

    return cohort_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2DM new-user cohorts from OMOP DuckDB")
    parser.add_argument("--db-path",    default="data/omop/omop.duckdb")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()

    cohort = build_cohort(args.db_path, args.output_dir)
    log.info("Cohort construction complete. %d patients in %s drug classes.",
             len(cohort), cohort["drug_class"].nunique())


if __name__ == "__main__":
    main()
