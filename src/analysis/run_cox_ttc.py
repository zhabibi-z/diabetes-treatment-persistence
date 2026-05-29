"""
run_cox_ttc.py — Cox proportional hazards model for time-to-comorbidity (TTC).

For each of the 15 comorbidities, fits a Cox model with time to that comorbidity
as the outcome. Predictors: drug class indicator + age + CCI + remaining comorbidities.
Reports HRs for drug class vs. metformin (reference).

Outputs:
  outputs/tables/cox_ttc_results.csv     — one row per comorbidity × predictor
  outputs/figures/forest_ttc.png          — forest plot of drug-class HRs per comorbidity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COMORBIDITY_CONCEPTS: dict[str, int] = {
    "hypertension": 316866, "obesity": 433736, "ckd": 46271022,
    "heart_failure": 316139, "hyperlipidemia": 432867, "nash": 4212540,
    "neuropathy": 378419, "retinopathy": 4226354, "depression": 440383,
    "atrial_fibrillation": 313217, "sleep_apnea": 4173636, "nafld": 4212540,
    "pvd": 321052, "stroke": 372924, "mi": 4329847,
}
COMORBIDITY_NAMES = list(COMORBIDITY_CONCEPTS.keys())


def run_ttc_cox(db_path: str, matched_cohort: str, output_dir: str) -> None:
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(matched_cohort)
    cohort["index_date"] = pd.to_datetime(cohort["index_date"])
    cohort["obs_end"]    = pd.to_datetime(cohort["obs_end"])

    person_ids  = tuple(cohort["person_id"].tolist())
    comorb_cids = tuple(set(COMORBIDITY_CONCEPTS.values()))

    conn = duckdb.connect(db_path, read_only=True)
    cond_df = conn.execute(f"""
        SELECT person_id, condition_concept_id,
               condition_start_date AS cond_date
        FROM condition_occurrence
        WHERE person_id IN {person_ids}
          AND condition_concept_id IN {comorb_cids}
    """).df()
    conn.close()

    cond_df["cond_date"] = pd.to_datetime(cond_df["cond_date"])
    cid_to_name = {v: k for k, v in COMORBIDITY_CONCEPTS.items()}
    cond_df["comorb_name"] = cond_df["condition_concept_id"].map(cid_to_name)

    all_results = []
    forest_data = []

    for comorb in COMORBIDITY_NAMES:
        # Restrict to patients free of this comorbidity at baseline
        if comorb not in cohort.columns:
            continue
        at_risk = cohort[cohort[comorb] == 0].copy()
        if len(at_risk) < 30:
            continue

        # Incident event: first onset after index date
        incident = cond_df[
            (cond_df["comorb_name"] == comorb) &
            (cond_df["person_id"].isin(at_risk["person_id"]))
        ]
        first_inc = (
            incident.merge(at_risk[["person_id", "index_date"]], on="person_id", how="inner")
        )
        first_inc = first_inc[first_inc["cond_date"] > first_inc["index_date"]]
        first_inc = (
            first_inc.sort_values("cond_date")
            .groupby("person_id")
            .first()
            .reset_index()
            [["person_id", "cond_date"]]
        )

        at_risk = at_risk.merge(first_inc, on="person_id", how="left")
        at_risk["ttc_days"]  = (
            at_risk["cond_date"].fillna(at_risk["obs_end"]) - at_risk["index_date"]
        ).dt.days.clip(lower=0)
        at_risk["ttc_event"] = at_risk["cond_date"].notna().astype(int)

        if at_risk["ttc_event"].sum() < 5:
            log.warning("Too few events for %s Cox (%d events) — skipping", comorb, at_risk["ttc_event"].sum())
            continue

        cox_cols = ["ttc_days", "ttc_event", "drug_class_num", "age_at_index", "cci"]
        # Other comorbidities as covariates (excluding current outcome)
        other_comorbs = [c for c in COMORBIDITY_NAMES if c != comorb and c in at_risk.columns]
        cox_cols += other_comorbs
        cox_data = at_risk[cox_cols].dropna()

        # Drop near-zero-variance covariates to prevent convergence failures
        covariate_cols = [c for c in cox_cols if c not in ("ttc_days", "ttc_event")]
        low_var = [c for c in covariate_cols if c in cox_data.columns and cox_data[c].var() < 0.001]
        if low_var:
            cox_data = cox_data.drop(columns=low_var)

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col="ttc_days", event_col="ttc_event")
            res = cph.summary.reset_index()
            res["comorbidity_outcome"] = comorb
            all_results.append(res)

            # Drug class HR (drug_class_num: 1=GLP1 vs met, 2=SGLT2 vs met relative)
            dc_row = res[res["covariate"] == "drug_class_num"]
            if not dc_row.empty:
                forest_data.append({
                    "comorbidity": comorb.replace("_", " ").title(),
                    "hr":          dc_row["exp(coef)"].values[0],
                    "hr_lower":    dc_row["exp(coef) lower 95%"].values[0],
                    "hr_upper":    dc_row["exp(coef) upper 95%"].values[0],
                    "p":           dc_row["p"].values[0],
                })
        except Exception as e:
            log.warning("Cox TTC failed for %s: %s", comorb, e)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(f"{output_dir}/tables/cox_ttc_results.csv", index=False)
        log.info("cox_ttc_results.csv: %d rows", len(results_df))

    # ── Forest plot ───────────────────────────────────────────────────────────
    if forest_data:
        fdf = pd.DataFrame(forest_data).sort_values("hr")
        fig, ax = plt.subplots(figsize=(7, max(4, len(fdf) * 0.4 + 1)))
        y = np.arange(len(fdf))
        ax.errorbar(
            fdf["hr"], y,
            xerr=[fdf["hr"] - fdf["hr_lower"], fdf["hr_upper"] - fdf["hr"]],
            fmt="o", color="#2C3E50", ecolor="#7F8C8D", capsize=4, markersize=6,
        )
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(fdf["comorbidity"], fontsize=9)
        ax.set_xlabel("Hazard Ratio (drug class on comorbidity onset)", fontsize=10)
        ax.set_title("TTC Cox — HR for Drug Class per Comorbidity", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/forest_ttc.png", dpi=150)
        plt.close()
        log.info("Forest plot saved: forest_ttc.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",        default="data/omop/omop.duckdb")
    parser.add_argument("--matched-cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--output-dir",     default="outputs")
    args = parser.parse_args()
    run_ttc_cox(args.db_path, args.matched_cohort, args.output_dir)


if __name__ == "__main__":
    main()
