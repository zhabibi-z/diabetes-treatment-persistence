"""
run_ttc.py — Time-to-comorbidity (TTC) Kaplan-Meier analysis.

For each of the 15 pre-specified comorbidities, computes the KM curve for
time from index date to first incident comorbidity onset, among patients
who did NOT have that comorbidity at baseline.

Censoring: discontinuation or end of observation, whichever is earlier.

Outputs:
  outputs/tables/ttc_summary.csv
  outputs/figures/km_ttc_{comorbidity}.png
  outputs/figures/km_ttc_grid.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

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
DRUG_COLORS = {"metformin": "#3498DB", "glp1": "#E74C3C", "sglt2": "#2ECC71"}
DRUG_LABELS = {"metformin": "Metformin", "glp1": "GLP-1 RA", "sglt2": "SGLT-2i"}


def run_ttc_analysis(db_path: str, matched_cohort: str, output_dir: str) -> None:
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(matched_cohort)
    cohort["index_date"] = pd.to_datetime(cohort["index_date"])
    cohort["obs_end"]    = pd.to_datetime(cohort["obs_end"])

    person_ids = tuple(cohort["person_id"].tolist())
    comorb_cids = tuple(set(COMORBIDITY_CONCEPTS.values()))

    conn = duckdb.connect(db_path, read_only=True)
    cond_df = conn.execute(f"""
        SELECT person_id, condition_concept_id, condition_start_date AS cond_date
        FROM condition_occurrence
        WHERE person_id IN {person_ids}
          AND condition_concept_id IN {comorb_cids}
    """).df()
    conn.close()

    cond_df["cond_date"] = pd.to_datetime(cond_df["cond_date"])
    cid_to_name = {v: k for k, v in COMORBIDITY_CONCEPTS.items()}
    cond_df["comorb_name"] = cond_df["condition_concept_id"].map(cid_to_name)
    cond_df = cond_df.dropna(subset=["comorb_name"])

    summary_rows = []

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for idx, comorb in enumerate(COMORBIDITY_NAMES):
        ax = axes[idx] if idx < len(axes) else None

        # Patients without baseline comorbidity
        if comorb not in cohort.columns:
            continue
        at_risk = cohort[cohort[comorb] == 0].copy()
        if len(at_risk) < 10:
            log.warning("Too few at-risk for %s (n=%d) — skipping", comorb, len(at_risk))
            continue

        # Incident events: first condition occurrence AFTER index date
        incident = cond_df[
            (cond_df["comorb_name"] == comorb) &
            (cond_df["person_id"].isin(at_risk["person_id"]))
        ]
        first_incident = (
            incident.merge(at_risk[["person_id", "index_date"]], on="person_id", how="inner")
        )
        first_incident = first_incident[first_incident["cond_date"] > first_incident["index_date"]]
        first_incident = (
            first_incident.sort_values("cond_date")
            .groupby("person_id")
            .first()
            .reset_index()
            [["person_id", "cond_date"]]
        )

        at_risk = at_risk.merge(first_incident, on="person_id", how="left")
        at_risk["ttc_days"] = (
            at_risk["cond_date"].fillna(at_risk["obs_end"]) - at_risk["index_date"]
        ).dt.days.clip(lower=0)
        at_risk["ttc_event"] = at_risk["cond_date"].notna().astype(int)

        lr_p = None
        for dc in ["metformin", "glp1", "sglt2"]:
            sub = at_risk[at_risk["drug_class"] == dc]
            if sub.empty:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["ttc_days"], sub["ttc_event"], label=DRUG_LABELS[dc])
            if ax is not None:
                kmf.plot_survival_function(ax=ax, color=DRUG_COLORS[dc], ci_show=False, linewidth=1.5)

        if ax is not None:
            ax.set_title(comorb.replace("_", " ").title(), fontsize=9)
            ax.set_xlabel("Days", fontsize=7)
            ax.set_ylabel("P(Comorbidity-Free)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

        # Log-rank p-value
        at_risk_clean = at_risk.dropna(subset=["ttc_days", "ttc_event", "drug_class"])
        if len(at_risk_clean["drug_class"].unique()) >= 2:
            try:
                lr = multivariate_logrank_test(
                    at_risk_clean["ttc_days"],
                    at_risk_clean["drug_class"],
                    at_risk_clean["ttc_event"],
                )
                lr_p = lr.p_value
            except Exception:
                lr_p = None

        summary_rows.append({
            "comorbidity": comorb,
            "n_at_risk": len(at_risk),
            "n_events": int(at_risk["ttc_event"].sum()),
            "logrank_p": round(lr_p, 4) if lr_p is not None else None,
        })

    # Remove empty subplots
    for i in range(len(COMORBIDITY_NAMES), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Time to Comorbidity Onset by Drug Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/km_ttc_grid.png", dpi=130, bbox_inches="tight")
    plt.close()
    log.info("TTC KM grid saved")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/tables/ttc_summary.csv", index=False)
    log.info("ttc_summary.csv written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",        default="data/omop/omop.duckdb")
    parser.add_argument("--matched-cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--output-dir",     default="outputs")
    args = parser.parse_args()
    run_ttc_analysis(args.db_path, args.matched_cohort, args.output_dir)


if __name__ == "__main__":
    main()
