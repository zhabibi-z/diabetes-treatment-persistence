"""
run_ttd.py — Time-to-discontinuation (TTD) analysis with 90-day grace period.

Primary outcome: first gap between end of supply and next dispensing > 90 days.
Patients who do not discontinue are right-censored at end of observation.

Grace period definition follows Lim et al. 2025 (Diabetologia).

Outputs:
  outputs/tables/ttd_events.csv     — one row per patient: time, event, covariates
  outputs/tables/ttd_summary.csv    — median TTD by drug class
  outputs/tables/cox_ttd_results.csv
  outputs/figures/km_ttd_overall.png
  outputs/figures/km_ttd_by_class.png
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
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GRACE_DAYS = 90

DRUG_COLORS = {"metformin": "#3498DB", "glp1": "#E74C3C", "sglt2": "#2ECC71"}
DRUG_LABELS = {"metformin": "Metformin", "glp1": "GLP-1 RA", "sglt2": "SGLT-2i"}

METFORMIN_CONCEPTS  = {1503297, 1503298, 1503299, 1503300, 1503301}
GLP1_CONCEPTS       = {2200644, 2200645, 1583722, 40170911, 1583723, 40239491}
SGLT2_CONCEPTS      = {1792455, 1488564, 1373463, 1488565, 1373464, 1792456}
ALL_STUDY_CONCEPTS  = METFORMIN_CONCEPTS | GLP1_CONCEPTS | SGLT2_CONCEPTS

COMORBIDITY_COLS = [
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]


def compute_ttd(drug_exposures: pd.DataFrame, cohort: pd.DataFrame, grace_days: int = GRACE_DAYS) -> pd.DataFrame:
    """Compute time-to-discontinuation per patient using the grace period approach."""
    drug_exposures = drug_exposures.copy()
    drug_exposures["rx_start"] = pd.to_datetime(drug_exposures["drug_exposure_start_date"])
    drug_exposures["rx_end"]   = pd.to_datetime(drug_exposures["drug_exposure_end_date"])
    drug_exposures = drug_exposures.sort_values(["person_id", "rx_start"])

    cohort_indexed = cohort.copy()
    cohort_indexed["index_date"] = pd.to_datetime(cohort_indexed["index_date"])
    cohort_indexed["obs_end"]    = pd.to_datetime(cohort_indexed["obs_end"])

    records = []
    for pid, grp in drug_exposures.groupby("person_id"):
        prow = cohort_indexed[cohort_indexed["person_id"] == pid]
        if prow.empty:
            continue
        prow = prow.iloc[0]
        index_dt = prow["index_date"]
        obs_end  = prow["obs_end"]

        # Filter to prescriptions on or after index date
        fills = grp[grp["rx_start"] >= index_dt].sort_values("rx_start")
        if fills.empty:
            records.append({
                "person_id": pid, "ttd_days": 0, "discontinued": 0,
                "disc_date": obs_end,
            })
            continue

        # Find first gap exceeding grace period (between consecutive fills)
        discontinued = False
        disc_date    = obs_end
        for i in range(len(fills) - 1):
            current_end  = fills.iloc[i]["rx_end"]
            next_start   = fills.iloc[i + 1]["rx_start"]
            gap = (next_start - current_end).days
            if gap > grace_days:
                disc_date    = current_end
                discontinued = True
                break

        # Also check gap after the LAST fill to end of observation
        if not discontinued:
            last_end = fills.iloc[-1]["rx_end"]
            if hasattr(obs_end, "to_pydatetime"):
                obs_end_dt = obs_end
            else:
                obs_end_dt = pd.Timestamp(obs_end)
            trailing_gap = (obs_end_dt - last_end).days
            if trailing_gap > grace_days:
                disc_date    = last_end
                discontinued = True

        ttd = (disc_date - index_dt).days
        records.append({
            "person_id":    pid,
            "ttd_days":     max(0, ttd),
            "discontinued": int(discontinued),
            "disc_date":    disc_date,
        })

    return pd.DataFrame(records)


def run_ttd_analysis(db_path: str, matched_cohort: str, output_dir: str) -> None:
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(matched_cohort)
    person_ids = tuple(cohort["person_id"].tolist())

    conn = duckdb.connect(db_path, read_only=True)
    study_cids = tuple(ALL_STUDY_CONCEPTS)
    drug_exp = conn.execute(f"""
        SELECT person_id, drug_concept_id,
               drug_exposure_start_date, drug_exposure_end_date,
               COALESCE(days_supply, 30) as days_supply
        FROM drug_exposure
        WHERE person_id IN {person_ids}
          AND drug_concept_id IN {study_cids}
    """).df()
    conn.close()

    # Fill end date from days_supply if missing
    drug_exp["drug_exposure_start_date"] = pd.to_datetime(drug_exp["drug_exposure_start_date"])
    drug_exp["drug_exposure_end_date"] = pd.to_datetime(drug_exp["drug_exposure_end_date"])
    missing_end = drug_exp["drug_exposure_end_date"].isna()
    drug_exp.loc[missing_end, "drug_exposure_end_date"] = (
        drug_exp.loc[missing_end, "drug_exposure_start_date"] +
        pd.to_timedelta(drug_exp.loc[missing_end, "days_supply"], unit="D")
    )

    log.info("Computing TTD with %d-day grace period", GRACE_DAYS)
    ttd = compute_ttd(drug_exp, cohort, GRACE_DAYS)

    events_df = cohort.merge(ttd, on="person_id", how="left")
    events_df["ttd_days"]     = events_df["ttd_days"].fillna(events_df["followup_days"])
    events_df["discontinued"] = events_df["discontinued"].fillna(0).astype(int)

    events_df.to_csv(f"{output_dir}/tables/ttd_events.csv", index=False)
    log.info("ttd_events.csv: %d rows, %d discontinuations (%.1f%%)",
             len(events_df), events_df["discontinued"].sum(),
             100 * events_df["discontinued"].mean())

    # ── TTD summary by drug class ─────────────────────────────────────────────
    summary_rows = []
    for dc in ["metformin", "glp1", "sglt2"]:
        sub = events_df[events_df["drug_class"] == dc]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["ttd_days"], sub["discontinued"], label=dc)
        med = kmf.median_survival_time_
        summary_rows.append({
            "drug_class": dc, "n": len(sub),
            "n_discontinued": int(sub["discontinued"].sum()),
            "pct_discontinued": round(100 * sub["discontinued"].mean(), 1),
            "median_ttd_days": round(med, 1) if not np.isinf(med) else ">study_end",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/tables/ttd_summary.csv", index=False)
    log.info("ttd_summary.csv written")

    # ── Kaplan-Meier plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dc in ["metformin", "glp1", "sglt2"]:
        sub = events_df[events_df["drug_class"] == dc]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["ttd_days"], sub["discontinued"], label=DRUG_LABELS[dc])
        kmf.plot_survival_function(ax=ax, color=DRUG_COLORS[dc], ci_show=True, linewidth=2)

    ax.set_xlabel("Days from Index Date", fontsize=12)
    ax.set_ylabel("Probability of Persistence", fontsize=12)
    ax.set_title("Treatment Persistence by Drug Class\n(90-day grace period, Lim 2025)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/km_ttd_by_class.png", dpi=150)
    plt.close()
    log.info("KM plot saved")

    # ── Log-rank test ─────────────────────────────────────────────────────────
    events_df_clean = events_df.dropna(subset=["ttd_days", "discontinued", "drug_class"])
    if len(events_df_clean["drug_class"].unique()) >= 2:
        lr = multivariate_logrank_test(
            events_df_clean["ttd_days"],
            events_df_clean["drug_class"],
            events_df_clean["discontinued"],
        )
        log.info("Log-rank test: p=%.4f (chi2=%.3f)", lr.p_value, lr.test_statistic)

    # ── Cox PH model ──────────────────────────────────────────────────────────
    cox_candidate = ["ttd_days", "discontinued", "drug_class_num", "age_at_index", "cci"]
    cox_candidate += [c for c in COMORBIDITY_COLS if c in events_df.columns]
    cox_data = events_df[cox_candidate].dropna()

    # Drop near-zero-variance covariates (variance < 0.001) to avoid convergence errors
    covariate_cols = [c for c in cox_candidate if c not in ("ttd_days", "discontinued")]
    low_var = [c for c in covariate_cols if c in cox_data.columns and cox_data[c].var() < 0.001]
    if low_var:
        log.warning("Dropping low-variance covariates from Cox model: %s", low_var)
        cox_data = cox_data.drop(columns=low_var)

    cox_cols_final = [c for c in cox_data.columns]

    if len(cox_data) >= 30:
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col="ttd_days", event_col="discontinued")
            cox_res = cph.summary.reset_index()
            cox_res.to_csv(f"{output_dir}/tables/cox_ttd_results.csv", index=False)
            log.info("Cox PH model complete — results in cox_ttd_results.csv")
            log.info("\n%s", cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())
        except Exception as e:
            log.error("Cox PH convergence failed: %s", e)
    else:
        log.warning("Insufficient data for Cox model (n=%d)", len(cox_data))


def main() -> None:
    parser = argparse.ArgumentParser(description="TTD analysis — 90-day grace period")
    parser.add_argument("--db-path",        default="data/omop/omop.duckdb")
    parser.add_argument("--matched-cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--output-dir",     default="outputs")
    args = parser.parse_args()
    run_ttd_analysis(args.db_path, args.matched_cohort, args.output_dir)


if __name__ == "__main__":
    main()
