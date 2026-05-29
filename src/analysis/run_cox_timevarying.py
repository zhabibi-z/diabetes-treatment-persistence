"""
run_cox_timevarying.py — Time-varying Cox proportional hazards model.

Models the effect of comorbidity onset (0 → 1 transition during follow-up)
on treatment discontinuation. Uses counting process format (tstart, tstop).

Follows Iskandar et al. 2018 (BADBIR) multi-class Cox approach adapted for
time-varying comorbidity indicators.

Outputs:
  outputs/tables/cox_timevarying_results.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd
from lifelines import CoxTimeVaryingFitter

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


def build_counting_process_data(
    cohort: pd.DataFrame,
    cond_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert patient data into counting process format with time-varying comorbidities."""
    cohort = cohort.copy()
    cohort["index_date"] = pd.to_datetime(cohort["index_date"])
    cohort["obs_end"]    = pd.to_datetime(cohort["obs_end"])

    cid_to_name = {v: k for k, v in COMORBIDITY_CONCEPTS.items()}
    cond_df = cond_df.copy()
    cond_df["comorb_name"] = cond_df["condition_concept_id"].map(cid_to_name)
    cond_df["cond_date"]   = pd.to_datetime(cond_df["cond_date"])

    rows = []
    for _, pat in cohort.iterrows():
        pid        = pat["person_id"]
        t0         = pat["index_date"]
        t_end      = pat["obs_end"]
        event      = int(pat.get("discontinued", 0))
        ttd        = float(pat.get("ttd_days", (t_end - t0).days))

        # Baseline comorbidity status
        baseline = {c: int(pat.get(c, 0)) for c in COMORBIDITY_NAMES}

        # Incident onsets during follow-up
        pat_conds = cond_df[
            (cond_df["person_id"] == pid) &
            (cond_df["cond_date"] > t0) &
            (cond_df["cond_date"] <= t_end) &
            (cond_df["comorb_name"].notna())
        ].copy()

        # Build time points: index date + each incident onset date
        onset_dates = {}
        for _, crow in pat_conds.iterrows():
            cname = crow["comorb_name"]
            if baseline.get(cname, 0) == 0:  # Only incident (not prevalent)
                if cname not in onset_dates or crow["cond_date"] < onset_dates[cname]:
                    onset_dates[cname] = crow["cond_date"]

        # Construct interval: [0, ttd] split at each onset
        changepoints = sorted(set(
            [(d - t0).days for d in onset_dates.values() if (d - t0).days > 0 and (d - t0).days < ttd]
        ))
        checkpoints = [0] + changepoints + [ttd]

        current_status = dict(baseline)
        for i in range(len(checkpoints) - 1):
            tstart = checkpoints[i]
            tstop  = checkpoints[i + 1]
            if tstop <= tstart:
                continue

            # Update time-varying status: any onset before tstop
            for cname, days in {c: (d - t0).days for c, d in onset_dates.items()}.items():
                if days <= tstart:
                    current_status[cname] = 1

            is_final = (i == len(checkpoints) - 2)
            row = {
                "id": pid,
                "tstart": tstart,
                "tstop": tstop,
                "event": int(event) if is_final else 0,
                "drug_class_num": int(pat.get("drug_class_num", 0)),
                "age_at_index": float(pat.get("age_at_index", 60)),
                "cci": float(pat.get("cci", 0)),
            }
            row.update({f"tv_{c}": current_status.get(c, 0) for c in COMORBIDITY_NAMES})
            rows.append(row)

    return pd.DataFrame(rows)


def run_cox_timevarying(db_path: str, matched_cohort: str, output_dir: str) -> None:
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(matched_cohort)

    # Try loading TTD events (pre-computed by run_ttd.py)
    ttd_path = f"{output_dir}/tables/ttd_events.csv"
    if Path(ttd_path).exists():
        ttd_events = pd.read_csv(ttd_path)[["person_id", "ttd_days", "discontinued"]]
        cohort = cohort.merge(ttd_events, on="person_id", how="left")
    cohort["ttd_days"]     = cohort.get("ttd_days",     cohort.get("followup_days", 365))
    cohort["discontinued"] = cohort.get("discontinued", pd.Series(0, index=cohort.index))

    person_ids = tuple(cohort["person_id"].tolist())
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

    log.info("Building counting process format")
    cp_data = build_counting_process_data(cohort, cond_df)

    if len(cp_data) < 50:
        log.warning("Insufficient data for time-varying Cox (n=%d rows)", len(cp_data))
        return

    tv_cols  = [f"tv_{c}" for c in COMORBIDITY_NAMES if f"tv_{c}" in cp_data.columns]
    fit_cols = ["drug_class_num", "age_at_index", "cci"] + tv_cols

    # Drop near-zero-variance covariates to prevent convergence failures
    low_var_tv = [c for c in fit_cols if c in cp_data.columns and cp_data[c].var() < 0.001]
    if low_var_tv:
        log.warning("Dropping low-variance TV covariates: %s", low_var_tv)
        fit_cols = [c for c in fit_cols if c not in low_var_tv]

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    try:
        ctv.fit(
            cp_data,
            id_col       = "id",
            start_col    = "tstart",
            stop_col     = "tstop",
            event_col    = "event",
            formula      = " + ".join(fit_cols),
        )
        results = ctv.summary.reset_index()
        results.to_csv(f"{output_dir}/tables/cox_timevarying_results.csv", index=False)
        log.info("Time-varying Cox results saved")
        log.info("\n%s", ctv.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())
    except Exception as e:
        log.error("Time-varying Cox failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",        default="data/omop/omop.duckdb")
    parser.add_argument("--matched-cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--output-dir",     default="outputs")
    args = parser.parse_args()
    run_cox_timevarying(args.db_path, args.matched_cohort, args.output_dir)


if __name__ == "__main__":
    main()
