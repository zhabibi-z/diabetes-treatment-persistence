"""
run_km_stratified.py — Per-comorbidity stratified Kaplan-Meier persistence curves.

For each of the 15 comorbidities: plots KM persistence (1 - discontinuation)
stratified by codx = 0 (absent at baseline) vs. codx = 1 (present at baseline).
Log-rank p-value reported on each plot.

Outputs:
  outputs/figures/km_stratified_{comorbidity}.png
  outputs/figures/km_stratified_grid.png
  outputs/tables/km_stratified_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COMORBIDITY_NAMES = [
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]


def run_km_stratified(matched_cohort: str, output_dir: str) -> None:
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(matched_cohort)

    # Load TTD events if available
    ttd_path = f"{output_dir}/tables/ttd_events.csv"
    if Path(ttd_path).exists():
        ttd = pd.read_csv(ttd_path)[["person_id", "ttd_days", "discontinued"]]
        df  = cohort.merge(ttd, on="person_id", how="inner")
    else:
        df = cohort.copy()
        df["ttd_days"]     = df.get("followup_days", pd.Series(365, index=df.index))
        df["discontinued"] = 0

    df = df.dropna(subset=["ttd_days"])

    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    axes = axes.flatten()

    summary_rows = []

    for idx, comorb in enumerate(COMORBIDITY_NAMES):
        ax = axes[idx]
        if comorb not in df.columns:
            ax.set_visible(False)
            continue

        grp0 = df[df[comorb] == 0]
        grp1 = df[df[comorb] == 1]

        if len(grp0) < 5 or len(grp1) < 5:
            ax.set_visible(False)
            continue

        kmf0 = KaplanMeierFitter()
        kmf1 = KaplanMeierFitter()
        kmf0.fit(grp0["ttd_days"], grp0["discontinued"], label=f"{comorb}=0 (n={len(grp0)})")
        kmf1.fit(grp1["ttd_days"], grp1["discontinued"], label=f"{comorb}=1 (n={len(grp1)})")

        kmf0.plot_survival_function(ax=ax, color="#2ECC71", ci_show=True, linewidth=1.8)
        kmf1.plot_survival_function(ax=ax, color="#E74C3C", ci_show=True, linewidth=1.8)

        # Log-rank test
        lr = logrank_test(
            grp0["ttd_days"], grp1["ttd_days"],
            grp0["discontinued"], grp1["discontinued"],
        )
        ax.set_title(
            f"{comorb.replace('_', ' ').title()}\np={lr.p_value:.3f}",
            fontsize=8, pad=3,
        )
        ax.set_xlabel("Days", fontsize=7)
        ax.set_ylabel("P(Persistent)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5)
        ax.set_ylim(0, 1.05)

        summary_rows.append({
            "comorbidity": comorb,
            "n_absent":    len(grp0),
            "n_present":   len(grp1),
            "median_ttd_absent":  kmf0.median_survival_time_,
            "median_ttd_present": kmf1.median_survival_time_,
            "logrank_p":   round(lr.p_value, 4),
        })

    for i in range(len(COMORBIDITY_NAMES), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Treatment Persistence by Comorbidity Status (codx=0 vs codx=1)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/km_stratified_grid.png", dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Stratified KM grid saved")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(f"{output_dir}/tables/km_stratified_summary.csv", index=False)
        log.info("km_stratified_summary.csv written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--output-dir",     default="outputs")
    args = parser.parse_args()
    run_km_stratified(args.matched_cohort, args.output_dir)


if __name__ == "__main__":
    main()
