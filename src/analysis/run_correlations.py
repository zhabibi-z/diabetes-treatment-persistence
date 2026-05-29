"""
run_correlations.py — Pearson correlation between each baseline comorbidity indicator
and observed time-to-discontinuation (TTD), with BH-FDR correction.

Note: correlations between binary predictors and continuous outcomes are
point-biserial r coefficients (mathematically equivalent to Pearson r).
p-values derived from the t-statistic with n−2 degrees of freedom.

Outputs:
  outputs/tables/correlations.csv
  outputs/figures/correlation_heatmap.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COMORBIDITY_NAMES = [
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction (returns adjusted p-values)."""
    n = len(p_values)
    order = np.argsort(p_values)
    p_adj = np.zeros(n)
    for rank, idx in enumerate(order):
        p_adj[idx] = p_values[idx] * n / (rank + 1)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return list(np.clip(p_adj, 0, 1))


def run_correlations(matched_cohort: str, ttd_file: str, output_dir: str) -> None:
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(matched_cohort)

    # Load TTD if separate file exists; otherwise fall back to cohort followup_days
    if Path(ttd_file).exists():
        ttd = pd.read_csv(ttd_file)[["person_id", "ttd_days", "discontinued"]]
        df  = cohort.merge(ttd, on="person_id", how="inner")
    else:
        df = cohort.copy()
        df["ttd_days"] = df.get("followup_days", pd.Series(365, index=df.index))

    df = df.dropna(subset=["ttd_days"])
    ttd_vals = df["ttd_days"].values.astype(float)

    results = []
    for comorb in COMORBIDITY_NAMES:
        if comorb not in df.columns:
            continue
        comorb_vals = df[comorb].values.astype(float)
        if np.std(comorb_vals) == 0:
            continue

        r, p = stats.pearsonr(comorb_vals, ttd_vals)
        n = len(df)
        results.append({
            "comorbidity": comorb,
            "pearson_r": round(r, 4),
            "p_value": p,
            "n": n,
        })

    if not results:
        log.warning("No correlations computed")
        return

    results_df = pd.DataFrame(results)
    results_df["p_adj_bh"] = bh_fdr(results_df["p_value"].tolist())
    results_df["significant_bh"] = results_df["p_adj_bh"] < 0.05
    results_df = results_df.sort_values("pearson_r")
    results_df.to_csv(f"{output_dir}/tables/correlations.csv", index=False)
    log.info("correlations.csv: %d rows", len(results_df))

    # ── Correlation bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E74C3C" if r < 0 else "#2ECC71" for r in results_df["pearson_r"]]
    bars = ax.barh(
        results_df["comorbidity"].str.replace("_", " ").str.title(),
        results_df["pearson_r"],
        color=colors, alpha=0.8, edgecolor="white",
    )
    # Mark significant after BH-FDR
    for bar, sig in zip(bars, results_df["significant_bh"]):
        if sig:
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    "*", va="center", fontsize=12, color="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r (comorbidity vs TTD)", fontsize=11)
    ax.set_title("Pearson Correlation: Comorbidity × Time-to-Discontinuation\n(* = significant after BH-FDR, α=0.05)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/correlation_heatmap.png", dpi=150)
    plt.close()
    log.info("Correlation bar chart saved")

    # Print top correlates
    log.info("\nTop comorbidity correlates with TTD:\n%s",
             results_df[["comorbidity", "pearson_r", "p_value", "p_adj_bh"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-cohort", default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--ttd-file",       default="outputs/tables/ttd_events.csv")
    parser.add_argument("--output-dir",     default="outputs")
    args = parser.parse_args()
    run_correlations(args.matched_cohort, args.ttd_file, args.output_dir)


if __name__ == "__main__":
    main()
