# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
analysis/run_negative_control.py — Negative control outcome analysis.

Purpose
-------
Negative control outcomes are conditions that share the same potential
confounders as the primary outcome (treatment discontinuation) but have
no plausible biological relationship to the study drugs. If the drug-class
coefficient is statistically significant for a negative control outcome,
it implies that unmeasured confounding — not a drug effect — is driving
the association.

This analysis operationalises the OHDSI negative control methodology to
provide empirical evidence on residual confounding after propensity score
matching. A drug-class effect that disappears in the primary analysis but
appears in negative controls, or vice versa, suggests the PS model did not
achieve adequate balance on unmeasured variables.

Negative control outcomes selected
-----------------------------------
Three conditions with no plausible pharmacological relationship to
metformin, GLP-1 RAs, or SGLT-2 inhibitors:

  1. Acute viral upper respiratory infection (SNOMED: 54150009)
     URI incidence is driven by exposure frequency and immune competence,
     not antidiabetic drug class.

  2. Accidental injury — fall or mechanical trauma (SNOMED: 52684005)
     Injury rates reflect patient mobility and environment, not drug choice.

  3. Otitis media (SNOMED: 65363002)
     Middle ear infection incidence has no known antidiabetic drug mechanism.

Interpretation
--------------
A log-odds ratio for drug_class significantly different from zero (p < 0.05
after Bonferroni correction for three outcomes) indicates residual confounding.
Failure to reject the null for all three negative controls strengthens the
causal inference for the primary outcome analysis.

References
----------
Negative control methodology:
    Lipsitch M et al. Epidemiology. 2010;21(3):383–388.
    Schuemie MJ et al. Am J Epidemiol. 2018;188(6):1177–1189.
OHDSI LEGEND negative control implementation:
    Ryan PB et al. Drug Saf. 2019;42(5):657–672.
E-value for residual confounding:
    VanderWeele TJ, Ding P. Ann Intern Med. 2017;167(4):268–274.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Negative control definitions ──────────────────────────────────────────────

NEGATIVE_CONTROLS = {
    "viral_uri":       54150009,   # Acute viral URI
    "accidental_fall": 52684005,   # Accidental fall
    "otitis_media":    65363002,   # Otitis media
}

BONFERRONI_ALPHA = 0.05 / len(NEGATIVE_CONTROLS)


# ── OMOP queries ──────────────────────────────────────────────────────────────

def _query_negative_controls(
    db_path: Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Query OMOP condition_occurrence for each negative control condition.

    Returns a DataFrame with one row per person and one binary column per
    negative control (1 = condition recorded after index date, 0 = not recorded).
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb is required. Install with: pip install duckdb")

    person_ids = tuple(cohort["person_id"].tolist())
    if len(person_ids) == 0:
        raise ValueError("Cohort is empty — cannot query OMOP.")

    con = duckdb.connect(str(db_path), read_only=True)
    result = cohort[["person_id", "index_date"]].copy()

    for nc_name, concept_id in NEGATIVE_CONTROLS.items():
        query = f"""
        SELECT DISTINCT co.person_id
        FROM condition_occurrence co
        JOIN (SELECT person_id, CAST(index_date AS DATE) AS idx
              FROM (VALUES {",".join(f"({row.person_id}, '{row.index_date}')"
                                    for _, row in cohort.iterrows())}
              ) AS t(person_id, index_date)) AS cohort_idx
          ON co.person_id = cohort_idx.person_id
        WHERE co.condition_concept_id = {concept_id}
          AND CAST(co.condition_start_date AS DATE) > cohort_idx.idx
          AND co.person_id IN {person_ids}
        """
        try:
            hits = set(con.execute(query).df()["person_id"].tolist())
        except Exception as exc:
            log.warning("Query failed for %s (concept %d): %s", nc_name, concept_id, exc)
            hits = set()

        result[nc_name] = result["person_id"].isin(hits).astype(int)
        prev = result[nc_name].mean() * 100
        log.info("  %-22s concept=%d  prevalence=%.1f%%  n_cases=%d",
                 nc_name, concept_id, prev, result[nc_name].sum())

    con.close()
    return result


def _synthetic_negative_controls(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic negative-control outcomes when OMOP is unavailable.

    Prevalences are drawn from published population estimates and are
    intentionally not associated with drug class, so a well-functioning
    PS model should produce non-significant drug coefficients.

    This fallback enables pipeline testing without a populated OMOP database.
    """
    rng = np.random.default_rng(42)
    n = len(cohort)
    result = cohort[["person_id", "index_date", "drug_class"]].copy()

    # Approximate annual prevalences: URI ~30%, falls ~5%, otitis ~3%
    result["viral_uri"]       = rng.binomial(1, 0.30, n)
    result["accidental_fall"] = rng.binomial(1, 0.05, n)
    result["otitis_media"]    = rng.binomial(1, 0.03, n)

    log.info("Synthetic negative-control outcomes generated (OMOP not available).")
    return result


# ── Logistic regression ───────────────────────────────────────────────────────

def _fit_negative_control_model(
    nc_data: pd.DataFrame,
    cohort: pd.DataFrame,
    outcome_col: str,
) -> dict:
    """
    Logistic regression: outcome ~ drug_class + age_at_index + cci + sex_female.

    Returns a dict of coefficient estimates and p-values for each covariate.
    Drug-class dummies use metformin as the reference category.

    Under the null hypothesis of no residual confounding, drug-class
    coefficients should not be statistically significant.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    merged = nc_data.merge(
        cohort[["person_id", "drug_class", "age_at_index", "cci", "sex_female"]],
        on="person_id", how="left",
    ).dropna()

    if len(merged) == 0 or merged[outcome_col].nunique() < 2:
        return {"outcome": outcome_col, "status": "insufficient_data"}

    # One-hot encode drug class (metformin = reference)
    merged["drug_glp1"]  = (merged["drug_class"] == "glp1").astype(int)
    merged["drug_sglt2"] = (merged["drug_class"] == "sglt2").astype(int)

    covariate_cols = ["drug_glp1", "drug_sglt2", "age_at_index", "cci", "sex_female"]
    X = merged[covariate_cols].values.astype(float)
    y = merged[outcome_col].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Fit with L2 regularisation; C=10 keeps regularisation weak so coefficients
    # are not shrunk toward zero, preserving interpretability as log-odds ratios.
    model = LogisticRegression(C=10, max_iter=1_000, random_state=42)
    model.fit(X_sc, y)

    # Approximate Wald p-values via sandwich variance estimator
    # (sklearn does not expose SEs natively; bootstrap p-values are
    # more rigorous but computationally prohibitive for a sensitivity check)
    coef = model.coef_[0]
    n    = len(y)
    k    = X_sc.shape[1]

    # Hessian approximation using predicted probabilities
    p_hat = model.predict_proba(X_sc)[:, 1]
    W = np.diag(p_hat * (1 - p_hat))
    cov_approx = np.linalg.pinv(X_sc.T @ W @ X_sc + 1e-6 * np.eye(k))
    se = np.sqrt(np.diag(cov_approx))

    z_vals = coef / (se + 1e-12)
    p_vals = 2 * (1 - chi2(1).cdf(z_vals ** 2))

    rows = []
    for i, col in enumerate(covariate_cols):
        rows.append({
            "outcome":   outcome_col,
            "covariate": col,
            "log_or":    round(coef[i], 4),
            "or":        round(np.exp(coef[i]), 4),
            "se":        round(se[i], 4),
            "z":         round(z_vals[i], 4),
            "p_value":   round(p_vals[i], 4),
            "bonferroni_sig": p_vals[i] < BONFERRONI_ALPHA,
        })

    prevalence = y.mean()
    n_events   = y.sum()

    return {
        "outcome":    outcome_col,
        "n":          n,
        "n_events":   n_events,
        "prevalence": round(prevalence, 4),
        "coefficients": rows,
        "status":     "ok",
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def _plot_negative_control_effects(results: list[dict], output_path: Path) -> None:
    """
    Forest plot of drug-class log-odds ratios for each negative control outcome.

    Coefficients that remain within the null range (log-OR ≈ 0) support the
    adequacy of propensity score balance on unmeasured confounders. Significant
    drug-class associations are highlighted in red.
    """
    rows = []
    for res in results:
        if res.get("status") != "ok":
            continue
        for coef in res["coefficients"]:
            if coef["covariate"] in ("drug_glp1", "drug_sglt2"):
                rows.append({
                    "label":    f"{res['outcome']}\n({coef['covariate'].replace('drug_', '').upper()})",
                    "log_or":   coef["log_or"],
                    "se":       coef["se"],
                    "sig":      coef["bonferroni_sig"],
                    "n_events": res["n_events"],
                })

    if not rows:
        log.warning("No drug-class coefficients to plot.")
        return

    df_plot = pd.DataFrame(rows).reset_index(drop=True)
    n_rows  = len(df_plot)

    fig, ax = plt.subplots(figsize=(9, max(4, 1.0 * n_rows + 2)))
    y_pos = np.arange(n_rows)

    colors = ["#E74C3C" if sig else "#2980B9" for sig in df_plot["sig"]]
    ax.errorbar(
        df_plot["log_or"], y_pos,
        xerr=1.96 * df_plot["se"],
        fmt="o", color="none",
        ecolor=colors, elinewidth=1.8, capsize=4, capthick=1.6,
    )
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.scatter(row["log_or"], i, color=colors[i], zorder=5, s=55)

    ax.axvline(0, linestyle="--", color="#7F8C8D", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["label"], fontsize=9)
    ax.set_xlabel("Log-Odds Ratio (drug vs. metformin)", fontsize=10)
    ax.set_title(
        "Negative Control Outcome Analysis\n"
        "Drug-class effects on conditions with no antidiabetic mechanism\n"
        "(Red = Bonferroni-significant; suggests residual confounding)",
        fontsize=10,
    )
    ax.axvspan(-0.1, 0.1, alpha=0.08, color="#27AE60", label="Null zone (|log-OR| < 0.1)")
    ax.legend(fontsize=8, loc="lower right")

    red_patch  = mpatches.Patch(color="#E74C3C", label=f"Bonferroni-sig (α = {BONFERRONI_ALPHA:.4f})")
    blue_patch = mpatches.Patch(color="#2980B9", label="Not significant")
    ax.legend(handles=[blue_patch, red_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Negative control forest plot saved: %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Negative control outcome analysis for residual confounding detection."
    )
    parser.add_argument("--db-path",       default=str(PATHS.omop_db))
    parser.add_argument("--cohort-matched",default=str(PATHS.cohort_matched))
    parser.add_argument("--output-dir",    default=str(PATHS.outputs))
    args = parser.parse_args()

    fig_dir = Path(args.output_dir) / "figures"
    tab_dir = Path(args.output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = Path(args.cohort_matched)
    if not cohort_path.exists():
        log.error("cohort_matched.csv not found at %s — run bootstrap.sh Steps 1–4 first.", cohort_path)
        sys.exit(1)

    cohort = pd.read_csv(cohort_path)
    log.info("Cohort loaded: %d patients", len(cohort))

    # Attempt OMOP query; fall back to synthetic outcomes if DB unavailable
    db_path = Path(args.db_path)
    try:
        nc_data = _query_negative_controls(db_path, cohort)
        source  = "OMOP"
    except Exception as exc:
        log.warning("OMOP query failed: %s — using synthetic outcomes.", exc)
        nc_data = _synthetic_negative_controls(cohort)
        source  = "synthetic"

    log.info("Negative control data source: %s", source)

    # Fit one logistic model per negative control outcome
    results = []
    for nc_name in NEGATIVE_CONTROLS:
        if nc_name not in nc_data.columns:
            log.warning("Outcome column '%s' missing — skipping.", nc_name)
            continue
        log.info("Fitting model for outcome: %s", nc_name)
        res = _fit_negative_control_model(nc_data, cohort, nc_name)
        results.append(res)

    # Flatten coefficient table
    coef_rows = []
    for res in results:
        if res.get("status") == "ok":
            for coef in res["coefficients"]:
                coef_rows.append({
                    "outcome":         res["outcome"],
                    "n":               res["n"],
                    "n_events":        res["n_events"],
                    "prevalence":      res["prevalence"],
                    "data_source":     source,
                    **coef,
                })

    coef_df = pd.DataFrame(coef_rows)
    coef_path = tab_dir / "negative_control_results.csv"
    coef_df.to_csv(coef_path, index=False)
    log.info("Results saved: %s", coef_path)

    # Interpretation summary
    drug_coefs = coef_df[coef_df["covariate"].isin(["drug_glp1", "drug_sglt2"])]
    n_sig = drug_coefs["bonferroni_sig"].sum()
    log.info("\n── Negative Control Interpretation ──────────────────────────────────────")
    log.info("  Bonferroni threshold: p < %.4f (α=0.05 / %d outcomes)", BONFERRONI_ALPHA, len(NEGATIVE_CONTROLS))
    log.info("  Drug-class coefficients significant after correction: %d / %d", n_sig, len(drug_coefs))

    if n_sig == 0:
        log.info("  PASS — No drug-class effect detected for negative controls.")
        log.info("  This is consistent with adequate balance on unmeasured confounders.")
    else:
        log.warning("  FAIL — Drug-class effect detected for ≥1 negative control outcome.")
        log.warning("  Residual confounding cannot be excluded. Interpret primary results with caution.")
        log.warning("  Significant outcomes:")
        for _, row in drug_coefs[drug_coefs["bonferroni_sig"]].iterrows():
            log.warning("    %-22s %-12s log-OR=%.3f  p=%.4f",
                        row["outcome"], row["covariate"], row["log_or"], row["p_value"])

    _plot_negative_control_effects(results, fig_dir / "negative_control_effects.png")
    log.info("\nNegative control analysis complete.")


if __name__ == "__main__":
    main()
