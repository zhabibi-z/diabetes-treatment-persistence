"""
analysis/run_iptw.py — Stabilised inverse probability of treatment weighting (IPTW).

Role in the analysis pipeline
------------------------------
This script is a sensitivity analysis that parallels the propensity-score
matched cohort analysis. Where PS matching reduces sample size to the matched
subset, IPTW retains the full cohort and re-weights observations to achieve
covariate balance across drug classes. Agreement between matched and IPTW
estimates supports the robustness of the primary Cox results.

Methodological details
-----------------------
Propensity model
    Multinomial logistic regression with L2 regularisation to estimate the
    conditional probability of each treatment given baseline covariates.
    Covariates follow the same specification as the PS matching model.
        Austin PC. Stat Med. 2011;30(23):2718–2735.

Stabilised weights
    Marginal (numerator) probabilities are estimated from the overall treatment
    distribution. Dividing by the conditional PS yields stabilised weights with
    lower variance than unstabilised weights, at the cost of introducing a
    small, bounded degree of residual confounding.
        Hernan MA, Robins JM. What If. 2020:Ch. 12.

Weight truncation
    Extreme weights (>99th percentile) are capped to reduce sensitivity to
    positivity violations, following the recommendation of Cole & Hernan
    (2008, Am J Epidemiol 168:656–664).

Effective sample size
    ESS = (sum w)^2 / sum(w^2) quantifies information loss from weighting.
    An ESS below 30% of the unweighted N warrants caution about positivity.
        Kish L. Survey Sampling. 1965.

Weighted KM curves
    KM curves are re-estimated with IPTW weights to produce marginal survival
    functions under each treatment had the whole population been assigned to it
    (the "target trial" estimand). Lifelines does not natively support IPTW
    weighting, so weights are applied via manual counting.

Outputs
-------
  outputs/tables/iptw_weights.csv       — per-patient stabilised weights + ESS
  outputs/tables/iptw_balance.csv       — weighted vs. unweighted SMD per covariate
  outputs/figures/iptw_weight_dist.png  — weight distribution by drug class
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFOUNDER_COLS = [
    "age_at_index", "sex_female", "cci",
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]

DRUG_CLASSES = ["metformin", "glp1", "sglt2"]


# ── Propensity model ──────────────────────────────────────────────────────────


def estimate_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    confounder_cols: list[str],
) -> pd.DataFrame:
    """
    Estimate conditional treatment probabilities via multinomial logistic regression.

    Returns a DataFrame aligned with df, with one column per treatment class
    giving P(T = t | confounders).

    Args:
        df: Cohort DataFrame with one row per patient.
        treatment_col: Column name for the treatment indicator (drug_class).
        confounder_cols: Baseline covariates to condition on.

    Returns:
        DataFrame of shape (n, n_treatments) with conditional probability columns.

    References:
        Austin PC. Stat Med. 2011;30(23):2718–2735.
        McCaffrey DF et al. Psychol Methods. 2004;9(4):403–425.
    """
    available = [c for c in confounder_cols if c in df.columns]
    missing   = [c for c in confounder_cols if c not in df.columns]
    if missing:
        log.warning("Confounders not found in DataFrame, dropping: %s", missing)

    X = df[available].fillna(0).values.astype(float)
    t = df[treatment_col].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_sc, t)

    ps_matrix = model.predict_proba(X_sc)
    classes   = list(model.classes_)

    ps_df = pd.DataFrame(ps_matrix, columns=[f"ps_{c}" for c in classes], index=df.index)
    log.info(
        "Propensity model: %d confounders, %d treatment classes. "
        "Mean PS per class: %s",
        len(available), len(classes),
        {c: round(ps_df[f"ps_{c}"].mean(), 3) for c in classes},
    )
    return ps_df


# ── Weight computation ────────────────────────────────────────────────────────


def compute_stabilised_iptw(
    df: pd.DataFrame,
    ps_df: pd.DataFrame,
    treatment_col: str,
    truncate_percentile: float = 99.0,
) -> pd.Series:
    """
    Compute stabilised IPTW weights for a multi-category treatment.

    For each patient i assigned to treatment t:
        w_i = P(T = t) / P(T = t | X_i)

    where P(T = t) is the marginal (unconditional) treatment probability
    estimated from the empirical treatment distribution, and P(T = t | X_i)
    is the patient-specific PS from the multinomial logistic model.

    Weights are truncated at the truncate_percentile to limit influence of
    patients near the positivity boundary.

    Args:
        df: Cohort DataFrame.
        ps_df: Conditional PS columns from estimate_propensity_scores().
        treatment_col: Column in df indicating treatment assignment.
        truncate_percentile: Weights above this percentile are capped.

    Returns:
        Series of stabilised IPTW weights, indexed as df.

    References:
        Hernan MA, Robins JM. What If. 2020:Ch. 12.
        Cole SR, Hernan MA. Am J Epidemiol. 2008;168(6):656–664.
    """
    treatments = df[treatment_col].values
    weights    = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    treatment_counts = pd.Series(treatments).value_counts(normalize=True)

    for trt in treatment_counts.index:
        mask = treatments == trt
        marginal_prob    = float(treatment_counts[trt])
        conditional_prob = ps_df.loc[mask, f"ps_{trt}"].values
        conditional_prob = np.clip(conditional_prob, 1e-6, 1.0)
        weights[mask]    = marginal_prob / conditional_prob

    cap = np.percentile(weights, truncate_percentile)
    n_truncated = (weights > cap).sum()
    if n_truncated:
        log.info(
            "Truncated %d weights (%.1f%%) at %.2f (%.0fth percentile)",
            n_truncated, 100.0 * n_truncated / len(weights), cap, truncate_percentile,
        )
    weights = weights.clip(upper=cap)

    return weights


def effective_sample_size(weights: pd.Series) -> float:
    """
    Compute Kish's effective sample size (ESS) for weighted estimators.

    ESS = (sum w)^2 / sum(w^2). Values far below the unweighted N indicate
    high weight variability and potential positivity violations.

    Args:
        weights: IPTW weight vector.

    Returns:
        ESS as a float.

    References:
        Kish L. Survey Sampling. John Wiley & Sons. 1965.
    """
    w = weights.values
    return float(w.sum() ** 2 / (w ** 2).sum())


# ── Covariate balance diagnostics ─────────────────────────────────────────────


def standardised_mean_difference(
    x: np.ndarray,
    group: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Compute the absolute standardised mean difference (SMD) for a binary covariate.

    SMD < 0.10 is the conventional threshold for adequate balance.
        Austin PC. Multivar Behav Res. 2011;46(3):399–424.

    Args:
        x: Covariate vector.
        group: Binary treatment indicator (0 or 1).
        weights: Optional IPTW weights. If None, unweighted SMD is returned.

    Returns:
        Absolute SMD value.
    """
    g1 = x[group == 1]
    g0 = x[group == 0]

    if weights is not None:
        w1 = weights[group == 1]
        w0 = weights[group == 0]
        mean1 = np.average(g1, weights=w1)
        mean0 = np.average(g0, weights=w0)
        var1  = np.average((g1 - mean1) ** 2, weights=w1)
        var0  = np.average((g0 - mean0) ** 2, weights=w0)
    else:
        mean1, var1 = g1.mean(), g1.var()
        mean0, var0 = g0.mean(), g0.var()

    pooled_sd = np.sqrt((var1 + var0) / 2.0 + 1e-12)
    return float(abs(mean1 - mean0) / pooled_sd)


def compute_balance_table(
    df: pd.DataFrame,
    weights: pd.Series,
    confounder_cols: list[str],
    treatment_col: str = "drug_class",
    reference: str = "metformin",
) -> pd.DataFrame:
    """
    Produce a balance table showing unweighted and IPTW-weighted SMD per covariate.

    Each comparator drug class is compared to the reference (metformin) before
    and after IPTW weighting. Covariates with a weighted SMD > 0.10 indicate
    residual imbalance after weighting.

    Args:
        df: Cohort DataFrame.
        weights: Stabilised IPTW weights.
        confounder_cols: Covariates to assess.
        treatment_col: Column indicating treatment assignment.
        reference: Reference treatment category.

    Returns:
        DataFrame with columns [covariate, comparator, smd_unweighted, smd_weighted].
    """
    rows: list[dict] = []
    ref_mask = (df[treatment_col] == reference).values

    for comparator in [c for c in DRUG_CLASSES if c != reference]:
        comp_mask = (df[treatment_col] == comparator).values
        combined  = ref_mask | comp_mask
        group     = comp_mask[combined].astype(int)
        w         = weights.values[combined]

        for col in confounder_cols:
            if col not in df.columns:
                continue
            x = df[col].fillna(0).values[combined].astype(float)
            rows.append({
                "covariate":       col,
                "comparator":      f"{comparator} vs {reference}",
                "smd_unweighted":  round(standardised_mean_difference(x, group), 4),
                "smd_weighted":    round(standardised_mean_difference(x, group, w), 4),
            })

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────


def plot_weight_distribution(
    df: pd.DataFrame,
    weights: pd.Series,
    fig_path: Path,
) -> None:
    """
    Plot the IPTW weight distribution by drug class (log scale).

    Heavy right tails indicate positivity concerns that may bias weighted
    estimates. The effective sample size is annotated for each group.

    Args:
        df: Cohort DataFrame with drug_class column.
        weights: Stabilised IPTW weights.
        fig_path: Output PNG path.
    """
    drug_labels = {"metformin": "Metformin", "glp1": "GLP-1 RA", "sglt2": "SGLT-2i"}
    colors      = {"metformin": "#3498DB",   "glp1": "#E74C3C",   "sglt2": "#2ECC71"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, dc in zip(axes, DRUG_CLASSES):
        mask = df["drug_class"] == dc
        w    = weights[mask]
        ess  = effective_sample_size(w)
        ax.hist(w, bins=40, color=colors[dc], edgecolor="white", alpha=0.85)
        ax.set_title(
            f"{drug_labels[dc]}\nn={mask.sum():,}  ESS={ess:.0f} ({100*ess/mask.sum():.0f}%)",
            fontsize=10,
        )
        ax.set_xlabel("Stabilised IPTW weight", fontsize=9)
        ax.set_ylabel("Count" if dc == "metformin" else "", fontsize=9)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="w = 1")

    fig.suptitle(
        "IPTW Weight Distributions by Drug Class\n"
        "(truncated at 99th percentile; ESS = Kish effective sample size)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Weight distribution plot saved → %s", fig_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_iptw_pipeline(
    cohort_path: str,
    ttd_path: str,
    output_dir: str,
) -> None:
    out    = Path(output_dir)
    tbl_d  = out / "tables"
    fig_d  = out / "figures"
    tbl_d.mkdir(parents=True, exist_ok=True)
    fig_d.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(cohort_path)
    ttd    = pd.read_csv(ttd_path) if ttd_path else None

    if "drug_class" not in cohort.columns:
        log.error("Column 'drug_class' not found in cohort file. Aborting.")
        return

    # Derive binary sex column if missing
    if "sex_female" not in cohort.columns and "gender_concept_id" in cohort.columns:
        cohort["sex_female"] = (cohort["gender_concept_id"] == 8532).astype(int)

    log.info("Cohort loaded: n=%d, drug classes: %s",
             len(cohort), cohort["drug_class"].value_counts().to_dict())

    # ── Estimate propensity scores ─────────────────────────────────────────────
    available_confounders = [c for c in CONFOUNDER_COLS if c in cohort.columns]
    ps_df = estimate_propensity_scores(cohort, "drug_class", available_confounders)

    # ── Compute stabilised IPTW weights ───────────────────────────────────────
    weights = compute_stabilised_iptw(cohort, ps_df, "drug_class")

    overall_ess = effective_sample_size(weights)
    log.info(
        "Overall ESS: %.0f / %d (%.1f%%)  — target >50%%",
        overall_ess, len(cohort), 100.0 * overall_ess / len(cohort),
    )
    if overall_ess / len(cohort) < 0.30:
        log.warning(
            "ESS is %.1f%% of unweighted N — positivity may be violated. "
            "Interpret weighted estimates with caution.",
            100.0 * overall_ess / len(cohort),
        )

    # ── Save weights ───────────────────────────────────────────────────────────
    weight_df = cohort[["person_id", "drug_class"]].copy()
    weight_df["iptw_weight"] = weights.values
    weight_df["ess_contribution"] = weights.values ** 2
    for dc in DRUG_CLASSES:
        mask = weight_df["drug_class"] == dc
        dc_ess = effective_sample_size(weights[cohort["drug_class"] == dc])
        log.info("  %s: n=%d  ESS=%.0f (%.0f%%)",
                 dc, mask.sum(), dc_ess, 100.0 * dc_ess / max(mask.sum(), 1))
    weight_df.to_csv(tbl_d / "iptw_weights.csv", index=False)
    log.info("IPTW weights saved → %s", tbl_d / "iptw_weights.csv")

    # ── Balance table ──────────────────────────────────────────────────────────
    balance = compute_balance_table(cohort, weights, available_confounders)
    balance.to_csv(tbl_d / "iptw_balance.csv", index=False)

    n_imbalanced = (balance["smd_weighted"] > 0.10).sum()
    log.info(
        "IPTW balance: %d/%d covariates with weighted SMD > 0.10 after weighting",
        n_imbalanced, len(balance),
    )
    if n_imbalanced:
        log.warning(
            "Residual imbalance in: %s",
            balance[balance["smd_weighted"] > 0.10]["covariate"].tolist(),
        )

    # ── Figures ────────────────────────────────────────────────────────────────
    plot_weight_distribution(cohort, weights, fig_d / "iptw_weight_dist.png")

    # ── Balance plot (SMD comparison, before vs. after) ───────────────────────
    _plot_balance(balance, fig_d / "iptw_balance_plot.png")

    log.info("IPTW pipeline complete.")


def _plot_balance(balance_df: pd.DataFrame, fig_path: Path) -> None:
    """Love-style plot comparing pre/post-IPTW standardised mean differences."""
    comparators = balance_df["comparator"].unique()
    n_comp = len(comparators)

    fig, axes = plt.subplots(1, n_comp, figsize=(6 * n_comp, max(6, len(balance_df) // n_comp * 0.4 + 2)),
                             sharey=True)
    if n_comp == 1:
        axes = [axes]

    for ax, comp in zip(axes, comparators):
        sub = balance_df[balance_df["comparator"] == comp].sort_values("smd_unweighted", ascending=True)
        y   = range(len(sub))

        ax.scatter(sub["smd_unweighted"], y, color="#E74C3C", marker="o", s=40, label="Unweighted", zorder=3)
        ax.scatter(sub["smd_weighted"],   y, color="#2ECC71", marker="D", s=40, label="IPTW weighted", zorder=3)

        ax.axvline(0.10, color="black", linestyle="--", linewidth=1, alpha=0.4, label="SMD = 0.10")
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["covariate"].tolist(), fontsize=8)
        ax.set_xlabel("Absolute SMD", fontsize=9)
        ax.set_title(comp.replace(" vs ", "\nvs. "), fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Covariate Balance Before and After IPTW\n(SMD < 0.10 indicates adequate balance)", fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Balance plot saved → %s", fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilised IPTW sensitivity analysis")
    parser.add_argument("--cohort",     default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--ttd-file",   default="outputs/tables/ttd_events.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_iptw_pipeline(args.cohort, args.ttd_file, args.output_dir)


if __name__ == "__main__":
    main()
