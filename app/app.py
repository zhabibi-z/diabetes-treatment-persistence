"""
app/app.py — T2DM Persistence RWE interactive dashboard (v2.0).

6 tabs:
  1. Overview     — study background, citations, data flow diagram
  2. Cohort       — paginated baseline characteristics, comorbidity prevalence, PS matching
  3. Survival     — KM curves, Cox HR tables, forest plot, stratified KM
  4. ML           — XGBoost CV performance, SHAP, UMAP
  5. Graph        — interactive streamlit-agraph knowledge graph + Cypher queries
  6. Chatbot      — LangChain + Groq (Llama 3.3 70B) Q&A

Run: streamlit run app/app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

st.set_page_config(
    page_title="T2DM Persistence RWE",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_csv_cached(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(ttl=3600)
def load_csv_columns(path: str, cols: list[str]) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p, usecols=lambda c: c in cols)
        return df
    return None


def load_csv(path: str, default_cols: list[str] | None = None) -> pd.DataFrame | None:
    return load_csv_cached(path)


def show_image(path: str, caption: str = "", width: int | None = None) -> None:
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_column_width=(width is None))
    else:
        st.info(f"Figure not yet generated: `{path}`\nRun `bash scripts/bootstrap.sh` to produce outputs.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("T2DM Persistence RWE")
    st.markdown("**Investigator:** Zia Habibi")
    st.markdown("**Data:** 30,000 synthetic patients, OMOP CDM v5.4")
    st.markdown("**Version:** v2.0 (30K scale)")
    st.divider()
    st.markdown("**Key References**")
    st.markdown("- Lim 2025 (90-day grace)")
    st.markdown("- Iskandar 2018 (Cox PH)")
    st.markdown("- Marcellusi 2019 (cohort design)")
    st.markdown("- Schneeweiss 2007 (new-user design)")
    st.markdown("- OHDSI LegendT2dm")
    st.divider()
    st.caption("Synthetic data only — no real PHI")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview", "👥 Cohort", "📈 Survival", "🤖 ML", "🕸️ Graph", "💬 Chatbot"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Comparative Treatment Persistence in Type 2 Diabetes")
    st.subheader("Metformin vs GLP-1 RA vs SGLT-2i — Real-World Evidence Study (v2.0)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Study Overview
        This real-world evidence (RWE) study characterises **comparative treatment persistence**
        across three major antidiabetic drug classes in T2DM, using 30,000 Synthea-generated
        synthetic patients structured in **OMOP CDM v5.4**.

        **Primary outcome:** Time-to-discontinuation (TTD) with a **90-day grace period** (Lim 2025).

        **15 comorbidities** are tracked as time-varying covariates (SNOMED-mapped).
        A **1:5 propensity-score matched** active-comparator new-user design is used,
        following the OHDSI LegendT2dm protocol (Schneeweiss 2007, Lund 2015).

        ### Study Design
        | Element | Design |
        |---------|--------|
        | Design | Active-comparator new-user cohort |
        | Scale | 30,000 synthetic T2DM patients |
        | Matching | 1:5 PS matching (MatchIt, cobalt) |
        | Primary outcome | TTD (90-day grace, Lim 2025) |
        | Secondary outcome | Time-to-comorbidity (TTC) |
        | Statistical methods | Cox PH, KM, time-varying Cox |
        | ML | XGBoost + SHAP (1-year discontinuation) |
        | Graph | Interactive NetworkX (streamlit-agraph) |
        | Chatbot | LangChain + Groq (Llama 3.3 70B) + RAG |

        ### Pipeline Architecture
        ```
        Synthetic Data Generator (30,000 T2DM patients)
            ↓
        ETL → OMOP CDM DuckDB
            ↓
        3 Mutually Exclusive New-User Cohorts
        (Metformin: ~17,928 | GLP-1 RA: ~5,955 | SGLT-2i: ~6,117)
            ↓
        1:5 PS Matching (R: MatchIt, cobalt)
            ├── TTD Analysis (90-day grace)
            ├── TTC Kaplan-Meier
            ├── Time-Varying Cox
            ├── TTC Cox + Forest Plot
            ├── Pearson Correlations
            ├── XGBoost + SHAP + UMAP
            └── Knowledge Graph (interactive)
            ↓
        Streamlit Dashboard
        ```

        ### Notes on ML Performance
        The previous v1.0 model on 5K patients reported AUC=0.961, which was a
        synthetic-data artifact at small scale (near-perfect class separation with
        simple lognormal TTD). The v2.0 model on 30K patients reports a more realistic
        AUC in the 0.70–0.85 range, consistent with real-world clinical prediction tasks.
        """)

    with col2:
        st.markdown("### Drug Classes")
        st.info("**Metformin** (Reference)\nFirst-line standard of care\nRxNorm: 1503297–1503301")
        st.error("**GLP-1 RA**\nSemaglutide, dulaglutide, liraglutide\nCV + weight benefit")
        st.success("**SGLT-2i**\nEmpagliflozin, dapagliflozin, canagliflozin\nCV + renal benefit")

        st.markdown("### Key Definitions")
        st.markdown("""
        - **Grace period**: 90 days (Lim 2025)
        - **Washout**: 365 days prior to index
        - **Minimum follow-up**: 90 days
        - **Cohort**: Mutually exclusive new-user
        - **New-user design**: Schneeweiss 2007
        """)

        # Quick metrics from cohort summary
        summary = load_csv("outputs/tables/cohort_summary.csv")
        if summary is not None:
            st.markdown("### Cohort Sizes")
            for _, row in summary.iterrows():
                st.metric(row["drug_class"].upper(), f"n={int(row['n']):,}")

    st.divider()
    st.markdown("### References")
    refs = [
        "Iskandar IYK et al. *Br J Dermatol* 2018;178(5):1083–1094 — Drug survival methodology",
        "Marcellusi A et al. *BMJ Open* 2019;9:e024596 — T2DM persistence, Italy",
        "Lim LL et al. *Diabetologia* 2025;68(3):412–427 — 90-day grace period validation",
        "Schneeweiss S et al. *Pharmacoepidemiol Drug Saf* 2007;16(5):565–570 — New-user design",
        "Lund JL et al. *Pharmacoepidemiol Drug Saf* 2015;24(10):1078–1086 — Active comparator",
        "OHDSI LegendT2dm — Active-comparator new-user protocol",
        "ADA Standards of Care 2024 — *Diabetes Care* 47(Suppl 1)",
    ]
    for ref in refs:
        st.markdown(f"- {ref}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COHORT (paginated, memory-optimised)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Cohort Characteristics")

    col1, col2 = st.columns(2)
    with col1:
        summary = load_csv("outputs/tables/cohort_summary.csv")
        if summary is not None:
            st.subheader("Cohort Summary")
            st.dataframe(summary.style.format(precision=2), use_container_width=True)
            for _, row in summary.iterrows():
                st.metric(
                    label=row["drug_class"].upper(),
                    value=f"n={int(row['n']):,}",
                )
        else:
            st.info("Cohort summary not yet generated. Run `bash scripts/bootstrap.sh`.")

    with col2:
        prev = load_csv("outputs/tables/comorbidity_prevalence.csv")
        if prev is not None:
            st.subheader("Comorbidity Prevalence (%)")
            pivot = prev.pivot(index="comorbidity", columns="drug_class", values="prevalence_pct")
            st.dataframe(pivot.style.background_gradient(cmap="YlOrRd").format("{:.1f}"),
                         use_container_width=True)
        else:
            st.info("Comorbidity prevalence not yet generated.")

    st.divider()
    st.subheader("Propensity Score Matching Diagnostics")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_image("outputs/figures/ps_distribution.png", "Propensity Score Distribution")
    with col2:
        show_image("outputs/figures/love_plot_glp1_vs_met.png", "Love Plot: GLP-1 RA vs Metformin")
    with col3:
        show_image("outputs/figures/love_plot_sglt2_vs_met.png", "Love Plot: SGLT-2i vs Metformin")

    # Paginated cohort explorer (max 1000 rows at a time)
    st.divider()
    st.subheader("Cohort Explorer (paginated — max 1,000 rows per page)")

    baseline_path = Path("outputs/tables/cohort_baseline.csv")
    if baseline_path.exists():
        display_cols = ["person_id", "drug_class", "age_at_index", "gender_concept_id",
                        "cci", "followup_days", "hypertension", "obesity", "ckd",
                        "heart_failure", "hyperlipidemia", "depression"]

        @st.cache_data(ttl=3600)
        def load_baseline_page(page: int, page_size: int = 1000) -> pd.DataFrame:
            skiprows = range(1, page * page_size + 1)
            df = pd.read_csv(baseline_path,
                             usecols=lambda c: c in display_cols,
                             skiprows=skiprows if page > 0 else None,
                             nrows=page_size)
            return df

        total_rows = sum(1 for _ in open(baseline_path)) - 1
        page_size = 1000
        n_pages = (total_rows + page_size - 1) // page_size
        page = st.number_input("Page", min_value=0, max_value=n_pages - 1, value=0, step=1)

        drug_filter = st.multiselect("Filter by drug class", ["metformin", "glp1", "sglt2"],
                                      default=["metformin", "glp1", "sglt2"])

        @st.cache_data(ttl=3600)
        def load_cohort_filtered(drug_classes: tuple) -> pd.DataFrame:
            df = pd.read_csv(baseline_path, usecols=lambda c: c in display_cols)
            return df[df["drug_class"].isin(drug_classes)]

        filtered = load_cohort_filtered(tuple(drug_filter))
        start = page * page_size
        end = min(start + page_size, len(filtered))
        st.dataframe(filtered.iloc[start:end], use_container_width=True)
        st.caption(f"Showing rows {start+1}–{end} of {len(filtered):,} filtered patients")
    else:
        st.info("Cohort baseline not yet generated.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SURVIVAL (only needed columns loaded)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Survival Analysis")

    st.subheader("Treatment Persistence — Kaplan-Meier")
    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/figures/km_ttd_by_class.png", "KM Persistence (Python/lifelines)")
    with col2:
        show_image("outputs/figures/km_persistence_survminer.png", "KM Persistence (R/survminer)")

    ttd_sum = load_csv_columns("outputs/tables/ttd_summary.csv",
                                ["drug_class", "median_ttd", "mean_ttd", "n", "n_discontinued"])
    if ttd_sum is not None:
        st.subheader("TTD Summary Statistics")
        st.dataframe(ttd_sum.style.format(precision=1), use_container_width=True)

    st.divider()
    st.subheader("Cox Proportional Hazards — Treatment Discontinuation")
    col1, col2 = st.columns([1, 1])
    with col1:
        cox = load_csv("outputs/tables/cox_ttd_results.csv")
        if cox is not None:
            st.dataframe(cox.style.format(precision=4), use_container_width=True)
    with col2:
        show_image("outputs/figures/forest_cox_ttd.png", "Forest Plot — Cox HR (TTD)")

    st.divider()
    st.subheader("Time-Varying Cox (Comorbidity 0→1 Transitions)")
    tv_cox = load_csv("outputs/tables/cox_timevarying_results.csv")
    if tv_cox is not None:
        st.dataframe(tv_cox.style.format(precision=4), use_container_width=True)
    else:
        st.info("Time-varying Cox results pending.")

    st.divider()
    st.subheader("Per-Comorbidity Stratified Kaplan-Meier (codx=0 vs codx=1)")
    show_image("outputs/figures/km_stratified_grid.png", "Stratified KM Grid (15 comorbidities)")

    km_sum = load_csv("outputs/tables/km_stratified_summary.csv")
    if km_sum is not None:
        st.dataframe(km_sum.style.format(precision=4), use_container_width=True)

    st.divider()
    st.subheader("Pearson Correlations: Comorbidity × TTD")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_image("outputs/figures/correlation_heatmap.png")
    with col2:
        corr = load_csv("outputs/tables/correlations.csv")
        if corr is not None:
            st.dataframe(
                corr.sort_values("pearson_r")
                .style.format({"pearson_r": "{:.4f}", "p_value": "{:.4f}", "p_adj_bh": "{:.4f}"}),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Schoenfeld Residuals (PH Assumption)")
    show_image("outputs/figures/schoenfeld_residuals.png", "Schoenfeld Residuals — Cox PH Test")

    st.divider()
    st.subheader("Time-to-Comorbidity (TTC) KM Grid")
    show_image("outputs/figures/km_ttc_grid.png", "TTC KM Grid by Drug Class")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML (lightweight inference)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Machine Learning — XGBoost + SHAP + UMAP")
    st.markdown(
        "**Outcome:** 1-year treatment discontinuation (binary)  "
        "**Model:** XGBoost, 5-fold stratified CV, 28 features  \n"
        "**Leakage correction:** `followup_days` removed from feature set (v2.1). "
        "Cohort restricted to patients with ≥365-day follow-up so every patient "
        "had a full observation window for the annual outcome. "
        "Expected AUROC is now ~0.65–0.78, consistent with published T2DM persistence models "
        "(Buysman et al. 2015; Pantalone et al. 2020)."
    )

    cv_res = load_csv("outputs/tables/ml_metrics.csv")
    if cv_res is not None:
        st.subheader("5-Fold Cross-Validation Results")
        st.dataframe(cv_res.style.format(precision=4), use_container_width=True)
        mean_row = cv_res[cv_res["split"] == "mean"]
        if not mean_row.empty:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("AUROC", f"{float(mean_row['auc'].iloc[0]):.3f}")
            col2.metric("95% CI", (
                f"{float(mean_row['auroc_ci_low'].iloc[0]):.3f}–"
                f"{float(mean_row['auroc_ci_high'].iloc[0]):.3f}"
                if "auroc_ci_low" in mean_row.columns else "—"
            ))
            col3.metric("F1", f"{float(mean_row['f1'].iloc[0]):.3f}")
            col4.metric("Brier", f"{float(mean_row['brier'].iloc[0]):.4f}")
            col5.metric("ECE", (
                f"{float(mean_row['ece'].iloc[0]):.4f}"
                if "ece" in mean_row.columns else "—"
            ))
            col6.metric("AUPRC", f"{float(mean_row['auprc'].iloc[0]):.3f}")
    else:
        st.info("ML results pending — run `bash scripts/bootstrap.sh`.")

    # ── Baseline model comparison ──────────────────────────────────────────────
    st.divider()
    st.subheader("Model Comparison — XGBoost vs. Baselines")
    st.caption(
        "XGBoost must outperform logistic regression and majority-class baselines to "
        "justify its complexity (Steyerberg et al. 2010, Ann Intern Med)."
    )
    comp = load_csv("outputs/tables/model_comparison.csv")
    if comp is not None:
        st.dataframe(
            comp.style.highlight_max(subset=["Mean AUROC"], color="#d4f0d4")
                      .highlight_min(subset=["Mean Brier"], color="#d4f0d4"),
            use_container_width=True,
        )
    else:
        st.info("Model comparison pending — re-run `python ml/train.py`.")

    # ── Calibration curve ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Calibration Curve (Reliability Diagram)")
    st.caption(
        "A perfectly calibrated model traces the diagonal. "
        "ECE = expected calibration error (lower is better). "
        "Van Calster et al. 2019, Stat Med."
    )
    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/figures/calibration_curve.png", "Reliability diagram with ECE annotation")
    with col2:
        show_image("outputs/figures/roc_curve.png", "ROC curve with bootstrap 95% CI")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SHAP Feature Importance (Beeswarm)")
        show_image("outputs/figures/shap_beeswarm.png")
    with col2:
        st.subheader("SHAP Waterfall — Example Patients")
        show_image("outputs/figures/shap_force_examples.png")

    # ── Fairness report ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Subgroup Fairness — Per-Group AUROC")
    st.caption(
        "AUC gaps >5 percentage points across sex or age band are flagged. "
        "Hardt et al. 2016 (NeurIPS); FDA AI/ML-SaMD Action Plan 2021."
    )
    fair_df = load_csv("outputs/tables/fairness_report.csv")
    if fair_df is not None:
        for axis in fair_df["axis"].unique():
            sub = fair_df[fair_df["axis"] == axis].drop(columns=["axis"])
            st.markdown(f"**{axis.replace('_', ' ').title()}**")
            styled = sub.style.applymap(
                lambda v: "background-color: #ffe0e0" if v is True else "",
                subset=["flagged"],
            ).format({"auc": "{:.3f}", "auc_gap": "{:+.3f}", "demographic_parity_diff": "{:.4f}"})
            st.dataframe(styled, use_container_width=True)
    else:
        st.info("Fairness report pending — re-run `python ml/train.py`.")

    st.divider()
    st.subheader("UMAP of XGBoost Leaf Embeddings")
    show_image("outputs/figures/umap_phenotypes.png", "UMAP coloured by drug class")

    st.divider()
    st.subheader("Interactive Prediction")
    st.markdown("Configure a hypothetical patient to get a discontinuation probability:")
    col1, col2, col3 = st.columns(3)
    with col1:
        drug_class = st.selectbox("Drug Class", ["metformin", "glp1", "sglt2"])
        age        = st.slider("Age at Index", 18, 90, 60)
        cci        = st.slider("Charlson Comorbidity Index", 0, 10, 2)
    with col2:
        hypertension = st.checkbox("Hypertension")
        ckd          = st.checkbox("CKD")
        heart_failure = st.checkbox("Heart Failure")
        depression    = st.checkbox("Depression")
    with col3:
        obesity      = st.checkbox("Obesity")
        stroke       = st.checkbox("Stroke/TIA")
        mi           = st.checkbox("Myocardial Infarction")
        nafld        = st.checkbox("NAFLD")

    if st.button("Predict Discontinuation Risk"):
        import numpy as np, xgboost as xgb_, pandas as _pd
        model_path = Path("outputs/models/xgb_model.ubj")
        if model_path.exists():
            @st.cache_resource
            def load_model():
                m = xgb_.XGBClassifier()
                m.load_model(str(model_path))
                return m
            model = load_model()
            # 28-feature set — followup_days excluded (leakage fix v2.1),
            # drug_class_num excluded (collinear with one-hot triplet),
            # sex_male excluded (collinear with sex_female).
            comorbidity_count = sum([
                int(hypertension), int(obesity), int(ckd),
                int(heart_failure), int(depression), int(nafld),
                int(stroke), int(mi),
            ])
            drug_met   = int(drug_class == "metformin")
            drug_glp1  = int(drug_class == "glp1")
            drug_sglt2 = int(drug_class == "sglt2")
            feature_vals = {
                # Demographics
                "age_at_index":        age,
                "age_over65":          int(age >= 65),
                "sex_female":          1,
                # Drug class (one-hot only)
                "drug_metformin":      drug_met,
                "drug_glp1":           drug_glp1,
                "drug_sglt2":          drug_sglt2,
                # Comorbidity burden
                "cci":                 cci,
                "comorbidity_count":   comorbidity_count,
                "days_since_t2dm_dx":  365,
                # Interactions
                "glp1_x_codx":         drug_glp1  * comorbidity_count,
                "sglt2_x_codx":        drug_sglt2 * comorbidity_count,
                "glp1_x_cci":          drug_glp1  * cci,
                "sglt2_x_cci":         drug_sglt2 * cci,
                # Comorbidity flags
                "hypertension":        int(hypertension),
                "obesity":             int(obesity),
                "ckd":                 int(ckd),
                "heart_failure":       int(heart_failure),
                "hyperlipidemia":      0,
                "nash":                0,
                "neuropathy":          0,
                "retinopathy":         0,
                "depression":          int(depression),
                "atrial_fibrillation": 0,
                "sleep_apnea":         0,
                "nafld":               int(nafld),
                "pvd":                 0,
                "stroke":              int(stroke),
                "mi":                  int(mi),
            }
            X = _pd.DataFrame([feature_vals])
            proba = model.predict_proba(X)[0][1]
            st.metric("Estimated 1-Year Discontinuation Probability", f"{proba:.1%}")
            if proba > 0.60:
                st.warning("High discontinuation risk — consider adherence support or closer follow-up.")
            elif proba > 0.40:
                st.info("Moderate discontinuation risk.")
            else:
                st.success("Lower discontinuation risk based on available features.")
            st.caption(
                "⚠ This prediction is based on synthetic Synthea data and has not been validated "
                "on real-world patient populations. Do not use for clinical decision-making."
            )
        else:
            st.error("Model not found — run `bash scripts/bootstrap.sh` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INTERACTIVE GRAPH (streamlit-agraph)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Knowledge Graph: Drug — Disease — Comorbidity")
    st.markdown(
        "Interactive graph encoding drug-class effects on comorbidities (TREATS), "
        "comorbidity associations with discontinuation (ASSOCIATED_WITH), "
        "and drug-class effects on TTD (DRUG_CLASS_EFFECT). "
        "**Click nodes** to see metadata. **Filter** by relationship type below."
    )

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        # ── Build graph data ──────────────────────────────────────────────────
        DRUG_COMORB_BENEFIT = {
            "glp1":      ["heart_failure", "ckd", "stroke", "mi", "obesity"],
            "sglt2":     ["heart_failure", "ckd", "pvd", "mi"],
            "metformin": ["obesity", "hyperlipidemia"],
        }
        COMORBIDITY_NAMES = [
            "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
            "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
            "sleep_apnea", "nafld", "pvd", "stroke", "mi",
        ]

        # Filter controls
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_treats = st.checkbox("Show TREATS edges (drug → comorbidity)", value=True)
            show_assoc  = st.checkbox("Show ASSOCIATED_WITH edges (comorbidity → outcome)", value=True)
        with col_f2:
            show_drug_effect = st.checkbox("Show DRUG_CLASS_EFFECT edges", value=True)
            highlight_type   = st.selectbox("Highlight node type",
                                             ["All", "DrugClass", "Comorbidity", "Outcome"])

        nodes: list[Node] = []
        edges: list[Edge] = []

        # Drug nodes (blue)
        drug_labels = {"metformin": "Metformin\n(Reference)", "glp1": "GLP-1 RA", "sglt2": "SGLT-2i"}
        for dc in ["metformin", "glp1", "sglt2"]:
            color = "#3498DB" if dc == "metformin" else ("#E74C3C" if dc == "glp1" else "#2ECC71")
            size = 30 if (highlight_type == "DrugClass" or highlight_type == "All") else 20
            nodes.append(Node(
                id=dc,
                label=drug_labels[dc],
                size=size,
                color=color,
                title=f"Type: DrugClass\nClass: {dc}\nRole: {'Reference' if dc=='metformin' else 'Comparator'}",
                font={"size": 14, "bold": True},
            ))

        # Comorbidity nodes (orange)
        corr_df = load_csv("outputs/tables/correlations.csv")
        corr_map = {}
        if corr_df is not None and "comorbidity" in corr_df.columns:
            corr_map = dict(zip(corr_df["comorbidity"], corr_df.get("pearson_r", [0]*len(corr_df))))

        for c in COMORBIDITY_NAMES:
            r = corr_map.get(c, 0.0)
            size = 20 if (highlight_type == "Comorbidity" or highlight_type == "All") else 15
            nodes.append(Node(
                id=c,
                label=c.replace("_", " ").title(),
                size=size,
                color="#F39C12",
                title=f"Type: Comorbidity\nName: {c.replace('_',' ').title()}\nPearson r with TTD: {r:.3f}",
            ))

        # Outcome node (purple)
        size = 35 if (highlight_type == "Outcome" or highlight_type == "All") else 25
        nodes.append(Node(
            id="discontinuation",
            label="Treatment\nDiscontinuation",
            size=size,
            color="#8E44AD",
            title="Type: Outcome\nDefinition: TTD (90-day grace period)\nReference: Lim 2025",
            font={"size": 14, "bold": True},
        ))

        # TREATS edges (green)
        if show_treats:
            for dc, comorbs in DRUG_COMORB_BENEFIT.items():
                for c in comorbs:
                    edges.append(Edge(
                        source=dc, target=c,
                        label="TREATS",
                        color="#27AE60",
                        width=2,
                        title="Relationship: TREATS (cardiorenal benefit, trial-based prior)",
                    ))

        # ASSOCIATED_WITH edges (red)
        if show_assoc:
            if corr_df is not None and "comorbidity" in corr_df.columns:
                for _, row in corr_df.iterrows():
                    c = row.get("comorbidity", "")
                    r = float(row.get("pearson_r", 0))
                    p = float(row.get("p_adj_bh", 1))
                    if c in COMORBIDITY_NAMES:
                        edges.append(Edge(
                            source=c, target="discontinuation",
                            label="ASSOC",
                            color="#E74C3C",
                            width=max(1, abs(r) * 5),
                            title=f"Relationship: ASSOCIATED_WITH\nPearson r: {r:.3f}\np-adj (BH): {p:.4f}",
                        ))
            else:
                for c in COMORBIDITY_NAMES:
                    edges.append(Edge(source=c, target="discontinuation",
                                      label="ASSOC", color="#E74C3C", width=1))

        # DRUG_CLASS_EFFECT edges (purple)
        if show_drug_effect:
            cox_ttd = load_csv("outputs/tables/cox_ttd_results.csv")
            for dc in ["glp1", "sglt2"]:
                hr_val = 1.0
                if cox_ttd is not None and "exp(coef)" in cox_ttd.columns:
                    dc_rows = cox_ttd[cox_ttd.get("covariate",
                                                    cox_ttd.columns[0]).isin([dc, f"drug_class_{dc}"])]
                    if not dc_rows.empty:
                        hr_val = float(dc_rows["exp(coef)"].iloc[0])
                edges.append(Edge(
                    source=dc, target="discontinuation",
                    label=f"HR={hr_val:.2f}",
                    color="#8E44AD",
                    width=2,
                    title=f"Relationship: DRUG_CLASS_EFFECT\nHR vs metformin: {hr_val:.3f}",
                ))

        config = Config(
            width=900,
            height=600,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#FFD700",
            collapsible=True,
            node={"labelProperty": "label"},
            link={"labelProperty": "label", "renderLabel": True},
        )

        st.info("**Tips:** Drag nodes to rearrange. Click a node to highlight its connections. Use checkboxes above to filter edge types.")
        agraph(nodes=nodes, edges=edges, config=config)

    except ImportError:
        st.warning("streamlit-agraph not installed. Falling back to static graph.")
        show_image("outputs/figures/knowledge_graph.png", "T2DM Knowledge Graph (NetworkX)")

    # Keep static PNG available
    st.divider()
    with st.expander("Static graph (for reference / README screenshot)"):
        show_image("outputs/figures/knowledge_graph.png", "T2DM Knowledge Graph (NetworkX)")

    st.divider()
    st.subheader("Cypher Export (Neo4j 5.15)")
    col1, col2 = st.columns(2)
    with col1:
        node_cypher = Path("graph/cypher_export/nodes.cypher")
        if node_cypher.exists():
            st.code(node_cypher.read_text()[:2000], language="cypher")
        else:
            st.info("Cypher not yet generated.")
    with col2:
        edge_cypher = Path("graph/cypher_export/edges.cypher")
        if edge_cypher.exists():
            st.code(edge_cypher.read_text()[:2000], language="cypher")

    st.divider()
    st.subheader("TTC Forest Plot — Drug-Class HR per Comorbidity")
    show_image("outputs/figures/forest_ttc.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Study Assistant — Llama 3.3 70B via Groq")
    st.markdown(
        "Ask questions about study results, statistical methods, or drug-class clinical profiles. "
        "The assistant has access to study outputs, the ADA 2024 guidelines, and the OMOP database."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the study..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from chatbot.chatbot import get_chatbot
                    bot = get_chatbot()
                    response = bot.get_response(prompt)
                except Exception as e:
                    response = f"Chatbot error: {e}\n\nEnsure GROQ_API_KEY is set in `.env`."
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            try:
                from chatbot.chatbot import get_chatbot
                get_chatbot().clear_history()
            except Exception:
                pass
            st.rerun()

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What is the median time-to-discontinuation for GLP-1 RA patients?",
        "Which comorbidities are most strongly associated with treatment discontinuation?",
        "Why is the 90-day grace period used instead of 30 days?",
        "What does the SHAP analysis show about comorbidity effects on discontinuation risk?",
        "Which drug class has the best evidence for CKD protection according to ADA 2024?",
        "How was propensity score matching performed in this study?",
        "What was the pre-matching imbalance and how did PS matching improve it?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")
