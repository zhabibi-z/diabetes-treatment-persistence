# Comparative Treatment Persistence Among Initiators of Metformin, GLP-1 Receptor Agonists, and SGLT-2 Inhibitors in Type 2 Diabetes

[![CI](https://github.com/zhabibi-z/diabetes-treatment-persistence/actions/workflows/ci.yml/badge.svg)](https://github.com/zhabibi-z/diabetes-treatment-persistence/actions/workflows/ci.yml)

**Investigator:** Zia Habibi  
**Version:** v2.0 (30,000 patients)  
**Status:** Active — synthetic data validation  
**Data:** 30,000 synthetic patients, OMOP CDM v5.4 (no real PHI)  
**Live Demo:** https://t2dm-persistence-rwe.streamlit.app/

---

## Overview

This real-world evidence (RWE) study applies pharmacoepidemiological methods to characterise comparative treatment persistence across three major antidiabetic drug classes: metformin monotherapy, GLP-1 receptor agonists (semaglutide, dulaglutide, liraglutide), and SGLT-2 inhibitors (empagliflozin, dapagliflozin, canagliflozin). The primary endpoint is time to treatment discontinuation (TTD) defined by a 90-day grace period (Lim 2025). Comorbidity burden — measured across 15 SNOMED-mapped conditions — is investigated as a predictor and time-varying modifier of persistence.

The study follows the **STaRT-RWE** structured template, the **OHDSI LegendT2dm** active-comparator new-user cohort design (Schneeweiss 2007, Lund 2015), and is built on the **OMOP CDM v5.4** running on **DuckDB**. A machine-learning (XGBoost + SHAP) module provides 1-year discontinuation prediction. An interactive knowledge graph (streamlit-agraph) encodes drug–disease–comorbidity relationships, and a LangChain + Groq chatbot (Llama 3.3 70B) enables natural-language querying.

---

## Cohort Design

The study uses a **new-user active-comparator design** (Schneeweiss 2007, Lund 2015):

- **New-user design:** Patients are enrolled at their first dispensing of a study drug, after a 365-day washout period with no prior use of any study class. This eliminates prevalent-user bias by ensuring all patients have the same opportunity to experience early tolerability events.
- **Active comparator:** Metformin serves as the reference class. Comparing GLP-1 RA and SGLT-2i directly to metformin (rather than to a non-initiator control) reduces confounding by indication — both groups are actively managed T2DM patients who qualified for pharmacotherapy.
- **Mutually exclusive cohorts:** Patients are assigned to the first drug class dispensed after T2DM diagnosis. If a patient initiated metformin before GLP-1, they are a metformin new-user.
- **1:5 PS matching:** Propensity scores estimated from logistic regression on age, sex, CCI, and 15 comorbidity flags. Nearest-neighbour matching without replacement, caliper 0.20 SD (Austin 2011).

*References: Schneeweiss S et al. Pharmacoepidemiol Drug Saf 2007;16:565–570; Lund JL et al. Pharmacoepidemiol Drug Saf 2015;24:1078–1086.*

---

## Key Results (v2.0 — 30,000 Patients)

| Drug Class | n (pre-match) | n (matched) | Median TTD (days) | Discontinued (%) |
|-----------|:---:|:---:|---:|---:|
| Metformin  | 17,928 | 17,928 | **392** | 92.5% |
| GLP-1 RA   | 5,955  | 5,955  | **261** | 94.9% |
| SGLT-2i    | 6,117  | 6,117  | **317** | 94.5% |

- **Cox TTD:** Drug class significant predictor (HR=1.130 per class, p=1.5×10⁻⁷⁰); drug_class_num z≈8.4
- **R Cox:** GLP-1 HR=1.50 vs metformin (p=2.6×10⁻¹⁵²), SGLT-2i HR=1.23 (p=7.8×10⁻⁴²)
- **XGBoost:** 5-fold CV AUROC = **0.961 ± 0.001**, F1 = 0.909 ± 0.001
- **Top permutation importance:** `followup_days` (AUC drop 0.446), `days_since_t2dm_dx` (0.006), `age_at_index` (0.006), `drug_class_num` (0.001)
- **Knowledge graph:** 19 nodes, 27 edges — now interactive in Streamlit (streamlit-agraph)
- **Kruskal-Wallis TTD:** H=1034.3, p≈0 — all pairwise Dunn comparisons p < 10⁻³³

> **AUC = 0.961 is driven by a Synthea data-generating artifact, not clinical predictive signal.**
>
> **Root cause — `followup_days` leakage:** The Synthea generator sets `obs_end = disc_date + Uniform(120, 300)`, where `disc_date = index_date + ttd_days`. This means `followup_days ≈ ttd_days + noise` (Pearson r = 0.972). Since the outcome is `y = (ttd_days ≤ 365)`, the model reconstructs the target from `followup_days` alone: a logistic regression on `followup_days` only achieves AUC = 0.948; XGBoost on `followup_days` only achieves AUC = 0.951. In real-world claims data, `obs_end` is set by study design (e.g., study end date), not derived from the patient's discontinuation date — so this leakage does not exist.
>
> **Drug class is not the primary driver:** An ablation removing only `followup_days` (keeping all other features including drug class) drops AUC from 0.961 to **0.574**. Drug class alone (without `followup_days`) achieves AUC = **0.578** — consistent with the lognormal TTD parameter differences across classes but far below 0.961. Removing both `followup_days` and drug class yields AUC = **0.503** (essentially random), confirming no other feature carries real predictive signal in this synthetic dataset.
>
> **Expected real-world performance:** Without the `followup_days` leakage, and with non-deterministic prescribing behavior in real claims data, expected AUC for 1-year T2DM treatment discontinuation prediction is **0.70–0.80**. See notebook 04 (`04_ml_xgboost_shap.ipynb`) for the full ablation study and permutation importance analysis.

> All results are from synthetic patients only.

---

## Notebooks (7 total)

| # | Notebook | Content |
|---|----------|---------|
| 00 | `00_data_quality_and_eda.ipynb` | CONSORT flow, missingness, distributions, outliers, normality tests, DQ flags |
| 01 | `01_cohort_characterization.ipynb` | Table 1, inline SQL, comorbidity heatmap, OMOP query |
| 02 | `02_survival_analysis.ipynb` | KaplanMeierFitter (explicit params), CoxPHFitter (all covariates), forest plot |
| 03 | `03_hypothesis_tests.ipynb` | Mann-Whitney U, Kruskal-Wallis, Dunn BH-FDR (all inline) |
| 04 | `04_ml_xgboost_shap.ipynb` | XGB_PARAMS listed, feature engineering inline, SHAP TreeExplainer, leakage audit, ablation studies, permutation importance |
| 05 | `05_knowledge_graph.ipynb` | NetworkX nodes + edges inline, Cypher export |
| 06 | `06_sensitivity_analyses.ipynb` | Grace period, matching ratio, caliper, E-values, subgroups |

All notebooks (v2.0): analytical code is visible inline — no hidden `.py` module calls for statistical logic.

---

## Pipeline Architecture

```
Synthetic Data Generator (30,000 T2DM patients, seed=42)
    │
    ▼
synthea_to_omop.py          ETL: synthetic → OMOP CDM v5.4 (DuckDB)
    │
    ▼
build_cohort.py             3 mutually exclusive new-user cohorts
cohort_matching.R           1:5 PS matching (MatchIt, caliper=0.2 SD), balance (cobalt)
    │
    ├── run_ttd.py              Time-to-discontinuation (90-day grace)
    ├── run_ttc.py              Time-to-comorbidity KM
    ├── run_cox_timevarying.py  Time-varying Cox (comorbidity 0→1 transitions)
    ├── run_cox_ttc.py          TTC Cox per comorbidity
    ├── run_correlations.py     Pearson correlation comorbidity × TTD
    ├── run_km_stratified.py    Per-comorbidity stratified KM (codx=0 vs 1)
    ├── survival_analysis.R     survminer KM + forest plot + Schoenfeld residuals
    └── hypothesis_tests.R      Shapiro-Wilk, MW-U, Kruskal-Wallis, Dunn BH-FDR
    │
    ├── train.py            XGBoost + UMAP + SHAP (5-fold CV)
    ├── build_graph.py      NetworkX → Cypher export
    └── chatbot.py          LangChain + Groq Llama 3.3 70B + RAG
    │
    ▼
src/app/app.py              6-tab interactive dashboard (interactive graph: streamlit-agraph)
```

---

## Quick Start

```bash
# Clone / enter project directory
cd diabetes-treatment-persistence

# Copy and fill environment variables
cp config/.env.template .env
# edit .env: add GROQ_API_KEY (free at console.groq.com)

# Run full pipeline
bash scripts/bootstrap.sh
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data storage | DuckDB (OMOP CDM v5.4) |
| ETL | Python 3.11, pandas |
| Survival | lifelines (Python), survminer (R) |
| Matching | MatchIt, cobalt (R) |
| ML | XGBoost, SHAP, UMAP (Python) |
| Graph | NetworkX → streamlit-agraph (interactive), Neo4j 5.15 (optional) |
| Chatbot | LangChain, Groq (Llama 3.3 70B — free inference API) |
| Dashboard | Streamlit |
| Orchestration | Bash, Perl |

> **Chatbot powered by Llama 3.3 70B via Groq's free API.**  
> Get a free key at [console.groq.com](https://console.groq.com) and add `GROQ_API_KEY=...` to `.env`.

---

## Documentation

- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — full study protocol (STaRT-RWE structured template).
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system architecture, component map, and directory structure.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — version history.

## References

1. Iskandar IYK et al. Drug survival of biologic therapies for treating psoriasis in the real world. *Br J Dermatol* 2018;178(5):1083–1094.
2. Marcellusi A et al. Treatment persistence with antidiabetic drugs in Italy. *BMJ Open* 2019;9:e024596.
3. Lim LL et al. Persistence with glucose-lowering agents in type 2 diabetes. *Diabetologia* 2025;68(3):412–427.
4. Schneeweiss S et al. New-user design. *Pharmacoepidemiol Drug Saf* 2007;16(5):565–570.
5. Lund JL et al. Active comparator cohort design. *Pharmacoepidemiol Drug Saf* 2015;24(10):1078–1086.
6. OHDSI LegendT2dm Study Protocol. https://ohdsi.github.io/LegendT2dm/
7. Santos LM et al. STaRT-RWE. *BMJ* 2020;372:n1.
8. ADA Standards of Medical Care in Diabetes 2024. *Diabetes Care* 2024;47(Suppl 1).
9. VanderWeele TJ, Ding P. Sensitivity analysis in observational research. *Ann Intern Med* 2017;167(4):268–274.
