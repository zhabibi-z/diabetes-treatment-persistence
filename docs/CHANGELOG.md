# Changelog

All notable changes to this project are documented here.

## [2.0.0] — 2026-05-13

### Scaled
- **30,000 synthetic patients** (seed=42): metformin n=17,928, GLP-1 RA n=5,955, SGLT-2i n=6,117 (pre-match)
- PS matching unchanged: 1:5 nearest-neighbour, caliper=0.20 SD; max SMD post-match 0.048 (SGLT-2i)

### Added
- **Notebook 00** `00_data_quality_and_eda.ipynb`: CONSORT flow, missingness heatmap, distributions, outlier detection, Shapiro-Wilk + KS normality tests, 10-check DQ flags table
- **Notebook 06** `06_sensitivity_analyses.ipynb`: grace period (60/90/120 d), matching ratio (1:1/1:3/1:5), caliper (0.1/0.2/0.5 SD), E-values (VanderWeele 2017), subgroup Cox by age/sex
- **Interactive knowledge graph**: Tab 5 replaced static PNG with streamlit-agraph (clickable nodes, edge-type filter, hover metadata); `streamlit-agraph>=0.0.45` added to requirements
- **Clinical interpretation cells**: 3–5 markdown cells per notebook 01–05 citing Marcellusi 2019, Lim 2025, Iskandar 2018, ADA 2024

### Changed
- **Notebooks 01–05**: all analytical code moved inline (no hidden `.py` module calls); SQL, Cox, KM, hypothesis tests, ML, graph logic all visible in notebook cells
- **Streamlit memory optimisation**: `@st.cache_data(ttl=3600)` on all loaders; cohort explorer paginated (max 1000 rows); survival tab loads only needed columns; ML tab uses `@st.cache_resource`; DuckDB filtered queries throughout
- **README**: v2.0 header, new Cohort Design section (Schneeweiss 2007, Lund 2015), updated key results table, honest AUC note (synthetic artifact), 7-notebook table, streamlit-agraph in stack

### Results (v2.0 — 30,000 synthetic patients)
- Metformin: n=17,928 pre-match, median TTD=392 days, 92.5% discontinued
- GLP-1 RA: n=5,955 pre-match, median TTD=261 days, 94.9% discontinued
- SGLT-2i: n=6,117 pre-match, median TTD=317 days, 94.5% discontinued
- Cox TTD: drug_class_num HR=1.130 (p=1.5×10⁻⁷⁰); R Cox GLP-1 HR=1.50 (p=2.6×10⁻¹⁵²), SGLT-2i HR=1.23 (p=7.8×10⁻⁴²)
- XGBoost: 5-fold CV AUROC=0.961±0.001, F1=0.909±0.001
- Kruskal-Wallis H=1034.3, p≈0; all pairwise Dunn p<10⁻³³

### AUC Note
The 0.961 AUC at 30K is identical to v1.0 at 5K. This is an expected synthetic data property: the lognormal TTD generating process (drug-class-specific μ/σ) creates inherently separable patterns. Real-world T2DM persistence models report AUC 0.70–0.85. XGBoost hyperparameters were not tuned to inflate this figure.

---

## [1.0.0] — 2026-05-12

### Added
- **ETL**: `synthea_to_omop.py` — Synthea CSV → OMOP CDM v5.4 (DuckDB)
- **Cohort build**: `build_cohort.py` — 3 mutually exclusive new-user cohorts (n=5,000)
- **PS matching**: `cohort_matching.R` — 1:5 nearest-neighbour PS matching (MatchIt), love plots (cobalt)
- **TTD analysis**: `run_ttd.py` — Time-to-discontinuation with 90-day grace period (Lim 2025)
- **TTC analysis**: `run_ttc.py` — Per-comorbidity Kaplan-Meier
- **Cox TTD**: `run_cox.py` — Cox proportional hazards for TTD
- **Time-varying Cox**: `run_cox_timevarying.py` — Comorbidity 0→1 transitions as time-varying covariates
- **TTC Cox**: `run_cox_ttc.py` — Cox with dummy comorbidity interactions
- **Correlations**: `run_correlations.py` — Pearson correlation comorbidity × TTD + heatmap
- **Stratified KM**: `run_km_stratified.py` — Per-comorbidity KM grid (codx=0 vs codx=1)
- **R survival**: `survival_analysis.R` — survminer KM plots, Schoenfeld residuals, forest plot
- **Hypothesis tests**: `hypothesis_tests.R` — Shapiro-Wilk, Kruskal-Wallis, Dunn BH-FDR, Mann-Whitney U
- **ML module**: `ml/train.py` — XGBoost 5-fold CV (AUROC=0.961), SHAP explainability, UMAP phenotyping
- **Knowledge graph**: `graph/build_graph.py` — NetworkX DiGraph (19 nodes, 27 edges), Cypher export
- **Chatbot**: `chatbot/chatbot.py` — LangChain + Groq Llama 3.3 70B + FAISS RAG + DuckDB SQL
- **Jupyter notebooks**: 5 notebooks in `analysis/notebooks/` (executed via nbconvert)
- **Streamlit dashboard**: `streamlit_app/app.py` — 6-tab interactive dashboard
- **R packages installer**: `scripts/install_r_packages.R`
- **Bootstrap script**: `scripts/bootstrap.sh` — full pipeline runner

### Results (v1.0 — 5,000 synthetic patients)
- Metformin: n=2,973, median TTD=407 days, 92.6% discontinued
- GLP-1 RA: n=1,056, median TTD=256 days, 95.5% discontinued
- SGLT-2i: n=971, median TTD=315 days, 93.7% discontinued
- XGBoost AUROC: 0.961 ± 0.008 (5-fold CV)
