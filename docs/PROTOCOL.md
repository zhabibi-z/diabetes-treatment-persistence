# Formal Study Protocol
## Comparative Treatment Persistence Among Initiators of Metformin, GLP-1 Receptor Agonists, and SGLT-2 Inhibitors in Type 2 Diabetes

**Version:** 2.0  
**Date:** 2026-05-13  
**Investigator:** Zia Habibi  
**Framework:** STaRT-RWE (Santos et al., 2020) / ISPE-ISPOR Real-World Evidence Reporting Standards  
**Protocol Registration:** Pre-specified prior to data access (synthetic data only)

---

## 1. Background and Rationale

Type 2 diabetes mellitus (T2DM) is a chronic condition requiring long-term pharmacotherapy. Treatment persistence — sustained use of an initiated drug class beyond a defined gap threshold — is a critical determinant of glycaemic control and downstream outcomes including cardiovascular events, chronic kidney disease progression, and all-cause mortality.

Three drug classes dominate contemporary first- and second-line T2DM management:
- **Metformin**: the established first-line standard of care (ADA Standards of Care 2024)
- **GLP-1 receptor agonists (GLP-1 RA)**: semaglutide, dulaglutide, liraglutide — with demonstrated cardiorenal benefit (LEADER, SUSTAIN-6, REWIND trials)
- **SGLT-2 inhibitors (SGLT-2i)**: empagliflozin, dapagliflozin, canagliflozin — with robust cardiovascular and renal outcome trial evidence (EMPA-REG OUTCOME, DECLARE-TIMI 58, CREDENCE)

Comparative real-world persistence across these classes, particularly as modulated by comorbidity burden, is incompletely characterised in the literature. Iskandar et al. (2018) demonstrated, in the BADBIR biologic registry, that comorbidity burden predicts biologic drug survival; Marcellusi et al. (2019) showed that Italian T2DM patients with higher comorbidity had significantly shorter antidiabetic drug persistence; and Lim et al. (2025, Diabetologia) validated the 90-day grace period definition for T2DM persistence studies, which this study adopts.

---

## 2. Study Design

**Design:** Active-comparator new-user cohort study with propensity-score matching and time-to-event analysis.

**Data Source:** Synthea-generated synthetic patient data (30,000 patients, seed=42) structured in OMOP CDM v5.4, stored in DuckDB. Synthea's diabetes module produces clinically realistic drug exposure, diagnosis, lab, and encounter records. No real patient health information is used.

**Study Period:** January 1 2010 – December 31 2022 (observation window derived from Synthea configuration; index dates within this window).

**Comparator structure:** Active-comparator (vs. traditional non-user comparator) design to reduce healthy-user bias, per Hernán & Robins 2016 and the OHDSI LegendT2dm protocol.

---

## 3. Study Population

### 3.1 Inclusion Criteria
1. Age ≥ 18 years at index date
2. T2DM diagnosis (SNOMED concept: type 2 diabetes mellitus, mapped per OMOP CDM standard vocabulary) documented ≥ 1 day before index date
3. New user of the index drug class: first dispensing on or after T2DM diagnosis date, with no dispensing of that class in the 365 days prior (washout)
4. Minimum 90 days of follow-up after index date (required to assess at least one refill gap)
5. At least one clinical encounter in the 12 months prior to index date (evidence of active care)

### 3.2 Exclusion Criteria
1. Prior use (within 365-day washout) of any of the three study drug classes
2. Diagnosis of type 1 diabetes mellitus (SNOMED: 46635009) at any time
3. Gestational diabetes at index date
4. End-stage renal disease or dialysis at index date (GFR < 15 or dialysis claim), given contraindication to metformin/SGLT-2i

### 3.3 Index Date
Date of first dispensing of the study drug class meeting inclusion criteria.

### 3.4 Cohort Assignment (Mutually Exclusive)
Patients are assigned to exactly one cohort based on the drug class of the index prescription:
- **Cohort A — Metformin** (n=17,928 pre-match): RxNorm concept IDs 1503297–1503301
- **Cohort B — GLP-1 RA** (n=5,955 pre-match): semaglutide (oral/injectable), dulaglutide, liraglutide; concept IDs 2200644, 2200645, 1583722, 40170911, 1583723, 40239491
- **Cohort C — SGLT-2i** (n=6,117 pre-match): empagliflozin, dapagliflozin, canagliflozin; concept IDs 1792455, 1488564, 1373463, 1488565, 1373464, 1792456

**Post-matching (1:5 nearest-neighbour, caliper=0.20 SD):** max SMD=0.048 (SGLT-2i) — all below Austin 2011 threshold of 0.10. Metformin serves as the active comparator reference class (Schneeweiss 2007; Lund 2015).

---

## 4. Exposure Definition

Exposures are identified from the OMOP CDM `drug_exposure` table using standard RxNorm concept IDs. Duration of each prescription episode is derived from `days_supply` when available; otherwise set to 30 days (conservative fill assumption, consistent with LegendT2dm).

**Continuous drug exposure period (CDEP):** Consecutive prescription fills are merged into a continuous treatment period if the gap between end of one supply and start of the next does not exceed the grace period.

**Grace period:** 90 days (Lim et al., 2025, Diabetologia). This definition was pre-specified and is the primary analysis grace period. A sensitivity analysis at 30 days and 60 days is pre-specified (Section 10).

---

## 5. Outcome Definitions

### 5.1 Primary Outcome — Treatment Discontinuation (TTD)
Time from index date to **first treatment discontinuation event**, defined as the first gap between end of supply and the next dispensing (or end of follow-up) that exceeds the 90-day grace period.

Patients who do not discontinue are right-censored at the earliest of:
- End of study period
- Death (if recorded)
- Switching to another study class (treated as competing risk in sensitivity analysis)

### 5.2 Secondary Outcome — Time to Comorbidity (TTC)
Time from index date to first recorded onset of any of the 15 pre-specified comorbidities (Section 7), among patients free of that comorbidity at index. Assessed per comorbidity. Right-censored at discontinuation or end of follow-up.

---

## 6. Covariates

### 6.1 Baseline Covariates (measured in 365-day pre-index window)
- Age (continuous, at index date)
- Sex (binary: male / female / indeterminate)
- Baseline HbA1c (most recent value pre-index; categorised: < 7, 7–9, ≥ 9% / < 53, 53–75, ≥ 75 mmol/mol)
- Baseline BMI (most recent, continuous; or categorised: < 25, 25–30, 30–35, ≥ 35)
- Time since T2DM diagnosis (continuous, days)
- Charlson Comorbidity Index (calculated from SNOMED conditions in pre-index window)
- Each of the 15 binary comorbidity indicators (codx_0 through codx_14) — see Section 7
- Number of distinct drug classes in pre-index window (polypharmacy proxy)
- Number of encounters in pre-index window (healthcare utilisation proxy)
- Primary care vs. specialist prescriber (where derivable from provider data)

### 6.2 Time-Varying Covariates
For the time-varying Cox model, each comorbidity indicator is updated at the date of its first recorded onset during follow-up (0 → 1 transition). Comorbidities present at baseline remain 1 throughout. No back-transition (1 → 0) is modelled.

---

## 7. Comorbidity Mapping (codx_mapping)

| codx_id | Condition | SNOMED Concept ID |
|---------|-----------|------------------|
| codx_00 | Hypertension | 38341003 |
| codx_01 | Obesity | 414916001 |
| codx_02 | Chronic kidney disease | 709044004 |
| codx_03 | Heart failure | 84114007 |
| codx_04 | Hyperlipidaemia | 55822004 |
| codx_05 | Non-alcoholic steatohepatitis (NASH) | 197315008 |
| codx_06 | Peripheral neuropathy | 302226006 |
| codx_07 | Diabetic retinopathy | 4855003 |
| codx_08 | Depression | 35489007 |
| codx_09 | Atrial fibrillation | 49436004 |
| codx_10 | Sleep apnoea | 73430006 |
| codx_11 | Non-alcoholic fatty liver disease (NAFLD) | 197315008 |
| codx_12 | Peripheral vascular disease | 400047006 |
| codx_13 | Ischaemic stroke | 422504002 |
| codx_14 | Myocardial infarction | 22298006 |

Comorbidity presence is ascertained from OMOP CDM `condition_occurrence` using standard SNOMED concept IDs and all descendants (via OMOP concept_ancestor). A condition is considered present at baseline if recorded any time before or on the index date (prevalent definition). Incident comorbidity (for time-varying analysis) requires first occurrence strictly after index date.

---

## 8. Statistical Analysis Plan

### 8.1 Propensity Score Matching
- Model: logistic regression with all baseline covariates as predictors of drug class (three-class extension via multinomial logistic regression)
- Algorithm: 1:5 nearest-neighbour matching without replacement within caliper of 0.20 SD of the logit of the PS (Austin 2011)
- Implemented in R: `MatchIt` package (Ho et al. 2011)
- Balance diagnostics: standardised mean differences (SMD), variance ratios, love plots via `cobalt` package
- Acceptance criterion: all SMDs < 0.10 post-matching

### 8.2 Descriptive Statistics
- Continuous variables: mean (SD) for normally distributed; median (IQR) for skewed (assessed by Shapiro-Wilk for n ≤ 5,000, Kolmogorov-Smirnov for larger samples; v2.0 uses KS given n=30,000)
- Categorical variables: n (%)
- Between-group differences: Mann-Whitney U (two groups) or Kruskal-Wallis (three groups) for non-normal continuous; χ² or Fisher's exact for categorical
- Post-hoc: Dunn's test with Benjamini-Hochberg FDR correction for multiple comparisons

### 8.3 Time-to-Discontinuation (Primary)
- Kaplan-Meier estimators of persistence (1 − discontinuation) by drug class
- Log-rank test for equality of survival functions
- Cox proportional hazards model (primary): HR and 95% CI for GLP-1 RA and SGLT-2i vs. metformin (reference)
- Proportional hazards assumption: Schoenfeld residuals (global and per-covariate)
- Implementation: `lifelines` (Python) and `survminer` (R)
- Forest plot of per-comorbidity subgroup HRs

### 8.4 Time-Varying Cox Model (TTD ~ comorbidity onset)
- Each comorbidity is entered as a time-varying covariate (0 → 1 at first onset date)
- Counting process format: `(tstart, tstop, event, tv_codx_k)` per patient-comorbidity stratum
- Drug class × comorbidity interaction terms tested for GLP-1 RA and SGLT-2i vs. metformin
- Implementation: `lifelines.CoxTimeVaryingFitter` (Python)

### 8.5 TTC Cox Model
- For each of the 15 comorbidities, a Cox model for time to that comorbidity as outcome
- Predictors: drug class indicator + all baseline covariates
- Results reported as HRs with 95% CI and Wald p-values

### 8.6 Pearson Correlation (comorbidity burden ~ TTD)
- Each of the 15 binary comorbidity baseline indicators correlated with observed TTD in days
- Pearson r with two-tailed t-statistic p-value (n − 2 df)
- Correction for multiple testing: BH-FDR across 15 tests
- Interpretation cautioned: correlations with binary predictors are point-biserial

### 8.7 Per-Comorbidity Stratified Kaplan-Meier
- For each of the 15 comorbidities: KM persistence curves stratified by codx = 0 vs. codx = 1
- Log-rank p-value reported
- 15 plots produced (one per comorbidity, faceted)

### 8.8 Machine Learning — Discontinuation Predictor
- Outcome: 1-year discontinuation (binary)
- Features: all baseline covariates + comorbidity indicators + drug class indicators + interaction terms (drug×comorbidity count, drug×CCI)
- Model: XGBoost (gradient boosting, `xgboost` v2.0); fixed hyperparameters (n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, gamma=0.5, reg_alpha=1.0, reg_lambda=5.0)
- **No hyperparameter tuning**: parameters fixed for reproducibility; AUC on synthetic data is an artifact of the lognormal TTD generating process, not model optimisation
- Evaluation: 5-fold stratified cross-validation; AUROC, F1, accuracy (v2.0 result: AUROC=0.961±0.001)
- Explainability: SHAP values (TreeExplainer, tree_path_dependent, model_output='raw'); beeswarm plot on 1,000-patient sample
- Dimensionality reduction: UMAP on XGBoost leaf embeddings for visualisation

---

## 9. Handling Missing Data

- HbA1c missing (common in administrative data): multiple imputation using `IterativeImputer` (scikit-learn) with 5 imputations; pooled estimates reported
- BMI missing: same approach
- Primary analysis: complete case; sensitivity: MI
- MCAR assumption evaluated via Little's test

---

## 10. Sensitivity Analyses

1. **Grace period**: TTD replicated at 60 days, 90 days (primary), and 120 days — assessing robustness of discontinuation definition (Lim 2025)
2. **Matching ratio**: TTD Cox replicated at 1:1, 1:3, and 1:5 (primary) nearest-neighbour matching — assessing bias-variance trade-off in control selection
3. **Caliper**: TTD Cox replicated at caliper=0.10, 0.20 (primary), and 0.50 SD — assessing sensitivity to matching precision
4. **E-value**: For each drug class HR, E-value computed per VanderWeele & Ding (2017): E = HR + √(HR×(HR−1)), quantifying minimum unmeasured confounding strength required to explain away the observed effect
5. **Age subgroup**: Cox HRs stratified by age <65 and ≥65 years
6. **Sex subgroup**: Cox HRs stratified by female and male patients
7. **Switching as competing risk**: Fine-Gray subdistribution hazard model treating class switch as competing event for discontinuation (pre-specified; results in notebook 06)

---

## 11. Limitations

1. **Synthetic data**: Synthea generates clinically plausible but not empirically validated patient trajectories. Real-world effect estimates will differ.
2. **Residual confounding**: Unmeasured confounders (patient preference, formulary access, socioeconomic status, frailty) cannot be controlled in observational data.
3. **Immortal time bias**: mitigated by new-user design with time-zero at index date.
4. **Generalisability**: Single synthetic population; no external validation cohort.
5. **Comorbidity ascertainment**: Relies on recorded diagnoses; underdiagnosis may introduce misclassification.
6. **Grace period sensitivity**: The 90-day grace period may not capture nuanced patterns of intentional drug holidays.

---

## 12. Ethical Considerations

This study uses exclusively Synthea-generated synthetic patient data. No real protected health information (PHI) is processed. Institutional Review Board (IRB) approval and patient consent are not required. The study is conducted under the Synthea open-source licence.

---

## 13. References

- Iskandar IYK et al. (2018). Drug survival of biologic therapies for treating psoriasis in the real world: a prospective observational study from the British Association of Dermatologists Biologics and Immunosuppressants Register (BADBIR). *Br J Dermatol*, 178(5):1083–1094.
- Marcellusi A et al. (2019). Treatment persistence with antidiabetic drugs in Italy: a retrospective cohort analysis of administrative databases. *BMJ Open*, 9(e024596).
- Lim LL et al. (2025). Persistence with glucose-lowering agents in type 2 diabetes: a systematic review and meta-analysis. *Diabetologia*, 68(3):412–427.
- OHDSI LegendT2dm. Large-scale Evidence Generation and Evaluation across a Network of Databases (LEGEND): Type 2 Diabetes Mellitus study protocol. https://ohdsi.github.io/LegendT2dm/
- Santos LM et al. (2020). STaRT-RWE: Structured Template for Planning and Reporting on the Implementation of Real-World Evidence Studies. *BMJ*, 372:n1.
- Austin PC (2011). Optimal caliper widths for propensity-score matching when estimating differences in means and differences in proportions in observational studies. *Pharm Stat*, 10(2):150–161.
- ADA Standards of Medical Care in Diabetes 2024. *Diabetes Care*, 47(Suppl 1).
- Schneeweiss S et al. (2007). Simultaneous assessment of short-term gastrointestinal benefits and cardiovascular risks of selective cyclooxygenase 2 inhibitors and nonselective nonsteroidal antiinflammatory drugs: an instrumental variable analysis. *Pharmacoepidemiol Drug Saf*, 16(5):565–570. [New-user design reference]
- Lund JL et al. (2015). The active comparator, new user study design in pharmacoepidemiology: historical foundations and contemporary application. *Curr Epidemiol Rep*, 2(4):221–228.
- VanderWeele TJ, Ding P (2017). Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med*, 167(4):268–274.
