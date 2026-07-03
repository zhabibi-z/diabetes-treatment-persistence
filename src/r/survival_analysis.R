#!/usr/bin/env Rscript
# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
# survival_analysis.R — Survival analysis suite: KM, Cox PH, Schoenfeld residuals,
#                        Harrell's C-index, and Fine-Gray competing risks model.
#
# Changes from v1
# ---------------
# 1. Harrell's C-index added via survival::concordance(). This is the survival-
#    analysis equivalent of AUROC and is required for any clinical prediction
#    model publication (Harrell FE. Regression Modelling Strategies. 2015).
#
# 2. Fine-Gray subdistribution hazard model added via cmprsk::crr(). In a
#    discontinuation study, patient death is a competing risk: a patient who
#    dies cannot subsequently discontinue medication. The standard Cox model
#    treats death as non-informative censoring, which overestimates the
#    cause-specific discontinuation hazard when death rates are non-trivial.
#    Fine JP, Gray RJ. JASA. 1999;94(446):496-509.
#
# 3. Cox model results and C-index now exported to CSV for downstream display
#    in the Streamlit dashboard.

suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(broom)
  library(optparse)
})

option_list <- list(
  make_option("--ttd-file", default = "outputs/tables/ttd_events.csv"),
  make_option("--cohort",   default = "outputs/tables/cohort_matched.csv"),
  make_option("--output",   default = "outputs/figures"),
  make_option("--tables",   default = "outputs/tables")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output, recursive = TRUE, showWarnings = FALSE)
dir.create(opt$tables, recursive = TRUE, showWarnings = FALSE)

message("Loading TTD events: ", opt$`ttd-file`)
if (!file.exists(opt$`ttd-file`)) stop("ttd_events.csv not found — run analysis/run_ttd.py first")

ttd    <- read_csv(opt$`ttd-file`, show_col_types = FALSE)
cohort <- read_csv(opt$cohort,     show_col_types = FALSE)

if (!"drug_class" %in% names(ttd)) {
  ttd <- ttd %>% left_join(select(cohort, person_id, drug_class), by = "person_id")
}

ttd <- ttd %>%
  filter(!is.na(ttd_days), ttd_days >= 0, !is.na(drug_class)) %>%
  mutate(drug_class = factor(drug_class,
                              levels = c("metformin", "glp1", "sglt2"),
                              labels = c("Metformin", "GLP-1 RA", "SGLT-2i")))

message(sprintf("Analytic sample: n=%d  events=%d (%.1f%%)",
                nrow(ttd), sum(ttd$discontinued), 100 * mean(ttd$discontinued)))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Kaplan-Meier (survminer)
# ══════════════════════════════════════════════════════════════════════════════

surv_obj <- Surv(ttd$ttd_days, ttd$discontinued)
km_fit   <- survfit(surv_obj ~ drug_class, data = ttd)

km_plot <- ggsurvplot(
  km_fit,
  data              = ttd,
  risk.table        = TRUE,
  pval              = TRUE,
  conf.int          = TRUE,
  palette           = c("#3498DB", "#E74C3C", "#2ECC71"),
  legend.labs       = c("Metformin", "GLP-1 RA", "SGLT-2i"),
  xlab              = "Days from Index Date",
  ylab              = "Probability of Persistence",
  title             = "Treatment Persistence by Drug Class\n(90-day grace period, Lim 2025)",
  ggtheme           = theme_bw(base_size = 12),
  risk.table.height = 0.28,
  surv.median.line  = "hv",
)
ggsave(
  file.path(opt$output, "km_persistence_survminer.png"),
  plot  = print(km_plot),
  width = 10, height = 7, dpi = 150
)
message("KM plot saved: km_persistence_survminer.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Cox Proportional Hazards + Forest Plot
# ══════════════════════════════════════════════════════════════════════════════

comorbidity_cols <- intersect(
  c("hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi"),
  names(ttd)
)

cox_formula_str <- paste(
  "Surv(ttd_days, discontinued) ~ drug_class + age_at_index + cci",
  if (length(comorbidity_cols) > 0) paste("+", paste(comorbidity_cols, collapse = " + ")) else ""
)
cox_formula <- tryCatch(
  as.formula(cox_formula_str),
  error = function(e) as.formula("Surv(ttd_days, discontinued) ~ drug_class + age_at_index + cci")
)

cox_fit     <- coxph(cox_formula, data = ttd, x = TRUE)
cox_summary <- tidy(cox_fit, exponentiate = TRUE, conf.int = TRUE)

# Export Cox results for Streamlit display
write_csv(cox_summary, file.path(opt$tables, "cox_ttd_results_r.csv"))
message("Cox PH results exported: cox_ttd_results_r.csv")
message("\nCox PH results:")
print(cox_summary)

# Forest plot
forest_data <- cox_summary %>%
  filter(grepl("drug_class|hypertension|ckd|obesity|heart|stroke|mi", term)) %>%
  mutate(term = gsub("drug_class", "", term))

if (nrow(forest_data) > 0) {
  forest_plot <- ggplot(forest_data, aes(x = estimate, y = term)) +
    geom_point(size = 3, color = "#2C3E50") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "#7F8C8D") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_x_log10() +
    labs(
      x        = "Hazard Ratio (log scale)", y = NULL,
      title    = "Cox PH — Hazard Ratios for Treatment Discontinuation",
      subtitle = "Reference: Metformin"
    ) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  ggsave(file.path(opt$output, "forest_cox_ttd.png"), forest_plot, width = 8, height = 5, dpi = 150)
  message("Forest plot saved: forest_cox_ttd.png")
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Harrell's C-index
#
# The concordance statistic (C-index) is the probability that, for a randomly
# selected pair of patients, the one who discontinues first has the higher
# predicted hazard. It generalises AUROC to censored data.
#
# Reference: Harrell FE et al. Biometrics. 1984;40(2):459–467.
#            Harrell FE. Regression Modelling Strategies. 2nd ed. 2015.
# ══════════════════════════════════════════════════════════════════════════════

message("\n--- Harrell's C-index (concordance statistic) ---")
c_stat <- concordance(cox_fit)
c_idx  <- c_stat$concordance
c_se   <- sqrt(c_stat$var)
c_lo   <- c_idx - 1.96 * c_se
c_hi   <- c_idx + 1.96 * c_se

message(sprintf(
  "C-index = %.3f  (95%% CI: %.3f – %.3f)",
  c_idx, c_lo, c_hi
))
message("Interpretation: 0.5 = random ranking; >0.70 = clinically useful discrimination.")

c_index_df <- data.frame(
  metric    = "Harrell C-index",
  estimate  = round(c_idx, 4),
  ci_lower  = round(c_lo,  4),
  ci_upper  = round(c_hi,  4),
  note      = "Concordance statistic for Cox PH model; Harrell et al. 1984"
)
write_csv(c_index_df, file.path(opt$tables, "c_index.csv"))
message("C-index saved: c_index.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Schoenfeld residuals (PH assumption test)
# ══════════════════════════════════════════════════════════════════════════════

ph_test <- cox.zph(cox_fit)
message("\nSchoenfeld residuals (global PH assumption test):")
print(ph_test)

ph_results <- as.data.frame(ph_test$table)
ph_results$covariate <- rownames(ph_results)
write_csv(ph_results, file.path(opt$tables, "schoenfeld_tests.csv"))

ph_plot <- ggcoxzph(ph_test, point.size = 0.5, point.alpha = 0.3)
ggsave(
  file.path(opt$output, "schoenfeld_residuals.png"),
  plot  = print(ph_plot),
  width = 10, height = 6, dpi = 130
)
message("Schoenfeld residuals plot saved")

if (ph_test$table["GLOBAL", "p"] < 0.05) {
  message(
    "WARNING: Global Schoenfeld test is significant (p=",
    round(ph_test$table["GLOBAL", "p"], 4), "). ",
    "The proportional hazards assumption may be violated. ",
    "Consider time-stratified Cox or a time-varying coefficient model."
  )
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Fine-Gray competing risks model
#
# Motivation: patient death is a competing risk for medication discontinuation.
# The standard Cox cause-specific hazard model incorrectly treats death as
# non-informative censoring, biasing subdistribution hazard estimates upward
# when mortality is present (Lau et al. 2009, Stat Med 28:2170–2197).
#
# The Fine-Gray model estimates the subdistribution hazard, which accounts for
# the competing event (death) and targets the cumulative incidence function
# directly — the clinically relevant quantity for intervention planning.
#
# Event coding:
#   0 = right-censored (still on medication at end of observation)
#   1 = discontinued (primary event)
#   2 = died (competing event)
#
# Reference: Fine JP, Gray RJ. JASA. 1999;94(446):496-509.
#            Lau B et al. Stat Med. 2009;28(15):2170–2197.
# ══════════════════════════════════════════════════════════════════════════════

message("\n--- Fine-Gray competing risks model ---")

# Construct event_type column.
# If a 'died' or 'death_date' column is available in the TTD file, use it.
# Otherwise create a proxy based on available information.
if ("died" %in% names(ttd)) {
  ttd <- ttd %>%
    mutate(event_type = case_when(
      died == 1             ~ 2L,   # death (competing event)
      discontinued == 1     ~ 1L,   # discontinuation (primary event)
      TRUE                  ~ 0L    # censored
    ))
  message("Using 'died' column for competing event indicator.")
} else {
  # Without death data, create a placeholder that assumes zero mortality.
  # This is declared explicitly so that the model structure is preserved
  # for when real OMOP death table data is available.
  ttd <- ttd %>% mutate(event_type = as.integer(discontinued))
  message(
    "NOTE: No 'died' column found in TTD events. ",
    "Fine-Gray model runs with event_type = discontinued only (no competing event). ",
    "To enable proper competing risks analysis, join OMOP death table records to ",
    "ttd_events.csv and add a 'died' binary column before running this script."
  )
}

# Require cmprsk
if (!requireNamespace("cmprsk", quietly = TRUE)) {
  message("cmprsk not installed — run Rscript scripts/install_r_packages.R")
  message("Fine-Gray model skipped.")
} else {
  library(cmprsk)

  # Build covariate matrix for Fine-Gray (requires numeric matrix, no factors)
  fg_covariates <- c("age_at_index", "cci", comorbidity_cols)
  fg_covariates <- intersect(fg_covariates, names(ttd))

  # Drug class as numeric (GLP-1 RA = 1 vs rest, SGLT-2i = 1 vs rest)
  ttd_fg <- ttd %>%
    mutate(
      drug_glp1  = as.integer(drug_class == "GLP-1 RA"),
      drug_sglt2 = as.integer(drug_class == "SGLT-2i"),
    )

  cov_matrix <- ttd_fg %>%
    select(all_of(fg_covariates), drug_glp1, drug_sglt2) %>%
    mutate(across(everything(), ~ as.numeric(replace_na(., 0)))) %>%
    as.matrix()

  fg_fit <- tryCatch(
    crr(
      ftime    = ttd_fg$ttd_days,
      fstatus  = ttd_fg$event_type,
      cov1     = cov_matrix,
      failcode = 1,    # primary event = discontinuation
      cencode  = 0,
    ),
    error = function(e) {
      message("Fine-Gray model error: ", e$message)
      NULL
    }
  )

  if (!is.null(fg_fit)) {
    fg_summary <- summary(fg_fit)

    # Extract subdistribution HRs for drug class indicators
    fg_df <- data.frame(
      covariate     = rownames(fg_summary$coef),
      subHR         = round(exp(fg_summary$coef[, "coef"]), 4),
      se            = round(fg_summary$coef[, "se(coef)"], 4),
      z             = round(fg_summary$coef[, "z"], 4),
      p_value       = round(fg_summary$coef[, "p-value"], 4),
      subHR_ci_low  = round(exp(fg_summary$conf.int[, "2.5%"]), 4),
      subHR_ci_high = round(exp(fg_summary$conf.int[, "97.5%"]), 4)
    )

    write_csv(fg_df, file.path(opt$tables, "finegray_results.csv"))
    message("\nFine-Gray subdistribution HRs:")
    print(fg_df %>% filter(grepl("drug_", covariate)))
    message("Fine-Gray results saved: finegray_results.csv")

    # Drug-class subdistribution HRs specifically
    drug_fg <- fg_df %>% filter(grepl("drug_", covariate))
    if (nrow(drug_fg) > 0) {
      message(sprintf(
        "GLP-1 RA vs Metformin — subdistribution HR: %.3f (95%% CI: %.3f–%.3f, p=%.4f)",
        drug_fg[drug_fg$covariate == "drug_glp1",  "subHR"],
        drug_fg[drug_fg$covariate == "drug_glp1",  "subHR_ci_low"],
        drug_fg[drug_fg$covariate == "drug_glp1",  "subHR_ci_high"],
        drug_fg[drug_fg$covariate == "drug_glp1",  "p_value"]
      ))
    }
  }
}

message("\nSurvival analysis complete.")
