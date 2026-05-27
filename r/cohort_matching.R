#!/usr/bin/env Rscript
# cohort_matching.R — 1:5 propensity score matching using MatchIt (Ho et al. 2011)
# with balance diagnostics via cobalt.
#
# Matching strategy: nearest-neighbour, caliper = 0.20 SD of logit PS,
# without replacement (Austin 2011). Multi-class: pairwise GLP1 vs metformin
# and SGLT2 vs metformin (metformin as reference).
#
# Outputs:
#   outputs/tables/cohort_matched.csv
#   outputs/figures/love_plot_glp1_vs_met.png
#   outputs/figures/love_plot_sglt2_vs_met.png
#   outputs/figures/ps_distribution.png

suppressPackageStartupMessages({
  library(MatchIt)
  library(cobalt)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(optparse)
})

option_list <- list(
  make_option("--input",   default = "outputs/tables/cohort_baseline.csv"),
  make_option("--output",  default = "outputs/tables/cohort_matched.csv"),
  make_option("--figures", default = "outputs/figures")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$figures, recursive = TRUE, showWarnings = FALSE)

message("Loading cohort: ", opt$input)
cohort <- read_csv(opt$input, show_col_types = FALSE)

comorbidity_cols <- c(
  "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
  "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
  "sleep_apnea", "nafld", "pvd", "stroke", "mi"
)
present_comorbidities <- intersect(comorbidity_cols, names(cohort))

covariate_formula <- paste(
  "~", paste(c("age_at_index", "gender_concept_id", "cci", present_comorbidities),
             collapse = " + ")
)

run_matching <- function(data, treatment_class, reference_class, label) {
  sub <- data %>% filter(drug_class %in% c(treatment_class, reference_class))
  sub$treated <- as.integer(sub$drug_class == treatment_class)

  if (sum(sub$treated == 1) < 5 || sum(sub$treated == 0) < 5) {
    warning(sprintf("Too few patients for %s matching — skipping", label))
    return(NULL)
  }

  formula_str <- paste("treated", covariate_formula)
  match_formula <- as.formula(formula_str)

  message(sprintf("Matching: %s vs %s (n=%d treated, n=%d control)",
                  treatment_class, reference_class,
                  sum(sub$treated == 1), sum(sub$treated == 0)))

  m_out <- tryCatch(
    matchit(
      match_formula,
      data    = sub,
      method  = "nearest",
      ratio   = 5,
      caliper = 0.20,
      std.caliper = TRUE,
      replace = FALSE
    ),
    error = function(e) {
      warning(sprintf("MatchIt failed for %s: %s", label, e$message))
      NULL
    }
  )
  if (is.null(m_out)) return(NULL)

  # Balance summary
  bal <- summary(m_out, standardize = TRUE)
  message(sprintf("  Max SMD post-match: %.3f", max(abs(bal$sum.matched[, "Std. Mean Diff."]), na.rm = TRUE)))

  # Love plot
  love_plot <- love.plot(
    m_out,
    binary  = "std",
    thresholds = c(m = 0.1),
    title   = sprintf("Covariate Balance: %s vs %s", label, reference_class),
    colors  = c("Before Matching" = "#E74C3C", "After Matching" = "#2ECC71")
  )
  ggsave(
    filename = file.path(opt$figures, sprintf("love_plot_%s_vs_met.png", label)),
    plot = love_plot, width = 8, height = 6, dpi = 150
  )

  matched_data <- match.data(m_out)
  matched_data$match_pair <- label
  matched_data
}

# Run pairwise matching
match_glp1  <- run_matching(cohort, "glp1",  "metformin", "glp1")
match_sglt2 <- run_matching(cohort, "sglt2", "metformin", "sglt2")

# Combine matched cohorts (metformin patients appear in both; deduplicate by taking union)
matched_list <- Filter(Negate(is.null), list(match_glp1, match_sglt2))

if (length(matched_list) == 0) {
  # Fallback: use unmatched cohort
  warning("No matched cohort produced — using unmatched cohort as fallback")
  write_csv(cohort, opt$output)
} else {
  matched <- bind_rows(matched_list) %>%
    group_by(person_id) %>%
    slice_head(n = 1) %>%
    ungroup()
  write_csv(matched, opt$output)
  message(sprintf("Matched cohort written: %s (%d rows)", opt$output, nrow(matched)))
}

# PS distribution plot
if (length(matched_list) > 0) {
  ps_data <- bind_rows(matched_list) %>%
    select(drug_class, distance) %>%
    filter(!is.na(distance))

  if (nrow(ps_data) > 0) {
    p_ps <- ggplot(ps_data, aes(x = distance, fill = drug_class)) +
      geom_density(alpha = 0.5) +
      scale_fill_manual(values = c("metformin" = "#3498DB", "glp1" = "#E74C3C", "sglt2" = "#2ECC71")) +
      labs(title = "Propensity Score Distribution (Post-Matching)",
           x = "Propensity Score", y = "Density", fill = "Drug Class") +
      theme_bw(base_size = 12)
    ggsave(file.path(opt$figures, "ps_distribution.png"), p_ps, width = 7, height = 4, dpi = 150)
    message("PS distribution plot saved")
  }
}

message("Cohort matching complete.")
