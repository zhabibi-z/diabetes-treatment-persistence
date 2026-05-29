#!/usr/bin/env Rscript
# hypothesis_tests.R — Normality tests, Mann-Whitney U, Kruskal-Wallis + Dunn BH-FDR.

suppressPackageStartupMessages({
  library(rstatix)
  library(nortest)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(optparse)
})

option_list <- list(
  make_option("--ttd-file", default = "outputs/tables/ttd_events.csv"),
  make_option("--cohort",   default = "outputs/tables/cohort_matched.csv"),
  make_option("--output",   default = "outputs/tables")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(opt$`ttd-file`)) stop("ttd_events.csv not found")

ttd    <- read_csv(opt$`ttd-file`, show_col_types = FALSE)
cohort <- read_csv(opt$cohort,     show_col_types = FALSE)

if (!"drug_class" %in% names(ttd)) {
  ttd <- ttd %>% left_join(select(cohort, person_id, drug_class), by = "person_id")
}

ttd <- ttd %>% filter(!is.na(ttd_days), ttd_days >= 0, !is.na(drug_class))

# ── Normality tests ───────────────────────────────────────────────────────────
normality_results <- list()

for (dc in unique(ttd$drug_class)) {
  x <- ttd$ttd_days[ttd$drug_class == dc]
  n <- length(x)
  message(sprintf("Normality tests for %s (n=%d)", dc, n))

  # Shapiro-Wilk (valid for n ≤ 5000)
  sw_p <- if (n <= 5000 && n >= 3) shapiro.test(x)$p.value else NA_real_

  # Kolmogorov-Smirnov (Lilliefors)
  ks_p <- tryCatch(lillie.test(x)$p.value, error = function(e) NA_real_)

  normality_results[[dc]] <- data.frame(
    drug_class = dc, n = n,
    shapiro_wilk_p = round(sw_p, 4),
    ks_lilliefors_p = round(ks_p, 4),
    normal = (sw_p > 0.05 && !is.na(sw_p))
  )
}

norm_df <- bind_rows(normality_results)
message("\nNormality test results:")
print(norm_df)
write_csv(norm_df, file.path(opt$output, "normality_tests.csv"))

# ── Kruskal-Wallis test (3-group) ─────────────────────────────────────────────
kw <- kruskal.test(ttd_days ~ drug_class, data = ttd)
message(sprintf("\nKruskal-Wallis: chi2=%.3f, df=%d, p=%.4f",
                kw$statistic, kw$parameter, kw$p.value))

kw_df <- data.frame(
  test = "Kruskal-Wallis",
  statistic = round(kw$statistic, 3),
  df = kw$parameter,
  p_value = round(kw$p.value, 4)
)

# ── Dunn post-hoc with BH-FDR correction ────────────────────────────────────
dunn_res <- dunn_test(ttd_days ~ drug_class, data = ttd, p.adjust.method = "BH")
message("\nDunn's post-hoc test (BH-FDR):")
print(dunn_res)

write_csv(dunn_res, file.path(opt$output, "dunn_posthoc.csv"))
write_csv(kw_df,   file.path(opt$output, "kruskal_results.csv"))

# ── Mann-Whitney U (pairwise) ────────────────────────────────────────────────
classes <- unique(ttd$drug_class)
mwu_results <- list()
pairs <- combn(classes, 2, simplify = FALSE)

for (pair in pairs) {
  x <- ttd$ttd_days[ttd$drug_class == pair[1]]
  y <- ttd$ttd_days[ttd$drug_class == pair[2]]
  w <- wilcox.test(x, y, exact = FALSE)
  mwu_results[[paste(pair, collapse = "_vs_")]] <- data.frame(
    group1 = pair[1], group2 = pair[2],
    W = w$statistic, p_value = round(w$p.value, 4)
  )
}

mwu_df <- bind_rows(mwu_results)
mwu_df$p_adj_bh <- p.adjust(mwu_df$p_value, method = "BH")
message("\nMann-Whitney U pairwise (BH-FDR):")
print(mwu_df)
write_csv(mwu_df, file.path(opt$output, "mann_whitney_results.csv"))

message("\nHypothesis tests complete.")
