#!/usr/bin/env Rscript
# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
# Install all R package dependencies for the T2DM Persistence RWE study.
# Run once before the pipeline: Rscript scripts/install_r_packages.R

# Keep an already-configured repository (e.g. Posit Public Package Manager on CI,
# which serves fast pre-built binaries); otherwise fall back to CRAN.
.repos <- getOption("repos")
if (is.null(.repos[["CRAN"]]) || .repos[["CRAN"]] == "@CRAN@") {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

required_packages <- c(
  "MatchIt",      # Propensity score matching
  "cobalt",       # Balance diagnostics and love plots
  "survival",     # Core survival functions (usually pre-installed)
  "survminer",    # KM plots with ggplot2 aesthetics
  "rstatix",      # Pipe-friendly stats (Kruskal-Wallis, Dunn)
  "nortest",      # Kolmogorov-Smirnov and other normality tests
  "ggpubr",       # ggplot2 publication-ready helpers
  "forestplot",   # Forest plots for HR tables
  "broom",        # Tidy model outputs
  "dplyr",        # Data manipulation
  "tidyr",        # Data reshaping
  "readr",        # CSV reading
  "ggplot2",      # Base plotting
  "scales",       # Axis formatting
  "purrr",        # Functional programming
  "stringr",      # String manipulation
  "lubridate",    # Date arithmetic
  "DBI",          # Database interface
  "duckdb",       # DuckDB R driver
  "FSA",          # Dunn's test (dunnTest)
  "DescTools",    # Descriptive statistics utilities
  "optparse",     # Command-line option parsing
  # ── Added for competing risks and calibration ──────────────────────────────
  "cmprsk"        # Fine-Gray subdistribution hazard model
                  # Fine JP, Gray RJ. JASA. 1999;94(446):496-509.
  # NOTE: 'pec' (Brier score for survival) was removed — it is not used by any R script and its
  # dependency 'rms'/'riskRegression' repeatedly failed to install on the CI mirror, breaking CI.
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing: %s", pkg))
    # dependencies = NA installs Depends/Imports/LinkingTo (what the packages
    # need to load), but NOT Suggests — which for MatchIt/cobalt pull in optional
    # optimal-matching solvers (Rsymphony, gurobi, designmatch) that need system
    # libraries and are unused here (the study uses nearest-neighbour matching).
    install.packages(pkg, dependencies = NA)
  } else {
    message(sprintf("Already installed: %s", pkg))
  }
}

invisible(lapply(required_packages, install_if_missing))

# Verify all installed
missing <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {
  stop(sprintf("Failed to install: %s", paste(missing, collapse = ", ")))
}

message("\nAll R packages successfully installed and verified.")
