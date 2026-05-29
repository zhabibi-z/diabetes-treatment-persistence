#!/usr/bin/env bash
# bootstrap.sh — One-command end-to-end pipeline runner for T2DM Persistence RWE.
# Run from project root: bash scripts/bootstrap.sh
# All steps log to logs/bootstrap_YYYYMMDD_HHMMSS.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bootstrap_$(date +%Y%m%d_%H%M%S).log"

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${BOLD}[STEP $1]${NC} $2"; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

echo "============================================================"
echo "  T2DM Persistence RWE — Bootstrap Pipeline"
echo "  $(date)"
echo "  Log: $LOG_FILE"
echo "============================================================"

# ── Load .env ────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    source .env
    set +a
    ok ".env loaded"
else
    warn ".env not found — using defaults. Some features (chatbot) may fail."
fi

# ── Activate venv ─────────────────────────────────────────────────
if [ ! -f venv/bin/activate ]; then
    fail "venv not found. Run: python3.11 -m venv venv && venv/bin/pip install -r requirements.txt"
fi
source venv/bin/activate
ok "Python venv activated: $(python --version)"

# ── Create output dirs ────────────────────────────────────────────
mkdir -p data/{synthea_output,omop} outputs/{tables,figures,models}
ok "Output directories ready"

# ── Step 1: Synthea data generation ──────────────────────────────
step 1 "Generating Synthea synthetic patients"
SYNTHEA_COUNT="${SYNTHEA_PATIENT_COUNT:-30000}"
SYNTHEA_SEED="${SYNTHEA_SEED:-42}"

if [ -d "synthea" ] && [ -f "synthea/run_synthea" ]; then
    cd synthea
    ./run_synthea \
        --exporter.csv.export=true \
        --exporter.baseDirectory="../data/synthea_output" \
        --generate.log_patients.detail=NONE \
        -p "$SYNTHEA_COUNT" \
        -s "$SYNTHEA_SEED" \
        diabetes
    cd "$PROJECT_ROOT"
    ok "Synthea generated $SYNTHEA_COUNT patients"
elif ls data/synthea_output/csv/*.csv &>/dev/null 2>&1; then
    warn "Synthea not found but CSV outputs already exist — skipping generation"
else
    warn "Synthea not installed. Generating synthetic fallback data via Python."
    python src/etl/synthea_to_omop.py --generate-synthetic --patients "$SYNTHEA_COUNT" --seed "$SYNTHEA_SEED"
fi

# ── Step 2: ETL — Synthea → OMOP DuckDB ──────────────────────────
step 2 "ETL: Loading Synthea output into OMOP CDM DuckDB"
python src/etl/synthea_to_omop.py \
    --synthea-dir data/synthea_output \
    --db-path "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --patients "$SYNTHEA_COUNT" \
    --seed "$SYNTHEA_SEED"
ok "OMOP CDM DuckDB loaded"

# ── Step 3: Cohort construction ───────────────────────────────────
step 3 "Building mutually exclusive new-user cohorts"
python src/cohort/build_cohort.py \
    --db-path "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --output-dir outputs/tables
ok "Cohorts built"

# ── Step 4: Propensity score matching (R) ────────────────────────
step 4 "Propensity score matching (MatchIt / cobalt in R)"
Rscript src/r/cohort_matching.R \
    --input  outputs/tables/cohort_baseline.csv \
    --output outputs/tables/cohort_matched.csv \
    --figures outputs/figures
ok "PS matching complete — balance diagnostics in outputs/figures/"

# ── Step 5: TTD analysis ──────────────────────────────────────────
step 5 "Time-to-discontinuation analysis (90-day grace)"
python src/analysis/run_ttd.py \
    --db-path "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --matched-cohort outputs/tables/cohort_matched.csv \
    --output-dir outputs
ok "TTD analysis complete"

# ── Step 6: TTC analysis ──────────────────────────────────────────
step 6 "Time-to-comorbidity Kaplan-Meier"
python src/analysis/run_ttc.py \
    --db-path "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --matched-cohort outputs/tables/cohort_matched.csv \
    --output-dir outputs
ok "TTC analysis complete"

# ── Step 7: Time-varying Cox ──────────────────────────────────────
step 7 "Time-varying Cox model (comorbidity 0→1 transitions)"
python src/analysis/run_cox_timevarying.py \
    --db-path "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --matched-cohort outputs/tables/cohort_matched.csv \
    --output-dir outputs
ok "Time-varying Cox complete"

# ── Step 8: TTC Cox ────────────────────────────────────────────────
step 8 "TTC Cox model (comorbidity onset as outcome)"
python src/analysis/run_cox_ttc.py \
    --db-path "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --matched-cohort outputs/tables/cohort_matched.csv \
    --output-dir outputs
ok "TTC Cox complete"

# ── Step 9: Pearson correlations ───────────────────────────────────
step 9 "Pearson correlations (comorbidity × TTD)"
python src/analysis/run_correlations.py \
    --matched-cohort outputs/tables/cohort_matched.csv \
    --ttd-file outputs/tables/ttd_events.csv \
    --output-dir outputs
ok "Correlation analysis complete"

# ── Step 10: Stratified KM ─────────────────────────────────────────
step 10 "Per-comorbidity stratified Kaplan-Meier"
python src/analysis/run_km_stratified.py \
    --matched-cohort outputs/tables/cohort_matched.csv \
    --output-dir outputs
ok "Stratified KM complete"

# ── Step 11: R survival + hypothesis tests ─────────────────────────
step 11 "R survival analysis (survminer KM + forest plot)"
Rscript src/r/survival_analysis.R \
    --ttd-file outputs/tables/ttd_events.csv \
    --cohort   outputs/tables/cohort_matched.csv \
    --output   outputs/figures

step 11b "R hypothesis tests (Shapiro-Wilk, Kruskal-Wallis, Dunn BH-FDR)"
Rscript src/r/hypothesis_tests.R \
    --ttd-file outputs/tables/ttd_events.csv \
    --cohort   outputs/tables/cohort_matched.csv \
    --output   outputs/tables
ok "R analyses complete"

# ── Step 12: Machine learning ──────────────────────────────────────
step 12 "XGBoost + UMAP + SHAP (5-fold CV, 28-feature leakage-corrected model)"
python src/ml/train.py \
    --cohort     outputs/tables/cohort_matched.csv \
    --ttd-file   outputs/tables/ttd_events.csv \
    --output-dir outputs
ok "ML training complete (calibration curve + baseline comparison + fairness report)"

# ── Step 12b: IPTW sensitivity analysis ────────────────────────────
step "12b" "Stabilised IPTW sensitivity analysis (Hernan & Robins 2020)"
python src/analysis/run_iptw.py \
    --cohort     outputs/tables/cohort_matched.csv \
    --ttd-file   outputs/tables/ttd_events.csv \
    --output-dir outputs
ok "IPTW complete — weights, balance table, weight distribution plot"

# ── Step 12c: Patient attrition CONSORT diagram ────────────────────
step "12c" "Patient attrition CONSORT flow diagram (Schulz et al. BMJ 2010)"
python src/analysis/run_attrition.py \
    --db-path        "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --cohort-baseline outputs/tables/cohort_baseline.csv \
    --cohort-matched  outputs/tables/cohort_matched.csv \
    --ttd-file        outputs/tables/ttd_events.csv \
    --output-dir      outputs
ok "Attrition diagram saved → outputs/figures/attrition_diagram.png"

# ── Step 12d: Negative control outcome analysis ────────────────────
step "12d" "Negative control outcome analysis (Lipsitch 2010; Schuemie 2018)"
python src/analysis/run_negative_control.py \
    --db-path        "${OMOP_DB_PATH:-data/omop/omop.duckdb}" \
    --cohort-matched outputs/tables/cohort_matched.csv \
    --output-dir     outputs
ok "Negative control results → outputs/tables/negative_control_results.csv"

# ── Step 13: Knowledge graph ───────────────────────────────────────
step 13 "Knowledge graph (NetworkX → Cypher)"
python src/graph/build_graph.py \
    --cohort     outputs/tables/cohort_matched.csv \
    --comorbidity cohort/codx_mapping.xlsx \
    --output-dir graph/cypher_export
ok "Graph built"

# ── Step 14: Report formatting (Perl) ─────────────────────────────
step 14 "Generating text summary report (Perl)"
perl scripts/report_formatter.pl outputs/tables/ > outputs/study_report.txt
ok "Report written to outputs/study_report.txt"

# ── Step 15: Launch Streamlit dashboard ───────────────────────────
step 15 "Streamlit dashboard"
echo ""
echo "============================================================"
echo -e "${GREEN}${BOLD}  Pipeline complete!${NC}"
echo "  Outputs: outputs/tables/, outputs/figures/"
echo "  Report:  outputs/study_report.txt"
echo ""
echo "  To launch the dashboard:"
echo "    source venv/bin/activate"
echo "    streamlit run src/app/app.py"
echo "============================================================"
