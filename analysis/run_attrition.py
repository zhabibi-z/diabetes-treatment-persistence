"""
analysis/run_attrition.py — CONSORT-style patient attrition flow diagram.

Traces every inclusion/exclusion step from the raw OMOP population to the
final machine-learning analytic sample, producing a publication-ready
flowchart. The attrition counts are queried or derived from pipeline
artefacts that must already exist (run after bootstrap.sh Steps 1–5).

Attrition sequence
------------------
1. Total persons in OMOP database
2. Persons with ≥1 T2DM diagnosis (ICD-10: E11.x / SNOMED: 44054006)
3. Persons initiating metformin, GLP-1 RA, or SGLT-2i after T2DM diagnosis
4. New users only (no antidiabetic use in prior 365-day washout window)
5. Post-cohort-matching sample (cohort_baseline → cohort_matched)
6. ML analytic sample (≥365-day follow-up restriction)

Each step reports: n retained, n excluded, and the exclusion reason.

References
----------
CONSORT 2010: Schulz KF et al. BMJ. 2010;340:c332.
STaRT-RWE attrition reporting: Wang SV et al. BMJ. 2021;372:m4856.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS, COHORT, ML

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Attrition step definition ─────────────────────────────────────────────────

def _collect_counts(
    db_path: Path,
    cohort_baseline: Path,
    cohort_matched: Path,
    ttd_events: Path,
) -> list[dict]:
    """
    Assemble patient counts for each attrition step.

    Falls back to CSV-derived counts when DuckDB is unavailable, because
    some users run the pipeline without Synthea and rely on pre-generated
    artefacts. The OMOP-query counts are informative; the CSV counts are
    authoritative for the matched and ML samples.
    """
    steps: list[dict] = []

    # Steps 1–4 require the OMOP database
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)

        n_total = con.execute("SELECT COUNT(*) FROM person").fetchone()[0]
        steps.append({
            "label": "Persons in OMOP database",
            "n": n_total,
            "excluded": 0,
            "reason": None,
        })

        n_t2dm = con.execute(
            "SELECT COUNT(DISTINCT person_id) FROM condition_occurrence "
            "WHERE condition_concept_id = 44054006"  # SNOMED T2DM
        ).fetchone()[0]
        steps.append({
            "label": "With ≥1 T2DM diagnosis",
            "n": n_t2dm,
            "excluded": n_total - n_t2dm,
            "reason": "No T2DM diagnosis on record",
        })

        # Drug initiators: count persons in cohort_baseline (already filtered to
        # new users with antidiabetic initiation; querying raw drug_exposure would
        # overcount by including all exposures, not just index prescriptions)
        con.close()

    except Exception as exc:
        log.warning("DuckDB query skipped (%s) — using CSV-derived counts only.", exc)
        steps = [
            {"label": "Persons in OMOP database", "n": None, "excluded": 0, "reason": None},
            {"label": "With ≥1 T2DM diagnosis",   "n": None, "excluded": None,
             "reason": "No T2DM diagnosis on record"},
        ]

    # Steps 5–6: derived from pipeline CSVs (always available post-pipeline)
    if cohort_baseline.exists():
        df_base = pd.read_csv(cohort_baseline)
        n_base = len(df_base)
        steps.append({
            "label": "Initiated study drug after T2DM\n(new-user criterion met)",
            "n": n_base,
            "excluded": None,
            "reason": "Prior antidiabetic use in 365-day washout",
        })
    else:
        n_base = None
        steps.append({
            "label": "Initiated study drug after T2DM\n(new-user criterion met)",
            "n": None, "excluded": None,
            "reason": "cohort_baseline.csv not found",
        })

    if cohort_matched.exists():
        df_matched = pd.read_csv(cohort_matched)
        n_matched = len(df_matched)
        n_excluded_match = (n_base - n_matched) if n_base else None
        steps.append({
            "label": "Propensity score matched\n(MatchIt, nearest-neighbour 1:1:1)",
            "n": n_matched,
            "excluded": n_excluded_match,
            "reason": "No match found within calliper",
        })
    else:
        n_matched = None
        steps.append({
            "label": "Propensity score matched",
            "n": None, "excluded": None, "reason": "cohort_matched.csv not found",
        })

    if ttd_events.exists():
        ttd = pd.read_csv(ttd_events)
        n_ttd = len(ttd)
        n_ml  = ttd[ttd["ttd_days"] >= ML.min_followup_days].shape[0] if "ttd_days" in ttd.columns else None
        n_excluded_ml = (n_ttd - n_ml) if (n_ttd and n_ml is not None) else None
        steps.append({
            "label": f"ML analytic sample\n(≥{ML.min_followup_days}-day follow-up for 1-year label)",
            "n": n_ml,
            "excluded": n_excluded_ml,
            "reason": f"Follow-up < {ML.min_followup_days} days",
        })
    else:
        steps.append({
            "label": "ML analytic sample",
            "n": None, "excluded": None, "reason": "ttd_events.csv not found",
        })

    return steps


def _fmt(n) -> str:
    return f"n = {n:,}" if n is not None else "n = N/A"


# ── Flowchart rendering ───────────────────────────────────────────────────────

def draw_attrition_diagram(steps: list[dict], output_path: Path) -> None:
    """
    Render a CONSORT-style attrition flowchart.

    Layout: inclusion boxes form a vertical spine on the left; exclusion
    boxes branch to the right. Box dimensions scale with step count to
    remain readable across 5–8 steps.

    Reference: Schulz KF et al. CONSORT 2010. BMJ. 2010;340:c332.
    """
    n_steps = len(steps)
    fig_h   = max(10, 2.2 * n_steps)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Style constants
    BOX_W      = 4.2
    BOX_H      = 1.3
    EX_BOX_W   = 3.8
    LEFT_X     = 0.4
    EX_X       = 5.8
    Y_STEP     = fig_h / n_steps
    BOX_STYLE  = dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#2980B9", linewidth=1.5)
    EX_STYLE   = dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor="#E74C3C", linewidth=1.2)
    ARROW_KW   = dict(arrowstyle="-|>", color="#2C3E50", lw=1.4)

    prev_cy = None

    for i, step in enumerate(steps):
        cy = fig_h - (i + 0.6) * Y_STEP

        # Main inclusion box
        label_text = f"{step['label']}\n{_fmt(step['n'])}"
        ax.text(
            LEFT_X + BOX_W / 2, cy,
            label_text,
            ha="center", va="center", fontsize=9.5, wrap=True,
            bbox=BOX_STYLE, zorder=3,
        )

        # Downward arrow from previous box
        if prev_cy is not None:
            ax.annotate(
                "", xy=(LEFT_X + BOX_W / 2, cy + BOX_H / 2),
                xytext=(LEFT_X + BOX_W / 2, prev_cy - BOX_H / 2),
                arrowprops=ARROW_KW, zorder=2,
            )

        # Exclusion branch (only when there are excluded patients)
        if step["excluded"] is not None and step["reason"]:
            ex_y = cy
            ax.annotate(
                "", xy=(EX_X, ex_y),
                xytext=(LEFT_X + BOX_W, ex_y),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.2),
                zorder=2,
            )
            ex_text = f"Excluded: {_fmt(step['excluded'])}\n{step['reason']}"
            ax.text(
                EX_X + EX_BOX_W / 2, ex_y,
                ex_text,
                ha="center", va="center", fontsize=8.5,
                bbox=EX_STYLE, zorder=3,
            )

        prev_cy = cy

    plt.title(
        "Patient Attrition — T2DM Persistence RWE Cohort\n"
        f"(CONSORT 2010, STaRT-RWE; grace period = {COHORT.grace_period_days} days)",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Attrition diagram saved: %s", output_path)


# ── CSV summary ───────────────────────────────────────────────────────────────

def save_attrition_table(steps: list[dict], output_path: Path) -> None:
    rows = []
    for i, s in enumerate(steps, 1):
        rows.append({
            "step":     i,
            "label":    s["label"].replace("\n", " "),
            "n":        s["n"],
            "excluded": s["excluded"],
            "reason":   s["reason"] or "",
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info("Attrition table saved: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CONSORT attrition flow diagram.")
    parser.add_argument("--db-path",        default=str(PATHS.omop_db))
    parser.add_argument("--cohort-baseline",default=str(PATHS.cohort_baseline))
    parser.add_argument("--cohort-matched", default=str(PATHS.cohort_matched))
    parser.add_argument("--ttd-file",       default=str(PATHS.ttd_events))
    parser.add_argument("--output-dir",     default=str(PATHS.outputs))
    args = parser.parse_args()

    out_fig = Path(args.output_dir) / "figures" / "attrition_diagram.png"
    out_csv = Path(args.output_dir) / "tables"  / "attrition_table.csv"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    steps = _collect_counts(
        db_path         = Path(args.db_path),
        cohort_baseline = Path(args.cohort_baseline),
        cohort_matched  = Path(args.cohort_matched),
        ttd_events      = Path(args.ttd_file),
    )

    for s in steps:
        log.info("  %-60s %s", s["label"].replace("\n", " "), _fmt(s["n"]))

    draw_attrition_diagram(steps, out_fig)
    save_attrition_table(steps, out_csv)


if __name__ == "__main__":
    main()
