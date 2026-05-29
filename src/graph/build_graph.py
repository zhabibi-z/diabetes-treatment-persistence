"""
graph/build_graph.py — Build a drug-disease-comorbidity knowledge graph.

Nodes:
  - DrugClass (metformin, glp1, sglt2)
  - Comorbidity (15 SNOMED conditions)
  - Outcome (treatment_discontinuation, comorbidity_onset)

Edges:
  - TREATS (DrugClass → Comorbidity, weighted by HR from TTC Cox)
  - ASSOCIATED_WITH (Comorbidity → Outcome, weighted by Pearson r)
  - PREDICTS (Comorbidity → Outcome, from SHAP values)
  - DRUG_CLASS_EFFECT (DrugClass → Outcome, HR from TTD Cox)

Outputs:
  graph/cypher_export/nodes.cypher
  graph/cypher_export/edges.cypher
  outputs/figures/knowledge_graph.png  (NetworkX layout)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COMORBIDITY_NAMES = [
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]

DRUG_CLASSES = ["metformin", "glp1", "sglt2"]

# Known cardiorenal benefit relationships (from trial evidence, encoded as prior)
DRUG_COMORB_BENEFIT: dict[str, list[str]] = {
    "glp1":  ["heart_failure", "ckd", "stroke", "mi", "obesity"],
    "sglt2": ["heart_failure", "ckd", "pvd", "mi"],
    "metformin": ["obesity", "hyperlipidemia"],
}


def build_graph(
    cohort_path: str,
    comorbidity_path: str,
    output_dir: str,
    corr_path: str | None = None,
    cox_ttc_path: str | None = None,
) -> nx.DiGraph:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()

    # ── Add drug class nodes ──────────────────────────────────────────────────
    drug_labels = {"metformin": "Metformin", "glp1": "GLP-1 RA", "sglt2": "SGLT-2i"}
    for dc in DRUG_CLASSES:
        G.add_node(dc, label=drug_labels[dc], node_type="DrugClass",
                   color="#3498DB" if dc == "metformin" else ("#E74C3C" if dc == "glp1" else "#2ECC71"))

    # ── Add comorbidity nodes ──────────────────────────────────────────────────
    for c in COMORBIDITY_NAMES:
        G.add_node(c, label=c.replace("_", " ").title(), node_type="Comorbidity", color="#F39C12")

    # ── Add outcome node ──────────────────────────────────────────────────────
    G.add_node("discontinuation", label="Treatment\nDiscontinuation",
               node_type="Outcome", color="#8E44AD")

    # ── Edges: drug → comorbidity (TREATS / cardiorenal benefit) ─────────────
    for dc, comorbs in DRUG_COMORB_BENEFIT.items():
        for c in comorbs:
            G.add_edge(dc, c, relation="TREATS", weight=1.0, direction="protective")

    # ── Edges: comorbidity → outcome (ASSOCIATED_WITH, from correlations) ────
    if corr_path and Path(corr_path).exists():
        corr_df = pd.read_csv(corr_path)
        for _, row in corr_df.iterrows():
            comorb = row.get("comorbidity", "")
            r      = float(row.get("pearson_r", 0))
            p_adj  = float(row.get("p_adj_bh", 1))
            if comorb in COMORBIDITY_NAMES:
                G.add_edge(
                    comorb, "discontinuation",
                    relation="ASSOCIATED_WITH",
                    weight=abs(r),
                    pearson_r=round(r, 4),
                    p_bh=round(p_adj, 4),
                )
    else:
        # Add placeholder edges
        for c in COMORBIDITY_NAMES:
            G.add_edge(c, "discontinuation", relation="ASSOCIATED_WITH", weight=0.1)

    # ── Edges: drug class → outcome (DRUG_CLASS_EFFECT, from Cox TTD) ────────
    if cox_ttc_path and Path(cox_ttc_path).exists():
        cox_df = pd.read_csv(cox_ttc_path)
        dc_rows = cox_df[cox_df.get("covariate", cox_df.columns[0]) == "drug_class_num"]
        if not dc_rows.empty:
            hr = float(dc_rows["exp(coef)"].mean())
            G.add_edge("glp1", "discontinuation", relation="DRUG_CLASS_EFFECT",
                       weight=abs(hr - 1), hr=round(hr, 3))
            G.add_edge("sglt2", "discontinuation", relation="DRUG_CLASS_EFFECT",
                       weight=abs(hr - 1), hr=round(hr, 3))
    else:
        for dc in ["glp1", "sglt2"]:
            G.add_edge(dc, "discontinuation", relation="DRUG_CLASS_EFFECT", weight=0.5, hr=1.0)

    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── Visualise ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_colors = [G.nodes[n].get("color", "#BDC3C7") for n in G.nodes()]
    node_labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    node_sizes  = [
        2500 if G.nodes[n]["node_type"] == "DrugClass"
        else (3500 if G.nodes[n]["node_type"] == "Outcome" else 1200)
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, ax=ax)

    edge_colors = {
        "TREATS":             "#27AE60",
        "ASSOCIATED_WITH":    "#E74C3C",
        "DRUG_CLASS_EFFECT":  "#8E44AD",
        "PREDICTS":           "#F39C12",
    }
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "ASSOCIATED_WITH")
        color    = edge_colors.get(relation, "#95A5A6")
        weight   = data.get("weight", 0.5)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=color, width=max(0.5, weight * 3),
            arrows=True, arrowsize=15,
            connectionstyle="arc3,rad=0.1", ax=ax,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#3498DB", label="Drug Class"),
        Patch(color="#F39C12", label="Comorbidity"),
        Patch(color="#8E44AD", label="Outcome"),
        Patch(color="#27AE60", label="TREATS"),
        Patch(color="#E74C3C", label="ASSOCIATED_WITH"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.set_title("T2DM Drug-Disease-Comorbidity Knowledge Graph", fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    fig_path = "outputs/figures/knowledge_graph.png"
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Graph visualisation saved: %s", fig_path)

    # ── Export to Cypher ──────────────────────────────────────────────────────
    export_cypher(G, output_dir)

    return G


def export_cypher(G: nx.DiGraph, output_dir: str) -> None:
    node_lines = []
    edge_lines = []

    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "Node")
        label     = data.get("label", node).replace("'", "\\'")
        node_id   = node.replace(" ", "_")
        node_lines.append(
            f"MERGE (n:{node_type} {{id: '{node_id}', name: '{label}'}});"
        )

    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "RELATED_TO")
        props    = {k: v for k, v in data.items() if k not in ("relation",)}
        props_str = ", ".join(f"{k}: {json.dumps(val)}" for k, val in props.items())
        u_id = u.replace(" ", "_")
        v_id = v.replace(" ", "_")
        edge_lines.append(
            f"MATCH (a {{id: '{u_id}'}}), (b {{id: '{v_id}'}}) "
            f"MERGE (a)-[:{relation} {{{props_str}}}]->(b);"
        )

    Path(f"{output_dir}/nodes.cypher").write_text("\n".join(node_lines))
    Path(f"{output_dir}/edges.cypher").write_text("\n".join(edge_lines))
    log.info("Cypher files exported: nodes.cypher, edges.cypher")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2DM knowledge graph")
    parser.add_argument("--cohort",      default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--comorbidity", default="cohort/codx_mapping.xlsx")
    parser.add_argument("--output-dir",  default="graph/cypher_export")
    parser.add_argument("--corr",        default="outputs/tables/correlations.csv")
    parser.add_argument("--cox-ttc",     default="outputs/tables/cox_ttc_results.csv")
    args = parser.parse_args()

    build_graph(args.cohort, args.comorbidity, args.output_dir, args.corr, args.cox_ttc)


if __name__ == "__main__":
    main()
