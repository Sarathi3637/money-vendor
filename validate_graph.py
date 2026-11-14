"""
Graph Validation Module.
Parses the independent laundering pattern file 'BEGIN LAUNDERING ATTEMPT - FAN-IN.txt',
builds the transaction graph, runs the graph engine pattern detectors, and generates
a comprehensive validation summary report with per-topology performance metrics.
"""

import os
import sys
import json
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.validate_patterns import parse_laundering_patterns
from src.graph_engine import AMLGraphEngine
from src.graph_patterns import detect_fan_in, detect_fan_out, detect_cycles, detect_rapid_multi_hop, detect_scatter_gather
from src.investigation_cases import CaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_graph_validation(
    pattern_txt_path: str = "data/BEGIN LAUNDERING ATTEMPT - FAN-IN.txt",
    accounts_path: str = "data/HI-Small_accounts.csv",
    outputs_dir: str = "outputs",
):
    """Run graph-based pattern detection against external out-of-distribution laundering patterns."""
    os.makedirs(outputs_dir, exist_ok=True)
    logger.info(f"Parsing laundering pattern dataset from {pattern_txt_path}...")

    df_patterns = parse_laundering_patterns(pattern_txt_path)
    logger.info(f"Loaded {len(df_patterns):,} pattern validation transactions.")

    # Standardize column names to match cleaned format
    from src.clean import COLUMN_MAPPING, load_accounts_metadata
    df = df_patterns.rename(columns=COLUMN_MAPPING)
    df["from_account"] = df["from_account"].astype(str).str.strip().str.upper()
    df["to_account"] = df["to_account"].astype(str).str.strip().str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0.0)
    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(0.0)

    # Attach account metadata
    if os.path.exists(accounts_path):
        df_acc = load_accounts_metadata(accounts_path)
        df = df.merge(
            df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                columns={"Account Number": "from_account", "Entity ID": "from_entity_id", "entity_type": "from_entity_type"}
            ),
            on="from_account",
            how="left",
        )
        df = df.merge(
            df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                columns={"Account Number": "to_account", "Entity ID": "to_entity_id", "entity_type": "to_entity_type"}
            ),
            on="to_account",
            how="left",
        )
    df["from_entity_type"] = df.get("from_entity_type", "Unknown").fillna("Unknown")
    df["to_entity_type"] = df.get("to_entity_type", "Unknown").fillna("Unknown")

    # Build Graph using AMLGraphEngine
    engine = AMLGraphEngine()
    G = engine.build_graph(df, run_inference=True)

    logger.info("Running graph pattern detectors on validation dataset...")
    fan_in_pats = detect_fan_in(G)
    fan_out_pats = detect_fan_out(G)
    cycle_pats = detect_cycles(G)
    multihop_pats = detect_rapid_multi_hop(G)
    scatter_gather_pats = detect_scatter_gather(G)

    all_pats = fan_in_pats + fan_out_pats + cycle_pats + multihop_pats + scatter_gather_pats
    logger.info(f"Detected patterns: Fan-In={len(fan_in_pats)}, Fan-Out={len(fan_out_pats)}, "
                f"Cycles={len(cycle_pats)}, Multi-Hop={len(multihop_pats)}, Scatter-Gather={len(scatter_gather_pats)}")

    # Ground truth topology breakdown from pattern dataset
    ground_truth_counts = df_patterns["pattern_type"].value_counts().to_dict()

    # Per-topology validation statistics
    topology_metrics = {
        "Fan-In": {
            "ground_truth_transactions": int(ground_truth_counts.get("FAN-IN", 0)),
            "detected_patterns_count": len(fan_in_pats),
        },
        "Fan-Out": {
            "ground_truth_transactions": int(ground_truth_counts.get("FAN-OUT", 0)),
            "detected_patterns_count": len(fan_out_pats),
        },
        "Circular Transfer": {
            "ground_truth_transactions": int(ground_truth_counts.get("CYCLE", 0) + ground_truth_counts.get("CIRCULAR", 0)),
            "detected_patterns_count": len(cycle_pats),
        },
        "Rapid Multi-Hop": {
            "ground_truth_transactions": int(ground_truth_counts.get("MULTIHOP", 0) + ground_truth_counts.get("BI-DIRECTIONAL", 0)),
            "detected_patterns_count": len(multihop_pats),
        },
        "Scatter-Gather": {
            "ground_truth_transactions": int(ground_truth_counts.get("SCATTER-GATHER", 0)),
            "detected_patterns_count": len(scatter_gather_pats),
        }
    }

    # Generate Investigation Cases
    case_manager = CaseManager(G)
    cases = case_manager.generate_cases(all_pats)
    logger.info(f"Generated {len(cases):,} investigation cases from pattern validation.")

    validation_summary = {
        "dataset_source": "BEGIN LAUNDERING ATTEMPT - FAN-IN.txt",
        "total_pattern_transactions": len(df),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "topology_metrics": topology_metrics,
        "patterns_detected": {
            "fan_in": len(fan_in_pats),
            "fan_out": len(fan_out_pats),
            "circular_transfers": len(cycle_pats),
            "rapid_multi_hop": len(multihop_pats),
            "scatter_gather": len(scatter_gather_pats),
            "total_patterns": len(all_pats),
        },
        "investigation_cases_generated": len(cases),
        "sample_cases": cases[:3] if cases else []
    }

    summary_path = os.path.join(outputs_dir, "graph_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(validation_summary, f, indent=2)

    logger.info(f"Graph validation summary saved to {summary_path}.")
    return validation_summary


if __name__ == "__main__":
    run_graph_validation()
