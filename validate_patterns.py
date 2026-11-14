"""
Independent Pattern & Graph-Based AML Validation Module.
Evaluates Random Forest and Isolation Forest on the 1,023 external pattern transactions
from 'BEGIN LAUNDERING ATTEMPT - FAN-IN.txt' completely separate from training.
"""

import os
import sys
import json
import logging
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.clean import COLUMN_MAPPING, load_accounts_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_laundering_patterns(txt_path: str = "data/BEGIN LAUNDERING ATTEMPT - FAN-IN.txt") -> pd.DataFrame:
    """
    Parse the multi-hop laundering pattern attempts from txt file,
    tagging each transaction with its pattern category.
    """
    logger.info(f"Parsing laundering pattern subgraphs from {txt_path}...")
    with open(txt_path, "r") as f:
        lines = f.readlines()

    current_pattern_name = "Unknown"
    current_pattern_type = "Unknown"
    records = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        if line_str.startswith("BEGIN LAUNDERING ATTEMPT"):
            match = re.search(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([^:]+)(?::\s*(.*))?", line_str)
            if match:
                current_pattern_type = match.group(1).strip()
                current_pattern_name = (match.group(2) or current_pattern_type).strip()
            continue
        elif line_str.startswith("END LAUNDERING ATTEMPT"):
            current_pattern_name = "Unknown"
            current_pattern_type = "Unknown"
            continue

        # Transaction row
        parts = line_str.split(",")
        if len(parts) == 11:
            records.append({
                "Timestamp": parts[0].strip(),
                "From Bank": parts[1].strip(),
                "Account": parts[2].strip(),
                "To Bank": parts[3].strip(),
                "Account.1": parts[4].strip(),
                "Amount Received": parts[5].strip(),
                "Receiving Currency": parts[6].strip(),
                "Amount Paid": parts[7].strip(),
                "Payment Currency": parts[8].strip(),
                "Payment Format": parts[9].strip(),
                "Is Laundering": int(parts[10].strip()),
                "pattern_type": current_pattern_type,
                "pattern_name": current_pattern_name,
            })

    df_patterns = pd.DataFrame(records)
    logger.info(f"Parsed {len(df_patterns):,} laundering transactions across {df_patterns['pattern_type'].nunique()} pattern types.")
    return df_patterns


def validate_on_patterns(
    pattern_txt_path: str = "data/BEGIN LAUNDERING ATTEMPT - FAN-IN.txt",
    accounts_path: str = "data/HI-Small_accounts.csv",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
):
    """
    Independent validation of models on out-of-distribution laundering patterns.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Parse raw pattern transactions
    df_raw = parse_laundering_patterns(pattern_txt_path)
    pattern_types = df_raw["pattern_type"].values
    pattern_names = df_raw["pattern_name"].values

    # 2. Standardize columns
    df = df_raw.rename(columns=COLUMN_MAPPING)
    df["from_account"] = df["from_account"].astype(str).str.strip().str.upper()
    df["to_account"] = df["to_account"].astype(str).str.strip().str.upper()
    df["from_bank"] = df["from_bank"].astype(str).str.strip()
    df["to_bank"] = df["to_bank"].astype(str).str.strip()
    df["payment_currency"] = df["payment_currency"].astype(str).str.strip()
    df["receiving_currency"] = df["receiving_currency"].astype(str).str.strip()
    df["payment_format"] = df["payment_format"].astype(str).str.strip()

    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(0.0)
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 3. Join account metadata
    if os.path.exists(accounts_path):
        df_acc = load_accounts_metadata(accounts_path)
        df = df.merge(
            df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                columns={
                    "Account Number": "from_account",
                    "Entity ID": "from_entity_id",
                    "entity_type": "from_entity_type",
                }
            ),
            on="from_account",
            how="left",
        )
        df = df.merge(
            df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                columns={
                    "Account Number": "to_account",
                    "Entity ID": "to_entity_id",
                    "entity_type": "to_entity_type",
                }
            ),
            on="to_account",
            how="left",
        )
    df["from_entity_type"] = df["from_entity_type"].fillna("Unknown")
    df["to_entity_type"] = df["to_entity_type"].fillna("Unknown")
    df["from_entity_id"] = df["from_entity_id"].fillna("Unknown")
    df["to_entity_id"] = df["to_entity_id"].fillna("Unknown")

    # 4. Load Models and Transform Features
    logger.info(f"Loading trained models and feature pipeline from '{models_dir}'...")
    rf_model = joblib.load(os.path.join(models_dir, "random_forest_aml.joblib"))
    iso_model = joblib.load(os.path.join(models_dir, "isolation_forest_aml.joblib"))
    fe = joblib.load(os.path.join(models_dir, "feature_engineer.joblib"))

    # Load optimal threshold if saved
    eval_json_path = os.path.join(models_dir, "evaluation_metrics.json")
    best_thresh = 0.5
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            metrics_data = json.load(f)
            best_thresh = metrics_data.get("random_forest", {}).get("optimal_threshold", {}).get("threshold", 0.5)

    X_pattern = fe.transform(df)
    logger.info(f"Transformed pattern feature matrix: {X_pattern.shape}")

    # 5. Model Inferences
    # Random Forest
    rf_probs = rf_model.predict_proba(X_pattern)[:, 1]
    rf_flags_default = (rf_probs >= 0.5).astype(int)
    rf_flags_opt = (rf_probs >= best_thresh).astype(int)

    # Isolation Forest
    iso_scores = iso_model.decision_function(X_pattern)
    iso_risk = -iso_scores
    iso_flags = (iso_model.predict(X_pattern) == -1).astype(int)

    df_results = pd.DataFrame({
        "pattern_type": pattern_types,
        "pattern_name": pattern_names,
        "amount_paid": df["amount_paid"],
        "payment_format": df["payment_format"],
        "rf_probability": rf_probs,
        "rf_detected_default": rf_flags_default,
        "rf_detected_opt": rf_flags_opt,
        "iso_risk_score": iso_risk,
        "iso_anomaly_detected": iso_flags,
    })

    # 6. Overall & Per-Pattern Metrics
    total_samples = len(df_results)
    summary = {
        "total_pattern_transactions": total_samples,
        "random_forest": {
            "mean_probability": float(np.mean(rf_probs)),
            "detected_count_default_thresh": int(sum(rf_flags_default)),
            "recall_default_thresh": float(np.mean(rf_flags_default)),
            "optimal_threshold": float(best_thresh),
            "detected_count_optimal_thresh": int(sum(rf_flags_opt)),
            "recall_optimal_thresh": float(np.mean(rf_flags_opt)),
        },
        "isolation_forest": {
            "mean_risk_score": float(np.mean(iso_risk)),
            "detected_anomalies_count": int(sum(iso_flags)),
            "anomaly_detection_rate": float(np.mean(iso_flags)),
        },
        "combined_coverage": {
            "detected_by_either": int(np.sum((rf_flags_opt == 1) | (iso_flags == 1))),
            "either_recall": float(np.mean((rf_flags_opt == 1) | (iso_flags == 1))),
            "detected_by_both": int(np.sum((rf_flags_opt == 1) & (iso_flags == 1))),
            "both_rate": float(np.mean((rf_flags_opt == 1) & (iso_flags == 1))),
        },
        "breakdown_by_pattern_type": {},
    }

    # Group by pattern type
    for ptype, grp in df_results.groupby("pattern_type"):
        summary["breakdown_by_pattern_type"][ptype] = {
            "total_transactions": len(grp),
            "rf_detection_rate_opt": float(grp["rf_detected_opt"].mean()),
            "iso_detection_rate": float(grp["iso_anomaly_detected"].mean()),
            "combined_coverage": float(((grp["rf_detected_opt"] == 1) | (grp["iso_anomaly_detected"] == 1)).mean()),
            "mean_rf_prob": float(grp["rf_probability"].mean()),
            "mean_iso_risk": float(grp["iso_risk_score"].mean()),
        }

    # 7. Visualization of Pattern Detection
    fig, ax = plt.subplots(figsize=(9, 5))
    types = list(summary["breakdown_by_pattern_type"].keys())
    rf_rates = [summary["breakdown_by_pattern_type"][t]["rf_detection_rate_opt"] * 100 for t in types]
    iso_rates = [summary["breakdown_by_pattern_type"][t]["iso_detection_rate"] * 100 for t in types]
    comb_rates = [summary["breakdown_by_pattern_type"][t]["combined_coverage"] * 100 for t in types]

    x = np.arange(len(types))
    width = 0.25

    ax.bar(x - width, rf_rates, width, label="Random Forest (Optimal Thresh)", color="#2b5c8f")
    ax.bar(x, iso_rates, width, label="Isolation Forest Anomaly", color="#e07a5f")
    ax.bar(x + width, comb_rates, width, label="Combined Dual Coverage", color="#2a9d8f")

    ax.set_ylabel("Detection Rate / Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Independent AML Pattern Subgraph Detection by Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=15, ha="right", fontsize=10)
    ax.legend(frameon=True)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "pattern_validation_performance.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    # 8. Save JSON & Markdown
    summary_path = os.path.join(outputs_dir, "pattern_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Pattern Validation Results:")
    logger.info(f"  - Total Pattern Transactions: {total_samples}")
    logger.info(f"  - Random Forest Detection (Optimal Thresh): {summary['random_forest']['recall_optimal_thresh']:.2%}")
    logger.info(f"  - Isolation Forest Detection: {summary['isolation_forest']['anomaly_detection_rate']:.2%}")
    logger.info(f"  - Combined Dual Coverage: {summary['combined_coverage']['either_recall']:.2%}")

    return summary


if __name__ == "__main__":
    validate_on_patterns()
