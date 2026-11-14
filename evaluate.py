"""
Evaluation Module for AML Detection Pipeline.
Calculates PR-AUC, ROC-AUC, Confusion Matrix, Feature Importance,
and generates publication-quality diagnostic plots.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def evaluate_models(
    models_dir: str = "models",
    outputs_dir: str = "outputs"
):
    """
    Run comprehensive evaluation on holdout test set for both RF and Isolation Forest.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load artifacts
    logger.info(f"Loading models and split data from '{models_dir}'...")
    rf_model = joblib.load(os.path.join(models_dir, "random_forest_aml.joblib"))
    iso_model = joblib.load(os.path.join(models_dir, "isolation_forest_aml.joblib"))
    feature_engineer = joblib.load(os.path.join(models_dir, "feature_engineer.joblib"))
    splits = joblib.load(os.path.join(models_dir, "split_data.joblib"))

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    logger.info(f"Test set shape: {X_test.shape}, Positives in test: {sum(y_test)}")

    metrics_dict = {
        "dataset": {
            "n_train": len(y_train),
            "n_train_positives": int(sum(y_train)),
            "n_test": len(y_test),
            "n_test_positives": int(sum(y_test)),
            "n_features": X_test.shape[1],
        },
        "random_forest": {},
        "isolation_forest": {},
    }

    # ==========================================
    # 2. EVALUATE RANDOM FOREST (SUPERVISED)
    # ==========================================
    logger.info("Evaluating Random Forest...")
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds_default = rf_model.predict(X_test)

    # PR-AUC & ROC-AUC
    pr_auc = float(average_precision_score(y_test, rf_probs))
    try:
        roc_auc = float(roc_auc_score(y_test, rf_probs))
    except Exception as e:
        logger.warning(f"ROC-AUC computation note: {e}")
        roc_auc = float("nan")

    # Optimal threshold based on F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_probs)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    rf_preds_opt = (rf_probs >= best_thresh).astype(int)

    cm = confusion_matrix(y_test, rf_preds_opt)
    tn, fp, fn, tp = cm.ravel()

    metrics_dict["random_forest"] = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "default_threshold_0.5": {
            "precision": float(precision_score(y_test, rf_preds_default, zero_division=0)),
            "recall": float(recall_score(y_test, rf_preds_default, zero_division=0)),
            "f1": float(f1_score(y_test, rf_preds_default, zero_division=0)),
        },
        "optimal_threshold": {
            "threshold": best_thresh,
            "f1_score": best_f1,
            "precision": float(precision_score(y_test, rf_preds_opt, zero_division=0)),
            "recall": float(recall_score(y_test, rf_preds_opt, zero_division=0)),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        },
    }

    logger.info(f"Random Forest PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}, Best F1: {best_f1:.4f} (at thresh={best_thresh:.4f})")

    # ==========================================
    # 3. EVALUATE ISOLATION FOREST (UNSUPERVISED)
    # ==========================================
    logger.info("Evaluating Isolation Forest...")
    # decision_function returns anomaly score (negative values are outliers)
    iso_scores = iso_model.decision_function(X_test)
    # in scikit-learn IsolationForest: -1 is anomaly, +1 is inlier
    iso_raw_pred = iso_model.predict(X_test)
    iso_anomaly_flag = (iso_raw_pred == -1).astype(int)

    # Invert scores so higher = more anomalous (risk score)
    iso_risk_scores = -iso_scores

    try:
        iso_roc_auc = float(roc_auc_score(y_test, iso_risk_scores))
        iso_pr_auc = float(average_precision_score(y_test, iso_risk_scores))
    except Exception as e:
        iso_roc_auc = float("nan")
        iso_pr_auc = float("nan")

    # Positive sample anomaly ranking in test set
    pos_indices = np.where(y_test == 1)[0]
    pos_risk_scores = iso_risk_scores[pos_indices]
    pos_percentiles = [
        float(np.mean(iso_risk_scores <= s) * 100) for s in pos_risk_scores
    ]

    metrics_dict["isolation_forest"] = {
        "n_anomalies_flagged": int(sum(iso_anomaly_flag)),
        "anomaly_rate": float(np.mean(iso_anomaly_flag)),
        "iso_roc_auc": iso_roc_auc,
        "iso_pr_auc": iso_pr_auc,
        "positives_anomaly_percentiles": pos_percentiles,
    }

    logger.info(f"Isolation Forest flagged {sum(iso_anomaly_flag)} test anomalies ({np.mean(iso_anomaly_flag):.2%}).")
    logger.info(f"Laundering samples anomaly percentiles (higher is more anomalous): {pos_percentiles}")

    # ==========================================
    # 4. PLOTTING DIAGNOSTICS
    # ==========================================
    logger.info("Generating evaluation visualization plots...")

    # Plot 1: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, color="#2b5c8f", lw=2.5, label=f"Random Forest (PR-AUC = {pr_auc:.4f})")
    ax.scatter([recalls[best_idx]], [precisions[best_idx]], color="#d9534f", s=80, zorder=5, label=f"Optimal Thresh ({best_thresh:.2f})")
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curve (Severe Class Imbalance)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    plt.tight_layout()
    pr_curve_path = os.path.join(outputs_dir, "pr_curve.png")
    fig.savefig(pr_curve_path, dpi=300)
    plt.close(fig)

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Random Forest (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=1.5, label="Random Guess")
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_curve_path = os.path.join(outputs_dir, "roc_curve.png")
    fig.savefig(roc_curve_path, dpi=300)
    plt.close(fig)

    # Plot 3: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal (0)", "Laundering (1)"],
        yticklabels=["Normal (0)", "Laundering (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix (Threshold = {best_thresh:.3f})", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    # Plot 4: Top 15 Feature Importances (Random Forest)
    importances = rf_model.feature_importances_
    feat_names = np.array(X_test.columns)
    sorted_idx = np.argsort(importances)[::-1][:15]
    top_feats = feat_names[sorted_idx]
    top_scores = importances[sorted_idx]

    metrics_dict["random_forest"]["top_features"] = [
        {"feature": str(f), "importance": float(s)} for f, s in zip(top_feats, top_scores)
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_scores, y=top_feats, palette="crest", ax=ax)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Engineered Feature", fontsize=11, fontweight="bold")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fi_path = os.path.join(outputs_dir, "feature_importance.png")
    fig.savefig(fi_path, dpi=300)
    plt.close(fig)

    # Plot 5: Isolation Forest Risk Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(iso_risk_scores, bins=60, kde=True, color="#3470a3", ax=ax, label="All Test Transactions")
    for i, p_score in enumerate(pos_risk_scores):
        ax.axvline(
            p_score,
            color="#d9534f",
            linestyle="--",
            lw=2.5,
            label=f"Known Laundering #{i+1} ({pos_percentiles[i]:.1f}%ile)" if i == 0 else "",
        )
    ax.set_xlabel("Isolation Forest Anomaly Risk Score (-decision_function)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Transaction Count", fontsize=11, fontweight="bold")
    ax.set_title("Isolation Forest Anomaly Distribution vs Known Laundering", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    iso_plot_path = os.path.join(outputs_dir, "isolation_forest_scores.png")
    fig.savefig(iso_plot_path, dpi=300)
    plt.close(fig)

    # 5. Save JSON summary
    metrics_json_path = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Evaluation metrics saved to {metrics_json_path}")
    logger.info(f"Plots saved to {outputs_dir}/")
    return metrics_dict


if __name__ == "__main__":
    evaluate_models()
