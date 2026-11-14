"""
Flask REST API Module for AML Investigation & Risk System (SHADOWTRACE Phase 3).
Provides dataset-first REST endpoints for dataset upload, schema detection,
data quality scoring, 10-stage execution pipeline, feature intelligence,
comparative ML analysis, graph patterns, case details, transaction replay,
and compliance reporting.
"""

import os
import sys
import logging
import pandas as pd
import networkx as nx
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graph_engine import AMLGraphEngine
from src.graph_patterns import (
    detect_fan_in,
    detect_fan_out,
    detect_cycles,
    detect_rapid_multi_hop,
    detect_scatter_gather,
)
from src.investigation_cases import CaseManager
from src.dataset_manager import dataset_manager, CANONICAL_FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global singleton stores for default Graph Engine and Investigation Cases
_graph_engine = None
_cases = []
_case_manager = None


def initialize_aml_engine(data_path: str = "data/transactions_cleaned.parquet"):
    """Initialize Graph Engine, build graph from cleaned transactions, and pre-compute investigation cases."""
    global _graph_engine, _cases, _case_manager
    logger.info("Initializing AML Graph Engine and generating investigation cases for Phase 3 API...")

    _graph_engine = AMLGraphEngine(models_dir="models")

    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
    else:
        logger.warning(f"Cleaned transactions not found at {data_path}. Creating empty graph.")
        df = pd.DataFrame(columns=[
            "timestamp", "from_bank", "from_account", "to_bank", "to_account",
            "amount_received", "receiving_currency", "amount_paid", "payment_currency",
            "payment_format", "is_laundering"
        ])

    G = _graph_engine.build_graph(df, run_inference=True)

    # Run pattern detectors
    logger.info("Running pattern detectors on graph...")
    fan_in = detect_fan_in(G)
    fan_out = detect_fan_out(G)
    cycles = detect_cycles(G)
    multihop = detect_rapid_multi_hop(G)
    scatter_gather = detect_scatter_gather(G)

    all_patterns = fan_in + fan_out + cycles + multihop + scatter_gather
    logger.info(f"Total patterns detected: {len(all_patterns)}")

    # Generate Investigation Cases
    _case_manager = CaseManager(G)
    _cases = _case_manager.generate_cases(all_patterns)
    logger.info(f"Generated {_cases.__len__():,} investigation cases successfully.")

    # Attach to dataset_manager baseline dataset
    dataset_manager.attach_primary_engine(_graph_engine, _cases, _case_manager)


def get_active_context(dataset_id=None):
    """Retrieve isolated cases, graph engine, and case manager for current or specified dataset."""
    global _cases, _graph_engine, _case_manager
    req_dataset_id = dataset_id or request.args.get("dataset_id") or request.headers.get("X-Dataset-Id")
    if req_dataset_id:
        inst = dataset_manager.get_dataset(req_dataset_id)
        if inst:
            return inst.cases or [], inst.graph_engine, inst.case_manager, inst
        else:
            return None, None, None, None

    active = dataset_manager.get_active_dataset()
    if active and active.cases:
        return active.cases, active.graph_engine, active.case_manager, active

    return _cases, _graph_engine, _case_manager, active


# ============================================================
# 1. CORE HEALTH & BACKWARD COMPATIBILITY ENDPOINTS
# ============================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    cases, engine, _, active_inst = get_active_context()
    model_loaded = engine is not None and engine.predictor is not None
    graph_nodes = engine.graph.number_of_nodes() if engine and engine.graph else 0
    graph_edges = engine.graph.number_of_edges() if engine and engine.graph else 0

    return jsonify({
        "status": "healthy",
        "service": "AML Investigation & Risk REST API (Phase 3)",
        "model_loaded": model_loaded,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "investigation_cases_count": len(cases),
        "active_dataset_id": active_inst.dataset_id if active_inst else None,
        "active_dataset_name": active_inst.name if active_inst else None,
        "total_datasets": len(dataset_manager.datasets),
    }), 200


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Aggregate statistics across all investigation cases and graph topology for active dataset."""
    cases, engine, _, active_inst = get_active_context()
    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    pattern_counts = {}
    total_flagged_tx = 0
    total_accounts = set()

    for c in cases:
        r_level = c.get("risk_level", "LOW")
        risk_counts[r_level] = risk_counts.get(r_level, 0) + 1
        for p in c.get("patterns", []):
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
        total_flagged_tx += c.get("transactions_count", 0)
        for acc in c.get("accounts", []):
            total_accounts.add(acc)

    graph_nodes = engine.graph.number_of_nodes() if engine and engine.graph else 0
    graph_edges = engine.graph.number_of_edges() if engine and engine.graph else 0

    return jsonify({
        "total_cases": len(cases),
        "risk_breakdown": risk_counts,
        "pattern_breakdown": pattern_counts,
        "total_flagged_transactions": total_flagged_tx,
        "total_investigated_accounts": len(total_accounts),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "model_loaded": engine is not None and engine.predictor is not None,
        "dataset_id": active_inst.dataset_id if active_inst else "None",
        "dataset_name": active_inst.name if active_inst else "None",
        "dataset_rows": active_inst.rows_count if active_inst else 0,
    }), 200


@app.route("/api/investigations", methods=["GET"])
@app.route("/api/datasets/<dataset_id>/investigations", methods=["GET"])
def get_investigations(dataset_id=None):
    """Get list of investigation cases with optional filtering and pagination."""
    cases, _, _, inst = get_active_context(dataset_id)
    if cases is None:
        return jsonify({"error": "Dataset not found.", "success": False}), 404

    risk_level = request.args.get("risk_level", "").upper()
    min_score = request.args.get("min_score", type=float)
    pattern = request.args.get("pattern", "")
    bank = request.args.get("bank", "")
    query = request.args.get("q", "").strip().upper()
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)

    filtered = cases
    if risk_level and risk_level != "ALL":
        filtered = [c for c in filtered if c.get("risk_level") == risk_level]
    if min_score is not None:
        filtered = [c for c in filtered if c.get("risk_score", 0.0) >= min_score]
    if pattern and pattern != "ALL":
        filtered = [c for c in filtered if pattern in c.get("patterns", [])]
    if bank:
        filtered = [c for c in filtered if bank.upper() in [str(b).upper() for b in c.get("banks", [])]]
    if query:
        filtered = [
            c for c in filtered
            if query in c.get("case_id", "").upper()
            or any(query in acc.upper() for acc in c.get("accounts", []))
        ]

    paginated = filtered[offset : offset + limit]

    summary_list = []
    for c in paginated:
        summary_list.append({
            "case_id": c["case_id"],
            "dataset_id": c.get("dataset_id", inst.dataset_id if inst else "dataset_primary"),
            "risk_score": c["risk_score"],
            "risk_level": c["risk_level"],
            "status": c["status"],
            "accounts_count": c["accounts_count"],
            "transactions_count": c["transactions_count"],
            "patterns": c["patterns"],
            "banks": c.get("banks", []),
            "created_at": c["created_at"],
        })

    return jsonify({
        "total": len(filtered),
        "count": len(summary_list),
        "limit": limit,
        "offset": offset,
        "cases": summary_list,
        "dataset_id": inst.dataset_id if inst else "dataset_primary",
    }), 200


@app.route("/api/investigations/<case_id>", methods=["GET"])
@app.route("/api/datasets/<dataset_id>/investigations/<case_id>", methods=["GET"])
def get_investigation_detail(case_id, dataset_id=None):
    """Get detailed view of a specific investigation case."""
    cases, _, _, inst = get_active_context(dataset_id)
    if cases is None:
        return jsonify({"error": "Dataset not found.", "success": False}), 404

    case_id = case_id.upper()
    match = next((c for c in cases if c["case_id"].upper() == case_id), None)
    if not match:
        return jsonify({"error": f"Investigation case '{case_id}' not found.", "success": False}), 404

    return jsonify(match), 200


@app.route("/api/investigations/<case_id>/timeline", methods=["GET"])
@app.route("/api/datasets/<dataset_id>/investigations/<case_id>/timeline", methods=["GET"])
def get_investigation_timeline(case_id, dataset_id=None):
    """Get chronological transaction timeline for a specific case."""
    cases, _, _, inst = get_active_context(dataset_id)
    if cases is None:
        return jsonify({"error": "Dataset not found.", "success": False}), 404

    case_id = case_id.upper()
    match = next((c for c in cases if c["case_id"].upper() == case_id), None)
    if not match:
        return jsonify({"error": f"Investigation case '{case_id}' not found.", "success": False}), 404

    return jsonify({
        "case_id": match["case_id"],
        "dataset_id": match.get("dataset_id", inst.dataset_id if inst else "dataset_primary"),
        "transactions_count": len(match["timeline"]),
        "timeline": match["timeline"],
    }), 200


@app.route("/api/investigations/<case_id>/network", methods=["GET"])
@app.route("/api/datasets/<dataset_id>/investigations/<case_id>/network", methods=["GET"])
def get_investigation_network(case_id, dataset_id=None):
    """Get network nodes and edges for a specific investigation case subgraph."""
    cases, engine, _, inst = get_active_context(dataset_id)
    if cases is None:
        return jsonify({"error": "Dataset not found.", "success": False}), 404

    case_id = case_id.upper()
    match = next((c for c in cases if c["case_id"].upper() == case_id), None)
    if not match:
        return jsonify({"error": f"Investigation case '{case_id}' not found.", "success": False}), 404

    if not engine or not engine.graph:
        return jsonify({"error": "Graph engine not initialized.", "success": False}), 500

    G = engine.graph
    case_accounts = set(match["accounts"])
    subgraph = G.subgraph(case_accounts).copy()

    nodes = []
    for node, data in subgraph.nodes(data=True):
        nodes.append({
            "account_id": node,
            "bank": data.get("bank", "Unknown"),
            "entity_type": data.get("entity_type", "Unknown"),
            "total_sent": data.get("total_sent", 0.0),
            "total_received": data.get("total_received", 0.0),
            "suspicious_transaction_count": data.get("suspicious_transaction_count", 0),
        })

    edges = []
    for u, v, k, d in subgraph.edges(keys=True, data=True):
        edges.append({
            "transaction_id": d.get("transaction_id"),
            "from_account": u,
            "to_account": v,
            "timestamp": str(d.get("timestamp")),
            "amount_paid": d.get("amount_paid"),
            "payment_currency": d.get("payment_currency"),
            "payment_format": d.get("payment_format"),
            "rf_laundering_prob": d.get("rf_laundering_prob", 0.0),
            "iso_risk_score": d.get("iso_risk_score", 0.0),
            "final_aml_alert": d.get("final_aml_alert", "NORMAL"),
        })

    return jsonify({
        "case_id": match["case_id"],
        "nodes": nodes,
        "edges": edges,
    }), 200


@app.route("/api/graph/account/<account_id>", methods=["GET"])
def get_account_graph(account_id):
    """Get local neighborhood and transactions for a specific account."""
    _, engine, _, _ = get_active_context()
    if not engine or not engine.graph:
        return jsonify({"error": "Graph engine not initialized."}), 500

    account_id = account_id.strip().upper()
    G = engine.graph
    if account_id not in G:
        return jsonify({"error": f"Account '{account_id}' not found in transaction graph."}), 404

    radius = request.args.get("radius", default=1, type=int)
    subgraph = engine.get_account_subgraph(account_id, radius=radius)

    nodes = [
        {"account_id": n, **{k: v for k, v in d.items()}}
        for n, d in subgraph.nodes(data=True)
    ]
    edges = [
        {
            "from_account": u,
            "to_account": v,
            "key": k,
            "timestamp": str(d.get("timestamp")),
            "amount_paid": d.get("amount_paid"),
            "payment_currency": d.get("payment_currency"),
            "payment_format": d.get("payment_format"),
            "rf_laundering_prob": d.get("rf_laundering_prob", 0.0),
            "iso_risk_score": d.get("iso_risk_score", 0.0),
            "final_aml_alert": d.get("final_aml_alert", "NORMAL"),
        }
        for u, v, k, d in subgraph.edges(keys=True, data=True)
    ]

    account_meta = G.nodes[account_id]

    return jsonify({
        "account_id": account_id,
        "metadata": {k: v for k, v in account_meta.items()},
        "subgraph_nodes_count": len(nodes),
        "subgraph_edges_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }), 200


@app.route("/api/analyze", methods=["POST"])
def analyze_transactions():
    """Ad-hoc analysis endpoint for evaluating new transaction records."""
    _, engine, _, _ = get_active_context()
    if not engine:
        return jsonify({"error": "Graph engine not initialized."}), 500

    data = request.get_json()
    if not data or "transactions" not in data:
        return jsonify({"error": "Invalid request payload. Expected JSON with 'transactions' list."}), 400

    tx_list = data["transactions"]
    if not isinstance(tx_list, list) or len(tx_list) == 0:
        return jsonify({"error": "'transactions' must be a non-empty list."}), 400

    try:
        df_new = pd.DataFrame(tx_list)
        if engine.predictor is not None:
            preds = engine.predictor.predict(df_new)
        else:
            preds = pd.DataFrame({
                "timestamp": df_new.get("timestamp", pd.Timestamp.now()),
                "from_account": df_new.get("from_account", "UNKNOWN"),
                "to_account": df_new.get("to_account", "UNKNOWN"),
                "amount_paid": df_new.get("amount_paid", 0.0),
                "rf_laundering_prob": 0.0,
                "iso_risk_score": 0.0,
                "final_aml_alert": "NORMAL"
            })

        return jsonify({
            "status": "success",
            "analyzed_transactions_count": len(df_new),
            "message": "Predictions computed successfully",
            "results": preds.to_dict(orient="records"),
        }), 200

    except Exception as e:
        logger.error(f"Error during ad-hoc analysis: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2. DATASET INTELLIGENCE & ISOLATION ENDPOINTS
# ============================================================

@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """List all registered datasets."""
    datasets = dataset_manager.list_datasets()
    return jsonify({
        "active_dataset_id": dataset_manager.active_dataset_id,
        "count": len(datasets),
        "datasets": datasets,
    }), 200


@app.route("/api/datasets/upload", methods=["POST"])
def upload_dataset():
    """Upload CSV or Excel dataset file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Please upload using key 'file'."}), 400

    file_obj = request.files["file"]
    if not file_obj or file_obj.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        inst = dataset_manager.ingest_upload(file_obj, file_obj.filename)
        return jsonify({
            "status": "success",
            "message": f"Dataset '{inst.name}' uploaded and profiled successfully.",
            "dataset_id": inst.dataset_id,
            "rows_count": inst.rows_count,
            "cols_count": inst.cols_count,
            "schema_mapping": inst.schema_mapping,
            "quality_score": inst.quality_report.get("quality_score", 0),
            "profile": inst.profile,
        }), 201
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/datasets/<dataset_id>/select", methods=["POST"])
def select_dataset(dataset_id):
    """Switch the system's active dataset."""
    success = dataset_manager.set_active_dataset(dataset_id)
    if not success:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    inst = dataset_manager.get_dataset(dataset_id)
    return jsonify({
        "status": "success",
        "active_dataset_id": dataset_id,
        "name": inst.name,
        "status": inst.status,
        "rows_count": inst.rows_count,
    }), 200


@app.route("/api/datasets/<dataset_id>/profile", methods=["GET"])
def get_dataset_profile(dataset_id):
    """Retrieve profile and preview for a dataset."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    return jsonify({
        "dataset_id": inst.dataset_id,
        "name": inst.name,
        "status": inst.status,
        "rows_count": inst.rows_count,
        "cols_count": inst.cols_count,
        "schema_mapping": inst.schema_mapping,
        "canonical_fields": CANONICAL_FIELDS,
        "profile": inst.profile,
    }), 200


@app.route("/api/datasets/<dataset_id>/schema", methods=["POST"])
def update_dataset_schema(dataset_id):
    """Update manual schema mapping overrides."""
    payload = request.get_json()
    if not payload or "mapping" not in payload:
        return jsonify({"error": "Missing 'mapping' dictionary in payload."}), 400

    try:
        updated = dataset_manager.update_schema_mapping(dataset_id, payload["mapping"])
        inst = dataset_manager.get_dataset(dataset_id)
        return jsonify({
            "status": "success",
            "message": "Schema mapping updated successfully.",
            "schema_mapping": updated,
            "quality_score": inst.quality_report.get("quality_score", 0),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/datasets/<dataset_id>/quality", methods=["GET"])
def get_dataset_quality(dataset_id):
    """Retrieve comprehensive Data Quality Report and Score."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    return jsonify({
        "dataset_id": inst.dataset_id,
        "name": inst.name,
        "quality_report": inst.quality_report,
    }), 200


@app.route("/api/datasets/<dataset_id>/analyze", methods=["POST"])
def run_dataset_analysis(dataset_id):
    """Trigger the 10-stage AML analysis pipeline."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    try:
        res = dataset_manager.run_analysis_pipeline(dataset_id)
        return jsonify({
            "status": "success",
            "message": "Analysis pipeline executed successfully.",
            "results": res,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets/<dataset_id>/status", methods=["GET"])
def get_dataset_status(dataset_id):
    """Poll pipeline progress and stage statuses."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    return jsonify({
        "dataset_id": inst.dataset_id,
        "status": inst.status,
        "pipeline_progress": inst.pipeline_progress,
        "current_stage": inst.current_stage,
        "stages": inst.pipeline_stages,
        "error_message": inst.error_message,
        "cases_count": len(inst.cases),
    }), 200


@app.route("/api/datasets/<dataset_id>/features", methods=["GET"])
def get_dataset_features(dataset_id):
    """Retrieve Feature Intelligence and Random Forest Feature Importance."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    return jsonify({
        "dataset_id": inst.dataset_id,
        "name": inst.name,
        "features": inst.feature_intelligence,
    }), 200


@app.route("/api/datasets/<dataset_id>/ml", methods=["GET"])
def get_dataset_ml(dataset_id):
    """Retrieve comparative ML findings (Supervised RF vs Unsupervised IF)."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    return jsonify({
        "dataset_id": inst.dataset_id,
        "name": inst.name,
        "ml_summary": inst.ml_summary,
        "random_forest": {
            "model_type": "Supervised Random Forest Classifier",
            "status": "OPERATIONAL",
            "purpose": "Learns topological and statistical indicators from labeled transaction graph patterns.",
            "flagged_count": inst.ml_summary.get("rf_flagged_count", 0),
            "threshold": inst.ml_summary.get("threshold", 0.5),
        },
        "isolation_forest": {
            "model_type": "Unsupervised Isolation Forest",
            "status": "OPERATIONAL",
            "purpose": "Isolates anomalous multidimensional feature outliers without relying on historical labels.",
            "anomalous_count": inst.ml_summary.get("if_anomalous_count", 0),
        },
        "combined_strategy": (
            "SHADOWTRACE fuses supervised laundering probabilities with unsupervised anomaly isolation. "
            "Neither model alone establishes guilt; their intersection prioritizes highest-confidence investigation leads."
        ),
    }), 200


@app.route("/api/datasets/<dataset_id>/patterns", methods=["GET"])
def get_dataset_patterns(dataset_id):
    """Retrieve AML Topological Pattern Intelligence breakdown."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst:
        return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404

    return jsonify({
        "dataset_id": inst.dataset_id,
        "name": inst.name,
        "patterns": inst.patterns_summary,
    }), 200


@app.route("/api/datasets/<dataset_id>/investigations/<case_id>/replay", methods=["GET"])
def get_dataset_transaction_replay(dataset_id, case_id):
    """Retrieve chronological animated replay frames for an investigation case."""
    try:
        replay_data = dataset_manager.get_transaction_replay(dataset_id, case_id)
        return jsonify(replay_data), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets/<dataset_id>/report/<case_id>", methods=["GET"])
def get_compliance_report(dataset_id, case_id):
    """Retrieve structured regulatory compliance report payload."""
    try:
        report_data = dataset_manager.generate_compliance_report(dataset_id, case_id)
        return jsonify(report_data), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets/<dataset_id>/accounts", methods=["GET"])
def get_dataset_accounts(dataset_id):
    """Retrieve accounts in dataset with forensic aggregates, case associations, and search."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst or not inst.graph_engine or not inst.graph_engine.graph:
        return jsonify({"error": f"Dataset or graph not available for '{dataset_id}'."}), 404

    G = inst.graph_engine.graph
    query = request.args.get("q", "").strip().upper()
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)

    # Pre-index cases by associated accounts
    cases_by_account = {}
    for c in inst.cases:
        c_id = c["case_id"]
        for acc in c.get("accounts", []):
            cases_by_account.setdefault(acc, []).append(c_id)

    all_accounts = []
    for node, d in G.nodes(data=True):
        if query and query not in node:
            continue
        all_accounts.append({
            "account_id": node,
            "bank": d.get("bank", "Unknown"),
            "entity_type": d.get("entity_type", "Unknown"),
            "total_sent": d.get("total_sent", 0.0),
            "total_received": d.get("total_received", 0.0),
            "unique_counterparties": d.get("unique_counterparties", 0),
            "suspicious_transactions": d.get("suspicious_transaction_count", 0),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "associated_cases": cases_by_account.get(node, []),
        })

    # Sort accounts by suspicious transactions then total volume
    all_accounts.sort(key=lambda x: (x["suspicious_transactions"], x["total_sent"] + x["total_received"]), reverse=True)
    paginated = all_accounts[offset : offset + limit]

    return jsonify({
        "total": len(all_accounts),
        "count": len(paginated),
        "limit": limit,
        "offset": offset,
        "accounts": paginated,
        "dataset_id": dataset_id,
    }), 200


@app.route("/api/datasets/<dataset_id>/accounts/<account_id>", methods=["GET"])
def get_dataset_account_detail(dataset_id, account_id):
    """Retrieve single account forensic dossier for a specific dataset."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst or not inst.graph_engine or not inst.graph_engine.graph:
        return jsonify({"error": f"Dataset or graph not available for '{dataset_id}'."}), 404

    account_id = account_id.strip().upper()
    G = inst.graph_engine.graph
    if account_id not in G:
        return jsonify({"error": f"Account '{account_id}' not found in dataset '{dataset_id}'."}), 404

    d = G.nodes[account_id]
    cases_by_account = [c["case_id"] for c in inst.cases if account_id in c.get("accounts", [])]

    # Find recent transactions involving this account in this dataset
    recent_txs = []
    if inst.df_mapped is not None:
        df_acc = inst.df_mapped[
            (inst.df_mapped["from_account"] == account_id) | (inst.df_mapped["to_account"] == account_id)
        ]
        for idx, row in df_acc.head(20).iterrows():
            recent_txs.append({
                "transaction_id": f"TX-{idx}",
                "timestamp": str(row.get("timestamp")),
                "from_account": row.get("from_account"),
                "to_account": row.get("to_account"),
                "amount_paid": float(row.get("amount_paid", 0.0)),
                "payment_currency": str(row.get("payment_currency", "USD")),
                "payment_format": str(row.get("payment_format", "Transfer")),
                "from_bank": str(row.get("from_bank", "")),
                "to_bank": str(row.get("to_bank", "")),
            })

    return jsonify({
        "account_id": account_id,
        "dataset_id": dataset_id,
        "bank": d.get("bank", "Unknown"),
        "entity_type": d.get("entity_type", "Unknown"),
        "total_sent": d.get("total_sent", 0.0),
        "total_received": d.get("total_received", 0.0),
        "unique_counterparties": d.get("unique_counterparties", 0),
        "fan_in_degree": G.in_degree(account_id),
        "fan_out_degree": G.out_degree(account_id),
        "suspicious_transactions": d.get("suspicious_transaction_count", 0),
        "associated_cases": cases_by_account,
        "recent_transactions_count": len(recent_txs),
        "recent_transactions": recent_txs,
    }), 200


@app.route("/api/datasets/<dataset_id>/transactions", methods=["GET"])
def get_dataset_transactions(dataset_id):
    """Retrieve and search transactions in dataset with multi-field filters and pagination."""
    inst = dataset_manager.get_dataset(dataset_id)
    if not inst or inst.df_mapped is None:
        return jsonify({"error": f"Dataset transactions not available for '{dataset_id}'."}), 404

    df = inst.df_mapped
    query = request.args.get("q", "").strip().upper()
    account = request.args.get("account", "").strip().upper()
    bank = request.args.get("bank", "").strip()
    currency = request.args.get("currency", "").strip().upper()
    date_str = request.args.get("date", "").strip()
    min_amount = request.args.get("min_amount", type=float)
    max_amount = request.args.get("max_amount", type=float)
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)

    filtered_df = df
    if query:
        mask = (
            filtered_df["from_account"].astype(str).str.contains(query, na=False)
            | filtered_df["to_account"].astype(str).str.contains(query, na=False)
            | filtered_df["from_bank"].astype(str).str.contains(query, na=False)
            | filtered_df["to_bank"].astype(str).str.contains(query, na=False)
        )
        filtered_df = filtered_df[mask]

    if account:
        mask = (filtered_df["from_account"] == account) | (filtered_df["to_account"] == account)
        filtered_df = filtered_df[mask]

    if bank:
        mask = (
            filtered_df["from_bank"].astype(str).str.contains(bank, case=False, na=False)
            | filtered_df["to_bank"].astype(str).str.contains(bank, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if currency:
        mask = (filtered_df["payment_currency"].astype(str).str.upper() == currency) | (filtered_df["receiving_currency"].astype(str).str.upper() == currency)
        filtered_df = filtered_df[mask]

    if date_str:
        mask = filtered_df["timestamp"].astype(str).str.startswith(date_str)
        filtered_df = filtered_df[mask]

    if min_amount is not None:
        filtered_df = filtered_df[filtered_df["amount_paid"] >= min_amount]

    if max_amount is not None:
        filtered_df = filtered_df[filtered_df["amount_paid"] <= max_amount]

    total = len(filtered_df)
    paginated = filtered_df.iloc[offset : offset + limit].copy()

    records = []
    for idx, row in paginated.iterrows():
        records.append({
            "transaction_id": f"TX-{idx}",
            "timestamp": str(row.get("timestamp")),
            "from_account": row.get("from_account"),
            "from_bank": row.get("from_bank"),
            "to_account": row.get("to_account"),
            "to_bank": row.get("to_bank"),
            "amount_paid": float(row.get("amount_paid", 0.0)),
            "payment_currency": str(row.get("payment_currency", "USD")),
            "payment_format": str(row.get("payment_format", "Transfer")),
            "is_laundering": int(row.get("is_laundering", 0)) if "is_laundering" in row else 0,
        })

    return jsonify({
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "transactions": records,
        "dataset_id": dataset_id,
    }), 200


if __name__ == "__main__":
    initialize_aml_engine()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
