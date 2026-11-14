"""
Graph Engine Module for AML Investigation.
Builds a directed multi-graph (NetworkX MultiDiGraph) from cleaned transaction data
and integrates ML predictions from the pre-trained Random Forest and Isolation Forest models.
"""

import os
import sys
import logging
import networkx as nx
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.infer import AMLPredictor
from src.clean import clean_and_enrich_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AMLGraphEngine:
    """
    Constructs and manages the AML transaction MultiDiGraph, enriching nodes and edges
    with transaction metadata and ML risk scores.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.graph = nx.MultiDiGraph()
        self.predictor = None
        self._load_predictor()

    def _load_predictor(self):
        try:
            self.predictor = AMLPredictor(models_dir=self.models_dir)
            logger.info("Loaded AMLPredictor successfully into AMLGraphEngine.")
        except Exception as e:
            logger.warning(f"Could not load AMLPredictor ({e}). Graph will be constructed without live ML inference enrichment.")
            self.predictor = None

    def build_graph(self, df_transactions: pd.DataFrame, run_inference: bool = True):
        """
        Build a NetworkX MultiDiGraph from transaction dataframe.
        Each edge represents a transaction. Nodes represent accounts.
        """
        logger.info(f"Building AML transaction MultiDiGraph from {len(df_transactions):,} transactions...")

        df = df_transactions.copy()

        # Ensure required columns exist
        required_cols = ["from_account", "to_account", "timestamp", "amount_paid", "amount_received"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from transaction dataframe.")

        # Run ML inference if predictor is available and requested
        if run_inference and self.predictor is not None:
            logger.info("Running ML inference on transactions for graph enrichment...")
            try:
                preds_df = self.predictor.predict(df)
                df["rf_laundering_prob"] = preds_df["rf_laundering_prob"].values
                df["rf_is_laundering_flag"] = preds_df["rf_is_laundering_flag"].values
                df["iso_risk_score"] = preds_df["iso_risk_score"].values
                df["iso_anomaly_flag"] = preds_df["iso_anomaly_flag"].values
                df["final_aml_alert"] = preds_df["final_aml_alert"].values
            except Exception as e:
                logger.error(f"ML inference enrichment failed: {e}. Proceeding without ML scores.")
                df["rf_laundering_prob"] = 0.0
                df["rf_is_laundering_flag"] = 0
                df["iso_risk_score"] = 0.0
                df["iso_anomaly_flag"] = 0
                df["final_aml_alert"] = "NORMAL"
        else:
            if "rf_laundering_prob" not in df.columns:
                df["rf_laundering_prob"] = 0.0
            if "iso_risk_score" not in df.columns:
                df["iso_risk_score"] = 0.0
            if "final_aml_alert" not in df.columns:
                df["final_aml_alert"] = "NORMAL"

        # Initialize MultiDiGraph
        G = nx.MultiDiGraph()

        # Add edges and transactions
        edge_records = []
        for idx, row in df.iterrows():
            sender = str(row["from_account"]).strip().upper()
            receiver = str(row["to_account"]).strip().upper()

            edge_data = {
                "transaction_id": str(idx),
                "timestamp": pd.to_datetime(row["timestamp"]),
                "amount_paid": float(row.get("amount_paid", 0.0)),
                "amount_received": float(row.get("amount_received", 0.0)),
                "payment_currency": str(row.get("payment_currency", "Unknown")),
                "receiving_currency": str(row.get("receiving_currency", "Unknown")),
                "payment_format": str(row.get("payment_format", "Unknown")),
                "from_bank": str(row.get("from_bank", "Unknown")),
                "to_bank": str(row.get("to_bank", "Unknown")),
                "from_entity_type": str(row.get("from_entity_type", "Unknown")),
                "to_entity_type": str(row.get("to_entity_type", "Unknown")),
                "is_laundering": int(row.get("is_laundering", 0)),
                "rf_laundering_prob": float(row.get("rf_laundering_prob", 0.0)),
                "iso_risk_score": float(row.get("iso_risk_score", 0.0)),
                "final_aml_alert": str(row.get("final_aml_alert", "NORMAL")),
            }
            G.add_edge(sender, receiver, key=idx, **edge_data)

        logger.info(f"Graph built with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")

        # Compute and attach node-level aggregates
        self._aggregate_node_features(G)
        self.graph = G
        return G

    def _aggregate_node_features(self, G: nx.MultiDiGraph):
        """Compute aggregated statistics for each node (fan-in, fan-out, totals, risk signals)."""
        logger.info("Computing node-level topological and risk aggregates...")

        for node in G.nodes():
            out_edges = G.out_edges(node, data=True)
            in_edges = G.in_edges(node, data=True)

            out_degree = G.out_degree(node)
            in_degree = G.in_degree(node)

            total_sent = sum(float(d.get("amount_paid", 0.0)) for _, _, d in out_edges)
            total_received = sum(
                float(d.get("amount_received", 0.0)) if float(d.get("amount_received", 0.0)) > 0 else float(d.get("amount_paid", 0.0))
                for _, _, d in in_edges
            )

            unique_destinations = len(set(v for _, v, _ in out_edges))
            unique_sources = len(set(u for u, _, _ in in_edges))

            # ML risk signals aggregation on node
            out_rf_probs = [d.get("rf_laundering_prob", 0.0) for _, _, d in out_edges]
            in_rf_probs = [d.get("rf_laundering_prob", 0.0) for _, _, d in in_edges]
            all_rf_probs = out_rf_probs + in_rf_probs
            max_rf_prob = max(all_rf_probs) if all_rf_probs else 0.0
            mean_rf_prob = np.mean(all_rf_probs) if all_rf_probs else 0.0

            out_iso_scores = [d.get("iso_risk_score", 0.0) for _, _, d in out_edges]
            in_iso_scores = [d.get("iso_risk_score", 0.0) for _, _, d in in_edges]
            all_iso_scores = out_iso_scores + in_iso_scores
            max_iso_score = max(all_iso_scores) if all_iso_scores else 0.0

            suspicious_tx_count = sum(1 for _, _, d in out_edges if d.get("final_aml_alert") == "FLAGGED") + \
                                  sum(1 for _, _, d in in_edges if d.get("final_aml_alert") == "FLAGGED")

            # Entity type from edges if available
            entity_type = "Unknown"
            for _, _, d in out_edges:
                if d.get("from_entity_type") != "Unknown":
                    entity_type = d.get("from_entity_type")
                    break
            if entity_type == "Unknown":
                for _, _, d in in_edges:
                    if d.get("to_entity_type") != "Unknown":
                        entity_type = d.get("to_entity_type")
                        break

            bank = "Unknown"
            for _, _, d in out_edges:
                if d.get("from_bank") != "Unknown":
                    bank = d.get("from_bank")
                    break

            G.nodes[node].update({
                "account_id": node,
                "bank": bank,
                "entity_type": entity_type,
                "total_sent": total_sent,
                "total_received": total_received,
                "transaction_count": out_degree + in_degree,
                "unique_counterparties": unique_destinations + unique_sources,
                "fan_in_degree": in_degree,
                "fan_out_degree": out_degree,
                "unique_sources": unique_sources,
                "unique_destinations": unique_destinations,
                "max_rf_probability": max_rf_prob,
                "mean_rf_probability": mean_rf_prob,
                "max_iso_risk_score": max_iso_score,
                "suspicious_transaction_count": suspicious_tx_count,
            })

    def get_account_subgraph(self, account_id: str, radius: int = 1) -> nx.MultiDiGraph:
        """Extract ego network subgraph around an account up to a given radius."""
        account_id = str(account_id).strip().upper()
        if account_id not in self.graph:
            return nx.MultiDiGraph()

        # Get nodes within radius exploring bidirectional counterparties
        simple_g = nx.Graph(self.graph)
        ego_nodes = nx.single_source_shortest_path_length(simple_g, account_id, cutoff=radius).keys()
        return self.graph.subgraph(ego_nodes).copy()

    def export_graph_json(self) -> dict:
        """Export graph into frontend-friendly JSON format."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "type": data.get("entity_type", "account"),
                "bank": data.get("bank", "Unknown"),
                "total_sent": data.get("total_sent", 0.0),
                "total_received": data.get("total_received", 0.0),
                "fan_in": data.get("fan_in_degree", 0),
                "fan_out": data.get("fan_out_degree", 0),
                "max_risk": data.get("max_rf_probability", 0.0),
            })

        edges = []
        for u, v, k, d in self.graph.edges(data=True, keys=True):
            edges.append({
                "id": f"{d.get('transaction_id', k)}",
                "source": u,
                "target": v,
                "amount": d.get("amount_paid", 0.0),
                "timestamp": str(d.get("timestamp", "")),
                "currency": d.get("payment_currency", ""),
                "payment_format": d.get("payment_format", ""),
                "rf_prob": d.get("rf_laundering_prob", 0.0),
                "alert": d.get("final_aml_alert", "NORMAL"),
            })

        return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    df = clean_and_enrich_data()
    engine = AMLGraphEngine()
    # Build on a sample for quick test
    G = engine.build_graph(df.head(1000), run_inference=True)
    print(f"Test graph constructed with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
    sample_node = list(G.nodes())[0]
    print(f"Sample node {sample_node} attributes:", G.nodes[sample_node])
