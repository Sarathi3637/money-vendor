"""
Investigation Cases & Evidence Generation Module.
Combines graph structural patterns, timelines, and pre-trained ML risk outputs into
unified Investigation Cases, computing prototype multi-faceted risk scores (0-100).

IMPORTANT TERMINOLOGY:
- Risk scores are prototype investigation risk scores and DO NOT represent official
  AML regulatory scores or real-world probabilities of money laundering.
- Cases are labeled as "Potential AML Risk", "Requires Investigation", "Investigation Case".
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import src.graph_config as config


class CaseManager:
    """Manages creation, enrichment, risk-scoring, and exporting of investigation cases."""

    def __init__(self, G):
        self.G = G
        self.cases = []

    def generate_cases(self, detected_patterns, dataset_id: str = "dataset_primary"):
        """
        Group overlapping patterns (sharing accounts or transactions) into unified investigation cases.
        Option B Compliance: Grouping is calculated strictly on suspicious/pattern-supporting nodes,
        edges, and transactions. Normal transaction edges that are not part of any detected pattern
        are excluded from investigation cases and timelines.
        """
        if not detected_patterns:
            return []

        # Build adjacency over pattern-supporting nodes
        patterns_by_node = {}
        for pat in detected_patterns:
            accounts = set()
            if "target_account" in pat: accounts.add(pat["target_account"])
            if "source_account" in pat: accounts.add(pat["source_account"])
            if "accounts" in pat: accounts.update(pat["accounts"])
            if "path" in pat: accounts.update(pat["path"])
            if "intermediate_account" in pat: accounts.add(pat["intermediate_account"])
            if "sources" in pat: accounts.update(pat["sources"])
            if "destinations" in pat: accounts.update(pat["destinations"])

            for acc in accounts:
                if acc not in patterns_by_node:
                    patterns_by_node[acc] = []
                patterns_by_node[acc].append(pat)

        # Merge patterns sharing nodes (connected components of suspicious patterns)
        grouped_cases = []
        visited_patterns = set()

        for pat in detected_patterns:
            pat_id = id(pat)
            if pat_id in visited_patterns:
                continue

            current_group = []
            queue = [pat]
            visited_patterns.add(pat_id)

            while queue:
                current_pat = queue.pop(0)
                current_group.append(current_pat)

                nodes = set()
                if "target_account" in current_pat: nodes.add(current_pat["target_account"])
                if "source_account" in current_pat: nodes.add(current_pat["source_account"])
                if "accounts" in current_pat: nodes.update(current_pat["accounts"])
                if "path" in current_pat: nodes.update(current_pat["path"])
                if "intermediate_account" in current_pat: nodes.add(current_pat["intermediate_account"])
                if "sources" in current_pat: nodes.update(current_pat["sources"])
                if "destinations" in current_pat: nodes.update(current_pat["destinations"])

                for node in nodes:
                    for neighbor_pat in patterns_by_node.get(node, []):
                        neighbor_id = id(neighbor_pat)
                        if neighbor_id not in visited_patterns:
                            visited_patterns.add(neighbor_id)
                            queue.append(neighbor_pat)

            grouped_cases.append(current_group)

        # Build cases with pre-indexed transaction lookup for O(1) timeline construction
        edge_by_tx = {}
        for u, v, k, d in self.G.edges(data=True, keys=True):
            tx_id = d.get("transaction_id")
            if tx_id is not None:
                edge_by_tx[tx_id] = (u, v, d)

        self.cases = []
        for idx, group in enumerate(grouped_cases):
            case_id = f"AML-{idx + 1:04d}"
            case_data = self._build_single_case(case_id, group, edge_by_tx, dataset_id=dataset_id)
            self.cases.append(case_data)

        # Sort cases by risk score descending
        self.cases.sort(key=lambda x: x["risk_score"], reverse=True)
        return self.cases

    def _build_single_case(self, case_id, patterns, edge_by_tx=None, dataset_id: str = "dataset_primary"):
        """Construct case dictionary with chronology, evidence, risk scores, and graph/ML signals."""
        accounts = set()
        transactions_ids = set()
        banks = set()
        evidence = []

        for pat in patterns:
            # Accounts
            if "target_account" in pat: accounts.add(pat["target_account"])
            if "source_account" in pat: accounts.add(pat["source_account"])
            if "accounts" in pat: accounts.update(pat["accounts"])
            if "path" in pat: accounts.update(pat["path"])
            if "intermediate_account" in pat: accounts.add(pat["intermediate_account"])
            if "sources" in pat: accounts.update(pat["sources"])
            if "destinations" in pat: accounts.update(pat["destinations"])

            # Transactions
            if "supporting_transactions" in pat:
                transactions_ids.update(pat["supporting_transactions"])

            # Generate structured evidence based on real-data patterns
            ptype = pat.get("type")
            if ptype == "Fan-In":
                evidence.append({
                    "indicator": "Fan-In Network",
                    "severity": "High" if pat.get("unique_senders", 0) >= 6 else "Medium",
                    "description": f"Account {pat['target_account']} received funds from {pat.get('unique_senders')} unique accounts within {pat.get('time_span_hours', 0):.1f} hours.",
                    "supporting_transactions": pat.get("supporting_transactions", [])
                })
            elif ptype == "Fan-Out":
                evidence.append({
                    "indicator": "Fan-Out Distribution",
                    "severity": "High" if pat.get("unique_destinations", 0) >= 6 else "Medium",
                    "description": f"Account {pat['source_account']} distributed funds to {pat.get('unique_destinations')} unique accounts within {pat.get('time_span_hours', 0):.1f} hours.",
                    "supporting_transactions": pat.get("supporting_transactions", [])
                })
            elif ptype == "Circular Transfer":
                evidence.append({
                    "indicator": "Circular Fund Loop",
                    "severity": "High",
                    "description": f"Funds moved loop-style through {pat.get('hops')} accounts: {' -> '.join(pat.get('accounts', []))}.",
                    "supporting_transactions": pat.get("supporting_transactions", [])
                })
            elif ptype == "Rapid Multi-Hop":
                evidence.append({
                    "indicator": "Rapid Multi-Hop Flow",
                    "severity": "High",
                    "description": f"Rapid chronological movement across {pat.get('hops')} hops within {pat.get('duration_hours', 0):.1f} hours: {' -> '.join(pat.get('path', []))}.",
                    "supporting_transactions": pat.get("supporting_transactions", [])
                })
            elif ptype == "Scatter-Gather":
                evidence.append({
                    "indicator": "Scatter-Gather Convergence",
                    "severity": "High",
                    "description": f"Intermediate account {pat['intermediate_account']} routed funds between multiple sources and destinations.",
                    "supporting_transactions": pat.get("supporting_transactions", [])
                })

        # Compile chronological timeline using ONLY suspicious supporting transactions (Option B)
        timeline = []
        if edge_by_tx is not None:
            for tx_id in transactions_ids:
                if tx_id in edge_by_tx:
                    u, v, d = edge_by_tx[tx_id]
                    banks.add(d.get("from_bank"))
                    banks.add(d.get("to_bank"))
                    timeline.append({
                        "timestamp": str(d.get("timestamp")),
                        "transaction_id": str(tx_id),
                        "sender": u,
                        "receiver": v,
                        "from_account": u,
                        "to_account": v,
                        "amount": d.get("amount_paid"),
                        "amount_paid": d.get("amount_paid"),
                        "currency": d.get("payment_currency"),
                        "payment_currency": d.get("payment_currency"),
                        "payment_format": d.get("payment_format", "Wire"),
                        "source_bank": d.get("from_bank"),
                        "from_bank": d.get("from_bank"),
                        "destination_bank": d.get("to_bank"),
                        "to_bank": d.get("to_bank"),
                        "rf_prob": d.get("rf_laundering_prob", 0.0),
                        "rf_laundering_prob": d.get("rf_laundering_prob", 0.0),
                        "iso_score": d.get("iso_risk_score", 0.0),
                        "iso_risk_score": d.get("iso_risk_score", 0.0),
                        "final_aml_alert": d.get("final_aml_alert", "FLAGGED"),
                    })
        else:
            for u, v, k, d in self.G.edges(data=True, keys=True):
                tx_id = d.get("transaction_id")
                if tx_id in transactions_ids:
                    banks.add(d.get("from_bank"))
                    banks.add(d.get("to_bank"))
                    timeline.append({
                        "timestamp": str(d.get("timestamp")),
                        "transaction_id": str(tx_id),
                        "sender": u,
                        "receiver": v,
                        "from_account": u,
                        "to_account": v,
                        "amount": d.get("amount_paid"),
                        "amount_paid": d.get("amount_paid"),
                        "currency": d.get("payment_currency"),
                        "payment_currency": d.get("payment_currency"),
                        "payment_format": d.get("payment_format", "Wire"),
                        "source_bank": d.get("from_bank"),
                        "from_bank": d.get("from_bank"),
                        "destination_bank": d.get("to_bank"),
                        "to_bank": d.get("to_bank"),
                        "rf_prob": d.get("rf_laundering_prob", 0.0),
                        "rf_laundering_prob": d.get("rf_laundering_prob", 0.0),
                        "iso_score": d.get("iso_risk_score", 0.0),
                        "iso_risk_score": d.get("iso_risk_score", 0.0),
                        "final_aml_alert": d.get("final_aml_alert", "FLAGGED"),
                    })

        # Sort chronological timeline
        timeline.sort(key=lambda x: x["timestamp"])

        # Aggregate node and edge ML stats
        rf_probs = [evt["rf_prob"] for evt in timeline]
        iso_scores = [evt["iso_score"] for evt in timeline]
        mean_rf = np.mean(rf_probs) if rf_probs else 0.0
        max_rf = max(rf_probs) if rf_probs else 0.0
        max_iso = max(iso_scores) if iso_scores else 0.0

        # Calculate multi-faceted transparent risk score (0-100)
        risk_score = self._calculate_risk_score(patterns, timeline, mean_rf, max_rf, max_iso)
        risk_level = "LOW"
        for level, boundary in sorted(config.RISK_TIERS.items(), key=lambda x: x[1], reverse=True):
            if risk_score >= boundary:
                risk_level = level
                break

        # Categorized evidence structure for frontend matrix & compliance reporting
        evidence_dict = {
            "pattern_evidence": [f"{e['indicator']}: {e['description']}" for e in evidence] if evidence else ["Structural topology matching AML pattern signatures."],
            "ml_signals": [
                f"Supervised RF Laundering Probability: Max {max_rf:.1%}, Mean {mean_rf:.1%}",
                f"Unsupervised Isolation Forest Outlier Score: {max_iso:.4f}"
            ],
            "velocity_flags": [
                f"{len(timeline)} chronological suspicious transactions recorded across {len(banks)} institutions",
                f"Time window spanning {timeline[0]['timestamp']} to {timeline[-1]['timestamp']}" if timeline else "Single time event"
            ],
            "network_complexity": [
                f"{len(accounts)} associated accounts involved in identified structural patterns",
                f"Identified topologies: {', '.join(set(pat['type'] for pat in patterns))}"
            ],
            "indicators": evidence
        }

        return {
            "case_id": case_id,
            "dataset_id": dataset_id,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "status": "Requires Investigation",
            "accounts_count": len(accounts),
            "transactions_count": len(transactions_ids),
            "suspicious_accounts": list(accounts),
            "suspicious_transactions": list(transactions_ids),
            "accounts": list(accounts),
            "banks": list(banks),
            "patterns": list(set(pat["type"] for pat in patterns)),
            "timeline": timeline,
            "evidence": evidence_dict,
            "ml_signals": {
                "mean_rf_probability": round(mean_rf, 4),
                "max_rf_probability": round(max_rf, 4),
                "max_iso_risk_score": round(max_iso, 4)
            },
            "graph_signals": {
                "unique_patterns_count": len(patterns),
                "accounts_involved": len(accounts),
                "transactions_involved": len(transactions_ids)
            },
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _calculate_risk_score(self, patterns, timeline, mean_rf, max_rf, max_iso):
        """Calculates risk score based on transparent configured weights (0-100 bounded)."""
        graph_types = set(pat["type"] for pat in patterns)
        graph_score = min(len(graph_types) * 25.0, 100.0)

        ml_score = (max_rf * 70.0) + (min(max(max_iso, 0), 1) * 30.0)
        ml_score = min(ml_score, 100.0)

        velocity_score = min(len(timeline) * 10.0, 100.0)

        accounts_involved = len(set(evt["sender"] for evt in timeline) | set(evt["receiver"] for evt in timeline))
        network_score = min(accounts_involved * 15.0, 100.0)

        banks_involved = len(set(evt["source_bank"] for evt in timeline) | set(evt["destination_bank"] for evt in timeline))
        cross_bank_score = min(banks_involved * 20.0, 100.0)

        overall_score = (
            graph_score * config.RISK_WEIGHTS["graph_score"] +
            ml_score * config.RISK_WEIGHTS["ml_score"] +
            velocity_score * config.RISK_WEIGHTS["velocity_score"] +
            network_score * config.RISK_WEIGHTS["network_complexity"] +
            cross_bank_score * config.RISK_WEIGHTS["cross_bank_context"]
        )

        return min(max(overall_score, 0.0), 100.0)
