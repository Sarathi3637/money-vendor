"""
Comprehensive Unit Tests for AML Graph Investigation Engine.
Covers graph construction, MultiDiGraph parallel edges, all pattern detectors (Fan-In, Fan-Out,
Cycles, Rapid Multi-Hop with timestamp ordering, Scatter-Gather), evidence generation,
timeline sorting, case grouping, duplicate case prevention, and normal connected transaction exclusion.
"""

import os
import sys
import pytest
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graph_engine import AMLGraphEngine
from src.graph_patterns import detect_fan_in, detect_fan_out, detect_cycles, detect_rapid_multi_hop, detect_scatter_gather
from src.investigation_cases import CaseManager


@pytest.fixture
def synthetic_transactions():
    """Create a synthetic transaction dataframe with known patterns and negative cases."""
    data = [
        # Fan-In pattern into ACC_X (4 senders within 48h)
        {"timestamp": "2022-09-01 10:00:00", "from_bank": "1", "from_account": "ACC_A1", "to_bank": "1", "to_account": "ACC_X", "amount_paid": 5000, "amount_received": 5000, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 10:05:00", "from_bank": "2", "from_account": "ACC_A2", "to_bank": "1", "to_account": "ACC_X", "amount_paid": 6000, "amount_received": 6000, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 10:10:00", "from_bank": "3", "from_account": "ACC_A3", "to_bank": "1", "to_account": "ACC_X", "amount_paid": 7000, "amount_received": 7000, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 10:15:00", "from_bank": "4", "from_account": "ACC_A4", "to_bank": "1", "to_account": "ACC_X", "amount_paid": 8000, "amount_received": 8000, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "ACH", "is_laundering": 1},

        # Parallel transactions between same accounts (MultiDiGraph test)
        {"timestamp": "2022-09-01 11:00:00", "from_bank": "1", "from_account": "ACC_X", "to_bank": "2", "to_account": "ACC_Y", "amount_paid": 1000, "amount_received": 1000, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "Wire", "is_laundering": 0},
        {"timestamp": "2022-09-01 11:30:00", "from_bank": "1", "from_account": "ACC_X", "to_bank": "2", "to_account": "ACC_Y", "amount_paid": 1500, "amount_received": 1500, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "Wire", "is_laundering": 0},

        # Normal connected transaction involving ACC_X (should be excluded from case timeline/evidence)
        {"timestamp": "2022-09-01 15:00:00", "from_bank": "1", "from_account": "ACC_X", "to_bank": "5", "to_account": "ACC_NORMAL_OUT", "amount_paid": 200, "amount_received": 200, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "Cash", "is_laundering": 0},

        # Fan-Out pattern from ACC_FO (4 destinations within 48h)
        {"timestamp": "2022-09-01 12:00:00", "from_bank": "1", "from_account": "ACC_FO", "to_bank": "2", "to_account": "ACC_D1", "amount_paid": 1000, "amount_received": 1000, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 12:05:00", "from_bank": "1", "from_account": "ACC_FO", "to_bank": "3", "to_account": "ACC_D2", "amount_paid": 1100, "amount_received": 1100, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 12:10:00", "from_bank": "1", "from_account": "ACC_FO", "to_bank": "4", "to_account": "ACC_D3", "amount_paid": 1200, "amount_received": 1200, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 12:15:00", "from_bank": "1", "from_account": "ACC_FO", "to_bank": "5", "to_account": "ACC_D4", "amount_paid": 1300, "amount_received": 1300, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},

        # Cycle pattern: C1 -> C2 -> C3 -> C1
        {"timestamp": "2022-09-01 13:00:00", "from_bank": "1", "from_account": "ACC_C1", "to_bank": "1", "to_account": "ACC_C2", "amount_paid": 10000, "amount_received": 10000, "payment_currency": "Euro", "receiving_currency": "Euro", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 13:05:00", "from_bank": "1", "from_account": "ACC_C2", "to_bank": "1", "to_account": "ACC_C3", "amount_paid": 9900, "amount_received": 9900, "payment_currency": "Euro", "receiving_currency": "Euro", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 13:10:00", "from_bank": "1", "from_account": "ACC_C3", "to_bank": "1", "to_account": "ACC_C1", "amount_paid": 9800, "amount_received": 9800, "payment_currency": "Euro", "receiving_currency": "Euro", "payment_format": "ACH", "is_laundering": 1},

        # Rapid Multi-Hop (Chronological forward: H1 -> H2 -> H3 -> H4)
        {"timestamp": "2022-09-01 14:00:00", "from_bank": "1", "from_account": "ACC_H1", "to_bank": "2", "to_account": "ACC_H2", "amount_paid": 5000, "amount_received": 5000, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "Wire", "is_laundering": 1},
        {"timestamp": "2022-09-01 14:02:00", "from_bank": "2", "from_account": "ACC_H2", "to_bank": "3", "to_account": "ACC_H3", "amount_paid": 4950, "amount_received": 4950, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "Wire", "is_laundering": 1},
        {"timestamp": "2022-09-01 14:04:00", "from_bank": "3", "from_account": "ACC_H3", "to_bank": "4", "to_account": "ACC_H4", "amount_paid": 4900, "amount_received": 4900, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "Wire", "is_laundering": 1},

        # Scatter-Gather (Sources S1, S2 -> Intermediate INT -> Destinations Dests D1, D2)
        {"timestamp": "2022-09-01 16:00:00", "from_bank": "1", "from_account": "ACC_S1", "to_bank": "2", "to_account": "ACC_INT", "amount_paid": 8000, "amount_received": 8000, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 16:05:00", "from_bank": "3", "from_account": "ACC_S2", "to_bank": "2", "to_account": "ACC_INT", "amount_paid": 9000, "amount_received": 9000, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 16:15:00", "from_bank": "2", "from_account": "ACC_INT", "to_bank": "4", "to_account": "ACC_DS1", "amount_paid": 8500, "amount_received": 8500, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},
        {"timestamp": "2022-09-01 16:20:00", "from_bank": "2", "from_account": "ACC_INT", "to_bank": "5", "to_account": "ACC_DS2", "amount_paid": 8400, "amount_received": 8400, "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "ACH", "is_laundering": 1},

        # Negative case: Normal isolated transaction
        {"timestamp": "2022-09-01 18:00:00", "from_bank": "9", "from_account": "ACC_NORMAL1", "to_bank": "9", "to_account": "ACC_NORMAL2", "amount_paid": 200, "amount_received": 200, "payment_currency": "US Dollar", "receiving_currency": "US Dollar", "payment_format": "Cash", "is_laundering": 0},
    ]
    # Assign unique transaction IDs
    for i, row in enumerate(data):
        row["transaction_id"] = f"TX_{i+1:04d}"
    return pd.DataFrame(data)


def test_graph_construction(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() == len(synthetic_transactions)


def test_multiedge_preservation(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    edge_data = G.get_edge_data("ACC_X", "ACC_Y")
    assert edge_data is not None
    assert len(edge_data) == 2


def test_fan_in_positive_and_negative(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    import src.graph_config as config
    orig_min = config.FAN_IN_MIN_SENDERS
    config.FAN_IN_MIN_SENDERS = 4

    fan_in_pats = detect_fan_in(G)
    config.FAN_IN_MIN_SENDERS = orig_min

    assert len(fan_in_pats) >= 1
    targets = [p["target_account"] for p in fan_in_pats]
    assert "ACC_X" in targets
    # Normal accounts should not trigger fan-in
    assert "ACC_NORMAL1" not in targets


def test_fan_out_positive_and_negative(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    import src.graph_config as config
    orig_min = config.FAN_OUT_MIN_DESTINATIONS
    config.FAN_OUT_MIN_DESTINATIONS = 4

    fan_out_pats = detect_fan_out(G)
    config.FAN_OUT_MIN_DESTINATIONS = orig_min

    assert len(fan_out_pats) >= 1
    sources = [p["source_account"] for p in fan_out_pats]
    assert "ACC_FO" in sources
    assert "ACC_NORMAL1" not in sources


def test_cycle_detection(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    cycles = detect_cycles(G)
    assert len(cycles) >= 1
    cycle_accounts = cycles[0]["accounts"]
    assert "ACC_C1" in cycle_accounts
    assert "ACC_C2" in cycle_accounts
    assert "ACC_C3" in cycle_accounts


def test_rapid_multi_hop_and_timestamp_ordering(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    multihop_pats = detect_rapid_multi_hop(G)
    assert len(multihop_pats) >= 1

    # Verify chronological timestamp ordering in multi-hop path
    for pat in multihop_pats:
        txs = pat["supporting_transactions"]
        timestamps = []
        for tx_id in txs:
            for u, v, k, d in G.edges(keys=True, data=True):
                if d.get("transaction_id") == tx_id:
                    timestamps.append(d.get("timestamp"))
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


def test_scatter_gather_positive_and_negative(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    sg_pats = detect_scatter_gather(G)
    assert len(sg_pats) >= 1
    intermediates = [p["intermediate_account"] for p in sg_pats]
    assert "ACC_INT" in intermediates
    assert "ACC_NORMAL1" not in intermediates


def test_investigation_cases_evidence_and_timeline(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    import src.graph_config as config
    config.FAN_IN_MIN_SENDERS = 4
    config.FAN_OUT_MIN_DESTINATIONS = 4

    fan_in_pats = detect_fan_in(G)
    cycle_pats = detect_cycles(G)
    all_pats = fan_in_pats + cycle_pats

    manager = CaseManager(G)
    cases = manager.generate_cases(all_pats)

    assert len(cases) > 0
    case = cases[0]
    assert "case_id" in case
    assert "risk_score" in case
    assert 0.0 <= case["risk_score"] <= 100.0
    assert "risk_level" in case
    assert "evidence" in case
    assert len(case["evidence"]) > 0
    assert "timeline" in case
    assert len(case["timeline"]) > 0

    # Verify timeline is sorted chronologically
    timestamps = [evt["timestamp"] for evt in case["timeline"]]
    assert timestamps == sorted(timestamps)


def test_normal_connected_transaction_exclusion(synthetic_transactions):
    """Proves that a normal connected transaction does not get incorrectly included in a suspicious case."""
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    import src.graph_config as config
    config.FAN_IN_MIN_SENDERS = 4
    fan_in_pats = detect_fan_in(G)

    manager = CaseManager(G)
    cases = manager.generate_cases(fan_in_pats)

    assert len(cases) == 1
    case = cases[0]

    # Verify that the normal transaction involving ACC_NORMAL_OUT (TX_0007) is NOT in the case timeline or accounts
    case_tx_ids = [evt["transaction_id"] for evt in case["timeline"]]
    for tx_id in case_tx_ids:
        # Find edge corresponding to tx_id
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("transaction_id") == tx_id:
                assert d.get("to_account") != "ACC_NORMAL_OUT"


def test_duplicate_case_prevention(synthetic_transactions):
    engine = AMLGraphEngine()
    engine.predictor = None
    G = engine.build_graph(synthetic_transactions, run_inference=False)

    import src.graph_config as config
    config.FAN_IN_MIN_SENDERS = 4
    fan_in_pats = detect_fan_in(G)

    manager = CaseManager(G)
    # Generate cases twice with same patterns
    cases1 = manager.generate_cases(fan_in_pats)
    cases2 = manager.generate_cases(fan_in_pats)

    assert len(cases1) == len(cases2)
    assert cases1[0]["case_id"] == cases2[0]["case_id"]
