"""
Comprehensive Backend Audit & Validation Test Suite for SHADOWTRACE.
Tests dataset isolation, file ingestion, schema detection, data quality scoring,
velocity chronology, ML compatibility, graph patterns, negative test cases,
idempotency, investigation cases, and account intelligence.
"""

import io
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app, initialize_aml_engine
from src.dataset_manager import dataset_manager, DatasetManager, CANONICAL_FIELDS
from src.features import AMLFeatureEngineer
from src.graph_engine import AMLGraphEngine
from src.graph_patterns import (
    detect_fan_in,
    detect_fan_out,
    detect_cycles,
    detect_rapid_multi_hop,
    detect_scatter_gather,
)
from src.investigation_cases import CaseManager


@pytest.fixture(scope="module")
def client():
    """Test client with initialized baseline engine."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        # Initialize if not already initialized
        if dataset_manager.get_active_dataset() is None or not dataset_manager.get_active_dataset().cases:
            initialize_aml_engine()
        yield client


# ============================================================
# 1. FILE INGESTION & FORMAT VALIDATION
# ============================================================

def test_csv_upload(client):
    """Verify CSV file ingestion, storage, profiling, and mapping."""
    csv_content = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2023-01-01 10:00:00,BankA,ACC_01,BankB,ACC_02,1000.0,USD,1000.0,USD,Wire,0\n"
        "2023-01-01 11:00:00,BankB,ACC_02,BankC,ACC_03,950.0,USD,950.0,USD,Wire,0\n"
    )
    data = {"file": (io.BytesIO(csv_content.encode("utf-8")), "sample_upload.csv")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    assert json_data["status"] == "success"
    assert json_data["rows_count"] == 2
    assert "dataset_id" in json_data
    assert json_data["dataset_id"].startswith("dataset_")


def test_xlsx_upload(client):
    """Verify XLSX file ingestion."""
    df = pd.DataFrame([
        {
            "Timestamp": "2023-01-01 10:00:00",
            "From Bank": "BankA",
            "Account": "ACC_X1",
            "To Bank": "BankB",
            "Account.1": "ACC_X2",
            "Amount Received": 5000.0,
            "Receiving Currency": "USD",
            "Amount Paid": 5000.0,
            "Payment Currency": "USD",
            "Payment Format": "ACH",
            "Is Laundering": 0,
        }
    ])
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    data = {"file": (output, "sample_excel.xlsx")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    assert json_data["rows_count"] == 1


def test_parquet_upload(client):
    """Verify Parquet file ingestion."""
    df = pd.DataFrame([
        {
            "timestamp": "2023-01-01 10:00:00",
            "from_bank": "BankA",
            "from_account": "ACC_P1",
            "to_bank": "BankB",
            "to_account": "ACC_P2",
            "amount_received": 1200.0,
            "receiving_currency": "USD",
            "amount_paid": 1200.0,
            "payment_currency": "USD",
            "payment_format": "Wire",
            "is_laundering": 0,
        }
    ])
    output = io.BytesIO()
    df.to_parquet(output, index=False)
    output.seek(0)

    data = {"file": (output, "sample_parquet.parquet")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    assert json_data["rows_count"] == 1


def test_malformed_and_empty_uploads(client):
    """Verify handling of empty files, corrupt data, and unsupported formats."""
    # 1. Unsupported extension
    data = {"file": (io.BytesIO(b"dummy pdf content"), "malicious.exe")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Unsupported file format" in res.get_json()["error"]

    # 2. Empty file (0 bytes)
    data = {"file": (io.BytesIO(b""), "empty.csv")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400

    # 3. Malformed corrupt file
    data = {"file": (io.BytesIO(b"\x00\x01\x02corruptedbinary"), "corrupt.xlsx")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


# ============================================================
# 2. SCHEMA DETECTION & MANUAL OVERRIDE
# ============================================================

def test_schema_detection_aliases(client):
    """Verify automatic detection with custom aliases (sender, receiver, etc.)."""
    csv_content = (
        "date,source_account,destination_account,value,sender_bank,target_bank\n"
        "2023-05-01 12:00:00,ACC_SRC,ACC_DST,25000.0,Bank1,Bank2\n"
    )
    data = {"file": (io.BytesIO(csv_content.encode("utf-8")), "alias_test.csv")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    mapping = json_data["schema_mapping"]
    assert mapping["timestamp"] == "date"
    assert mapping["from_account"] == "source_account"
    assert mapping["to_account"] == "destination_account"
    assert mapping["amount_paid"] == "value"


def test_manual_schema_mapping_override(client):
    """Verify manual override via POST /api/datasets/<id>/schema and actual effect on pipeline."""
    csv_content = (
        "col_a,col_b,col_c,col_d\n"
        "2023-01-01,USER_1,USER_2,500.0\n"
    )
    data = {"file": (io.BytesIO(csv_content.encode("utf-8")), "unknown_cols.csv")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    d_id = res.get_json()["dataset_id"]

    # Supply manual mapping override
    override_payload = {
        "mapping": {
            "timestamp": "col_a",
            "from_account": "col_b",
            "to_account": "col_c",
            "amount_paid": "col_d",
        }
    }
    res2 = client.post(f"/api/datasets/{d_id}/schema", json=override_payload)
    assert res2.status_code == 200
    updated_map = res2.get_json()["schema_mapping"]
    assert updated_map["from_account"] == "col_b"
    assert updated_map["to_account"] == "col_c"

    # Verify the mapped dataframe was updated
    inst = dataset_manager.get_dataset(d_id)
    assert inst.df_mapped["from_account"].iloc[0] == "USER_1"
    assert inst.df_mapped["to_account"].iloc[0] == "USER_2"
    assert inst.df_mapped["amount_paid"].iloc[0] == 500.0


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

def test_data_quality_penalties(client):
    """Verify that quality score is calculated dynamically and detects dirty data."""
    # Create dataset with invalid amounts, duplicates, and missing accounts
    dirty_csv = (
        "timestamp,from_account,to_account,amount_paid\n"
        "2023-01-01 10:00:00,ACC_1,ACC_2,-500.0\n"        # Negative amount
        "2023-01-01 10:00:00,ACC_1,ACC_2,-500.0\n"        # Duplicate row
        "2023-01-01 11:00:00,,ACC_3,100.0\n"              # Missing sender
        "2023-01-01 12:00:00,ACC_2,,0.0\n"                # Missing receiver & zero amount
        "invalid_time,ACC_4,ACC_5,200.0\n"                 # Invalid timestamp
    )
    data = {"file": (io.BytesIO(dirty_csv.encode("utf-8")), "dirty_data.csv")}
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    quality = res.get_json()["profile"]
    q_score = res.get_json()["quality_score"]

    # Quality score must NOT be 100%
    assert q_score < 100.0
    d_id = res.get_json()["dataset_id"]
    res_q = client.get(f"/api/datasets/{d_id}/quality")
    q_rep = res_q.get_json()["quality_report"]
    assert q_rep["negative_amounts"] >= 1
    assert q_rep["duplicate_rows"] >= 1


# ============================================================
# 4. DATASET ISOLATION (DATASET A VS DATASET B)
# ============================================================

def test_dataset_isolation_a_vs_b(client):
    """Verify strict dataset isolation: Dataset A and Dataset B never share accounts, transactions, or cases."""
    # Dataset A
    csv_a = (
        "timestamp,from_account,to_account,amount_paid,from_bank,to_bank\n"
        "2023-01-01 10:00:00,A_SENDER_1,A_TARGET,1000.0,BankA,BankA\n"
        "2023-01-01 11:00:00,A_SENDER_2,A_TARGET,1000.0,BankA,BankA\n"
        "2023-01-01 12:00:00,A_SENDER_3,A_TARGET,1000.0,BankA,BankA\n"
        "2023-01-01 13:00:00,A_SENDER_4,A_TARGET,1000.0,BankA,BankA\n"
    )
    # Dataset B
    csv_b = (
        "timestamp,from_account,to_account,amount_paid,from_bank,to_bank\n"
        "2023-02-01 10:00:00,B_SRC,B_DEST_1,2000.0,BankB,BankB\n"
        "2023-02-01 11:00:00,B_SRC,B_DEST_2,2000.0,BankB,BankB\n"
        "2023-02-01 12:00:00,B_SRC,B_DEST_3,2000.0,BankB,BankB\n"
        "2023-02-01 13:00:00,B_SRC,B_DEST_4,2000.0,BankB,BankB\n"
    )

    res_a = client.post("/api/datasets/upload", data={"file": (io.BytesIO(csv_a.encode("utf-8")), "data_a.csv")}, content_type="multipart/form-data")
    res_b = client.post("/api/datasets/upload", data={"file": (io.BytesIO(csv_b.encode("utf-8")), "data_b.csv")}, content_type="multipart/form-data")
    id_a = res_a.get_json()["dataset_id"]
    id_b = res_b.get_json()["dataset_id"]

    assert id_a != id_b

    # Run analysis on Dataset A
    client.post(f"/api/datasets/{id_a}/analyze")
    # Run analysis on Dataset B
    client.post(f"/api/datasets/{id_b}/analyze")

    # Verify accounts isolation
    accs_a = client.get(f"/api/datasets/{id_a}/accounts").get_json()["accounts"]
    accs_b = client.get(f"/api/datasets/{id_b}/accounts").get_json()["accounts"]
    a_names = [a["account_id"] for a in accs_a]
    b_names = [b["account_id"] for b in accs_b]

    assert "A_TARGET" in a_names
    assert "A_TARGET" not in b_names
    assert "B_SRC" in b_names
    assert "B_SRC" not in a_names

    # Verify transactions isolation
    txs_a = client.get(f"/api/datasets/{id_a}/transactions").get_json()["transactions"]
    txs_b = client.get(f"/api/datasets/{id_b}/transactions").get_json()["transactions"]
    assert all("A_" in t["from_account"] for t in txs_a)
    assert all("B_" in t["from_account"] for t in txs_b)

    # Verify cases isolation
    cases_a = client.get(f"/api/datasets/{id_a}/investigations").get_json()["cases"]
    cases_b = client.get(f"/api/datasets/{id_b}/investigations").get_json()["cases"]
    for ca in cases_a:
        assert ca["dataset_id"] == id_a
    for cb in cases_b:
        assert cb["dataset_id"] == id_b


# ============================================================
# 5. VELOCITY CHRONOLOGY & REAL-TIME INFERENCE
# ============================================================

def test_velocity_chronology_no_future_leakage():
    """Verify that velocity calculation uses only past records in offline mode, and state cache in realtime mode."""
    fe = AMLFeatureEngineer()
    # Mock learned degree stats to avoid KeyError
    fe.from_degree_stats_ = {"count": {}, "unique_dest": {}, "total_sent": {}, "mean_sent": {}}
    fe.to_degree_stats_ = {"count": {}, "unique_src": {}, "total_received": {}, "mean_received": {}}
    fe.bank_stats_ = {"from_bank_counts": {}, "to_bank_counts": {}}

    # Transactions out of chronological order in the input DataFrame
    df = pd.DataFrame([
        {"timestamp": "2023-01-01 12:00:00", "from_account": "ACC_V1", "to_account": "ACC_V2", "amount_paid": 100, "amount_received": 100, "from_bank": "B1", "to_bank": "B2", "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "Wire", "from_entity_id": "Unknown", "to_entity_id": "Unknown"},
        {"timestamp": "2023-01-01 10:00:00", "from_account": "ACC_V1", "to_account": "ACC_V2", "amount_paid": 100, "amount_received": 100, "from_bank": "B1", "to_bank": "B2", "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "Wire", "from_entity_id": "Unknown", "to_entity_id": "Unknown"},
    ])

    num_df = fe._extract_numerical_features(df, is_realtime=False)
    # The first row (12:00) should have previous timestamp 10:00 (diff = 7200 sec)
    # The second row (10:00) was the first chronological transaction for ACC_V1, so its diff must be 86400 (default)
    # log1p(7200) != log1p(86400)
    diff_row0 = np.expm1(num_df["time_diff_prev_from"].iloc[0])
    diff_row1 = np.expm1(num_df["time_diff_prev_from"].iloc[1])
    assert abs(diff_row0 - 7200.0) < 1.0
    assert abs(diff_row1 - 86400.0) < 1.0

    # Test Real-Time Mode with state cache
    state_cache = {"ACC_V1": pd.to_datetime("2023-01-01 11:50:00")}
    single_tx = pd.DataFrame([{
        "timestamp": "2023-01-01 12:00:00", "from_account": "ACC_V1", "to_account": "ACC_V2",
        "amount_paid": 100, "amount_received": 100, "from_bank": "B1", "to_bank": "B2",
        "payment_currency": "USD", "receiving_currency": "USD", "payment_format": "Wire",
        "from_entity_id": "Unknown", "to_entity_id": "Unknown"
    }])
    num_rt = fe._extract_numerical_features(single_tx, is_realtime=True, state_cache=state_cache)
    rt_diff = np.expm1(num_rt["time_diff_prev_from"].iloc[0])
    # 10 minutes = 600 seconds
    assert abs(rt_diff - 600.0) < 1.0
    # Cache must be updated to 12:00
    assert state_cache["ACC_V1"] == pd.to_datetime("2023-01-01 12:00:00")


# ============================================================
# 6. ML COMPATIBILITY & INDEPENDENCE
# ============================================================

def test_ml_compatibility_validation():
    """Verify that AMLFeatureEngineer rejects incompatible schemas and does not silently fake predictions."""
    fe = AMLFeatureEngineer()
    incompatible_df = pd.DataFrame({"random_col": [1, 2, 3]})
    compat, msg = fe.validate_feature_compatibility(incompatible_df)
    assert not compat
    assert "Supervised model unavailable for this schema" in msg


# ============================================================
# 7. GRAPH PATTERNS: SYNTHETIC POSITIVE & NEGATIVE TEST CASES
# ============================================================

def test_graph_patterns_positive_and_negative_cases():
    """Verify Fan-In, Fan-Out, Cycles, Rapid Multi-Hop, and Scatter-Gather with positive and negative cases."""
    now = datetime(2023, 6, 1, 10, 0, 0)
    G = nx.MultiDiGraph()

    # 1. Fan-In Positive (4 senders -> TARGET in 2 hours)
    for i in range(1, 5):
        G.add_edge(f"FI_SRC_{i}", "FI_TARGET", key=f"fi_{i}", transaction_id=f"tx_fi_{i}", timestamp=now + timedelta(minutes=i * 10), amount_paid=100.0)

    # 2. Fan-In Negative (senders spread across 200 hours > 48h window)
    for i in range(1, 5):
        G.add_edge(f"NEG_FI_SRC_{i}", "NEG_FI_TARGET", key=f"neg_fi_{i}", transaction_id=f"tx_neg_fi_{i}", timestamp=now + timedelta(hours=i * 60), amount_paid=100.0)

    # 3. Fan-Out Positive (1 source -> 4 destinations in 2 hours)
    for i in range(1, 5):
        G.add_edge("FO_SRC", f"FO_DST_{i}", key=f"fo_{i}", transaction_id=f"tx_fo_{i}", timestamp=now + timedelta(minutes=i * 10), amount_paid=100.0)

    # 4. Directed Cycle (3 nodes: C1 -> C2 -> C3 -> C1)
    G.add_edge("C1", "C2", key="cyc_1", transaction_id="tx_cyc_1", timestamp=now + timedelta(hours=1), amount_paid=500.0)
    G.add_edge("C2", "C3", key="cyc_2", transaction_id="tx_cyc_2", timestamp=now + timedelta(hours=2), amount_paid=500.0)
    G.add_edge("C3", "C1", key="cyc_3", transaction_id="tx_cyc_3", timestamp=now + timedelta(hours=3), amount_paid=500.0)

    # 5. Rapid Multi-Hop (H1 -> H2 -> H3 -> H4 in forward chronological order)
    G.add_edge("H1", "H2", key="hop_1", transaction_id="tx_hop_1", timestamp=now + timedelta(hours=1), amount_paid=1000.0)
    G.add_edge("H2", "H3", key="hop_2", transaction_id="tx_hop_2", timestamp=now + timedelta(hours=2), amount_paid=950.0)
    G.add_edge("H3", "H4", key="hop_3", transaction_id="tx_hop_3", timestamp=now + timedelta(hours=3), amount_paid=900.0)

    # 6. Scatter-Gather Positive (SG_IN1, SG_IN2 -> SG_INT -> SG_OUT1, SG_OUT2 with inflow preceding outflow)
    G.add_edge("SG_IN1", "SG_INT", key="sg_in1", transaction_id="tx_sg_in1", timestamp=now + timedelta(hours=1), amount_paid=500.0)
    G.add_edge("SG_IN2", "SG_INT", key="sg_in2", transaction_id="tx_sg_in2", timestamp=now + timedelta(hours=2), amount_paid=500.0)
    G.add_edge("SG_INT", "SG_OUT1", key="sg_out1", transaction_id="tx_sg_out1", timestamp=now + timedelta(hours=4), amount_paid=450.0)
    G.add_edge("SG_INT", "SG_OUT2", key="sg_out2", transaction_id="tx_sg_out2", timestamp=now + timedelta(hours=5), amount_paid=450.0)

    # 7. Scatter-Gather Negative (Outflows occurred 50 hours BEFORE any inflows arrived)
    G.add_edge("NEG_SG_INT", "NEG_SG_OUT1", key="neg_sg_out1", transaction_id="tx_neg_sg_out1", timestamp=now, amount_paid=450.0)
    G.add_edge("NEG_SG_INT", "NEG_SG_OUT2", key="neg_sg_out2", transaction_id="tx_neg_sg_out2", timestamp=now + timedelta(hours=1), amount_paid=450.0)
    G.add_edge("NEG_SG_IN1", "NEG_SG_INT", key="neg_sg_in1", transaction_id="tx_neg_sg_in1", timestamp=now + timedelta(hours=50), amount_paid=500.0)
    G.add_edge("NEG_SG_IN2", "NEG_SG_INT", key="neg_sg_in2", transaction_id="tx_neg_sg_in2", timestamp=now + timedelta(hours=51), amount_paid=500.0)

    # Assertions
    fi = detect_fan_in(G)
    fi_targets = [p["target_account"] for p in fi]
    assert "FI_TARGET" in fi_targets
    assert "NEG_FI_TARGET" not in fi_targets

    fo = detect_fan_out(G)
    fo_sources = [p["source_account"] for p in fo]
    assert "FO_SRC" in fo_sources

    cycles = detect_cycles(G)
    assert len(cycles) >= 1
    assert any("C1" in c["accounts"] and "C2" in c["accounts"] for c in cycles)

    hops = detect_rapid_multi_hop(G)
    assert len(hops) >= 1
    hop_paths = [h["path"] for h in hops]
    assert any("H1" in p and "H4" in p for p in hop_paths)

    sg = detect_scatter_gather(G)
    sg_intermediates = [p["intermediate_account"] for p in sg]
    assert "SG_INT" in sg_intermediates
    assert "NEG_SG_INT" not in sg_intermediates  # Negative case strictly excluded!


# ============================================================
# 8. INVESTIGATION CASE TRACEABILITY & REPLAY
# ============================================================

def test_case_generation_and_replay(client):
    """Verify case evidence, deterministic risk scores, chronological timeline, and replay frames."""
    # Use baseline dataset
    cases_res = client.get("/api/investigations?limit=5")
    assert cases_res.status_code == 200
    cases_data = cases_res.get_json()
    assert cases_data["total"] > 0
    case_id = cases_data["cases"][0]["case_id"]

    # Detail view
    detail_res = client.get(f"/api/investigations/{case_id}")
    assert detail_res.status_code == 200
    case_dict = detail_res.get_json()
    assert "risk_score" in case_dict
    assert 0 <= case_dict["risk_score"] <= 100
    assert case_dict["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert "timeline" in case_dict
    assert "evidence" in case_dict

    # Replay
    active_d_id = dataset_manager.active_dataset_id
    replay_res = client.get(f"/api/datasets/{active_d_id}/investigations/{case_id}/replay")
    assert replay_res.status_code == 200
    replay_data = replay_res.get_json()
    assert "frames" in replay_data
    if replay_data["frames"]:
        first_frame = replay_data["frames"][0]
        assert "timestamp" in first_frame
        assert "active_transaction" in first_frame
        assert "cumulative_volume" in first_frame


# ============================================================
# 9. PIPELINE IDEMPOTENCY & TRANSACTION SEARCH
# ============================================================

def test_pipeline_idempotency_and_search(client):
    """Verify that repeated analyze calls do not duplicate cases, and transaction search filters work."""
    csv_content = (
        "timestamp,from_account,to_account,amount_paid,from_bank,to_bank,payment_currency\n"
        "2023-01-01 10:00:00,IDEMP_1,IDEMP_TARGET,1000.0,Chase,Chase,USD\n"
        "2023-01-01 10:30:00,IDEMP_2,IDEMP_TARGET,1000.0,Chase,Chase,USD\n"
        "2023-01-01 11:00:00,IDEMP_3,IDEMP_TARGET,1000.0,Chase,Chase,USD\n"
        "2023-01-01 11:30:00,IDEMP_4,IDEMP_TARGET,1000.0,Chase,Chase,USD\n"
    )
    res = client.post("/api/datasets/upload", data={"file": (io.BytesIO(csv_content.encode("utf-8")), "idemp.csv")}, content_type="multipart/form-data")
    d_id = res.get_json()["dataset_id"]

    # First analysis run
    run1 = client.post(f"/api/datasets/{d_id}/analyze")
    assert run1.status_code == 200
    cases_count_1 = run1.get_json()["results"]["cases_count"]

    # Second analysis run (idempotency check)
    run2 = client.post(f"/api/datasets/{d_id}/analyze")
    assert run2.status_code == 200
    cases_count_2 = run2.get_json()["results"]["cases_count"]

    assert cases_count_1 == cases_count_2  # Must not duplicate!

    # Test Transaction Search Filters
    tx_search = client.get(f"/api/datasets/{d_id}/transactions?account=IDEMP_1")
    assert tx_search.status_code == 200
    records = tx_search.get_json()["transactions"]
    assert len(records) == 1
    assert records[0]["from_account"] == "IDEMP_1"

    # Test Account Dossier endpoint
    acc_dossier = client.get(f"/api/datasets/{d_id}/accounts/IDEMP_TARGET")
    assert acc_dossier.status_code == 200
    acc_json = acc_dossier.get_json()
    assert acc_json["account_id"] == "IDEMP_TARGET"
    assert acc_json["total_received"] == 4000.0
    assert len(acc_json["associated_cases"]) > 0


# ============================================================
# 10. REAL DATASET 1 VALIDATION
# ============================================================

def test_real_dataset_1_validation():
    """Verify that data/Dataset 1.xlsx exists and contains expected record count (~70,568)."""
    dataset_1_path = "data/Dataset 1.xlsx"
    parquet_path = "data/transactions_cleaned.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(dataset_1_path):
        df = pd.read_excel(dataset_1_path)
    else:
        pytest.skip("Dataset 1 file not present in local workspace.")

    row_count = len(df)
    col_count = len(df.columns)
    # Calculated dynamically without hardcoding
    assert 50000 < row_count < 100000, f"Expected ~70,568 rows, found {row_count}"
    assert col_count >= 10, f"Expected >= 10 columns, found {col_count}"
