"""
Unit Tests for Dataset Manager and Dataset-First API Endpoints.
Tests upload, schema detection, data quality score, profile preview,
and dataset isolation.
"""

import os
import io
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app, initialize_aml_engine
from src.dataset_manager import dataset_manager, DatasetManager


@pytest.fixture(scope="module", autouse=True)
def client():
    """Initialize app engine and set up test client."""
    app.config["TESTING"] = True
    initialize_aml_engine()
    with app.test_client() as client:
        yield client


def test_list_datasets(client):
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.get_json()
    assert "datasets" in data
    assert "active_dataset_id" in data
    assert data["count"] >= 1
    # Check dataset_primary is registered
    dataset_ids = [d["dataset_id"] for d in data["datasets"]]
    assert "dataset_primary" in dataset_ids


def test_get_dataset_profile(client):
    res = client.get("/api/datasets/dataset_primary/profile")
    assert res.status_code == 200
    data = res.get_json()
    assert data["dataset_id"] == "dataset_primary"
    assert "schema_mapping" in data
    assert "profile" in data
    assert len(data["profile"]["preview"]) <= 20


def test_get_dataset_quality(client):
    res = client.get("/api/datasets/dataset_primary/quality")
    assert res.status_code == 200
    data = res.get_json()
    assert "quality_report" in data
    qr = data["quality_report"]
    assert "quality_score" in qr
    assert 0 <= qr["quality_score"] <= 100
    assert len(qr["checks"]) >= 5


def test_get_dataset_features(client):
    res = client.get("/api/datasets/dataset_primary/features")
    assert res.status_code == 200
    data = res.get_json()
    assert "features" in data
    f_info = data["features"]
    assert f_info["total_features"] >= 25
    assert len(f_info["feature_importance"]) >= 5


def test_get_dataset_ml(client):
    res = client.get("/api/datasets/dataset_primary/ml")
    assert res.status_code == 200
    data = res.get_json()
    assert "random_forest" in data
    assert "isolation_forest" in data
    assert "ml_summary" in data


def test_get_dataset_patterns(client):
    res = client.get("/api/datasets/dataset_primary/patterns")
    assert res.status_code == 200
    data = res.get_json()
    assert "patterns" in data
    p = data["patterns"]
    assert "fan_in" in p
    assert "fan_out" in p
    assert "cycle" in p


def test_upload_and_schema_detection(client):
    # Create sample CSV in memory with non-standard column names
    csv_content = """Tx_Time,Sender_Account,Receiver_Account,Source_Bank,Target_Bank,Amount_Paid,Currency,Method
2023-01-01 10:00:00,ACC_001,ACC_002,BANK_A,BANK_B,5000.0,USD,WIRE
2023-01-01 10:05:00,ACC_002,ACC_003,BANK_B,BANK_C,4800.0,USD,WIRE
2023-01-01 10:10:00,ACC_003,ACC_001,BANK_C,BANK_A,4700.0,USD,WIRE
"""
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "sample_laundering.csv")
    }
    res = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    upload_res = res.get_json()
    new_dataset_id = upload_res["dataset_id"]
    assert new_dataset_id.startswith("dataset_")
    assert upload_res["rows_count"] == 3

    # Verify schema auto-detection
    mapping = upload_res["schema_mapping"]
    assert mapping["timestamp"] == "Tx_Time"
    assert mapping["from_account"] == "Sender_Account"
    assert mapping["to_account"] == "Receiver_Account"
    assert mapping["amount_paid"] == "Amount_Paid"

    # Verify quality score
    assert upload_res["quality_score"] > 80

    # Test running analysis pipeline on the new dataset
    analyze_res = client.post(f"/api/datasets/{new_dataset_id}/analyze")
    assert analyze_res.status_code == 200
    analyze_data = analyze_res.get_json()
    assert analyze_data["status"] == "success"

    # Test status endpoint
    status_res = client.get(f"/api/datasets/{new_dataset_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.get_json()
    assert status_data["status"] == "COMPLETED"
    assert status_data["pipeline_progress"] == 100
