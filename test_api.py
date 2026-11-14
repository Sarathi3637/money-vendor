"""
Comprehensive Unit Tests for AML Investigation & Risk REST API (Phase 3).
Covers health check, investigation cases listing and filtering, case detail retrieval,
timeline inspection, network subgraph extraction, account graph neighborhood lookup,
and ad-hoc transaction analysis.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app, initialize_aml_engine


@pytest.fixture(scope="module", autouse=True)
def client():
    """Initialize app engine and set up test client."""
    app.config["TESTING"] = True
    # Initialize engine with a small sample or full cleaned data if available
    initialize_aml_engine()
    with app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "investigation_cases_count" in data


def test_get_investigations(client):
    response = client.get("/api/investigations")
    assert response.status_code == 200
    data = response.get_json()
    assert "cases" in data
    assert "total" in data

    # Test filtering parameters
    response_filtered = client.get("/api/investigations?limit=5&min_score=0.0")
    assert response_filtered.status_code == 200
    data_filtered = response_filtered.get_json()
    assert len(data_filtered["cases"]) <= 5


def test_investigation_detail_and_timeline(client):
    # First get list of cases
    response = client.get("/api/investigations?limit=1")
    assert response.status_code == 200
    cases = response.get_json()["cases"]

    if len(cases) > 0:
        case_id = cases[0]["case_id"]

        # Test case detail
        res_detail = client.get(f"/api/investigations/{case_id}")
        assert res_detail.status_code == 200
        detail_data = res_detail.get_json()
        assert detail_data["case_id"] == case_id
        assert "evidence" in detail_data
        assert "risk_score" in detail_data

        # Test case timeline
        res_timeline = client.get(f"/api/investigations/{case_id}/timeline")
        assert res_timeline.status_code == 200
        timeline_data = res_timeline.get_json()
        assert timeline_data["case_id"] == case_id
        assert "timeline" in timeline_data

        # Test case network
        res_network = client.get(f"/api/investigations/{case_id}/network")
        assert res_network.status_code == 200
        network_data = res_network.get_json()
        assert "nodes" in network_data
        assert "edges" in network_data
    else:
        pytest.skip("No investigation cases generated to test detail endpoints.")


def test_investigation_not_found(client):
    res = client.get("/api/investigations/AML-9999")
    assert res.status_code == 404


def test_account_graph(client):
    # Get an account from cases if available
    response = client.get("/api/investigations?limit=1")
    cases = response.get_json()["cases"]
    if len(cases) > 0:
        case_id = cases[0]["case_id"]
        res_detail = client.get(f"/api/investigations/{case_id}")
        accounts = res_detail.get_json().get("accounts", [])
        if accounts:
            acc_id = accounts[0]
            res_acc = client.get(f"/api/graph/account/{acc_id}?radius=1")
            assert res_acc.status_code == 200
            acc_data = res_acc.get_json()
            assert acc_data["account_id"] == acc_id
            assert "nodes" in acc_data
            assert "edges" in acc_data
    else:
        pytest.skip("No accounts available in investigation cases.")


def test_analyze_endpoint(client):
    payload = {
        "transactions": [
            {
                "timestamp": "2022-09-01 12:00:00",
                "from_bank": "1",
                "from_account": "ACC_TEST_1",
                "to_bank": "2",
                "to_account": "ACC_TEST_2",
                "amount_paid": 5000.0,
                "amount_received": 5000.0,
                "payment_currency": "US Dollar",
                "receiving_currency": "US Dollar",
                "payment_format": "Wire",
                "is_laundering": 0
            }
        ]
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["analyzed_transactions_count"] == 1
    assert "results" in data
