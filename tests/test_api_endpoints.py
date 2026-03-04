"""
Test suite for HT-HGNN v2.0 FastAPI Backend Endpoints.

Exercises every REST endpoint exposed by ``frontend/backend_server.py``
using the FastAPI ``TestClient`` (no live server needed):

- GET  /health
- GET  /model/info
- POST /predict
- POST /analyze
- POST /upload/predict
- GET  /training/history
- GET  /datasets
- POST /datasets/{dataset_id}/load
- POST /explain
- POST /simulate/cascade  (async task, polled via GET /tasks/{id})

All tests use small synthetic payloads and run against the app's built-in
mock / fallback logic so no GPU, real model weights, or real datasets are
required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "frontend"))

import pytest
import io
import time

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from backend_server import app
    HAS_APP = True
except ImportError:
    HAS_APP = False
    TestClient = None
    app = None

pytestmark = pytest.mark.skipif(not HAS_APP, reason="FastAPI app or TestClient not importable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient so startup events fire once."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def small_features():
    """Return a small 3-node feature matrix for prediction requests."""
    return [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]]


# ---------------------------------------------------------------------------
# Tests -- v1.0 endpoints
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health should return 200 with status 'healthy'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "timestamp" in body


def test_model_info_endpoint(client):
    """GET /model/info should return model metadata (may be mock)."""
    resp = client.get("/model/info")
    # 200 if MODEL_INFO was populated during startup, 503 otherwise
    if resp.status_code == 200:
        body = resp.json()
        assert "name" in body
        assert "version" in body
        assert "parameters" in body
        assert "metrics" in body
    else:
        assert resp.status_code == 503


def test_predict_endpoint(client, small_features):
    """POST /predict should return per-node predictions."""
    resp = client.post("/predict", json={"nodeFeatures": small_features})
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["pricePredictions"]) == 3
    assert len(body["changePredictions"]) == 3
    assert len(body["criticalityScores"]) == 3
    assert len(body["nodeIds"]) == 3


def test_analyze_endpoint(client, small_features):
    """POST /analyze should wrap predictions with an averageRisk score."""
    resp = client.post("/analyze", json={"nodeFeatures": small_features})
    assert resp.status_code == 200
    body = resp.json()

    assert "timestamp" in body
    assert "predictions" in body
    assert "averageRisk" in body
    avg_risk = body["averageRisk"]
    assert isinstance(avg_risk, float)


def test_upload_predict_endpoint(client):
    """POST /upload/predict should accept a CSV and return predictions."""
    csv_content = (
        "f1,f2,f3,f4,f5,f6,f7,f8\n"
        "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0\n"
        "9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0\n"
    )
    file_bytes = csv_content.encode("utf-8")
    resp = client.post(
        "/upload/predict",
        files={"file": ("test.csv", io.BytesIO(file_bytes), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["pricePredictions"]) == 2
    assert len(body["nodeIds"]) == 2


def test_training_history_endpoint(client):
    """GET /training/history should return epoch-loss data (real or mock)."""
    resp = client.get("/training/history")
    assert resp.status_code == 200
    body = resp.json()

    assert "epochs" in body
    assert "total_loss" in body
    assert len(body["epochs"]) > 0


# ---------------------------------------------------------------------------
# Tests -- v2.0 endpoints
# ---------------------------------------------------------------------------

def test_datasets_list_endpoint(client):
    """GET /datasets should return the full dataset catalog."""
    resp = client.get("/datasets")
    assert resp.status_code == 200
    datasets = resp.json()

    assert isinstance(datasets, list)
    assert len(datasets) >= 5, "Should expose at least 5 dataset entries"

    ids = {d["id"] for d in datasets}
    assert "dataco" in ids
    assert "bom" in ids
    assert "ports" in ids
    assert "maintenance" in ids
    assert "retail" in ids


def test_dataset_load_endpoint(client):
    """POST /datasets/{id}/load should return a graph summary."""
    resp = client.post(
        "/datasets/dataco/load",
        json={
            "temporal_window": 30,
            "min_hyperedge_size": 3,
            "dynamic_edges": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "loaded"
    assert body["dataset_id"] == "dataco"
    gs = body["graph_summary"]
    assert gs["nodes"] > 0
    assert gs["hyperedges"] > 0
    assert gs["avg_hyperedge_size"] > 0


def test_explain_endpoint(client):
    """POST /explain should return HyperSHAP-style explanations (real or mock)."""
    resp = client.post(
        "/explain",
        json={
            "node_ids": ["node_0", "node_1"],
            "prediction_type": "criticality",
            "max_hyperedges": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert "explanations" in body
    assert len(body["explanations"]) == 2

    first = body["explanations"][0]
    assert first["node_id"] == "node_0"
    assert first["prediction_type"] == "criticality"
    assert "top_hyperedge_attributions" in first
    assert "top_feature_attributions" in first
    assert "recommendations" in first
    assert len(first["recommendations"]) > 0


def test_cascade_simulate_endpoint(client):
    """POST /simulate/cascade should accept and return a task_id (async)."""
    resp = client.post(
        "/simulate/cascade",
        json={
            "shock_nodes": ["node_0", "node_1"],
            "shock_magnitude": 0.8,
            "propagation_steps": 20,
            "cascade_threshold": 0.5,
        },
    )
    assert resp.status_code == 202
    body = resp.json()

    assert "task_id" in body
    assert body["status"] == "PENDING"
    assert "poll_url" in body

    # Poll for completion (give it a few seconds for the async mock)
    task_id = body["task_id"]
    poll_url = f"/tasks/{task_id}"

    for _ in range(20):
        poll_resp = client.get(poll_url)
        assert poll_resp.status_code == 200
        poll_body = poll_resp.json()

        if poll_body["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    assert poll_body["status"] == "COMPLETED", (
        f"Expected task COMPLETED, got {poll_body['status']}"
    )
    assert poll_body["result"] is not None
    result = poll_body["result"]
    assert "total_disrupted" in result
    assert "timeline" in result
