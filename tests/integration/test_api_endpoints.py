import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import pytest
from src.cloud_controller.api_gateway import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_submit_task_returns_201(client):
    resp = client.post("/task/submit",
                       data=json.dumps({"priority": "normal"}),
                       content_type="application/json")
    assert resp.status_code == 201


def test_submit_task_missing_fields_returns_400(client):
    resp = client.post("/task/submit",
                       data=json.dumps({}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_submit_task_bad_priority_returns_400(client):
    resp = client.post("/task/submit",
                       data=json.dumps({"priority": "ultra"}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_get_server_status_returns_200(client):
    resp = client.get("/servers/status")
    assert resp.status_code == 200


def test_get_server_status_response_schema(client):
    resp = client.get("/servers/status")
    data = json.loads(resp.data)
    assert "servers" in data
    assert isinstance(data["servers"], list)


def test_get_latest_predictions_returns_200(client):
    resp = client.get("/predictions/latest")
    assert resp.status_code == 200


def test_get_latest_predictions_returns_list(client):
    resp = client.get("/predictions/latest")
    data = json.loads(resp.data)
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_get_metrics_summary_returns_200(client):
    resp = client.get("/metrics/summary")
    assert resp.status_code == 200


def test_get_metrics_summary_contains_all_7_metrics(client):
    resp = client.get("/metrics/summary")
    data = json.loads(resp.data)
    for key in [
        "energy_consumption_wh",
        "execution_time_s",
        "latency_ms",
        "throughput_tasks_per_s",
        "reliability_pct",
        "cost_usd",
        "recovery_time_s",
    ]:
        assert key in data, f"Missing key: {key}"


def test_invalid_route_returns_404(client):
    resp = client.get("/does_not_exist")
    assert resp.status_code == 404


def test_method_not_allowed_returns_405(client):
    resp = client.delete("/servers/status")
    assert resp.status_code == 405
