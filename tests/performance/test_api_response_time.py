import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import time
import pytest
from src.cloud_controller.api_gateway import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _elapsed_ms(client, method, path, **kwargs):
    start = time.perf_counter()
    resp = getattr(client, method)(path, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return resp, elapsed


def test_submit_task_responds_under_200ms(client):
    _, ms = _elapsed_ms(
        client, "post", "/task/submit",
        data=json.dumps({"priority": "normal"}),
        content_type="application/json",
    )
    assert ms < 200, f"Response took {ms:.1f}ms (> 200ms)"


def test_server_status_responds_under_100ms(client):
    _, ms = _elapsed_ms(client, "get", "/servers/status")
    assert ms < 100, f"Response took {ms:.1f}ms (> 100ms)"


def test_prediction_endpoint_responds_under_500ms(client):
    _, ms = _elapsed_ms(client, "get", "/predictions/latest")
    assert ms < 500, f"Response took {ms:.1f}ms (> 500ms)"


def test_metrics_summary_responds_under_150ms(client):
    _, ms = _elapsed_ms(client, "get", "/metrics/summary")
    assert ms < 150, f"Response took {ms:.1f}ms (> 150ms)"
