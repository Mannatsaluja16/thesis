import sys, os
# Must be set BEFORE any src imports so api_gateway skips start_monitoring()
os.environ["DISABLE_MONITORING"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


def pytest_configure(config):
    """Create a minimal trained model if one doesn't exist yet (needed for CI)."""
    model_path = "models/fault_predictor.pt"
    if not os.path.exists(model_path):
        try:
            import torch
            from src.fault_prediction.model import FaultPredictorLSTM

            os.makedirs("models", exist_ok=True)
            model = FaultPredictorLSTM()
            X = torch.rand(16, 10, 5)
            y = torch.randint(0, 2, (16, 1)).float()
            loss = torch.nn.BCEWithLogitsLoss()(model(X), y)
            loss.backward()
            torch.save(model.state_dict(), model_path)
        except Exception:
            pass  # torch not available; tests will skip themselves


@pytest.fixture(scope="session", autouse=True)
def _cleanup_torch_model():
    """Null-out cached PyTorch model at session end to avoid SIGABRT on GC."""
    yield
    try:
        import src.fault_prediction.predict as pp
        pp._model = None
        import gc
        gc.collect()
    except Exception:
        pass


@pytest.fixture(scope="session")
def flask_client():
    from src.cloud_controller.api_gateway import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
