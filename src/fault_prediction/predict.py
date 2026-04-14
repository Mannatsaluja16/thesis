import logging
import numpy as np
import torch
from src.config import FAULT_THRESHOLD
from src.fault_prediction.model import FaultPredictorLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = FaultPredictorLSTM()
        _model.load_state_dict(
            torch.load("models/fault_predictor.pt", map_location="cpu")
        )
        _model.eval()
        logger.info("Fault predictor model loaded.")
    return _model


def predict_fault(metrics_window: np.ndarray, threshold: float = FAULT_THRESHOLD,
                  instance_id: str = "") -> dict:
    """
    Predict fault from a (10, 5) metrics window.
    Returns dict with fault_predicted, confidence, instance_id.
    """
    model = _load_model()
    tensor = torch.tensor(metrics_window, dtype=torch.float32).unsqueeze(0)  # (1, 10, 5)
    with torch.no_grad():
        logit = model(tensor).item()
        confidence = float(torch.sigmoid(torch.tensor(logit)).item())
    return {
        "fault_predicted": confidence >= threshold,
        "confidence": confidence,
        "instance_id": instance_id,
    }


def main():
    # Demo with random input
    window = np.random.rand(10, 5).astype(np.float32)
    result = predict_fault(window, instance_id="i-demo")
    logger.info("Prediction result: %s", result)


if __name__ == "__main__":
    main()
