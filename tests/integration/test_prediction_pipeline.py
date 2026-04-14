import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from src.data_collector.real_data_loader import load_single_vm
from src.data_collector.preprocessor import normalize_features, create_sequences
from src.config import WINDOW_SIZE, INPUT_SIZE


def test_raw_data_to_sequences_shape():
    # Use first VM file from dataset as a small real-data sample
    vm_file = "dataset/15.csv"
    if not __import__('os').path.exists(vm_file):
        pytest.skip("dataset/15.csv not found")
    df = load_single_vm(vm_file)
    df = normalize_features(df)
    X, y = create_sequences(df, window_size=WINDOW_SIZE)
    assert X.shape[1] == WINDOW_SIZE
    assert X.shape[2] == INPUT_SIZE


def test_prediction_returns_correct_keys():
    pytest.importorskip("torch")
    if not os.path.exists("models/fault_predictor.pt"):
        pytest.skip("Model not trained yet.")
    from src.fault_prediction.predict import predict_fault
    window = np.random.rand(WINDOW_SIZE, INPUT_SIZE).astype(np.float32)
    result = predict_fault(window, instance_id="test-inst")
    assert "fault_predicted" in result
    assert "confidence" in result
    assert "instance_id" in result


def test_high_cpu_input_predicts_fault():
    pytest.importorskip("torch")
    if not os.path.exists("models/fault_predictor.pt"):
        pytest.skip("Model not trained yet.")
    from src.fault_prediction.predict import predict_fault
    # All high CPU, network
    window = np.ones((WINDOW_SIZE, INPUT_SIZE), dtype=np.float32)
    result = predict_fault(window, instance_id="high-cpu")
    assert isinstance(result["confidence"], float)


def test_normal_input_predicts_no_fault():
    pytest.importorskip("torch")
    if not os.path.exists("models/fault_predictor.pt"):
        pytest.skip("Model not trained yet.")
    from src.fault_prediction.predict import predict_fault
    window = np.zeros((WINDOW_SIZE, INPUT_SIZE), dtype=np.float32)
    result = predict_fault(window, instance_id="low-load")
    assert result["confidence"] < 0.9
