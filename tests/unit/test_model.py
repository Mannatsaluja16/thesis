import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.fault_prediction.model import FaultPredictorLSTM


@pytest.fixture
def model():
    return FaultPredictorLSTM()


def test_model_instantiates(model):
    assert model is not None


def test_forward_output_shape(model):
    x = torch.rand(4, 10, 5)  # (batch=4, window=10, features=5)
    out = model(x)
    assert out.shape == (4, 1)


def test_output_range(model):
    # Model now returns logits (unbounded); just check shape and dtype
    x = torch.rand(8, 10, 5)
    out = model(x)
    assert out.shape == (8, 1)
    assert out.dtype == torch.float32


def test_model_trains_one_step(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = torch.rand(4, 10, 5)
    y = torch.rand(4, 1)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0
