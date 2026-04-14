import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import pytest

from src.data_collector.preprocessor import normalize_features, create_sequences, FEATURE_COLS


def _sample_df(n=200):
    data = {
        "CPUUtilization":  np.random.uniform(0, 100, n),
        "NetworkIn":       np.random.uniform(0, 1e7, n),
        "NetworkOut":      np.random.uniform(0, 1e7, n),
        "DiskReadOps":     np.random.uniform(0, 200, n),
        "DiskWriteOps":    np.random.uniform(0, 200, n),
        "StatusCheckFailed": np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


def test_normalize_bounds():
    df = _sample_df()
    norm = normalize_features(df)
    for col in FEATURE_COLS:
        assert norm[col].min() >= -1e-9
        assert norm[col].max() <= 1.0 + 1e-9


def test_sequence_shape():
    df = normalize_features(_sample_df(200))
    X, y = create_sequences(df, window_size=10)
    assert X.ndim == 3
    assert X.shape[1] == 10
    assert X.shape[2] == len(FEATURE_COLS)


def test_labels_binary():
    df = normalize_features(_sample_df(200))
    X, y = create_sequences(df)
    assert set(y.astype(int)).issubset({0, 1})
