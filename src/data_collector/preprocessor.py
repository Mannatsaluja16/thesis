import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import RANDOM_SEED, WINDOW_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = ["CPUUtilization", "NetworkIn", "NetworkOut", "DiskReadOps", "DiskWriteOps"]
LABEL_COL    = "StatusCheckFailed"


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature columns to [0, 1] using MinMaxScaler."""
    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().fillna(0)
    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df


def create_sequences(df: pd.DataFrame, window_size: int = WINDOW_SIZE):
    """
    Create sliding-window sequences.
    Label = 1 if a fault occurs within the next 2 timesteps.
    Returns X: (n, window_size, n_features), y: (n,)
    """
    df = df.reset_index(drop=True)
    features = df[FEATURE_COLS].values
    labels   = df[LABEL_COL].values

    X, y = [], []
    lookahead = 4   # "fault within next 4 minutes" — wider window = more positive sequences
    for i in range(len(df) - window_size - lookahead):
        X.append(features[i : i + window_size])
        future_label = int(labels[i + window_size : i + window_size + lookahead].any())
        y.append(future_label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_dataset(X, y, train: float = 0.70, val: float = 0.15, shuffle: bool = True):
    """
    Split arrays into train / validation / test.
    Shuffles by default so test patterns differ from training.
    """
    n = len(X)
    rng = np.random.default_rng(RANDOM_SEED)

    if shuffle:
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)

    train_end = int(n * train)
    val_end   = int(n * (train + val))

    ti = idx[:train_end]
    vi = idx[train_end:val_end]
    sti = idx[val_end:]

    return (
        X[ti], y[ti],
        X[vi], y[vi],
        X[sti], y[sti],
    )


def create_sequences_per_vm(df: pd.DataFrame, window_size: int = WINDOW_SIZE):
    """
    Create sequences separately for each VM so no window crosses VM boundaries.
    Normalise each VM independently (its own scaler) to prevent data leakage.
    Returns X: (n, window_size, n_features), y: (n,)
    """
    all_X, all_y = [], []
    lookahead = 2
    for vm_id, vm_df in df.groupby("instance_id"):
        vm_df = vm_df.sort_values("timestamp").reset_index(drop=True)
        # Normalise this VM's features independently
        vm_df[FEATURE_COLS] = vm_df[FEATURE_COLS].ffill().fillna(0)
        scaler = MinMaxScaler()
        vm_df[FEATURE_COLS] = scaler.fit_transform(vm_df[FEATURE_COLS])
        features = vm_df[FEATURE_COLS].values
        labels   = vm_df[LABEL_COL].values
        for i in range(len(vm_df) - window_size - lookahead):
            all_X.append(features[i : i + window_size])
            future_label = int(labels[i + window_size : i + window_size + lookahead].any())
            all_y.append(future_label)
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


def stratified_sample(X, y, n_total: int = 20000, seed: int = RANDOM_SEED):
    """Sample n_total sequences keeping the original class ratio."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    ratio   = len(pos_idx) / len(y)
    n_pos   = min(int(n_total * ratio), len(pos_idx))
    n_neg   = min(n_total - n_pos, len(neg_idx))
    chosen  = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False),
        rng.choice(neg_idx, n_neg, replace=False),
    ])
    chosen = rng.permutation(chosen)
    return X[chosen], y[chosen]


def main():
    src_path = "data/raw/real_workload.csv"
    if not os.path.exists(src_path):
        logger.info("Real workload data not found — loading from dataset/ folder...")
        from src.data_collector.real_data_loader import main as loader_main
        loader_main()
    logger.info("Using real workload dataset: %s", src_path)
    df = pd.read_csv(src_path)

    # Build sequences per VM (prevents cross-VM window contamination)
    X, y = create_sequences_per_vm(df)
    logger.info("Sequences: X=%s  |  fault=%.1f%%", X.shape, y.mean() * 100)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, shuffle=True)
    logger.info(
        "Train: %d (fault %.1f%%)  Val: %d (fault %.1f%%)  Test: %d (fault %.1f%%)",
        len(X_train), y_train.mean()*100,
        len(X_val),   y_val.mean()*100,
        len(X_test),  y_test.mean()*100,
    )

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_val.npy",   X_val)
    np.save("data/processed/y_val.npy",   y_val)
    np.save("data/processed/X_test.npy",  X_test)
    np.save("data/processed/y_test.npy",  y_test)
    logger.info("Saved processed data to data/processed/")


if __name__ == "__main__":
    main()
