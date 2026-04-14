import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from src.config import RANDOM_SEED
from src.fault_prediction.model import FaultPredictorLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_processed_data():
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    y_val   = np.load("data/processed/y_val.npy")
    return X_train, y_train, X_val, y_val


def train(epochs: int = 50, batch_size: int = 512, lr: float = 0.001, patience: int = 7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    X_train, y_train, X_val, y_val = load_processed_data()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=4, pin_memory=pin, persistent_workers=True)

    model = FaultPredictorLSTM().to(device)

    # Class-weighted loss to handle imbalance
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    logger.info("pos_weight=%.2f  (neg=%d pos=%d)", pos_weight.item(), int(n_neg), int(n_pos))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1    = -1.0
    patience_count = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Training pass ───────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_ds)

        # ── Validation pass ─────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        all_preds     = []
        all_labels    = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss   = criterion(logits, yb)
                val_loss += loss.item() * len(Xb)
                probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
                preds  = (probs >= 0.5).astype(int)
                all_preds.extend(preds.tolist() if probs.ndim > 0 else [int(preds)])
                all_labels.extend(yb.squeeze().cpu().numpy().tolist())
        val_loss /= len(val_ds)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        scheduler.step()
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  lr=%.6f",
            epoch, epochs, train_loss, val_loss, val_f1,
            optimizer.param_groups[0]["lr"],
        )

        # ── Early stopping on val F1 ─────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), "models/fault_predictor.pt")
            logger.info("  -> saved best model (val_f1=%.4f)", best_val_f1)
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_val_f1)
                break

    logger.info("Training complete. Best val_f1=%.4f", best_val_f1)


def main():
    if not os.path.exists("data/processed/X_train.npy"):
        logger.info("Processed data not found. Running preprocessor...")
        from src.data_collector.preprocessor import main as preprocess_main
        preprocess_main()
    train()


if __name__ == "__main__":
    main()
