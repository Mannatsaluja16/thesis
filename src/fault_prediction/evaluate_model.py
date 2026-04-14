import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.fault_prediction.model import FaultPredictorLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(threshold: float = 0.5):
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultPredictorLSTM()
    model.load_state_dict(torch.load("models/fault_predictor.pt", map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(tensor).squeeze()
        preds  = torch.sigmoid(logits).cpu().numpy()

    y_pred = (preds >= threshold).astype(int)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    logger.info("Accuracy:  %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall:    %.4f", rec)
    logger.info("F1 Score:  %.4f", f1)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    evaluate()


if __name__ == "__main__":
    main()
