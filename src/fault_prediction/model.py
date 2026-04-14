import torch
import torch.nn as nn
from src.config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS


class FaultPredictorLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.3)
        self.fc1     = nn.Linear(hidden_size, 64)
        self.fc2     = nn.Linear(64, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, window_size, input_size)
        out, _ = self.lstm(x)                            # (batch, window_size, hidden_size)
        out    = out[:, -1, :]                           # last timestep
        out    = self.dropout(self.relu(self.fc1(out)))  # (batch, 64)
        return self.fc2(out)                             # (batch, 1) — raw logits
