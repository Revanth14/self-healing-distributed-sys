"""
ml/lstm_autoencoder.py

LSTM Autoencoder for sequence anomaly detection.
Learns the normal temporal pattern of metrics over a sliding window.
Anomalies are detected when reconstruction error is high —
the model "can't explain" the sequence it's seeing.

Why LSTM over Isolation Forest alone:
- IF catches point anomalies (one bad reading)
- LSTM catches sequence anomalies (gradual drift, phase shifts)
- Together they catch the cascading failure pattern:
  CPU rises slowly → latency follows → errors spike
  Each step alone looks normal; the sequence is the signal.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lstm_autoencoder.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ──────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Encoder compresses a sequence of feature vectors into a latent vector.
    Decoder reconstructs the original sequence from the latent vector.
    High reconstruction error = the sequence doesn't look like training data.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (hidden, cell) = self.encoder(x)

        # Repeat latent vector across seq_len for decoder input
        seq_len = x.shape[1]
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)


# ── Detector ───────────────────────────────────────────────────────────────

class LSTMDetector:
    """
    Wraps LSTMAutoencoder with training, scoring, and persistence.
    Maintains a rolling window of recent feature vectors per node
    to form sequences for inference.
    """

    def __init__(
        self,
        seq_len: int    = 15,   # 15 snapshots = 30s of history
        hidden_size: int = 64,
        num_layers: int  = 2,
        epochs: int      = 30,
        lr: float        = 1e-3,
        batch_size: int  = 32,
    ):
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size

        self._model: LSTMAutoencoder = None
        self._threshold: float = 0.0
        self._mean: np.ndarray = None
        self._std:  np.ndarray = None
        self._trained = False

        # Per-node rolling window for inference
        self._windows: dict[str, list] = {}

    # ── training ───────────────────────────────────────────────────────────

    def fit(self, feature_vectors: list) -> "LSTMDetector":
        """
        Train on healthy baseline feature vectors.
        Builds overlapping sequences of length seq_len.
        """
        if len(feature_vectors) < self.seq_len + 10:
            raise ValueError(
                f"Need at least {self.seq_len + 10} samples, got {len(feature_vectors)}"
            )

        X = np.array([fv.to_list() for fv in feature_vectors], dtype=np.float32)

        # Normalise
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0) + 1e-8
        X = (X - self._mean) / self._std

        # Build overlapping sequences
        sequences = np.array([
            X[i: i + self.seq_len]
            for i in range(len(X) - self.seq_len)
        ])

        input_size = X.shape[1]
        self._model = LSTMAutoencoder(input_size, self.hidden_size, self.num_layers).to(DEVICE)

        dataset    = TensorDataset(torch.tensor(sequences))
        loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer  = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion  = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                reconstructed = self._model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                log.info(f"LSTM epoch {epoch+1}/{self.epochs} loss={total_loss/len(loader):.5f}")

        # Set threshold = mean + 3*std of training reconstruction errors
        errors = self._reconstruction_errors(sequences)
        self._threshold = float(errors.mean() + 3 * errors.std())
        self._trained = True

        log.info(
            f"LSTM trained on {len(sequences)} sequences "
            f"(threshold={self._threshold:.5f}, device={DEVICE})"
        )
        return self

    # ── inference ──────────────────────────────────────────────────────────

    def score(self, feature_vector) -> float:
        """
        Returns anomaly score in [0, 1] for a single feature vector.
        Maintains a per-node rolling window internally.
        Score is 0 until the window is full.
        """
        if not self._trained:
            return 0.0

        node_id = feature_vector.node_id
        if node_id not in self._windows:
            self._windows[node_id] = []

        self._windows[node_id].append(feature_vector.to_list())
        if len(self._windows[node_id]) > self.seq_len:
            self._windows[node_id].pop(0)

        if len(self._windows[node_id]) < self.seq_len:
            return 0.0

        seq = np.array(self._windows[node_id], dtype=np.float32)
        seq = (seq - self._mean) / self._std
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        self._model.eval()
        with torch.no_grad():
            recon = self._model(seq_t)
            error = float(nn.MSELoss()(recon, seq_t).item())

        # Normalise against threshold
        score = min(error / (self._threshold + 1e-8), 2.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def _reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        self._model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = torch.tensor(sequences[i: i + self.batch_size]).to(DEVICE)
                recon = self._model(batch)
                mse = nn.MSELoss(reduction="none")(recon, batch)
                errors.extend(mse.mean(dim=(1, 2)).cpu().numpy())
        return np.array(errors)

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state":  self._model.state_dict(),
            "threshold":    float(self._threshold),
            "mean":         self._mean.tolist(),
            "std":          self._std.tolist(),
            "seq_len":      self.seq_len,
            "hidden_size":  self.hidden_size,
            "num_layers":   self.num_layers,
            "input_size":   self._model.input_size,
            }, path)
        log.info(f"LSTM saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "LSTMDetector":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        detector = cls(
            seq_len=checkpoint["seq_len"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        )
        detector._model = LSTMAutoencoder(
            checkpoint["input_size"],
            checkpoint["hidden_size"],
            checkpoint["num_layers"],
        ).to(DEVICE)
        detector._model.load_state_dict(checkpoint["model_state"])
        detector._threshold = checkpoint["threshold"]
        detector._mean      = np.array(checkpoint["mean"])
        detector._std       = np.array(checkpoint["std"])
        detector._trained   = True
        log.info(f"LSTM loaded from {path}")
        return detector

    @property
    def is_trained(self) -> bool:
        return self._trained