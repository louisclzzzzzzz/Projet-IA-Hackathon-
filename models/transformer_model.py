"""
models/transformer_model.py
==============================
Modèle Transformer avec :
  - Encodage positionnel sinusoïdal (Vaswani et al., 2017)
  - 4 têtes d'attention, 2 couches encoder
  - Entrée multi-features (5 features par timestep)
  - Prédiction directe multi-pas (horizon sorties)

L'encodage positionnel sinusoïdal est adapté aux séquences intraday :
il permet au modèle de distinguer les positions relatives sans apprentissage
supplémentaire.

Repris et amélioré depuis main.py (Louis Cluzel) — architecture identique
mais avec entrée multifeature et prédiction multi-pas directe.
"""

import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class _PositionalEncoding(nn.Module):
        """
        Encodage positionnel sinusoïdal (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Adapté des séquences intraday (max_len étendu à 512).
        Repris de main.py avec correction de la dimension des indices.
        """

        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            # Gérer d_model impair
            n_cos = d_model // 2
            pe[:, 1::2] = torch.cos(position * div_term[:n_cos])
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]

    class _TransformerNet(nn.Module):
        """Transformer encoder-only avec projection d'entrée et sortie multi-pas."""

        def __init__(self, input_size: int, d_model: int, nhead: int,
                     num_layers: int, horizon: int, dropout: float):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc = _PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, horizon)

        def forward(self, x):  # x: (batch, seq, input_size)
            x = self.input_proj(x)
            x = self.pos_enc(x)
            out = self.transformer(x)
            return self.fc(out[:, -1, :])  # (batch, horizon)


# ─────────────────────────────────────────────────────────────
# WRAPPER
# ─────────────────────────────────────────────────────────────

class TransformerPredictor:
    """
    Prédicteur Transformer avec feature engineering et prédiction directe.

    Paramètres
    ----------
    d_model     : int   — dimension du modèle (défaut 64)
    nhead       : int   — nombre de têtes d'attention (défaut 4)
    num_layers  : int   — nombre de couches encoder (défaut 2)
    dropout     : float — taux de dropout (défaut 0.1)
    epochs      : int   — époques maximum (défaut 150)
    lr          : float — taux d'apprentissage Adam (défaut 1e-3)
    batch_size  : int   — taille mini-lots (défaut 64)
    patience    : int   — patience early stopping (défaut 25)
    lr_patience : int   — patience ReduceLROnPlateau (défaut 15)
    lr_factor   : float — facteur de réduction LR (défaut 0.5)
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 128,
        patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        device: str = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        if device is None:
            self.device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else None

        self._model = None
        self._horizon = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TransformerPredictor":
        if not TORCH_AVAILABLE:
            return self

        n_seq, lookback, n_features = X_train.shape
        _, horizon = y_train.shape
        self._horizon = horizon

        # d_model doit être divisible par nhead
        d_model = self.d_model
        if d_model % self.nhead != 0:
            d_model = (d_model // self.nhead + 1) * self.nhead

        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model = _TransformerNet(
            n_features, d_model, self.nhead,
            self.num_layers, horizon, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.lr_patience, factor=self.lr_factor
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred = self._model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self._model.eval()
        return self

    def predict(self, X_pred: np.ndarray, scaler) -> np.ndarray:
        if not TORCH_AVAILABLE or self._model is None:
            return np.full(self._horizon or 15, np.nan)

        if X_pred.ndim == 2:
            X_pred = X_pred[np.newaxis, ...]

        with torch.no_grad():
            x_t = torch.FloatTensor(X_pred).to(self.device)
            pred_norm = self._model(x_t).cpu().numpy()[0]

        pred = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        return pred

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_pred: np.ndarray, scaler
    ) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict(X_pred, scaler)
