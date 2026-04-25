"""
models/lstm_model.py
=====================
Réseau LSTM PyTorch avec :
  - Entrée multi-features (5 features par timestep)
  - Prédiction directe multi-pas (horizon = 15 sorties simultanées)
  - Fenêtre lookback étendue à 120 pas
  - Early stopping (patience 25), ReduceLROnPlateau (patience 15)
  - Gradient clipping (max_norm 1.0)
  - 150 époques maximum

Amélioration par rapport à main.py :
  - Multi-features : value, return, ma10, ma20, std10
  - Prédiction directe (évite l'accumulation d'erreur du mode autorégressif)
  - Lookback doublé (120 au lieu de 60) pour exploiter les données intraday
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE RÉSEAU
# ─────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        """LSTM multi-couches avec sortie multi-pas directe."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     horizon: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


# ─────────────────────────────────────────────────────────────
# WRAPPER SCIKIT-LIKE
# ─────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Prédicteur LSTM avec feature engineering et prédiction directe multi-pas.

    Paramètres
    ----------
    hidden_size  : int   — dimension état caché (défaut 128)
    num_layers   : int   — nombre de couches LSTM (défaut 2)
    dropout      : float — taux de dropout entre couches (défaut 0.2)
    epochs       : int   — époques maximum (défaut 150)
    lr           : float — taux d'apprentissage Adam (défaut 1e-3)
    batch_size   : int   — taille des mini-lots (défaut 64)
    patience     : int   — patience early stopping (défaut 25)
    lr_patience  : int   — patience ReduceLROnPlateau (défaut 15)
    lr_factor    : float — facteur de réduction LR (défaut 0.5)
    device       : str   — 'cpu' ou 'cuda' (défaut auto-détection)
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 128,
        patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        device: str = None,
    ):
        self.hidden_size = hidden_size
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
        self._scaler = None
        self._horizon = None

    # ── Entraînement ──────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMPredictor":
        """
        Entraîne le réseau LSTM.

        Paramètres
        ----------
        X_train : (n_seq, lookback, n_features)
        y_train : (n_seq, horizon)
        """
        if not TORCH_AVAILABLE:
            return self

        n_seq, lookback, n_features = X_train.shape
        _, horizon = y_train.shape
        self._horizon = horizon

        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model = _LSTMNet(n_features, self.hidden_size, self.num_layers,
                               horizon, self.dropout).to(self.device)
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

    # ── Prédiction ────────────────────────────────────────────

    def predict(self, X_pred: np.ndarray, scaler) -> np.ndarray:
        """
        Prédit l'horizon à partir de la fenêtre X_pred.

        Paramètres
        ----------
        X_pred : (1, lookback, n_features) ou (lookback, n_features)
        scaler : MinMaxScaler ajusté sur la série d'entraînement

        Retourne
        --------
        np.ndarray (horizon,) dans l'espace original (inverse-transformé)
        """
        if not TORCH_AVAILABLE or self._model is None:
            return np.full(self._horizon or 15, np.nan)

        if X_pred.ndim == 2:
            X_pred = X_pred[np.newaxis, ...]

        with torch.no_grad():
            x_t = torch.FloatTensor(X_pred).to(self.device)
            pred_norm = self._model(x_t).cpu().numpy()[0]  # (horizon,)

        # Inverse transform (colonne 'value' = col 0, scaler 1D)
        pred = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        return pred

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_pred: np.ndarray, scaler
    ) -> np.ndarray:
        """Raccourci : fit + predict."""
        self.fit(X_train, y_train)
        return self.predict(X_pred, scaler)
