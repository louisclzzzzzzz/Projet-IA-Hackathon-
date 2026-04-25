"""
models/ensemble_model.py
=========================
Ensemble des modèles Deep Learning : moyenne pondérée de
LSTM + GRU + Transformer.

Les poids sont appris par validation : on minimise le MSE
sur les séquences de validation (derniers K% du train) en
cherchant les poids optimaux par simplex de Nelder-Mead.

Si l'optimisation échoue, on utilise des poids égaux (1/3, 1/3, 1/3).
"""

import numpy as np
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


class EnsemblePredictor:
    """
    Ensemble pondéré de LSTM, GRU et Transformer.

    Les prédictions des trois modèles sont combinées :
        y_ensemble = w1 * y_lstm + w2 * y_gru + w3 * y_transformer
    avec w1 + w2 + w3 = 1, wi >= 0.

    Les poids sont optimisés sur la validation (optionnel).
    Sans données de validation, poids égaux.

    Paramètres
    ----------
    weights : list/array de 3 poids (optionnel) — si None, poids égaux
    """

    def __init__(self, weights=None):
        self.weights = weights  # (w_lstm, w_gru, w_transformer)
        self._fitted_weights = None

    def fit_weights(
        self,
        pred_lstm: np.ndarray,
        pred_gru: np.ndarray,
        pred_trans: np.ndarray,
        y_val: np.ndarray,
    ) -> "EnsemblePredictor":
        """
        Apprend les poids optimaux sur des données de validation.

        Paramètres
        ----------
        pred_lstm/gru/trans : (n_val, horizon) — prédictions sur validation
        y_val               : (n_val, horizon) — vraies valeurs sur validation

        Retourne
        --------
        self (chainable)
        """
        if self.weights is not None:
            self._fitted_weights = np.array(self.weights)
            return self

        predictions = np.stack([pred_lstm, pred_gru, pred_trans], axis=0)  # (3, n_val, horizon)
        y_val_flat = y_val.flatten()

        def loss(w):
            w = np.abs(w)
            w = w / (w.sum() + 1e-12)
            ensemble = (w[0] * predictions[0] + w[1] * predictions[1] + w[2] * predictions[2])
            return np.mean((ensemble.flatten() - y_val_flat) ** 2)

        # Optimisation des poids : contrainte simplex (somme = 1, >= 0)
        result = minimize(
            loss,
            x0=[1/3, 1/3, 1/3],
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
        )

        if result.success or result.fun < loss([1/3, 1/3, 1/3]):
            raw = np.abs(result.x)
            self._fitted_weights = raw / (raw.sum() + 1e-12)
        else:
            self._fitted_weights = np.array([1/3, 1/3, 1/3])

        return self

    def predict(
        self,
        pred_lstm: np.ndarray,
        pred_gru: np.ndarray,
        pred_trans: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule la prédiction d'ensemble.

        Paramètres
        ----------
        pred_lstm/gru/trans : (horizon,) — prédictions de chaque modèle

        Retourne
        --------
        np.ndarray (horizon,)
        """
        # Gérer les NaN (modèle non entraîné = fallback sur les autres)
        preds = []
        weights = []
        base_w = self._fitted_weights if self._fitted_weights is not None else [1/3, 1/3, 1/3]

        for p, w in zip([pred_lstm, pred_gru, pred_trans], base_w):
            if p is not None and not np.any(np.isnan(p)):
                preds.append(p)
                weights.append(w)

        if not preds:
            return np.zeros(15)

        weights = np.array(weights)
        weights = weights / weights.sum()

        return sum(w * p for w, p in zip(weights, preds))
