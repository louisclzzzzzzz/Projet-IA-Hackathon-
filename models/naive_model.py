"""
models/naive_model.py
======================
Baseline : Random Walk with Drift (RWD).

Prédit les prochaines valeurs en prolongeant la tendance linéaire
calculée sur les données d'entraînement (drift = pente moyenne).

Sert de plancher de comparaison : si un modèle fait moins bien que
RWD, il est inutile. Inspiré de projet_ia/app1/pipeline/ (Charbonnier & Claveau).

Prédiction :
    drift = (train[-1] - train[0]) / (len(train) - 1)
    y_h   = train[-1] + h * drift    pour h = 1, 2, ..., horizon
"""

import numpy as np


class NaivePredictor:
    """
    Random Walk with Drift.

    Baseline minimaliste : extrapolation linéaire de la tendance passée.
    Aucun paramètre à ajuster. Temps d'entraînement instantané.
    """

    def __init__(self):
        self._last = None
        self._drift = None

    def fit(self, train: np.ndarray) -> "NaivePredictor":
        """Calcule le drift (variation moyenne par pas) sur les données train."""
        train = np.asarray(train, dtype=float)
        n = len(train)
        if n < 2:
            self._drift = 0.0
        else:
            self._drift = (train[-1] - train[0]) / (n - 1)
        self._last = float(train[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Génère `horizon` prédictions par extrapolation linéaire."""
        if self._last is None:
            raise RuntimeError("Appeler fit() avant predict()")
        return np.array(
            [self._last + (h + 1) * self._drift for h in range(horizon)],
            dtype=float,
        )

    def fit_predict(self, train: np.ndarray, horizon: int) -> np.ndarray:
        """Raccourci : fit + predict en une seule appel."""
        return self.fit(train).predict(horizon)
