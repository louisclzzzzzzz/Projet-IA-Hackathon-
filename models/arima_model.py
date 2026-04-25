"""
models/arima_model.py
======================
Modèle ARIMA avec stratégie rolling forecast.

À chaque pas de l'horizon, le modèle est ré-ajusté sur l'historique
augmenté de la prédiction précédente. Cette approche donne de meilleurs
résultats qu'un forecast multi-pas unique pour les séries courtes.

Ordre ARIMA(5,1,2) par défaut — testé sur indicateurs MACD et Stochastique.
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ARIMAPredictor:
    """
    Prédicteur ARIMA avec rolling forecast.

    Paramètres
    ----------
    order : tuple (p, d, q) — ordre ARIMA, défaut (5, 1, 2)
    """

    def __init__(self, order: tuple = (5, 1, 2)):
        self.order = order
        self._train = None

    def fit(self, train: np.ndarray) -> "ARIMAPredictor":
        self._train = np.asarray(train, dtype=float)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Rolling forecast : ré-ajustement à chaque pas.
        La valeur prédite est ajoutée à l'historique pour le pas suivant
        (approche récursive, nécessaire car les vraies valeurs futures sont inconnues).
        """
        if self._train is None:
            raise RuntimeError("Appeler fit() avant predict()")

        if not STATSMODELS_AVAILABLE:
            # Fallback Random Walk with Drift
            from models.naive_model import NaivePredictor
            return NaivePredictor().fit_predict(self._train, horizon)

        history = list(self._train)
        predictions = []

        for _ in range(horizon):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = _ARIMA(history, order=self.order)
                    fit = model.fit()
                    yhat = float(fit.forecast(steps=1)[0])
            except Exception:
                # Fallback : dernière valeur connue
                yhat = history[-1]

            predictions.append(yhat)
            history.append(yhat)  # rolling : la prédiction alimente l'historique

        return np.array(predictions, dtype=float)

    def fit_predict(self, train: np.ndarray, horizon: int) -> np.ndarray:
        return self.fit(train).predict(horizon)
