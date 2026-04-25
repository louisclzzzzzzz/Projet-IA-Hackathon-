"""
preprocessing/features.py
===========================
Feature engineering pour les modèles Deep Learning.

Pour chaque série d'indicateur (MACD, Stoch_K), construit une matrice
de features multi-variée utilisée comme entrée des modèles LSTM/GRU/Transformer.

Features calculées sur la série normalisée :
  1. Valeur de l'indicateur (normalisée)
  2. Retour relatif (variation % / pas précédent)
  3. Moyenne mobile 10 pas
  4. Moyenne mobile 20 pas
  5. Écart-type 10 pas

Inspiré de l'idée de feature engineering de projet_ia/app1/pipeline/models/ml.py.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────
# CONSTRUCTION DE LA MATRICE DE FEATURES
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(series: np.ndarray, eps: float = 1e-8) -> pd.DataFrame:
    """
    Construit une matrice de features (N, 5) à partir d'une série 1D.

    Features :
      - value  : valeur brute de la série
      - return : retour relatif (x[t] - x[t-1]) / (|x[t-1]| + eps)
      - ma10   : moyenne mobile 10 pas
      - ma20   : moyenne mobile 20 pas
      - std10  : écart-type 10 pas

    Les premières lignes contenant des NaN (warmup rolling) sont conservées
    — la suppression des NaN se fait dans prepare_dl_data().

    Paramètres
    ----------
    series : np.ndarray (1D) — série temporelle brute
    eps    : float — plancher pour éviter la division par zéro

    Retourne
    --------
    pd.DataFrame (N, 5) avec colonnes ['value', 'return', 'ma10', 'ma20', 'std10']
    """
    s = pd.Series(series.astype(float))

    ret = s.diff() / (s.abs().shift(1) + eps)
    ma10 = s.rolling(10).mean()
    ma20 = s.rolling(20).mean()
    std10 = s.rolling(10).std()

    return pd.DataFrame(
        {"value": s, "return": ret, "ma10": ma10, "ma20": ma20, "std10": std10}
    )


# ─────────────────────────────────────────────────────────────
# CRÉATION DES SÉQUENCES D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple:
    """
    Crée des paires (X, y) pour l'entraînement supervisé.

    X[i] = features[i : i+lookback]         shape (lookback, n_features)
    y[i] = targets[i+lookback : i+lookback+horizon]  shape (horizon,)

    Paramètres
    ----------
    features : np.ndarray (N, n_features)
    targets  : np.ndarray (N,) — valeurs cibles (série principale)
    lookback : int — fenêtre temporelle d'entrée
    horizon  : int — nombre de pas à prédire

    Retourne
    --------
    X : np.ndarray (n_seq, lookback, n_features)
    y : np.ndarray (n_seq, horizon)
    """
    X, y = [], []
    n = len(targets)
    for i in range(lookback, n - horizon + 1):
        X.append(features[i - lookback : i])
        y.append(targets[i : i + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# PIPELINE COMPLET POUR UN MODÈLE DL
# ─────────────────────────────────────────────────────────────

def prepare_dl_data(
    series: np.ndarray,
    horizon: int,
    lookback: int,
    max_train_bars: int = 3000,
) -> dict:
    """
    Pipeline de préparation complet pour les modèles Deep Learning.

    Étapes :
      1. Limitation aux max_train_bars barres les plus récentes (avant le test)
         → évite des temps d'entraînement prohibitifs sur données intraday
      2. Ajustement du scaler sur les données d'entraînement uniquement
      3. Normalisation de la série complète (train + test)
      4. Calcul des features sur la série normalisée
      5. Suppression des NaN (warmup rolling)
      6. Création des séquences d'entraînement
      7. Extraction de la fenêtre de prédiction

    Paramètres
    ----------
    series         : np.ndarray (N,) — série brute de l'indicateur
    horizon        : int — nombre de pas futurs à prédire
    lookback       : int — fenêtre d'observation pour les modèles DL
    max_train_bars : int — nombre maximum de barres d'entraînement (défaut 6000)
                     Limite les séquences d'entraînement pour rester < 1 min sur CPU.
                     6000 barres ≈ 78 jours de trading intraday 5min → suffisant.

    Retourne
    --------
    dict avec :
      'X_train'  : (n_seq, lookback, 5)
      'y_train'  : (n_seq, horizon)
      'X_pred'   : (1, lookback, 5)  — fenêtre pour la prédiction finale
      'scaler'   : MinMaxScaler ajusté sur les données d'entraînement
      'n_features': int (5)
    """
    n = len(series)
    train_end = n - horizon  # index de fin de train (exclusif)

    if train_end <= 0:
        raise ValueError(f"Série trop courte : {n} points pour horizon={horizon}")

    # Limiter les barres d'entraînement pour rester rapide sur CPU.
    # On garde les max_train_bars barres les plus récentes (hors test),
    # plus le lookback nécessaire pour la fenêtre de prédiction.
    min_bars_needed = lookback + horizon + 20  # 20 = warmup rolling features
    effective_max = max(max_train_bars, min_bars_needed)
    if train_end > effective_max:
        train_start = train_end - effective_max
    else:
        train_start = 0

    # Sous-série utilisée : train_start → fin (incluant la zone de test pour la fenêtre pred)
    sub_series = series[train_start:]
    sub_train_end = train_end - train_start  # position de fin de train dans la sous-série

    train_series = sub_series[:sub_train_end]

    # 1. Ajustement du scaler sur le train uniquement
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series.reshape(-1, 1))

    # 2. Normalisation de la sous-série complète (train + test)
    series_norm = scaler.transform(sub_series.reshape(-1, 1)).flatten()

    # 3. Calcul des features sur la série normalisée
    feat_df = build_feature_matrix(series_norm)

    # 4. Suppression des NaN (les ~20 premières lignes)
    valid_mask = ~feat_df.isnull().any(axis=1)
    feat_clean = feat_df[valid_mask].values.astype(np.float32)  # (M, 5)
    ser_clean = series_norm[valid_mask.values]                   # (M,)

    # Recalculer la position de fin de train dans les données nettoyées
    valid_indices = np.where(valid_mask.values)[0]
    train_valid_count = int((valid_indices < sub_train_end).sum())

    if train_valid_count < lookback + horizon:
        raise ValueError(
            f"Données d'entraînement insuffisantes après suppression NaN : "
            f"{train_valid_count} points (besoin : {lookback + horizon})"
        )

    train_feat = feat_clean[:train_valid_count]
    train_ser = ser_clean[:train_valid_count]

    # 5. Création des séquences d'entraînement
    X_train, y_train = create_sequences(train_feat, train_ser, lookback, horizon)

    # 6. Fenêtre de prédiction : lookback points juste avant la zone de test
    X_pred_window = feat_clean[train_valid_count - lookback : train_valid_count]
    if len(X_pred_window) < lookback:
        raise ValueError(
            f"Fenêtre de prédiction insuffisante : {len(X_pred_window)} < {lookback}"
        )
    X_pred = X_pred_window.reshape(1, lookback, -1)  # (1, lookback, 5)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_pred": X_pred,
        "scaler": scaler,
        "n_features": feat_clean.shape[1],
    }
