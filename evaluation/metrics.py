"""
evaluation/metrics.py
======================
Métriques d'évaluation conformes aux spécifications du hackathon.

Métriques principales (hackathon) :
  - Tendance  : proportion de directions correctement prédites ∈ [0, 1]
  - Corrélation : coefficient de Pearson ∈ [-1, 1]
  - Score combiné : (T_MACD + max(0, ρ_MACD) + T_Stoch + max(0, ρ_Stoch)) / 4

Métriques de diagnostic :
  - MAE  : Mean Absolute Error
  - RMSE : Root Mean Squared Error

Adapté de projet_ia/app1/app/evaluation/metrics.py (Charbonnier & Claveau) —
même calcul de tendance (ignorer les variations nulles) et de corrélation.

Évaluation par indicateur :
  - MACD Line, MACD Histogramme, Stochastique %K, Stochastique %D
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ─────────────────────────────────────────────────────────────
# MÉTRIQUES PRINCIPALES
# ─────────────────────────────────────────────────────────────

def trend_score(real: np.ndarray, pred: np.ndarray) -> float:
    """
    Score de tendance : proportion de directions correctement prédites.

    Direction réelle : signe(y_h - y_{h-1})
    Direction prédite : signe(ŷ_h - ŷ_{h-1})
    Les variations nulles sont ignorées (ambiguës).

    Retourne float ∈ [0, 1].
    0.5 = performance aléatoire, 1.0 = parfait.

    Repris de projet_ia/app1/app/evaluation/metrics.py.
    """
    n = min(len(real), len(pred))
    if n < 2:
        return 0.5

    real_dirs = np.sign(np.diff(real[:n].astype(float)))
    pred_dirs = np.sign(np.diff(pred[:n].astype(float)))

    matches, total = 0, 0
    for dr, dp in zip(real_dirs, pred_dirs):
        if dr == 0:
            continue
        total += 1
        if dr == dp:
            matches += 1

    return matches / total if total > 0 else 0.5


def pearson_correlation(real: np.ndarray, pred: np.ndarray) -> float:
    """
    Coefficient de corrélation de Pearson ∈ [-1, 1].
    Retourne 0.0 si l'une des séries est constante.

    Repris de projet_ia/app1/app/evaluation/metrics.py.
    """
    n = min(len(real), len(pred))
    if n < 2:
        return 0.0

    r = real[:n].astype(float)
    p = pred[:n].astype(float)

    if np.std(r) < 1e-10 or np.std(p) < 1e-10:
        return 0.0

    corr, _ = pearsonr(r, p)
    return float(corr) if not np.isnan(corr) else 0.0


# ─────────────────────────────────────────────────────────────
# MÉTRIQUES DE DIAGNOSTIC
# ─────────────────────────────────────────────────────────────

def mae(real: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    n = min(len(real), len(pred))
    return float(np.mean(np.abs(real[:n].astype(float) - pred[:n].astype(float))))


def rmse(real: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    n = min(len(real), len(pred))
    return float(np.sqrt(np.mean((real[:n].astype(float) - pred[:n].astype(float)) ** 2)))


# ─────────────────────────────────────────────────────────────
# SCORE COMBINÉ HACKATHON
# ─────────────────────────────────────────────────────────────

def hackathon_score(
    trend_macd: float,
    corr_macd: float,
    trend_stoch: float,
    corr_stoch: float,
) -> float:
    """
    Score combiné du hackathon :
        (T_MACD + max(0, ρ_MACD) + T_Stoch + max(0, ρ_Stoch)) / 4

    Les corrélations négatives sont pénalisées (remplacées par 0).
    Le score maximum est 1.0 (tendance parfaite + corrélation = 1 sur les deux).

    Paramètres
    ----------
    trend_macd/stoch : float ∈ [0, 1]  — score de tendance
    corr_macd/stoch  : float ∈ [-1, 1] — corrélation de Pearson

    Retourne
    --------
    float ∈ [0, 1]
    """
    return (
        trend_macd + max(0.0, corr_macd) +
        trend_stoch + max(0.0, corr_stoch)
    ) / 4.0


# ─────────────────────────────────────────────────────────────
# ÉVALUATION GLOBALE — TOUTES LES MÉTRIQUES PAR MODÈLE
# ─────────────────────────────────────────────────────────────

def evaluate_all(
    results: dict,
    macd_real: np.ndarray,
    stoch_real: np.ndarray,
) -> pd.DataFrame:
    """
    Évalue tous les modèles sur MACD et Stochastique.

    Paramètres
    ----------
    results    : dict — {indicateur: {modèle: np.ndarray de prédictions}}
                 indicateur ∈ {'macd', 'stoch'}
    macd_real  : np.ndarray — valeurs réelles MACD sur l'horizon
    stoch_real : np.ndarray — valeurs réelles Stoch sur l'horizon

    Retourne
    --------
    pd.DataFrame — une ligne par modèle, triée par score combiné décroissant
    Colonnes : Modèle, Tendance MACD (%), Corrélation MACD, MAE MACD, RMSE MACD,
               Tendance Stoch (%), Corrélation Stoch, MAE Stoch, RMSE Stoch,
               Score combiné
    """
    model_names = sorted(
        set(results.get("macd", {}).keys()) | set(results.get("stoch", {}).keys())
    )
    rows = []

    for name in model_names:
        macd_pred  = results.get("macd",  {}).get(name)
        stoch_pred = results.get("stoch", {}).get(name)

        def safe_metrics(real, pred):
            if pred is None or len(pred) == 0 or np.any(np.isnan(pred)):
                return 0.5, 0.0, float("nan"), float("nan")
            return (
                trend_score(real, pred),
                pearson_correlation(real, pred),
                mae(real, pred),
                rmse(real, pred),
            )

        m_t, m_c, m_mae, m_rmse = safe_metrics(macd_real, macd_pred)
        s_t, s_c, s_mae, s_rmse = safe_metrics(stoch_real, stoch_pred)
        score = hackathon_score(m_t, m_c, s_t, s_c)

        rows.append({
            "Modèle":             name,
            "Tendance MACD (%)":  round(m_t * 100, 1),
            "Corrélation MACD":   round(m_c, 4),
            "MAE MACD":           round(m_mae, 4) if not np.isnan(m_mae) else float("nan"),
            "RMSE MACD":          round(m_rmse, 4) if not np.isnan(m_rmse) else float("nan"),
            "Tendance Stoch (%)": round(s_t * 100, 1),
            "Corrélation Stoch":  round(s_c, 4),
            "MAE Stoch":          round(s_mae, 4) if not np.isnan(s_mae) else float("nan"),
            "RMSE Stoch":         round(s_rmse, 4) if not np.isnan(s_rmse) else float("nan"),
            "Score combiné":      round(score, 4),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("Score combiné", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rang"
    return df


# ─────────────────────────────────────────────────────────────
# ÉVALUATION DÉTAILLÉE PAR INDICATEUR
# ─────────────────────────────────────────────────────────────

def evaluate_indicator(
    real: np.ndarray,
    predictions: dict,
    indicator_name: str,
) -> pd.DataFrame:
    """
    Évalue tous les modèles sur un seul indicateur.

    Paramètres
    ----------
    real            : np.ndarray — valeurs réelles
    predictions     : dict {modèle: np.ndarray}
    indicator_name  : str

    Retourne
    --------
    pd.DataFrame avec Modèle, Tendance (%), Corrélation, MAE, RMSE
    """
    rows = []
    for name, pred in predictions.items():
        if pred is None or len(pred) == 0 or np.any(np.isnan(pred)):
            continue
        rows.append({
            "Indicateur":    indicator_name,
            "Modèle":        name,
            "Tendance (%)":  round(trend_score(real, pred) * 100, 1),
            "Corrélation":   round(pearson_correlation(real, pred), 4),
            "MAE":           round(mae(real, pred), 4),
            "RMSE":          round(rmse(real, pred), 4),
        })
    return pd.DataFrame(rows)
