"""
preprocessing/indicators.py
=============================
Calcul des indicateurs techniques MACD(12,26,9) et Stochastique(14,3,5).

Repris et adapté de projet_ia/app1/app/preprocessing/indicators.py
(Charbonnier & Claveau) — logique de calcul identique garantissant la
comparabilité des scores entre les deux projets.

Indicateurs :
  - MACD (Moving Average Convergence Divergence) : (12, 26, 9)
  - Stochastique : (14, 3, 5)  — %K lissé, %D double lissage
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# UTILITAIRES DE BASE
# ─────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Moyenne Mobile Exponentielle (EMA).
    α = 2 / (period + 1), adjust=False pour comportement incrémental.
    Repris de projet_ia/app1/app/preprocessing/indicators.py.
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Moyenne Mobile Simple (SMA) sur fenêtre glissante."""
    return series.rolling(window=period, min_periods=period).mean()


# ─────────────────────────────────────────────────────────────
# MACD (12, 26, 9)
# ─────────────────────────────────────────────────────────────

def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Calcule le MACD(12,26,9) : ligne MACD, signal et histogramme.

    Formules :
        EMA_fast   = EMA(close, 12)
        EMA_slow   = EMA(close, 26)
        MACD       = EMA_fast - EMA_slow
        Signal     = EMA(MACD, 9)
        Histogramme= MACD - Signal

    Retourne pd.DataFrame avec colonnes ['macd', 'signal', 'histogram'].
    Repris de projet_ia/app1/app/preprocessing/indicators.py.
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )


# ─────────────────────────────────────────────────────────────
# STOCHASTIQUE (14, 3, 5)
# ─────────────────────────────────────────────────────────────

def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 5,
) -> pd.DataFrame:
    """
    Calcule l'oscillateur Stochastique(14,3,5).

    Formules :
        %K_raw = (close - lowest_low(14)) / (highest_high(14) - lowest_low(14)) × 100
        %K     = SMA(%K_raw, 3)    — premier lissage
        %D     = SMA(%K, 5)        — second lissage (ligne de signal)

    Gestion de la division par zéro : valeur neutre 50 si range = 0.
    Retourne pd.DataFrame avec colonnes ['k', 'd'].
    Repris de projet_ia/app1/app/preprocessing/indicators.py.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    range_hl = highest_high - lowest_low
    k_raw = np.where(
        range_hl > 1e-10,
        (close - lowest_low) / range_hl * 100,
        50.0,  # valeur neutre si aucune variation
    )
    k_raw_series = pd.Series(k_raw, index=close.index)

    k_line = sma(k_raw_series, k_smooth)  # %K lissé
    d_line = sma(k_line, d_period)        # %D double lissage

    return pd.DataFrame({"k": k_line, "d": d_line}, index=close.index)


# ─────────────────────────────────────────────────────────────
# CALCUL COMPLET SUR UN DATAFRAME OHLCV
# ─────────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule MACD et Stochastique sur un DataFrame OHLCV complet.
    Les colonnes du DataFrame doivent être en minuscules : close, high, low.

    Retourne le DataFrame enrichi avec les colonnes :
        macd, macd_signal, macd_hist, stoch_k, stoch_d
    Les lignes initiales avec NaN (warmup des EMA/SMA) sont conservées.
    """
    macd_df = compute_macd(df["close"])
    stoch_df = compute_stochastic(df["high"], df["low"], df["close"])

    df = df.copy()
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["signal"]
    df["macd_hist"] = macd_df["histogram"]
    df["stoch_k"] = stoch_df["k"]
    df["stoch_d"] = stoch_df["d"]

    return df
