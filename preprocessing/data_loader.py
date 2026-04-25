"""
preprocessing/data_loader.py
=============================
Chargement des données Sibyllium au format TICKER_YYYY-MM-DD.xlsx
et depuis un fichier Parquet pré-converti.

Adapté de projet_ia/app1/app/app.py et preprocessing/indicators.py
(Charbonnier & Claveau) pour intégrer la lecture XLSX Sibyllium dans le
pipeline de Louis Cluzel.

Format Sibyllium :
  - Nom de fichier : TICKER_YYYY-MM-DD.xlsx
  - Colonnes attendues : date, open, high, low, close, volume
  - Barres intraday 5 minutes
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DEPUIS PARQUET (rapide, recommandé)
# ─────────────────────────────────────────────────────────────

def load_from_parquet(parquet_path: str) -> dict:
    """
    Charge toutes les données depuis un fichier Parquet pré-converti.
    Retourne un dict {ticker: pd.DataFrame} trié par date.

    Le Parquet doit avoir les colonnes : ticker, date, open, high, low, close, volume.
    Généré par projet_ia/app1/app/convert_to_parquet.py.

    Paramètres
    ----------
    parquet_path : str — chemin vers data.parquet

    Retourne
    --------
    dict {ticker (str): pd.DataFrame}
    """
    df = pd.read_parquet(parquet_path)
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"])

    result = {}
    for ticker, group in df.groupby("ticker"):
        g = group.drop(columns="ticker").sort_values("date").reset_index(drop=True)
        result[ticker] = g

    return result


def load_ticker_from_parquet(parquet_path: str, ticker: str) -> pd.DataFrame:
    """
    Charge les données d'un seul ticker depuis le Parquet.
    Plus efficace que load_from_parquet() pour un ticker unique.

    Paramètres
    ----------
    parquet_path : str — chemin vers data.parquet
    ticker       : str — symbole de l'actif (ex: 'AAPL')

    Retourne
    --------
    pd.DataFrame trié par date (colonnes: date, open, high, low, close, volume)
    """
    df = pd.read_parquet(parquet_path, filters=[("ticker", "=", ticker)])
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' introuvable dans {parquet_path}")
    df["date"] = pd.to_datetime(df["date"])
    if "ticker" in df.columns:
        df = df.drop(columns="ticker")
    return df.sort_values("date").reset_index(drop=True)


def list_tickers_in_parquet(parquet_path: str) -> list:
    """Retourne la liste triée des tickers disponibles dans le Parquet."""
    df = pd.read_parquet(parquet_path, columns=["ticker"])
    return sorted(df["ticker"].unique().tolist())


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DEPUIS DOSSIER XLSX SIBYLLIUM
# ─────────────────────────────────────────────────────────────

def load_from_folder(folder_path: str, ticker: str = None) -> dict:
    """
    Charge et agrège les fichiers XLSX Sibyllium depuis un dossier local.
    Filtre les fichiers fantômes macOS (._*) et les fichiers invalides.

    Logique reprise de projet_ia/app1/app/app.py — load_data() et
    preprocessing/indicators.py — load_and_aggregate().

    Paramètres
    ----------
    folder_path : str  — dossier contenant les fichiers TICKER_YYYY-MM-DD.xlsx
    ticker      : str  — si fourni, ne charge que ce ticker (sinon tous)

    Retourne
    --------
    dict {ticker (str): pd.DataFrame} trié par date
    """
    xlsx_files = sorted(
        glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)
    )

    all_data: dict = {}

    for path in xlsx_files:
        fname = os.path.basename(path)

        # Ignorer les fichiers fantômes macOS et fichiers cachés
        if fname.startswith("._") or fname.startswith("__") or fname.startswith("."):
            continue

        # Extraire le ticker depuis le nom de fichier (TICKER_YYYY-MM-DD.xlsx)
        parts = fname.replace(".xlsx", "").split("_")
        file_ticker = parts[0].upper()

        # Filtrer par ticker si demandé
        if ticker is not None and file_ticker != ticker.upper():
            continue

        try:
            df = pd.read_excel(path, header=0, engine="openpyxl")
            df.columns = df.columns.str.lower().str.strip()

            if "close" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["date"])

            if file_ticker not in all_data:
                all_data[file_ticker] = []
            all_data[file_ticker].append(df)

        except Exception:
            continue  # Fichier corrompu ou invalide : ignoré

    # Concaténer et trier par date pour chaque ticker
    result = {}
    for t, frames in all_data.items():
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("date").reset_index(drop=True)
        result[t] = merged

    return result


# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide et nettoie un DataFrame OHLCV.
    - Vérifie la présence des colonnes obligatoires
    - Supprime les lignes avec NaN sur close
    - Assure le tri chronologique
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df.dropna(subset=["close"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """Retourne un résumé des méta-données du DataFrame OHLCV."""
    return {
        "n_bars": len(df),
        "n_days": df["date"].dt.date.nunique() if "date" in df.columns else 0,
        "start": df["date"].iloc[0] if len(df) > 0 else None,
        "end": df["date"].iloc[-1] if len(df) > 0 else None,
        "last_close": float(df["close"].iloc[-1]) if len(df) > 0 else None,
    }
