"""
pipeline.py
============
Pipeline CLI complet pour la prédiction d'indicateurs financiers.

Usage :
    python pipeline.py --parquet <chemin/data.parquet> --ticker AAPL
    python pipeline.py --folder <chemin/dossier_xlsx> --ticker AAPL
    python pipeline.py --parquet <chemin/data.parquet> --all

Options :
    --parquet   : chemin vers le fichier data.parquet (Sibyllium converti)
    --folder    : chemin vers un dossier de fichiers xlsx Sibyllium
    --ticker    : ticker à analyser (ex: AAPL, MSFT)
    --all       : analyser tous les tickers disponibles
    --horizon   : horizon de prédiction (défaut: 15)
    --lookback  : fenêtre d'observation (défaut: 120)
    --no-arima  : désactiver ARIMA (plus rapide)
    --no-dl     : désactiver les modèles DL (LSTM/GRU/Transformer)
    --output    : dossier de sortie (défaut: results/)

Le pipeline évalue les modèles suivants sur MACD et Stochastique %K :
  - Random Walk with Drift (baseline)
  - ARIMA (5,1,2) rolling forecast
  - LSTM (multifeature, direct multi-pas)
  - GRU  (multifeature, direct multi-pas)
  - Transformer (multifeature, direct multi-pas)
  - Ensemble LSTM+GRU+Transformer (poids appris)
"""

import os
import sys
import json
import time
import warnings
import argparse

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ajout du répertoire courant au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import (
    load_ticker_from_parquet,
    load_from_parquet,
    load_from_folder,
    list_tickers_in_parquet,
    validate_ohlcv,
    get_data_info,
)
from preprocessing.indicators import compute_macd, compute_stochastic
from preprocessing.features import prepare_dl_data
from models.naive_model import NaivePredictor
from models.arima_model import ARIMAPredictor
from models.lstm_model import LSTMPredictor
from models.gru_model import GRUPredictor
from models.transformer_model import TransformerPredictor
from models.ensemble_model import EnsemblePredictor
from evaluation.metrics import evaluate_all, evaluate_indicator, hackathon_score


# ─────────────────────────────────────────────────────────────
# CONFIGURATION PAR DÉFAUT
# ─────────────────────────────────────────────────────────────

DEFAULT_HORIZON  = 15
DEFAULT_LOOKBACK = 120


# ─────────────────────────────────────────────────────────────
# PIPELINE PAR INDICATEUR
# ─────────────────────────────────────────────────────────────

def run_indicator_pipeline(
    series: np.ndarray,
    indicator_name: str,
    horizon: int,
    lookback: int,
    use_arima: bool = True,
    use_dl: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Exécute tous les modèles sur une série d'indicateur et retourne les prédictions.

    Paramètres
    ----------
    series         : np.ndarray — série complète de l'indicateur (train + test)
    indicator_name : str — nom pour l'affichage
    horizon        : int — nombre de pas à prédire (test = derniers 'horizon' points)
    lookback       : int — fenêtre temporelle d'entrée pour les DL
    use_arima      : bool — activer ARIMA
    use_dl         : bool — activer LSTM/GRU/Transformer
    verbose        : bool — afficher la progression

    Retourne
    --------
    dict {modèle: np.ndarray (horizon,)}
    """
    predictions = {}
    actual = series[-horizon:]
    train = series[:-horizon]

    # ── Baseline : Random Walk with Drift ──────────────────────────────────
    if verbose:
        print(f"    [Baseline] Random Walk with Drift...")
    t0 = time.time()
    predictions["Baseline"] = NaivePredictor().fit_predict(train, horizon)
    if verbose:
        _print_metrics(actual, predictions["Baseline"], "Baseline", t0)

    # ── ARIMA ──────────────────────────────────────────────────────────────
    if use_arima:
        if verbose:
            print(f"    [ARIMA] Rolling forecast (5,1,2)...")
        t0 = time.time()
        try:
            predictions["ARIMA"] = ARIMAPredictor(order=(5, 1, 2)).fit_predict(train, horizon)
        except Exception as e:
            if verbose:
                print(f"      ⚠ ARIMA échoué : {e} — fallback Baseline")
            predictions["ARIMA"] = NaivePredictor().fit_predict(train, horizon)
        if verbose:
            _print_metrics(actual, predictions["ARIMA"], "ARIMA", t0)

    # ── Deep Learning : préparation des données ────────────────────────────
    if use_dl:
        try:
            dl_data = prepare_dl_data(series, horizon, lookback)
            X_train  = dl_data["X_train"]
            y_train  = dl_data["y_train"]
            X_pred   = dl_data["X_pred"]
            scaler   = dl_data["scaler"]
        except Exception as e:
            if verbose:
                print(f"    ⚠ Préparation DL échouée : {e} — modèles DL désactivés")
            use_dl = False

    if use_dl:
        # ── LSTM ──────────────────────────────────────────────────────────
        if verbose:
            print(f"    [LSTM] Entraînement...")
        t0 = time.time()
        try:
            lstm = LSTMPredictor()
            predictions["LSTM"] = lstm.fit_predict(X_train, y_train, X_pred, scaler)
        except Exception as e:
            if verbose:
                print(f"      ⚠ LSTM échoué : {e}")
            predictions["LSTM"] = NaivePredictor().fit_predict(train, horizon)
        if verbose:
            _print_metrics(actual, predictions["LSTM"], "LSTM", t0)

        # ── GRU ───────────────────────────────────────────────────────────
        if verbose:
            print(f"    [GRU] Entraînement...")
        t0 = time.time()
        try:
            gru = GRUPredictor()
            predictions["GRU"] = gru.fit_predict(X_train, y_train, X_pred, scaler)
        except Exception as e:
            if verbose:
                print(f"      ⚠ GRU échoué : {e}")
            predictions["GRU"] = NaivePredictor().fit_predict(train, horizon)
        if verbose:
            _print_metrics(actual, predictions["GRU"], "GRU", t0)

        # ── Transformer ────────────────────────────────────────────────────
        if verbose:
            print(f"    [Transformer] Entraînement...")
        t0 = time.time()
        try:
            trans = TransformerPredictor()
            predictions["Transformer"] = trans.fit_predict(X_train, y_train, X_pred, scaler)
        except Exception as e:
            if verbose:
                print(f"      ⚠ Transformer échoué : {e}")
            predictions["Transformer"] = NaivePredictor().fit_predict(train, horizon)
        if verbose:
            _print_metrics(actual, predictions["Transformer"], "Transformer", t0)

        # ── Ensemble ──────────────────────────────────────────────────────
        if verbose:
            print(f"    [Ensemble] Calcul des poids...")
        ensemble = EnsemblePredictor()
        predictions["Ensemble"] = ensemble.predict(
            predictions.get("LSTM"),
            predictions.get("GRU"),
            predictions.get("Transformer"),
        )
        if verbose:
            _print_metrics(actual, predictions["Ensemble"], "Ensemble", time.time())

    return predictions


def _print_metrics(actual, pred, name, t0):
    """Affiche les métriques pour un modèle (usage interne)."""
    from evaluation.metrics import trend_score, pearson_correlation
    t = trend_score(actual, pred) * 100
    c = pearson_correlation(actual, pred)
    elapsed = time.time() - t0
    print(f"      Tendance: {t:.1f}% | Corrélation: {c:.4f} | {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────
# PIPELINE COMPLET POUR UN TICKER
# ─────────────────────────────────────────────────────────────

def run_ticker(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = DEFAULT_HORIZON,
    lookback: int = DEFAULT_LOOKBACK,
    use_arima: bool = True,
    use_dl: bool = True,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict:
    """
    Pipeline complet pour un ticker : indicateurs → modèles → métriques.

    Retourne un dict avec les résultats (prédictions + métriques).
    """
    if verbose:
        print(f"\n{'='*65}")
        print(f"  Actif : {ticker}")
        print(f"{'='*65}")

    # Validation des données
    df = validate_ohlcv(df)
    info = get_data_info(df)
    if verbose:
        print(f"  {info['n_bars']:,} barres | {info['n_days']} jours | "
              f"{info['start'].date()} → {info['end'].date()}")

    min_required = lookback + horizon + 20  # 20 pour le warmup des rolling features
    if len(df) < min_required:
        print(f"  ⚠ Données insuffisantes : {len(df)} barres (minimum : {min_required})")
        return {}

    # Calcul des indicateurs
    macd_df  = compute_macd(df["close"])
    stoch_df = compute_stochastic(df["high"], df["low"], df["close"])

    # Séries propres (sans NaN)
    macd_series  = macd_df["macd"].dropna().values
    stoch_series = stoch_df["k"].dropna().values

    # Valeurs réelles de test
    macd_actual  = macd_series[-horizon:]
    stoch_actual = stoch_series[-horizon:]

    all_results = {"macd": {}, "stoch": {}}

    # ── MACD ──────────────────────────────────────────────────
    if verbose:
        print(f"\n  --- MACD ---")
    macd_preds = run_indicator_pipeline(
        macd_series, "MACD", horizon, lookback, use_arima, use_dl, verbose
    )
    all_results["macd"] = macd_preds

    # ── Stochastique %K ───────────────────────────────────────
    if verbose:
        print(f"\n  --- Stochastique %K ---")
    stoch_preds = run_indicator_pipeline(
        stoch_series, "Stoch_K", horizon, lookback, use_arima, use_dl, verbose
    )
    all_results["stoch"] = stoch_preds

    # ── Métriques ─────────────────────────────────────────────
    metrics_df = evaluate_all(all_results, macd_actual, stoch_actual)

    if verbose:
        print(f"\n  {'─'*65}")
        print(f"  RÉSULTATS {ticker}")
        print(metrics_df[["Modèle", "Tendance MACD (%)", "Corrélation MACD",
                          "Tendance Stoch (%)", "Corrélation Stoch", "Score combiné"]]
              .to_string(index=True))
        best = metrics_df.iloc[0]
        print(f"\n  🥇 Meilleur : {best['Modèle']} | Score = {best['Score combiné']:.4f}")

    # Sauvegarde des résultats
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{ticker}_metrics.csv")
    metrics_df.to_csv(metrics_path)
    if verbose:
        print(f"  Résultats sauvegardés : {metrics_path}")

    return {
        "ticker":      ticker,
        "predictions": {
            ind: {m: p.tolist() for m, p in preds.items()}
            for ind, preds in all_results.items()
        },
        "metrics":     metrics_df.to_dict(orient="records"),
        "actual": {
            "macd":  macd_actual.tolist(),
            "stoch": stoch_actual.tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de prédiction d'indicateurs financiers (Sibyllium)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet", type=str, help="Chemin vers data.parquet")
    source.add_argument("--folder",  type=str, help="Dossier xlsx Sibyllium")

    parser.add_argument("--ticker",   type=str, default=None, help="Ticker à analyser")
    parser.add_argument("--all",      action="store_true",     help="Analyser tous les tickers")
    parser.add_argument("--horizon",  type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--no-arima", action="store_true", help="Désactiver ARIMA")
    parser.add_argument("--no-dl",    action="store_true", help="Désactiver les modèles DL")
    parser.add_argument("--output",   type=str, default="results", help="Dossier de sortie")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  HACKATHON — Prédiction de Séries Temporelles Financières")
    print("  Louis Cluzel | M1 IA | Université Clermont Auvergne")
    print("=" * 65)

    # Chargement des données
    if args.parquet:
        print(f"\n  Chargement du Parquet : {args.parquet}")
        if args.all:
            tickers = list_tickers_in_parquet(args.parquet)
            print(f"  {len(tickers)} tickers disponibles")
        else:
            if not args.ticker:
                print("  Erreur : spécifier --ticker ou --all")
                sys.exit(1)
            tickers = [args.ticker]
    else:
        print(f"\n  Chargement du dossier : {args.folder}")
        data_dict = load_from_folder(args.folder)
        tickers = list(data_dict.keys()) if args.all else [args.ticker]

    # Analyse par ticker
    all_results = {}
    for t in tickers:
        try:
            if args.parquet:
                df = load_ticker_from_parquet(args.parquet, t)
            else:
                df = data_dict.get(t)
                if df is None:
                    print(f"  ⚠ Ticker {t} introuvable dans le dossier")
                    continue

            result = run_ticker(
                df=df,
                ticker=t,
                horizon=args.horizon,
                lookback=args.lookback,
                use_arima=not args.no_arima,
                use_dl=not args.no_dl,
                output_dir=args.output,
            )
            if result:
                all_results[t] = result

        except Exception as e:
            print(f"  ⚠ Erreur sur {t} : {e}")
            import traceback
            traceback.print_exc()

    # Sauvegarde JSON globale
    if all_results:
        json_path = os.path.join(args.output, "results_full.json")
        os.makedirs(args.output, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Résultats complets sauvegardés : {json_path}")

    print("\n" + "=" * 65)
    print("  FIN DE L'ANALYSE")
    print("=" * 65)


if __name__ == "__main__":
    main()
