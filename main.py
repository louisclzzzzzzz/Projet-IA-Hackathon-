"""
Hackathon : Prediction de series temporelles appliquee aux indicateurs financiers
=================================================================================
Ce script implemente et compare plusieurs methodes de prediction de series temporelles
sur les indicateurs MACD(12,26,9) et Stochastique(14,3,5).

Methodes implementees :
  1. ARIMA (modele statistique classique)
  2. LSTM (deep learning recurrent)
  3. GRU (deep learning recurrent)
  4. Transformer (attention-based)

Metriques d'evaluation :
  - Tendance : pourcentage de directions correctement predites
  - Correlation : coefficient de Pearson entre predictions et valeurs reelles
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HORIZON = 15          # horizon de prediction
LOOKBACK = 60         # fenetre d'observation pour les modeles DL
EPOCHS_DL = 100       # nombre d'epoques pour LSTM/GRU/Transformer
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Actifs financiers a analyser
TICKERS = ["AAPL", "MSFT", "GOOGL"]
DATA_PERIOD = "5y"

np.random.seed(42)
torch.manual_seed(42)


# ===========================================================================
# PARTIE 1 : Recuperation et preparation des donnees
# ===========================================================================
def fetch_data(ticker: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """Telecharge les donnees OHLCV depuis Yahoo Finance."""
    print(f"  Telechargement des donnees pour {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """Calcule le MACD(12,26,9) : ligne MACD, signal et histogramme."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    result = pd.DataFrame({
        "MACD_line": macd_line,
        "MACD_signal": signal_line,
        "MACD_hist": histogram
    }, index=df.index)
    return result


def compute_stochastic(df: pd.DataFrame, k_period=14, d_period=3, dd_period=5) -> pd.DataFrame:
    """Calcule le Stochastique(14,3,5) : %K, %D (lissage simple) et %DD (double lissage)."""
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    stoch_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_period).mean()       # premier lissage
    stoch_dd = stoch_d.rolling(window=dd_period).mean()      # double lissage
    result = pd.DataFrame({
        "Stoch_K": stoch_k,
        "Stoch_D": stoch_d,
        "Stoch_DD": stoch_dd
    }, index=df.index)
    return result


def prepare_dataset(ticker: str) -> pd.DataFrame:
    """Pipeline complet : telechargement + calcul indicateurs."""
    df = fetch_data(ticker)
    macd = compute_macd(df)
    stoch = compute_stochastic(df)
    dataset = pd.concat([df, macd, stoch], axis=1).dropna()
    return dataset


# ===========================================================================
# PARTIE 2 : Metriques d'evaluation
# ===========================================================================
def evaluate_trend(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pourcentage de directions correctement predites (tendance)."""
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    correct = np.sum(true_dir == pred_dir)
    return correct / len(true_dir) * 100


def evaluate_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de correlation de Pearson."""
    if np.std(y_pred) < 1e-10:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr


# ===========================================================================
# PARTIE 3 : Modele ARIMA
# ===========================================================================
def predict_arima(series: np.ndarray, horizon: int = HORIZON, order=(5, 1, 2)) -> np.ndarray:
    """Prediction ARIMA avec rolling forecast."""
    train = list(series)
    predictions = []
    for _ in range(horizon):
        try:
            model = ARIMA(train, order=order)
            fit = model.fit()
            yhat = fit.forecast(steps=1)[0]
        except Exception:
            yhat = train[-1]
        predictions.append(yhat)
        train.append(yhat)
    return np.array(predictions)


# ===========================================================================
# PARTIE 4 : Modeles Deep Learning (LSTM, GRU, Transformer)
# ===========================================================================
def create_sequences(data: np.ndarray, lookback: int = LOOKBACK):
    """Cree les sequences (X, y) pour l'entrainement des modeles DL."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out


def train_dl_model(model, train_loader, epochs=EPOCHS_DL, lr=LR):
    """Entraine un modele Deep Learning."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    return model


def predict_dl(model, last_sequence: np.ndarray, scaler, horizon: int = HORIZON) -> np.ndarray:
    """Prediction auto-regressive pour les modeles DL."""
    model.eval()
    seq = last_sequence.copy()
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            pred = model(x).cpu().numpy()[0, 0]
            predictions.append(pred)
            seq = np.append(seq[1:], pred)
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()
    return predictions


def run_dl_pipeline(series: np.ndarray, model_class, model_name: str,
                    horizon: int = HORIZON, lookback: int = LOOKBACK):
    """Pipeline complet pour un modele DL : scaling, training, prediction."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    train_data = scaled[:-horizon]
    X, y = create_sequences(train_data, lookback)
    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y).unsqueeze(-1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = model_class(input_size=1)
    model = train_dl_model(model, loader)

    last_seq = scaled[-(horizon + lookback):-horizon]
    predictions = predict_dl(model, last_seq, scaler, horizon)
    return predictions


# ===========================================================================
# PARTIE 5 : Pipeline d'evaluation complet
# ===========================================================================
def run_evaluation(dataset: pd.DataFrame, indicator_col: str, indicator_name: str,
                   ticker: str) -> dict:
    """Execute tous les modeles sur un indicateur et retourne les resultats."""
    series = dataset[indicator_col].values
    actual = series[-HORIZON:]
    train_series = series[:-HORIZON]

    results = {}

    # --- ARIMA ---
    print(f"    [ARIMA] Prediction {indicator_name}...")
    pred_arima = predict_arima(train_series, HORIZON)
    trend_arima = evaluate_trend(actual, pred_arima)
    corr_arima = evaluate_correlation(actual, pred_arima)
    results["ARIMA"] = {"predictions": pred_arima, "trend": trend_arima, "correlation": corr_arima}
    print(f"      Tendance: {trend_arima:.1f}% | Correlation: {corr_arima:.4f}")

    # --- LSTM ---
    print(f"    [LSTM] Prediction {indicator_name}...")
    pred_lstm = run_dl_pipeline(series, LSTMModel, "LSTM")
    trend_lstm = evaluate_trend(actual, pred_lstm)
    corr_lstm = evaluate_correlation(actual, pred_lstm)
    results["LSTM"] = {"predictions": pred_lstm, "trend": trend_lstm, "correlation": corr_lstm}
    print(f"      Tendance: {trend_lstm:.1f}% | Correlation: {corr_lstm:.4f}")

    # --- GRU ---
    print(f"    [GRU] Prediction {indicator_name}...")
    pred_gru = run_dl_pipeline(series, GRUModel, "GRU")
    trend_gru = evaluate_trend(actual, pred_gru)
    corr_gru = evaluate_correlation(actual, pred_gru)
    results["GRU"] = {"predictions": pred_gru, "trend": trend_gru, "correlation": corr_gru}
    print(f"      Tendance: {trend_gru:.1f}% | Correlation: {corr_gru:.4f}")

    # --- Transformer ---
    print(f"    [Transformer] Prediction {indicator_name}...")
    pred_trans = run_dl_pipeline(series, TransformerModel, "Transformer")
    trend_trans = evaluate_trend(actual, pred_trans)
    corr_trans = evaluate_correlation(actual, pred_trans)
    results["Transformer"] = {"predictions": pred_trans, "trend": trend_trans, "correlation": corr_trans}
    print(f"      Tendance: {trend_trans:.1f}% | Correlation: {corr_trans:.4f}")

    # --- Graphique de comparaison ---
    plot_predictions(actual, results, indicator_name, ticker,
                     dataset.index[-HORIZON:])

    return results


# ===========================================================================
# PARTIE 6 : Visualisations
# ===========================================================================
def plot_predictions(actual, results, indicator_name, ticker, dates):
    """Trace les predictions de chaque modele vs valeurs reelles."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, actual, "k-o", linewidth=2, markersize=4, label="Valeurs reelles", zorder=5)

    colors = {"ARIMA": "#e74c3c", "LSTM": "#3498db", "GRU": "#2ecc71", "Transformer": "#9b59b6"}
    for name, res in results.items():
        ax.plot(dates, res["predictions"], "--s", color=colors.get(name, "gray"),
                linewidth=1.5, markersize=3,
                label=f"{name} (T={res['trend']:.0f}%, r={res['correlation']:.3f})")

    ax.set_title(f"Predictions {indicator_name} - {ticker} (horizon={HORIZON})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(indicator_name)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()

    fname = f"{ticker}_{indicator_name.replace(' ', '_')}_predictions.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Figure sauvegardee : figures/{fname}")


def plot_indicator_history(dataset, ticker):
    """Trace l'historique complet des indicateurs pour un actif."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Prix
    axes[0].plot(dataset.index, dataset["Close"], color="#2c3e50", linewidth=0.8)
    axes[0].set_title(f"{ticker} - Prix de cloture", fontsize=12)
    axes[0].set_ylabel("Prix ($)")
    axes[0].grid(True, alpha=0.3)

    # MACD
    axes[1].plot(dataset.index, dataset["MACD_line"], label="MACD", color="#e74c3c", linewidth=0.8)
    axes[1].plot(dataset.index, dataset["MACD_signal"], label="Signal", color="#3498db", linewidth=0.8)
    axes[1].bar(dataset.index, dataset["MACD_hist"], color=np.where(dataset["MACD_hist"] >= 0, "#2ecc71", "#e74c3c"),
                alpha=0.4, width=1.5, label="Histogramme")
    axes[1].set_title("MACD (12, 26, 9)", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Stochastique
    axes[2].plot(dataset.index, dataset["Stoch_K"], label="%K", color="#e74c3c", linewidth=0.8)
    axes[2].plot(dataset.index, dataset["Stoch_D"], label="%D", color="#3498db", linewidth=0.8)
    axes[2].plot(dataset.index, dataset["Stoch_DD"], label="%DD", color="#2ecc71", linewidth=0.8)
    axes[2].axhline(80, color="gray", linestyle="--", alpha=0.5)
    axes[2].axhline(20, color="gray", linestyle="--", alpha=0.5)
    axes[2].fill_between(dataset.index, 80, 100, alpha=0.1, color="red", label="Surachat")
    axes[2].fill_between(dataset.index, 0, 20, alpha=0.1, color="green", label="Survente")
    axes[2].set_title("Stochastique (14, 3, 5)", fontsize=12)
    axes[2].set_ylabel("%")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fname = f"{ticker}_indicators_overview.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure sauvegardee : figures/{fname}")


def plot_summary_table(all_results):
    """Cree un tableau recapitulatif des performances."""
    rows = []
    for ticker, indicators in all_results.items():
        for ind_name, models in indicators.items():
            for model_name, metrics in models.items():
                rows.append({
                    "Actif": ticker,
                    "Indicateur": ind_name,
                    "Modele": model_name,
                    "Tendance (%)": f"{metrics['trend']:.1f}",
                    "Correlation": f"{metrics['correlation']:.4f}"
                })

    df_results = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.35 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=df_results.values,
        colLabels=df_results.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#ecf0f1")

    plt.title("Tableau recapitulatif des performances", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure sauvegardee : figures/summary_table.png")

    return df_results


def plot_bar_comparison(all_results):
    """Graphique en barres comparant les modeles."""
    models = ["ARIMA", "LSTM", "GRU", "Transformer"]
    metrics_trend = {m: [] for m in models}
    metrics_corr = {m: [] for m in models}

    for ticker, indicators in all_results.items():
        for ind_name, model_results in indicators.items():
            for m in models:
                if m in model_results:
                    metrics_trend[m].append(model_results[m]["trend"])
                    metrics_corr[m].append(model_results[m]["correlation"])

    avg_trend = {m: np.mean(v) if v else 0 for m, v in metrics_trend.items()}
    avg_corr = {m: np.mean(v) if v else 0 for m, v in metrics_corr.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    bars1 = ax1.bar(models, [avg_trend[m] for m in models], color=colors, edgecolor="white")
    ax1.set_title("Tendance moyenne (%)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, [avg_trend[m] for m in models]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(models, [avg_corr[m] for m in models], color=colors, edgecolor="white")
    ax2.set_title("Correlation moyenne (Pearson)", fontsize=12, fontweight="bold")
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, [avg_corr[m] for m in models]):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "bar_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure sauvegardee : figures/bar_comparison.png")


# ===========================================================================
# PARTIE 7 : Main
# ===========================================================================
def main():
    print("=" * 70)
    print("  HACKATHON - Prediction de series temporelles financieres")
    print("=" * 70)

    all_results = {}

    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"  Actif : {ticker}")
        print(f"{'='*70}")

        dataset = prepare_dataset(ticker)
        print(f"  Dataset : {len(dataset)} lignes, de {dataset.index[0].date()} a {dataset.index[-1].date()}")

        # Sauvegarde en xlsx
        xlsx_path = os.path.join(OUTPUT_DIR, f"data_{ticker}.xlsx")
        dataset.to_excel(xlsx_path)
        print(f"  Donnees sauvegardees : data_{ticker}.xlsx")

        # Visualisation des indicateurs
        plot_indicator_history(dataset, ticker)

        all_results[ticker] = {}

        # Indicateurs a predire
        indicators = {
            "MACD_line": "MACD Line",
            "MACD_hist": "MACD Histogramme",
            "Stoch_K": "Stochastique %K",
            "Stoch_D": "Stochastique %D",
        }

        for col, name in indicators.items():
            print(f"\n  --- Indicateur : {name} ---")
            results = run_evaluation(dataset, col, name, ticker)
            all_results[ticker][name] = results

    # --- Synthese globale ---
    print(f"\n{'='*70}")
    print("  SYNTHESE GLOBALE")
    print(f"{'='*70}")

    df_summary = plot_summary_table(all_results)
    plot_bar_comparison(all_results)

    # Sauvegarde des resultats en JSON
    json_results = {}
    for ticker, indicators in all_results.items():
        json_results[ticker] = {}
        for ind_name, models in indicators.items():
            json_results[ticker][ind_name] = {}
            for model_name, metrics in models.items():
                json_results[ticker][ind_name][model_name] = {
                    "trend": float(metrics["trend"]),
                    "correlation": float(metrics["correlation"]),
                    "predictions": metrics["predictions"].tolist()
                }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    print("\n  Resultats sauvegardes dans results.json")
    print(f"\n  Toutes les figures sont dans le dossier figures/")

    # Affichage tableau final
    print("\n" + "=" * 70)
    print("  RESULTATS FINAUX")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
