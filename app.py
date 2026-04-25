"""
app.py — Application Streamlit
================================
Projet IA — Louis Cluzel | M1 IA | Université Clermont Auvergne

Interface interactive pour la prédiction d'indicateurs financiers sur
données Sibyllium (barres intraday 5 min, format TICKER_YYYY-MM-DD.xlsx).

Fonctionnalités :
  - Chargement depuis fichier Parquet pré-converti ou dossier xlsx Sibyllium
  - Calcul automatique des indicateurs MACD(12,26,9) et Stochastique(14,3,5)
  - Comparaison de 6 modèles : Baseline, ARIMA, LSTM, GRU, Transformer, Ensemble
  - Tableau de classement avec toutes les métriques (tendance, corrélation, score hackathon)
  - Visualisations interactives Plotly (dark theme)

Lancement :
    streamlit run app.py

Dépendances :
    pip install -r requirements.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
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
from evaluation.metrics import evaluate_all, hackathon_score


# ─────────────────────────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prédiction Séries Temporelles Financières",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
    [data-testid="metric-container"] {
        background-color: #161b22; border: 1px solid #21262d;
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px; }
    h1 { color: #e6edf3 !important; font-size: 24px !important; font-weight: 700 !important; }
    h2 { color: #e6edf3 !important; font-size: 18px !important; font-weight: 600 !important;
         border-bottom: 1px solid #21262d; padding-bottom: 6px; }
    h3 { color: #8b949e !important; font-size: 14px !important; }
    hr  { border-color: #21262d; }
    .stButton > button {
        background-color: #238636; color: #ffffff; border: 1px solid #2ea043;
        border-radius: 6px; font-weight: 600; padding: 8px 20px;
        width: 100%; transition: background-color 0.2s;
    }
    .stButton > button:hover { background-color: #2ea043; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

for key in ["data_loaded", "raw_data", "computed", "results", "metrics_df"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "data_loaded" else False


# ─────────────────────────────────────────────────────────────
# CONSTANTES VISUELLES
# ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#8b949e", size=11, family="monospace"),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
)

MODEL_COLORS = {
    "Baseline":    "#6e7681",
    "ARIMA":       "#58a6ff",
    "LSTM":        "#f0883e",
    "GRU":         "#3fb950",
    "Transformer": "#bc8cff",
    "Ensemble":    "#ffa657",
}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Prédiction Indicateurs")
    st.markdown(
        "<p style='color:#8b949e;font-size:12px'>Louis Cluzel · M1 IA · UCA</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📂 Données Sibyllium")

    parquet_path = st.text_input(
        "Fichier data.parquet",
        placeholder="/chemin/vers/data.parquet",
        help="Fichier Parquet pré-converti depuis convert_to_parquet.py",
    )
    st.markdown(
        "<p style='color:#8b949e;font-size:11px;text-align:center'>— ou —</p>",
        unsafe_allow_html=True,
    )
    folder_path = st.text_input(
        "Dossier xlsx Sibyllium",
        placeholder="/chemin/vers/dossier/",
        help="Dossier contenant TICKER_YYYY-MM-DD.xlsx",
    )

    load_button = st.button("📂 Charger les données")

    st.markdown("---")

    ticker_options = (
        list(st.session_state.raw_data.keys())
        if st.session_state.data_loaded and isinstance(st.session_state.raw_data, dict)
        else []
    )
    selected_ticker = st.selectbox(
        "Actif analysé",
        options=ticker_options if ticker_options else ["— Charger des données —"],
    )

    st.markdown("---")
    st.markdown("### 🤖 Modèles")

    horizon  = st.slider("Horizon (pas)", 5, 30, 15, 1)
    lookback = st.slider("Lookback (pas)", 30, 200, 120, 10)

    use_baseline    = st.checkbox("Baseline (RWD)",  value=True)
    use_arima       = st.checkbox("ARIMA",            value=True)
    use_lstm        = st.checkbox("LSTM",             value=True)
    use_gru         = st.checkbox("GRU",              value=True)
    use_transformer = st.checkbox("Transformer",      value=True)
    use_ensemble    = st.checkbox("Ensemble",         value=True)

    st.markdown("---")
    run_button = st.button("▶ Lancer l'analyse", type="primary")


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────

if load_button:
    with st.spinner("Chargement des données..."):
        try:
            if parquet_path and os.path.isfile(parquet_path):
                raw = load_from_parquet(parquet_path)
                st.session_state.raw_data = raw
                st.session_state.data_loaded = True
                n_tickers = len(raw)
                n_rows = sum(len(v) for v in raw.values())
                st.success(
                    f"✅ {n_rows:,} lignes | {n_tickers} tickers — "
                    f"{', '.join(sorted(raw.keys())[:8])}{'...' if n_tickers > 8 else ''}"
                )
            elif folder_path and os.path.isdir(folder_path):
                raw = load_from_folder(folder_path)
                if not raw:
                    st.error("Aucun fichier valide trouvé dans le dossier.")
                else:
                    st.session_state.raw_data = raw
                    st.session_state.data_loaded = True
                    n_rows = sum(len(v) for v in raw.values())
                    st.success(
                        f"✅ {n_rows:,} lignes | {len(raw)} tickers : "
                        f"{', '.join(sorted(raw.keys()))}"
                    )
            else:
                st.warning("Renseignez un chemin valide vers le Parquet ou le dossier xlsx.")
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")


# ─────────────────────────────────────────────────────────────
# LANCEMENT DE L'ANALYSE
# ─────────────────────────────────────────────────────────────

if run_button:
    if not st.session_state.data_loaded:
        st.warning("Veuillez d'abord charger des données.")
        st.stop()

    if selected_ticker not in (st.session_state.raw_data or {}):
        st.warning("Sélectionnez un ticker valide.")
        st.stop()

    df_raw = st.session_state.raw_data[selected_ticker].copy()

    with st.spinner("Calcul des indicateurs..."):
        try:
            df = validate_ohlcv(df_raw)
            macd_df  = compute_macd(df["close"])
            stoch_df = compute_stochastic(df["high"], df["low"], df["close"])
            macd_series  = macd_df["macd"].dropna().values
            stoch_series = stoch_df["k"].dropna().values
            st.session_state.computed = {
                "df": df,
                "macd_df": macd_df,
                "stoch_df": stoch_df,
                "macd_series": macd_series,
                "stoch_series": stoch_series,
            }
        except Exception as e:
            st.error(f"Erreur calcul indicateurs : {e}")
            st.stop()

    # Préparation DL (une seule fois pour MACD et Stoch)
    use_dl = use_lstm or use_gru or use_transformer or use_ensemble

    macd_dl_data = stoch_dl_data = None
    if use_dl:
        with st.spinner("Préparation des données DL..."):
            try:
                macd_dl_data  = prepare_dl_data(macd_series, horizon, lookback)
                stoch_dl_data = prepare_dl_data(stoch_series, horizon, lookback)
            except Exception as e:
                st.warning(f"⚠ Préparation DL : {e} — modèles DL désactivés")
                use_dl = False

    results = {"macd": {}, "stoch": {}}
    active_models = []
    if use_baseline:    active_models.append("Baseline")
    if use_arima:       active_models.append("ARIMA")
    if use_lstm:        active_models.append("LSTM")
    if use_gru:         active_models.append("GRU")
    if use_transformer: active_models.append("Transformer")

    progress_bar = st.progress(0, text="Entraînement des modèles...")

    lstm_preds = {"macd": None, "stoch": None}
    gru_preds  = {"macd": None, "stoch": None}
    trans_preds = {"macd": None, "stoch": None}

    for i, name in enumerate(active_models):
        progress_bar.progress(i / len(active_models), text=f"Entraînement {name}...")

        macd_train  = macd_series[:-horizon]
        stoch_train = stoch_series[:-horizon]

        for ind, series, train, dl_data, store_key in [
            ("macd",  macd_series,  macd_train,  macd_dl_data,  lstm_preds),
            ("stoch", stoch_series, stoch_train, stoch_dl_data, gru_preds),
        ]:
            pass  # handled below

        # ── Prédiction MACD et Stoch par modèle ───────────────────────────
        for ind, series_full, train_ser, dl_data_ind in [
            ("macd",  macd_series,  macd_series[:-horizon],  macd_dl_data),
            ("stoch", stoch_series, stoch_series[:-horizon], stoch_dl_data),
        ]:
            try:
                if name == "Baseline":
                    pred = NaivePredictor().fit_predict(train_ser, horizon)
                elif name == "ARIMA":
                    pred = ARIMAPredictor().fit_predict(train_ser, horizon)
                elif name == "LSTM" and use_dl and dl_data_ind:
                    m = LSTMPredictor()
                    m.fit(dl_data_ind["X_train"], dl_data_ind["y_train"])
                    pred = m.predict(dl_data_ind["X_pred"], dl_data_ind["scaler"])
                    if ind == "macd":  lstm_preds["macd"] = pred
                    else:              lstm_preds["stoch"] = pred
                elif name == "GRU" and use_dl and dl_data_ind:
                    m = GRUPredictor()
                    m.fit(dl_data_ind["X_train"], dl_data_ind["y_train"])
                    pred = m.predict(dl_data_ind["X_pred"], dl_data_ind["scaler"])
                    if ind == "macd":  gru_preds["macd"] = pred
                    else:              gru_preds["stoch"] = pred
                elif name == "Transformer" and use_dl and dl_data_ind:
                    m = TransformerPredictor()
                    m.fit(dl_data_ind["X_train"], dl_data_ind["y_train"])
                    pred = m.predict(dl_data_ind["X_pred"], dl_data_ind["scaler"])
                    if ind == "macd":  trans_preds["macd"] = pred
                    else:              trans_preds["stoch"] = pred
                else:
                    continue
                results[ind][name] = pred
            except Exception as e:
                st.warning(f"{name} ({ind.upper()}) : {e}")
                results[ind][name] = NaivePredictor().fit_predict(
                    series_full[:-horizon], horizon
                )

    # Ensemble
    if use_ensemble and use_dl:
        progress_bar.progress(0.95, text="Calcul Ensemble...")
        ensemble = EnsemblePredictor()
        for ind in ["macd", "stoch"]:
            results[ind]["Ensemble"] = ensemble.predict(
                results[ind].get("LSTM"),
                results[ind].get("GRU"),
                results[ind].get("Transformer"),
            )

    progress_bar.progress(1.0, text="Terminé !")
    progress_bar.empty()

    macd_actual  = macd_series[-horizon:]
    stoch_actual = stoch_series[-horizon:]

    metrics_df = evaluate_all(results, macd_actual, stoch_actual)

    st.session_state.results    = results
    st.session_state.metrics_df = metrics_df
    st.session_state["macd_actual"]   = macd_actual
    st.session_state["stoch_actual"]  = stoch_actual
    st.session_state["macd_context"]  = macd_series[max(0, -horizon - 50):-horizon]
    st.session_state["stoch_context"] = stoch_series[max(0, -horizon - 50):-horizon]
    st.session_state["ticker_used"]   = selected_ticker
    st.session_state["horizon_used"]  = horizon


# ─────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown("# 📈 Prédiction de Séries Temporelles Financières")
with col_t2:
    st.markdown(
        "<p style='text-align:right;color:#8b949e;font-size:12px;margin-top:14px'>"
        "Louis Cluzel · M1 IA · UCA</p>",
        unsafe_allow_html=True,
    )
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# AFFICHAGE DES DONNÉES ET INDICATEURS
# ─────────────────────────────────────────────────────────────

computed = st.session_state.computed
if computed:
    df = computed["df"]
    info = get_data_info(df)

    last_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
    delta_pct  = (last_close - prev_close) / (abs(prev_close) + 1e-8) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Actif",        st.session_state.get("ticker_used", "—"))
    c2.metric("Dernier prix", f"${last_close:.4f}", f"{delta_pct:+.2f}%")
    c3.metric("Barres (5min)", f"{info['n_bars']:,}")
    c4.metric("Jours tradés", f"{info['n_days']}")
    c5.metric("Horizon prédit", f"{st.session_state.get('horizon_used', 15)} pas")

    st.markdown("---")
    st.markdown("## Données & Indicateurs Techniques")

    # ── Prix + Volume ──────────────────────────────────────────
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04
    )
    fig_price.add_trace(go.Scatter(
        x=df["date"], y=df["close"], name="Close",
        line=dict(color="#58a6ff", width=1.2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.05)"
    ), row=1, col=1)
    vol_colors = [
        "#3fb950" if df["close"].iloc[i] >= df["open"].iloc[i] else "#f85149"
        for i in range(len(df))
    ]
    fig_price.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.6
    ), row=2, col=1)
    fig_price.update_layout(**PLOTLY_LAYOUT, height=380, title="Prix de clôture & Volume")
    st.plotly_chart(fig_price, use_container_width=True)

    # ── MACD + Stochastique ────────────────────────────────────
    col_m, col_s = st.columns(2)

    macd_df  = computed["macd_df"]
    stoch_df = computed["stoch_df"]

    with col_m:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=macd_df.index, y=macd_df["macd"],
            name="MACD", line=dict(color="#58a6ff", width=1.5)
        ))
        fig_macd.add_trace(go.Scatter(
            x=macd_df.index, y=macd_df["signal"],
            name="Signal", line=dict(color="#f0883e", width=1.5)
        ))
        h_colors = ["#3fb950" if v >= 0 else "#f85149" for v in macd_df["histogram"].fillna(0)]
        fig_macd.add_trace(go.Bar(
            x=macd_df.index, y=macd_df["histogram"],
            name="Histogramme", marker_color=h_colors, opacity=0.7
        ))
        fig_macd.update_layout(**PLOTLY_LAYOUT, height=280, title="MACD (12, 26, 9)")
        st.plotly_chart(fig_macd, use_container_width=True)

    with col_s:
        fig_stoch = go.Figure()
        fig_stoch.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.07)", line_width=0)
        fig_stoch.add_hrect(y0=0,  y1=20,  fillcolor="rgba(63,185,80,0.07)",  line_width=0)
        fig_stoch.add_hline(y=80, line_dash="dot", line_color="#f85149", opacity=0.5)
        fig_stoch.add_hline(y=20, line_dash="dot", line_color="#3fb950", opacity=0.5)
        fig_stoch.add_trace(go.Scatter(
            x=stoch_df.index, y=stoch_df["k"],
            name="%K", line=dict(color="#bc8cff", width=1.5)
        ))
        fig_stoch.add_trace(go.Scatter(
            x=stoch_df.index, y=stoch_df["d"],
            name="%D", line=dict(color="#f0883e", width=1.5, dash="dot")
        ))
        layout_s = {**PLOTLY_LAYOUT, "yaxis": {**PLOTLY_LAYOUT["yaxis"], "range": [0, 100]}}
        fig_stoch.update_layout(**layout_s, height=280, title="Stochastique (14, 3, 5)")
        st.plotly_chart(fig_stoch, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PRÉDICTIONS
# ─────────────────────────────────────────────────────────────

results     = st.session_state.results
metrics_df  = st.session_state.metrics_df

if results and metrics_df is not None:
    st.markdown("---")
    st.markdown("## Prédictions")

    for ind_key, ind_label, actual_key, context_key in [
        ("macd",  "MACD",           "macd_actual",  "macd_context"),
        ("stoch", "Stochastique %K", "stoch_actual", "stoch_context"),
    ]:
        actual  = st.session_state.get(actual_key, np.array([]))
        context = st.session_state.get(context_key, np.array([]))

        fig_pred = go.Figure()
        n_ctx = len(context)
        x_ctx  = list(range(-n_ctx, 0))
        x_test = list(range(len(actual)))

        fig_pred.add_trace(go.Scatter(
            x=x_ctx, y=context.tolist(),
            name="Historique", line=dict(color="#484f58", width=1.0)
        ))
        fig_pred.add_trace(go.Scatter(
            x=x_test, y=actual.tolist(),
            name="Réel", line=dict(color="#e6edf3", width=2),
            mode="lines+markers", marker=dict(size=5)
        ))
        for name, pred in results.get(ind_key, {}).items():
            if pred is None or np.any(np.isnan(pred)):
                continue
            color = MODEL_COLORS.get(name, "#58a6ff")
            dash  = "dash" if name == "Baseline" else "solid"
            fig_pred.add_trace(go.Scatter(
                x=x_test[:len(pred)], y=pred.tolist(),
                name=name, line=dict(color=color, width=1.8, dash=dash), opacity=0.9
            ))

        fig_pred.add_vline(x=0, line_dash="dot", line_color="#484f58",
                           annotation_text="  t=0", annotation_font_color="#6e7681")
        fig_pred.update_layout(
            **PLOTLY_LAYOUT, height=350,
            title=f"Prédictions — {ind_label} (horizon={len(actual)} pas)"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # TABLEAU DE CLASSEMENT
    # ─────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("## 🏆 Classement des modèles")

    def _color_corr(val):
        if isinstance(val, float):
            if val > 0.5:  return "color: #3fb950"
            if val < 0:    return "color: #f85149"
        return "color: #e6edf3"

    def _color_trend(val):
        if isinstance(val, (int, float)):
            if val > 55:   return "color: #3fb950"
            if val < 45:   return "color: #f85149"
        return "color: #e6edf3"

    display_cols = [
        "Modèle",
        "Tendance MACD (%)", "Corrélation MACD",
        "Tendance Stoch (%)", "Corrélation Stoch",
        "Score combiné",
    ]
    styled = (
        metrics_df[display_cols]
        .style
        .applymap(_color_corr,  subset=["Corrélation MACD",  "Corrélation Stoch"])
        .applymap(_color_trend, subset=["Tendance MACD (%)", "Tendance Stoch (%)"])
        .set_properties(**{
            "background-color": "#161b22", "color": "#e6edf3", "border": "1px solid #21262d"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#21262d"), ("color", "#8b949e"),
                ("font-size", "12px"), ("border", "1px solid #30363d")
            ]},
        ])
        .highlight_max(subset=["Score combiné"], color="#1a3a2a")
        .format({
            "Tendance MACD (%)": "{:.1f}%",
            "Tendance Stoch (%)": "{:.1f}%",
            "Corrélation MACD": "{:.4f}",
            "Corrélation Stoch": "{:.4f}",
            "Score combiné": "{:.4f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=280)

    # Détails complets
    with st.expander("Voir toutes les métriques (MAE, RMSE)"):
        st.dataframe(metrics_df, use_container_width=True)

    # Meilleur modèle
    best = metrics_df.iloc[0]
    st.success(
        f"🥇 **Meilleur modèle** : **{best['Modèle']}** — "
        f"Score hackathon : **{best['Score combiné']:.4f}** · "
        f"Tendance MACD : **{best['Tendance MACD (%)']:.1f}%** · "
        f"Corrélation MACD : **{best['Corrélation MACD']:.4f}**"
    )

    # Critère de succès
    score_max = float(metrics_df["Score combiné"].max())
    if score_max >= 0.55:
        st.balloons()
        st.success(f"🎯 Objectif hackathon atteint : score ≥ 0.55 ✓")
    elif score_max >= 0.45:
        st.info(f"📊 Score combiné max : {score_max:.4f} — proche de l'objectif (0.55)")
    else:
        st.warning(f"📊 Score combiné max : {score_max:.4f} — en dessous de l'objectif (0.55)")

else:
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style='text-align:center;padding:80px 20px;color:#484f58'>
            <div style='font-size:48px;margin-bottom:16px'>📊</div>
            <h2 style='color:#484f58 !important;border:none'>Chargez vos données pour commencer</h2>
            <p style='color:#6e7681;max-width:500px;margin:0 auto;font-size:14px;line-height:1.6'>
                Renseignez le chemin vers le fichier <code>data.parquet</code> ou un dossier
                contenant des fichiers Sibyllium (<code>TICKER_YYYY-MM-DD.xlsx</code>),
                puis cliquez sur <strong>Charger les données</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif not results:
        st.info("Sélectionnez un ticker et cliquez sur **▶ Lancer l'analyse**.")
