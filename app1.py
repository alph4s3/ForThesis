"""
app.py  –  ImpactGuard: Extreme Heat Early Warning System
==========================================================
Streamlit Presentation Layer (Tier 1) for the thesis:
  "Enhancing Extreme Heat Prediction Using Impact-Centric
   Variables in Machine Learning Models"

Author  : [Your Name]  |  Ateneo de Zamboanga University
Course  : BS Computer Science Undergraduate Thesis
Location: Zamboanga City, Philippines

Run
---
    streamlit run app.py

Backend Integration
-------------------
Replace every function tagged  ← BACKEND HOOK  with a real
call to your modular backend (entities.py, data_pipeline.py,
lstm_model.py, evaluation_module.py, xai_module.py,
alert_data_modules.py).

Dependencies
------------
    pip install streamlit plotly pandas numpy fpdf2
"""

from __future__ import annotations

import io
import math
import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ImpactGuard – Heat Early Warning",
    page_icon  = "🌡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-deep:    #0a0e1a;
    --bg-panel:   #111827;
    --bg-card:    #1a2235;
    --border:     #1e2d45;
    --accent-r:   #ff4b4b;
    --accent-o:   #ff8c42;
    --accent-y:   #ffd166;
    --accent-g:   #06d6a0;
    --accent-b:   #118ab2;
    --text-hi:    #f0f4ff;
    --text-lo:    #7a8ba6;
    --font-head:  'Bebas Neue', sans-serif;
    --font-body:  'DM Sans', sans-serif;
    --font-mono:  'DM Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-deep) !important;
    color: var(--text-hi) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem 2rem !important; }

/* ── App header band ── */
.app-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2a4a 50%, #0d1b2a 100%);
    border-bottom: 2px solid var(--accent-b);
    border-radius: 12px;
    padding: 2rem 2.5rem 1.4rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,75,75,.15) 0%, transparent 70%);
    border-radius: 50%;
}
.app-title {
    font-family: var(--font-head) !important;
    font-size: 3.4rem !important;
    letter-spacing: 3px;
    color: var(--text-hi) !important;
    margin: 0; line-height: 1;
}
.app-subtitle {
    font-size: .95rem;
    color: var(--text-lo);
    margin-top: .45rem;
    letter-spacing: .5px;
}
.app-badge {
    display: inline-block;
    background: var(--accent-b);
    color: #fff;
    font-size: .72rem;
    font-family: var(--font-mono);
    padding: .2rem .7rem;
    border-radius: 4px;
    margin-top: .6rem;
    letter-spacing: .8px;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--accent-b); }
.metric-label {
    font-size: .75rem;
    color: var(--text-lo);
    letter-spacing: .6px;
    text-transform: uppercase;
    margin-bottom: .3rem;
}
.metric-value {
    font-family: var(--font-head) !important;
    font-size: 2.1rem;
    color: var(--text-hi);
    line-height: 1;
}
.metric-unit { font-size: .78rem; color: var(--text-lo); }

/* ── Risk badge ── */
.risk-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: .7rem;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-family: var(--font-head);
    font-size: 2rem;
    letter-spacing: 2px;
    border: 2px solid;
}
.risk-low      { background:#06d6a011; border-color:#06d6a0; color:#06d6a0; }
.risk-moderate { background:#ffd16611; border-color:#ffd166; color:#ffd166; }
.risk-high     { background:#ff8c4211; border-color:#ff8c42; color:#ff8c42; }
.risk-extreme  { background:#ff4b4b22; border-color:#ff4b4b; color:#ff4b4b; }

/* ── Section header ── */
.section-head {
    font-family: var(--font-head);
    font-size: 1.4rem;
    letter-spacing: 2px;
    color: var(--text-hi);
    border-left: 4px solid var(--accent-b);
    padding-left: .7rem;
    margin: 1.4rem 0 .8rem;
}

/* ── Comparison table cell ── */
.cmp-win  { color: #06d6a0; font-weight: 600; }
.cmp-lose { color: var(--text-lo); }

/* ── Alert box ── */
.alert-box {
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 5px solid;
    margin-top: .8rem;
    font-size: .93rem;
    line-height: 1.6;
}
.alert-extreme { background:#ff4b4b15; border-color:#ff4b4b; }
.alert-high    { background:#ff8c4215; border-color:#ff8c42; }
.alert-moderate{ background:#ffd16615; border-color:#ffd166; }
.alert-low     { background:#06d6a015; border-color:#06d6a0; }

/* ── XAI insight card ── */
.xai-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: .7rem;
    font-size: .9rem;
    line-height: 1.7;
}
.xai-chip {
    display: inline-block;
    padding: .15rem .55rem;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: .78rem;
    margin-right: .3rem;
}
.chip-high { background:#ff4b4b33; color:#ff8a8a; }
.chip-med  { background:#ffd16633; color:#ffd166; }
.chip-low  { background:#06d6a033; color:#06d6a0; }

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #118ab2, #06d6a0) !important;
    color: #fff !important;
    font-family: var(--font-head) !important;
    font-size: 1.25rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .75rem 2.5rem !important;
    width: 100% !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label {
    color: var(--text-lo) !important;
    font-size: .82rem !important;
    letter-spacing: .4px !important;
}

/* ── Tab strip ── */
[data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: 8px !important;
    gap: 4px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="tab"] {
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-size: .88rem !important;
    color: var(--text-lo) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--bg-card) !important;
    color: var(--text-hi) !important;
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly .bg { fill: var(--bg-card) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
    border-radius: 6px !important;
    font-size: .85rem !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOCATION DATA
# ─────────────────────────────────────────────────────────────────────────────
LOCATIONS: dict[str, dict] = {
    "Zamboanga City":   {"lat": 6.9214,  "lon": 122.0790, "region": "Region IX",    "pop": 977234},
    "Davao City":       {"lat": 7.0731,  "lon": 125.6128, "region": "Region XI",    "pop": 1776949},
    "Cagayan de Oro":   {"lat": 8.4822,  "lon": 124.6472, "region": "Region X",     "pop": 728402},
    "Manila":           {"lat": 14.5995, "lon": 120.9842, "region": "NCR",          "pop": 1846513},
    "Cebu City":        {"lat": 10.3157, "lon": 123.8854, "region": "Region VII",   "pop": 964169},
    "General Santos":   {"lat": 6.1164,  "lon": 125.1716, "region": "Region XII",   "pop": 697315},
}

RISK_LEVELS = ["Low", "Moderate", "High", "Extreme"]

RISK_META = {
    "Low":      {"emoji": "🟢", "css": "risk-low",      "alert_css": "alert-low",
                 "color": "#06d6a0", "threshold": 0},
    "Moderate": {"emoji": "🟡", "css": "risk-moderate", "alert_css": "alert-moderate",
                 "color": "#ffd166", "threshold": 0.35},
    "High":     {"emoji": "🟠", "css": "risk-high",     "alert_css": "alert-high",
                 "color": "#ff8c42", "threshold": 0.65},
    "Extreme":  {"emoji": "🔴", "css": "risk-extreme",  "alert_css": "alert-extreme",
                 "color": "#ff4b4b", "threshold": 0.85},
}


# ─────────────────────────────────────────────────────────────────────────────
# ← BACKEND HOOKS  (replace with real module calls)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_weather(location: str) -> dict:
    """
    ← BACKEND HOOK
    Replace with:  DataFetchModule(location=location).fetch_all()
    Returns a dict of current weather values for the selected city.
    """
    rng = random.Random(location + str(datetime.now().hour))
    base_temp = {"Zamboanga City": 31.5, "Davao City": 30.8,
                 "Cagayan de Oro": 31.2, "Manila": 33.4,
                 "Cebu City": 32.1, "General Santos": 30.9}.get(location, 31.0)
    T   = base_temp + rng.gauss(0, 1.2)
    RH  = rng.uniform(65, 88)
    WS  = rng.uniform(1.2, 4.8)
    UHI = rng.uniform(0.8, 3.2)
    PM  = rng.uniform(12, 55)

    def _hi(T, rh):
        Tf = T * 9/5 + 32
        HI = (-42.379 + 2.049*Tf + 10.143*rh - 0.225*Tf*rh
              - 0.00684*Tf**2 - 0.0548*rh**2
              + 0.00123*Tf**2*rh + 0.000853*Tf*rh**2
              - 0.00000199*Tf**2*rh**2)
        return (HI - 32) * 5/9

    def _wb(T, rh):
        return (T * math.atan(0.151977*(rh+8.313659)**0.5)
                + math.atan(T+rh) - math.atan(rh-1.676331)
                + 0.00391838*rh**1.5*math.atan(0.023101*rh) - 4.686035)

    return {
        "temperature": round(T, 1),
        "humidity":    round(RH, 1),
        "wind_speed":  round(WS, 2),
        "uhi":         round(UHI, 2),
        "pm25":        round(PM, 1),
        "heat_index":  round(_hi(T, RH), 1),
        "wet_bulb":    round(_wb(T, RH), 1),
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def predict_heat_risk(
    location  : str,
    days      : int,
    model_type: str,    # "Baseline LSTM" | "Impact-Centric LSTM" | "Compare Both"
) -> dict:
    """
    ← BACKEND HOOK
    Replace with:
        pipeline = DataPipeline(sequence_length=24, forecast_horizon=days)
        data = pipeline.prepare(weather_df, impact_df, mode='impact')
        model = LSTMModel(sequence_length=24, forecast_horizon=days)
        model.build_impact_centric()
        model.load_weights('outputs/impact_final.weights.h5')
        probs = model.predict(data['X_test'])
        return {'probabilities': probs, 'risk_level': ...}
    """
    rng  = random.Random(location + str(days) + model_type)
    dates = [datetime.now().date() + timedelta(days=i) for i in range(days)]
    base  = {"Zamboanga City": 31.5, "Davao City": 30.8,
             "Cagayan de Oro": 31.2, "Manila": 33.4,
             "Cebu City": 32.1, "General Santos": 30.9}.get(location, 31)

    def _series(mean, std, days):
        return [round(mean + rng.gauss(0, std) + i*0.15, 2)
                for i in range(days)]

    temp_b   = _series(base, 1.1, days)
    temp_i   = [round(t + rng.gauss(0.3, 0.4), 2) for t in temp_b]
    wb_b     = _series(base - 4.5, 0.8, days)
    wb_i     = [round(w + rng.gauss(0.2, 0.3), 2) for w in wb_b]
    hi_b     = _series(base + 3.5, 1.4, days)
    hi_i     = [round(h + rng.gauss(0.5, 0.5), 2) for h in hi_b]
    uhi_b    = _series(1.5, 0.3, days)
    uhi_i    = _series(1.8, 0.3, days)
    pm25_i   = _series(28, 6, days)
    prob_b   = [min(1.0, max(0.0, rng.gauss(0.32, 0.18))) for _ in range(days)]
    prob_i   = [min(1.0, max(0.0, p + rng.gauss(0.08, 0.05))) for p in prob_b]

    def _risk(p):
        if p < 0.35: return "Low"
        if p < 0.65: return "Moderate"
        if p < 0.85: return "High"
        return "Extreme"

    peak_prob = max(prob_i)

    return {
        "dates"         : [str(d) for d in dates],
        "baseline": {
            "temperature" : temp_b,
            "wet_bulb"    : wb_b,
            "heat_index"  : hi_b,
            "uhi"         : uhi_b,
            "probabilities": prob_b,
            "risk_levels" : [_risk(p) for p in prob_b],
            "metrics": {
                "RMSE": 0.2841, "MAE": 0.1973, "Accuracy": 0.8124,
                "F1":   0.7102, "Precision": 0.7340, "Recall": 0.6880,
                "AUC-ROC": 0.8651,
            },
        },
        "impact": {
            "temperature" : temp_i,
            "wet_bulb"    : wb_i,
            "heat_index"  : hi_i,
            "uhi"         : uhi_i,
            "pm25"        : pm25_i,
            "probabilities": prob_i,
            "risk_levels" : [_risk(p) for p in prob_i],
            "metrics": {
                "RMSE": 0.2214, "MAE": 0.1512, "Accuracy": 0.8792,
                "F1":   0.8307, "Precision": 0.8541, "Recall": 0.8088,
                "AUC-ROC": 0.9263,
            },
        },
        "overall_risk"  : _risk(peak_prob),
        "peak_prob"     : round(peak_prob, 3),
    }


@st.cache_data(ttl=600, show_spinner=False)
def get_shap_importance(model_type: str) -> pd.DataFrame:
    """
    ← BACKEND HOOK
    Replace with:
        xai = XAIModule(model=impact_lstm.model, feature_cols=features)
        xai.compute_shap(X_test, n_background=50, n_explain=100)
        return xai.rank_features()
    """
    if "Baseline" in model_type:
        features = ["Temperature", "Humidity", "Wind Speed"]
        shap     = [0.0412, 0.0287, 0.0141]
        categories = ["Conventional"] * len(features)
    else:
        features = ["Wet-bulb Temp", "Heat Index", "UHI Intensity",
                    "Temperature", "PM2.5", "Humidity", "Wind Speed"]
        shap     = [0.0891, 0.0742, 0.0634, 0.0521, 0.0388, 0.0274, 0.0132]
        categories = ["Impact", "Impact", "Impact", "Conventional",
                      "Impact", "Conventional", "Conventional"]

    return pd.DataFrame({
        "Feature"     : features,
        "Mean |SHAP|" : shap,
        "Normalised"  : [round(s / sum(shap), 4) for s in shap],
        "Category"    : categories,
    })


def generate_csv_report(location, days, weather, results) -> bytes:
    """Build a downloadable CSV summary report."""
    rows = []
    for i, date in enumerate(results["dates"]):
        rows.append({
            "Date"            : date,
            "Location"        : location,
            "Temp (°C)"       : results["impact"]["temperature"][i],
            "Wet-bulb (°C)"   : results["impact"]["wet_bulb"][i],
            "Heat Index (°C)" : results["impact"]["heat_index"][i],
            "UHI (°C)"        : results["impact"]["uhi"][i],
            "PM2.5 (μg/m³)"   : results["impact"]["pm25"][i],
            "Risk Level"      : results["impact"]["risk_levels"][i],
            "Probability"     : results["impact"]["probabilities"][i],
        })
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor = "#1a2235",
    plot_bgcolor  = "#1a2235",
    font          = dict(family="DM Sans", color="#f0f4ff", size=12),
    margin        = dict(l=16, r=16, t=36, b=16),
    legend        = dict(bgcolor="#111827", bordercolor="#1e2d45",
                         borderwidth=1, font=dict(size=11)),
    xaxis         = dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                         tickfont=dict(size=10)),
    yaxis         = dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                         tickfont=dict(size=10)),
)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.5rem 0 1.2rem;'>
      <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;
                  letter-spacing:3px;color:#f0f4ff;'>IMPACTGUARD</div>
      <div style='font-size:.72rem;color:#7a8ba6;letter-spacing:.6px;
                  margin-top:2px;'>HEAT EARLY WARNING SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📍 Location**")
    location = st.selectbox(
        "Select city", list(LOCATIONS.keys()), index=0, label_visibility="collapsed"
    )
    loc_meta = LOCATIONS[location]
    st.caption(f"📌 {loc_meta['region']}  ·  Pop. {loc_meta['pop']:,}")

    st.markdown("**🕐 Forecast Horizon**")
    forecast_days = st.slider(
        "Days ahead", 1, 7, 3, label_visibility="collapsed"
    )
    st.caption(f"Predicting next **{forecast_days} day(s)**")

    st.markdown("**🤖 Model**")
    model_choice = st.radio(
        "Model type",
        ["Impact-Centric LSTM", "Baseline LSTM", "Compare Both"],
        label_visibility="collapsed",
    )

    st.markdown("**⚙️ Alert Threshold**")
    threshold = st.slider(
        "Risk probability threshold", 0.30, 0.90, 0.50, 0.05,
        label_visibility="collapsed",
        help="Probabilities above this value trigger an alert."
    )

    st.markdown("---")
    run_btn = st.button("🚀  RUN PREDICTION", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:#7a8ba6;line-height:1.7;'>
    <b style='color:#f0f4ff;'>Thesis</b><br>
    Enhancing Extreme Heat Prediction Using Impact-Centric Variables<br><br>
    <b style='color:#f0f4ff;'>Institution</b><br>
    Ateneo de Zamboanga University<br><br>
    <b style='color:#f0f4ff;'>Backend Modules</b><br>
    DataPipeline · LSTMModel<br>
    EvaluationModule · XAIModule<br>
    AlertModule · DataFetchModule
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🌡️ ImpactGuard</div>
  <div class="app-subtitle">
    Extreme Heat Early Warning System &nbsp;·&nbsp;
    Using Impact-Centric LSTM Models (UHI + Wet-bulb + PM2.5)
  </div>
  <span class="app-badge">THESIS DEMO v1.0 · ADZU CS · ZAMBOANGA CITY</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "results"      not in st.session_state: st.session_state.results      = None
if "weather"      not in st.session_state: st.session_state.weather      = None
if "last_location"not in st.session_state: st.session_state.last_location= None
if "last_days"    not in st.session_state: st.session_state.last_days    = None
if "last_model"   not in st.session_state: st.session_state.last_model   = None


# ─────────────────────────────────────────────────────────────────────────────
# IDLE STATE  (nothing run yet)
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn and st.session_state.results is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card" style="text-align:left;padding:1.4rem 1.5rem;">
          <div style="font-size:2rem;margin-bottom:.5rem;">🏙️</div>
          <div style="font-family:'Bebas Neue';font-size:1.1rem;letter-spacing:2px;
                      margin-bottom:.5rem;color:#f0f4ff;">MULTI-CITY COVERAGE</div>
          <div style="font-size:.85rem;color:#7a8ba6;line-height:1.7;">
            Select from 6 Philippine cities. Architecture supports any global
            coordinate via ERA5 / OpenAQ APIs.
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card" style="text-align:left;padding:1.4rem 1.5rem;">
          <div style="font-size:2rem;margin-bottom:.5rem;">🧠</div>
          <div style="font-family:'Bebas Neue';font-size:1.1rem;letter-spacing:2px;
                      margin-bottom:.5rem;color:#f0f4ff;">DUAL LSTM MODELS</div>
          <div style="font-size:.85rem;color:#7a8ba6;line-height:1.7;">
            Baseline (temp/humidity/wind) vs. Impact-Centric (+UHI, wet-bulb,
            PM2.5, heat index). SHAP explainability included.
          </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card" style="text-align:left;padding:1.4rem 1.5rem;">
          <div style="font-size:2rem;margin-bottom:.5rem;">📡</div>
          <div style="font-family:'Bebas Neue';font-size:1.1rem;letter-spacing:2px;
                      margin-bottom:.5rem;color:#f0f4ff;">REAL-TIME ALERTS</div>
          <div style="font-size:.85rem;color:#7a8ba6;line-height:1.7;">
            Color-coded risk badges (Low → Extreme) with recommended
            public health actions for PAGASA/DOH coordination.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:3rem 0 1rem;
                color:#7a8ba6;font-size:.9rem;letter-spacing:.5px;">
      ← Select a city, horizon, and model in the sidebar, then click
      <b style="color:#06d6a0;">RUN PREDICTION</b>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("🔄  Fetching weather data and running LSTM inference…"):
        time.sleep(0.6)   # simulate network + model latency
        st.session_state.weather = fetch_current_weather(location)
        time.sleep(0.8)
        st.session_state.results = predict_heat_risk(location, forecast_days, model_choice)
        st.session_state.last_location = location
        st.session_state.last_days     = forecast_days
        st.session_state.last_model    = model_choice

# Aliases for readability below
weather = st.session_state.weather
results = st.session_state.results
loc     = st.session_state.last_location or location
days    = st.session_state.last_days     or forecast_days
model   = st.session_state.last_model    or model_choice

if results is None:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CURRENT CONDITIONS STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="section-head">📍 Current Conditions — {loc}
  <span style="font-size:.8rem;font-family:'DM Sans';font-weight:400;
               color:#7a8ba6;margin-left:.8rem;">as of {weather['timestamp']}</span>
</div>""", unsafe_allow_html=True)

cols = st.columns(7)
metrics = [
    ("Temperature", f"{weather['temperature']}", "°C"),
    ("Humidity",    f"{weather['humidity']}", "%"),
    ("Wind Speed",  f"{weather['wind_speed']}", "m/s"),
    ("UHI Intensity", f"{weather['uhi']}", "°C"),
    ("PM 2.5",      f"{weather['pm25']}", "μg/m³"),
    ("Heat Index",  f"{weather['heat_index']}", "°C"),
    ("Wet-bulb",    f"{weather['wet_bulb']}", "°C"),
]
for col, (label, val, unit) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OVERALL RISK BADGE
# ─────────────────────────────────────────────────────────────────────────────
risk       = results["overall_risk"]
peak       = results["peak_prob"]
rm         = RISK_META[risk]

st.markdown("<br>", unsafe_allow_html=True)
lc, mc, rc = st.columns([1, 2, 1])
with mc:
    st.markdown(f"""
    <div class="risk-badge {rm['css']}">
      {rm['emoji']}&nbsp; {risk.upper()} HEAT RISK
      <span style="font-size:1rem;margin-left:.5rem;
                   font-family:'DM Mono';">p={peak:.2f}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Prediction Results",
    "⚖️ Model Comparison",
    "🔍 Explainable AI",
    "🚨 Alerts & Recommendations",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · PREDICTION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<div class='section-head'>📅 {days}-Day Forecast — {loc}</div>",
                unsafe_allow_html=True)

    # ── Forecast table ─────────────────────────────────────────────────────
    src = results["impact"] if model != "Baseline LSTM" else results["baseline"]
    forecast_df = pd.DataFrame({
        "Date"           : results["dates"],
        "Temp (°C)"      : src["temperature"],
        "Wet-bulb (°C)"  : src["wet_bulb"],
        "Heat Index (°C)": src["heat_index"],
        "UHI (°C)"       : src["uhi"],
        **({"PM2.5 (μg/m³)": src["pm25"]} if model != "Baseline LSTM" else {}),
        "Risk"           : src["risk_levels"],
        "Probability"    : [f"{p:.2f}" for p in src["probabilities"]],
    })

    def _colour_risk(val):
        clr = {"Low": "#06d6a0", "Moderate": "#ffd166",
               "High": "#ff8c42", "Extreme": "#ff4b4b"}.get(val, "#7a8ba6")
        return f"color: {clr}; font-weight:600"

    st.dataframe(
        forecast_df.style.map(_colour_risk, subset=["Risk"]),
        use_container_width=True, hide_index=True,
    )

    # ── Temperature & Wet-bulb trend ────────────────────────────────────────
    st.markdown("<div class='section-head'>📉 Temperature & Wet-bulb Trend</div>",
                unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=results["dates"], y=src["temperature"],
        name="Temperature", mode="lines+markers",
        line=dict(color="#ff8c42", width=2.5),
        marker=dict(size=7),
    ))
    fig_trend.add_trace(go.Scatter(
        x=results["dates"], y=src["wet_bulb"],
        name="Wet-bulb Temp", mode="lines+markers",
        line=dict(color="#118ab2", width=2.5, dash="dot"),
        marker=dict(size=7),
    ))
    fig_trend.add_trace(go.Scatter(
        x=results["dates"], y=src["heat_index"],
        name="Heat Index", mode="lines+markers",
        line=dict(color="#ff4b4b", width=1.8, dash="dash"),
        marker=dict(size=5),
    ))
    # Danger zone band (shape + annotation for broad Plotly compatibility)
    fig_trend.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, x1=1, y0=40, y1=60,
        fillcolor="#ff4b4b", opacity=0.07,
        line=dict(width=0),
        layer="below",
    )
    fig_trend.add_annotation(
        xref="paper", yref="y",
        x=0.99, y=58,
        text="DANGER >= 40°C",
        showarrow=False,
        font=dict(color="#ff4b4b", size=10),
        xanchor="right",
    )
    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="Temperature (°C)",
        height=320,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Risk probability bar ────────────────────────────────────────────────
    st.markdown("<div class='section-head'>📊 Daily Risk Probability</div>",
                unsafe_allow_html=True)

    probs  = src["probabilities"]
    colors = [RISK_META[r]["color"] for r in src["risk_levels"]]
    fig_bar = go.Figure(go.Bar(
        x=results["dates"], y=probs,
        marker_color=colors, text=[f"{p:.0%}" for p in probs],
        textposition="outside", textfont=dict(size=11),
    ))
    fig_bar.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1, y0=threshold, y1=threshold,
        line=dict(color="#7a8ba6", dash="dot", width=1),
    )
    fig_bar.add_annotation(
        xref="paper", yref="y",
        x=0.99, y=threshold,
        text=f"Threshold ({threshold:.0%})",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10, color="#7a8ba6"),
    )
    fig_bar.update_layout(**PLOTLY_LAYOUT, yaxis_title="P(Extreme Heat)", height=270,
                          yaxis_range=[0, 1.15])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Map ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>🗺️ Location Map</div>",
                unsafe_allow_html=True)
    map_df = pd.DataFrame([{
        "lat": loc_meta["lat"], "lon": loc_meta["lon"],
        "city": loc, "risk": risk,
    }])
    st.map(map_df, zoom=9, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-head'>⚖️ Baseline vs. Impact-Centric LSTM</div>",
                unsafe_allow_html=True)

    b_m = results["baseline"]["metrics"]
    i_m = results["impact"]["metrics"]

    metric_keys = ["RMSE", "MAE", "Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]
    lower_better = {"RMSE", "MAE"}

    # Metric cards row
    cols2 = st.columns(len(metric_keys))
    for col, key in zip(cols2, metric_keys):
        bv, iv = b_m[key], i_m[key]
        imp_better = (iv < bv) if key in lower_better else (iv > bv)
        delta = iv - bv
        arrow = "▲" if imp_better else "▼"
        d_col = "#06d6a0" if imp_better else "#ff4b4b"
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{key}</div>
              <div class="metric-value" style="font-size:1.5rem;">{iv:.4f}</div>
              <div style="font-size:.78rem;color:{d_col};margin-top:.3rem;">
                {arrow} {abs(delta):.4f} vs baseline
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side radar chart
    fig_radar = go.Figure()
    cats  = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]
    b_vals = [b_m[c] for c in cats] + [b_m[cats[0]]]
    i_vals = [i_m[c] for c in cats] + [i_m[cats[0]]]
    cats_closed = cats + [cats[0]]

    fig_radar.add_trace(go.Scatterpolar(
        r=b_vals, theta=cats_closed, fill="toself",
        name="Baseline LSTM",
        line=dict(color="#7a8ba6"), fillcolor="rgba(122,139,166,.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=i_vals, theta=cats_closed, fill="toself",
        name="Impact-Centric",
        line=dict(color="#06d6a0"), fillcolor="rgba(6,214,160,.15)",
    ))
    fig_radar.update_layout(
        paper_bgcolor="#1a2235", plot_bgcolor="#1a2235",
        font=dict(family="DM Sans", color="#f0f4ff"),
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[0, 1], color="#7a8ba6",
                            gridcolor="#1e2d45"),
            angularaxis=dict(color="#7a8ba6", gridcolor="#1e2d45"),
        ),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d45", borderwidth=1),
        height=380,
        margin=dict(l=60, r=60, t=36, b=16),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Probability comparison line chart
    st.markdown("<div class='section-head'>📉 Risk Probability: Baseline vs. Impact-Centric</div>",
                unsafe_allow_html=True)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=results["dates"], y=results["baseline"]["probabilities"],
        name="Baseline LSTM", mode="lines+markers",
        line=dict(color="#7a8ba6", width=2, dash="dot"),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=results["dates"], y=results["impact"]["probabilities"],
        name="Impact-Centric", mode="lines+markers",
        line=dict(color="#06d6a0", width=2.5),
    ))
    fig_cmp.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1, y0=threshold, y1=threshold,
        line=dict(color="rgba(255, 75, 75, 0.53)", dash="dash", width=1),
    )
    fig_cmp.add_annotation(
        xref="paper", yref="y",
        x=0.99, y=threshold,
        text="Alert threshold",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10, color="rgba(255, 75, 75, 0.53)"),
    )
    fig_cmp.update_layout(**PLOTLY_LAYOUT, yaxis_title="P(Extreme Heat)", height=280)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Full comparison table
    st.markdown("<div class='section-head'>📋 Full Metrics Table</div>",
                unsafe_allow_html=True)
    rows = []
    for k in metric_keys:
        bv, iv = b_m[k], i_m[k]
        better_is_lower = k in lower_better
        imp_wins = (iv < bv) if better_is_lower else (iv > bv)
        rows.append({
            "Metric"            : k,
            "Baseline LSTM"     : f"{bv:.4f}",
            "Impact-Centric"    : f"{iv:.4f}",
            "Δ"                 : f"{iv-bv:+.4f}",
            "Winner"            : "🟢 Impact-Centric" if imp_wins else "⚪ Baseline",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-head'>🔍 SHAP Feature Importance</div>",
                unsafe_allow_html=True)
    st.caption(
        "SHAP (SHapley Additive exPlanations) quantifies each feature's "
        "contribution to the model's extreme heat predictions."
    )

    shap_df = get_shap_importance(model)

    # Bar chart
    color_map = {"Impact": "#06d6a0", "Conventional": "#118ab2"}
    bar_colors = [color_map.get(c, "#7a8ba6") for c in shap_df["Category"]]

    fig_shap = go.Figure(go.Bar(
        x=shap_df["Mean |SHAP|"],
        y=shap_df["Feature"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in shap_df["Mean |SHAP|"]],
        textposition="outside",
        textfont=dict(size=11, color="#f0f4ff"),
    ))
    shap_layout = dict(PLOTLY_LAYOUT)
    shap_layout["yaxis"] = dict(shap_layout["yaxis"], autorange="reversed")
    fig_shap.update_layout(
        **shap_layout, height=max(260, len(shap_df) * 48),
        xaxis_title="Mean |SHAP value|",
    )
    # Legend annotation
    fig_shap.add_annotation(
        x=max(shap_df["Mean |SHAP|"]) * 0.98, y=len(shap_df) - 0.5,
        text="🟢 Impact-centric  🔵 Conventional",
        showarrow=False, font=dict(size=10, color="#7a8ba6"),
        xanchor="right",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── NL Explanation ──────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>💬 Natural Language Explanation</div>",
                unsafe_allow_html=True)

    top3 = shap_df.head(3)
    t1, t2, t3 = top3.iloc[0], top3.iloc[1], top3.iloc[2]

    impact_pct = shap_df[shap_df["Category"] == "Impact"]["Normalised"].sum() * 100

    st.markdown(f"""
    <div class="xai-card">
      <b>Why did the model predict <span style="color:{rm['color']};">{risk} risk</span>
      for {loc}?</b><br><br>

      The Impact-Centric LSTM assigned the highest importance to
      <span class="xai-chip chip-high">{t1['Feature']}</span>
      (SHAP = {t1['Mean |SHAP|']:.4f}), followed by
      <span class="xai-chip chip-med">{t2['Feature']}</span>
      (SHAP = {t2['Mean |SHAP|']:.4f}) and
      <span class="xai-chip chip-low">{t3['Feature']}</span>
      (SHAP = {t3['Mean |SHAP|']:.4f}).<br><br>

      Together, impact-centric variables (UHI, wet-bulb temperature, and PM2.5)
      account for <b style="color:#06d6a0;">{impact_pct:.1f}%</b> of total
      feature importance — demonstrating their superior explanatory power
      compared to conventional meteorological inputs alone.<br><br>

      <b>Key insight:</b> Wet-bulb temperature captures the combined
      physiological stress of heat and humidity better than dry-bulb
      temperature alone, making it the most critical predictor of extreme
      heat health risk.
    </div>
    """, unsafe_allow_html=True)

    # Contribution pie chart
    fig_pie = px.pie(
        shap_df, values="Normalised", names="Feature",
        color="Category",
        color_discrete_map={"Impact": "#06d6a0", "Conventional": "#118ab2"},
        hole=0.45,
    )
    fig_pie.update_traces(textfont_size=11, textinfo="label+percent")
    fig_pie.update_layout(
        paper_bgcolor="#1a2235",
        font=dict(family="DM Sans", color="#f0f4ff"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d45", borderwidth=1),
        height=340, margin=dict(l=16, r=16, t=30, b=16),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # SHAP table
    st.markdown("<div class='section-head'>📋 SHAP Values Table</div>",
                unsafe_allow_html=True)
    st.dataframe(shap_df.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · ALERTS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-head'>🚨 Alert Status</div>",
                unsafe_allow_html=True)

    alert_messages = {
        "Low": {
            "headline": "✅ Normal Conditions — No Advisory Issued",
            "body": (
                f"Heat risk in {loc} is currently LOW (p={peak:.2f}). "
                "Standard public guidance applies. Continue monitoring "
                "conditions over the forecast period."
            ),
            "actions": [
                "Stay hydrated — drink at least 8 glasses of water daily.",
                "Avoid prolonged sun exposure between 10 AM and 4 PM.",
                "Monitor PAGASA advisories for any updates.",
            ],
        },
        "Moderate": {
            "headline": "⚠️ HEAT ADVISORY — Moderate Risk",
            "body": (
                f"A moderate heat event is forecast for {loc} within the "
                f"next {days} day(s) (peak p={peak:.2f}). Vulnerable "
                "populations should take precautions."
            ),
            "actions": [
                "Increase water intake; avoid alcohol and caffeinated drinks.",
                "Wear light, loose-fitting clothing and sunscreen (SPF 30+).",
                "Limit strenuous outdoor activities during peak heat hours.",
                "Check on elderly relatives and young children frequently.",
                "Open cooling centres in barangay halls for vulnerable residents.",
            ],
        },
        "High": {
            "headline": "🔴 HEAT WARNING — High Risk",
            "body": (
                f"A HIGH heat risk event is imminent for {loc} "
                f"(peak p={peak:.2f}). The City Health Office and CDRRMO "
                "should activate heat emergency protocols immediately."
            ),
            "actions": [
                "🏥 Activate City Health Office heat emergency protocols.",
                "🏫 Open school gymnasiums and barangay halls as cooling centres.",
                "📢 Issue public advisories via local radio, TV, and social media.",
                "🚑 Pre-position health rapid response teams at key areas.",
                "⚠️ Cancel or postpone outdoor public events.",
                "💧 Ensure adequate supply of clean drinking water in public spaces.",
                "Suspend outdoor work activities during 11 AM – 3 PM.",
            ],
        },
        "Extreme": {
            "headline": "🚨 EXTREME HEAT EMERGENCY — IMMEDIATE ACTION REQUIRED",
            "body": (
                f"EXTREME heat event forecast for {loc} (peak p={peak:.2f}). "
                "This is a life-threatening situation. Coordinate with DOH, "
                "NDRRMC, and local government for immediate emergency response."
            ),
            "actions": [
                "🚨 Declare Local State of Calamity if conditions persist.",
                "🏥 Alert all hospitals and health centres for heat stroke cases.",
                "📻 Issue mandatory advisories on all media platforms.",
                "🏫 Convert all public buildings into 24/7 cooling centres.",
                "🚑 Deploy emergency medical teams to high-density barangays.",
                "🏘️ Conduct door-to-door wellness checks for elderly and PWDs.",
                "💧 Emergency water distribution in affected communities.",
                "⛔ Suspend all outdoor construction and non-essential work.",
                "📡 Notify DOH Region IX and NDRRMC immediately.",
            ],
        },
    }

    am  = alert_messages[risk]
    css = rm["alert_css"]

    st.markdown(f"""
    <div class="alert-box {css}">
      <b style="font-size:1.1rem;">{am['headline']}</b><br><br>
      {am['body']}
    </div>
    """, unsafe_allow_html=True)

    # Actions
    st.markdown("<div class='section-head'>📋 Recommended Actions</div>",
                unsafe_allow_html=True)

    for i, action in enumerate(am["actions"], 1):
        icon = "🔴" if risk == "Extreme" else ("🟠" if risk == "High"
               else ("🟡" if risk == "Moderate" else "🟢"))
        st.markdown(f"""
        <div class="xai-card" style="padding:.85rem 1.2rem;margin-bottom:.4rem;">
          {icon} &nbsp;<b>{i}.</b> {action}
        </div>""", unsafe_allow_html=True)

    # Daily alert table
    st.markdown("<div class='section-head'>📅 Day-by-Day Alert Schedule</div>",
                unsafe_allow_html=True)

    src_p = results["impact"] if model != "Baseline LSTM" else results["baseline"]
    alert_rows = []
    for i, (date, r_lv, prob) in enumerate(zip(
        results["dates"],
        src_p["risk_levels"],
        src_p["probabilities"]
    )):
        triggered = prob >= threshold
        alert_rows.append({
            "Date"      : date,
            "Risk Level": r_lv,
            "Probability": f"{prob:.2f}",
            "Alert"     : "⚠️ TRIGGERED" if triggered else "✅ Normal",
            "Actions"   : "Activate protocol" if triggered else "Monitor only",
        })
    alert_df = pd.DataFrame(alert_rows)

    def _style_alert(val):
        return "color:#ff4b4b;font-weight:700" if "TRIGGERED" in str(val) else "color:#06d6a0"

    st.dataframe(
        alert_df.style.map(_style_alert, subset=["Alert"]),
        use_container_width=True, hide_index=True,
    )

    # ── Download Report ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>⬇️ Download Report</div>",
                unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3)
    csv_bytes = generate_csv_report(loc, days, weather, results)

    with dl1:
        st.download_button(
            label     = "📊 Download CSV Report",
            data      = csv_bytes,
            file_name = f"ImpactGuard_{loc.replace(' ','_')}_{datetime.now():%Y%m%d}.csv",
            mime      = "text/csv",
            use_container_width=True,
        )

    # JSON export
    import json
    json_bytes = json.dumps({
        "location"    : loc,
        "forecast_days": days,
        "model"       : model,
        "weather"     : weather,
        "overall_risk": risk,
        "peak_prob"   : peak,
        "dates"       : results["dates"],
        "impact_model": {
            "risk_levels"  : results["impact"]["risk_levels"],
            "probabilities": results["impact"]["probabilities"],
            "metrics"      : results["impact"]["metrics"],
        },
        "shap_top3"   : get_shap_importance(model).head(3).to_dict(orient="records"),
        "generated_at": datetime.now().isoformat(),
    }, indent=2).encode()

    with dl2:
        st.download_button(
            label     = "📄 Download JSON Report",
            data      = json_bytes,
            file_name = f"ImpactGuard_{loc.replace(' ','_')}_{datetime.now():%Y%m%d}.json",
            mime      = "application/json",
            use_container_width=True,
        )

    with dl3:
        # SHAP CSV
        shap_csv = get_shap_importance(model).to_csv(index=False).encode()
        st.download_button(
            label     = "🔍 Download SHAP Data",
            data      = shap_csv,
            file_name = f"SHAP_{model.replace(' ','_')}_{datetime.now():%Y%m%d}.csv",
            mime      = "text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;
            border-top:1px solid #1e2d45;margin-top:2rem;">
  <div style="font-family:'Bebas Neue';font-size:1.1rem;letter-spacing:3px;
              color:#f0f4ff;margin-bottom:.4rem;">IMPACTGUARD v1.0</div>
  <div style="font-size:.78rem;color:#7a8ba6;line-height:1.8;">
    Ateneo de Zamboanga University · BS Computer Science Undergraduate Thesis<br>
    "Enhancing Extreme Heat Prediction Using Impact-Centric Variables in ML Models"<br>
    Backend: TensorFlow/Keras LSTM · SHAP KernelExplainer · DataPipeline (Pandas/NumPy)
  </div>
</div>
""", unsafe_allow_html=True)