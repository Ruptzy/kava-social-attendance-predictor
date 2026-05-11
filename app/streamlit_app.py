"""
KavaCast - Kava Social Chess Attendance Predictor.

A "luxury tournament lounge command center" Streamlit dashboard.

Design synthesis (per the user's two reference images):
- UI/UX inspiration: a dark analytics dashboard with a top KPI row, chart-card
  grid, intentional sidebar, and tabbed sub-views.
- Mood / palette inspiration: Rolex / Porsche / cocktail-lounge --
  near-black green base, deep green surfaces, burgundy richness, bronze
  metallic accents, soft silver typography. Quiet confidence; nothing neon.

Run locally:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_for_date, load_models  # noqa: E402


# ----------------------------------------------------------------------------
# Page config & color tokens
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="KavaCast · Kava Social Chess Attendance Predictor",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette (from the user's moodboard, with derived helpers)
NEAR_BLACK = "#03110D"   # page background
DEEP_GREEN = "#16302B"   # card surface
DEEP_GREEN_2 = "#1a3933"  # subtle card highlight
SURFACE_2 = "#0a1a14"    # sidebar / panels
BURGUNDY = "#390517"     # rich state / accent
BURGUNDY_GLOW = "#5a1124"
BRONZE = "#A38560"       # primary metallic
BRONZE_BRIGHT = "#C4A77D"
BRONZE_DIM = "#6F5A41"
SILVER = "#E0E0E0"       # main text
SILVER_DIM = "#9BA09B"
SILVER_MUTE = "#6B7570"
BORDER = "rgba(163, 133, 96, 0.18)"
BORDER_STRONG = "rgba(163, 133, 96, 0.36)"


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {
            --kc-bg: #03110D;
            --kc-surface: #0a1a14;
            --kc-card: #16302B;
            --kc-card-2: #1a3933;
            --kc-burgundy: #390517;
            --kc-burgundy-glow: #5a1124;
            --kc-bronze: #A38560;
            --kc-bronze-bright: #C4A77D;
            --kc-bronze-dim: #6F5A41;
            --kc-silver: #E0E0E0;
            --kc-silver-dim: #9BA09B;
            --kc-silver-mute: #6B7570;
            --kc-border: rgba(163, 133, 96, 0.18);
            --kc-border-strong: rgba(163, 133, 96, 0.36);
        }

        html, body, .stApp, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
            color: var(--kc-silver);
        }

        .stApp {
            background:
                radial-gradient(ellipse 1100px 600px at 0% 0%, rgba(57, 5, 23, 0.18) 0%, transparent 55%),
                radial-gradient(ellipse 900px 500px at 100% 100%, rgba(163, 133, 96, 0.06) 0%, transparent 55%),
                var(--kc-bg) !important;
        }

        .block-container {
            max-width: 1500px !important;
            padding-top: 1.4rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* HERO ---------------------------------------------------------- */
        .kc-hero {
            position: relative;
            border-radius: 22px;
            padding: 2.6rem 2.6rem 2.4rem;
            margin: 0 0 1.75rem;
            background:
                radial-gradient(circle at 92% -10%, rgba(163, 133, 96, 0.18) 0%, transparent 55%),
                radial-gradient(circle at 0% 110%, rgba(57, 5, 23, 0.55) 0%, transparent 55%),
                linear-gradient(135deg, #061812 0%, #0e251f 50%, var(--kc-card) 100%);
            border: 1px solid var(--kc-border);
            box-shadow:
                0 24px 60px rgba(0,0,0,0.55),
                inset 0 1px 0 rgba(224, 224, 224, 0.04);
            overflow: hidden;
        }
        .kc-hero::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, var(--kc-bronze-dim) 18%, var(--kc-bronze-bright) 50%, var(--kc-bronze-dim) 82%, transparent 100%);
        }
        .kc-hero::after {
            content: "";
            position: absolute;
            bottom: -120px; right: -80px;
            width: 360px; height: 360px;
            background: radial-gradient(circle, rgba(163, 133, 96, 0.06), transparent 70%);
            pointer-events: none;
        }
        .kc-eyebrow {
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.28em;
            color: var(--kc-bronze);
            font-weight: 500;
            margin-bottom: 0.75rem;
        }
        .kc-eyebrow::before {
            content: "—— ";
            color: var(--kc-bronze-dim);
            margin-right: 0.35rem;
        }
        .kc-title {
            font-size: 3.4rem;
            font-weight: 700;
            letter-spacing: -0.035em;
            margin: 0;
            color: var(--kc-silver);
            line-height: 1.0;
        }
        .kc-title .kc-title-accent {
            color: var(--kc-bronze);
            font-weight: 300;
        }
        .kc-subtitle {
            font-size: 1.05rem;
            font-weight: 500;
            color: var(--kc-bronze-bright);
            margin: 0.6rem 0 1.05rem;
            letter-spacing: 0.005em;
        }
        .kc-desc {
            font-size: 0.95rem;
            line-height: 1.7;
            max-width: 760px;
            color: var(--kc-silver-dim);
            margin: 0 0 1.35rem 0;
            font-weight: 400;
        }
        .kc-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .kc-badge {
            padding: 0.42rem 0.9rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 500;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .kc-badge--bronze {
            background: rgba(163, 133, 96, 0.1);
            color: var(--kc-bronze-bright);
            border: 1px solid rgba(163, 133, 96, 0.45);
        }
        .kc-badge--burgundy {
            background: rgba(57, 5, 23, 0.55);
            color: #d4a3b1;
            border: 1px solid rgba(57, 5, 23, 0.9);
        }
        .kc-badge--silver {
            background: rgba(224, 224, 224, 0.04);
            color: var(--kc-silver-dim);
            border: 1px solid rgba(224, 224, 224, 0.1);
        }

        /* SECTION HEADERS ---------------------------------------------- */
        .kc-section-h {
            display: flex; align-items: baseline; gap: 0.85rem;
            margin: 1.8rem 0 1.1rem;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid var(--kc-border);
        }
        .kc-section-h .eyebrow {
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.28em;
            color: var(--kc-bronze);
            font-weight: 500;
        }
        .kc-section-h h3 {
            margin: 0;
            font-size: 1.32rem;
            font-weight: 600;
            color: var(--kc-silver);
            letter-spacing: -0.01em;
        }

        /* METRIC CARDS ------------------------------------------------- */
        .kc-card {
            position: relative;
            background: linear-gradient(180deg, var(--kc-card) 0%, #122922 100%);
            border: 1px solid var(--kc-border);
            border-radius: 16px;
            padding: 1.35rem 1.55rem 1.55rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.32), inset 0 1px 0 rgba(224, 224, 224, 0.03);
            height: 100%;
            min-height: 178px;
            overflow: hidden;
            transition: transform 0.18s ease, border-color 0.18s ease;
        }
        .kc-card:hover {
            transform: translateY(-1px);
            border-color: var(--kc-border-strong);
        }
        .kc-card::before {
            content: "";
            position: absolute;
            top: 0; left: 1.5rem; right: 1.5rem;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--kc-bronze) 50%, transparent);
            opacity: 0.5;
        }
        .kc-card--feature::before { height: 2px; opacity: 1.0; }
        .kc-card-label {
            text-transform: uppercase;
            font-size: 0.66rem;
            letter-spacing: 0.22em;
            font-weight: 500;
            color: var(--kc-bronze);
            margin-bottom: 0.95rem;
        }
        .kc-card-value {
            font-size: 3.2rem;
            font-weight: 700;
            line-height: 1;
            color: var(--kc-silver);
            letter-spacing: -0.04em;
            font-variant-numeric: tabular-nums;
        }
        .kc-card-value-mid { font-size: 2.0rem; letter-spacing: -0.02em; }
        .kc-card-value--bronze { color: var(--kc-bronze-bright); }
        .kc-card-value--burgundy { color: #d4a3b1; }
        .kc-card-unit {
            font-size: 0.92rem;
            font-weight: 400;
            color: var(--kc-silver-mute);
            margin-left: 0.45rem;
            letter-spacing: 0.02em;
        }
        .kc-card-sub {
            margin-top: 0.9rem;
            font-size: 0.78rem;
            color: var(--kc-silver-dim);
            line-height: 1.55;
            font-weight: 400;
        }
        .kc-card-sub b { color: var(--kc-silver); font-weight: 600; }
        .kc-card-body {
            font-size: 0.92rem;
            line-height: 1.6;
            color: var(--kc-silver);
            font-weight: 400;
        }
        .kc-card-list { margin: 0.85rem 0 0; padding: 0; list-style: none; font-size: 0.82rem; }
        .kc-card-list li {
            display: flex; justify-content: space-between;
            padding: 0.4rem 0;
            border-top: 1px solid var(--kc-border);
        }
        .kc-card-list li:first-child { border-top: 0; }
        .kc-card-list .label {
            color: var(--kc-silver-dim);
            text-transform: uppercase; letter-spacing: 0.12em;
            font-size: 0.7rem; font-weight: 500;
        }
        .kc-card-list .value { color: var(--kc-bronze-bright); font-weight: 600; }

        /* Probability progress track */
        .kc-progress-track {
            height: 6px; background: rgba(224, 224, 224, 0.06); border-radius: 999px;
            margin-top: 0.95rem; overflow: hidden;
        }
        .kc-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--kc-burgundy) 0%, var(--kc-bronze) 100%);
            border-radius: 999px;
            box-shadow: 0 0 10px rgba(163, 133, 96, 0.35);
        }

        /* CHART PANEL -------------------------------------------------- */
        .kc-chart-panel {
            background: linear-gradient(180deg, var(--kc-card) 0%, #122922 100%);
            border: 1px solid var(--kc-border);
            border-radius: 16px;
            padding: 1.05rem 1.15rem 0.5rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.32), inset 0 1px 0 rgba(224, 224, 224, 0.03);
            margin-bottom: 1.05rem;
            position: relative;
        }
        .kc-chart-panel::before {
            content: "";
            position: absolute;
            top: 0; left: 1.55rem; right: 1.55rem;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(163, 133, 96, 0.4), transparent);
        }
        .kc-chart-panel h4 {
            font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.22em;
            color: var(--kc-bronze); margin: 0 0 0.25rem; font-weight: 500;
        }
        .kc-chart-caption {
            font-size: 0.78rem;
            color: var(--kc-silver-dim);
            line-height: 1.5;
            padding: 0.2rem 0.2rem 0.7rem;
        }

        /* SIDEBAR ------------------------------------------------------ */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #061812 0%, #0a1a14 100%) !important;
            border-right: 1px solid var(--kc-border) !important;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.75rem !important; padding-bottom: 2rem !important; }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: var(--kc-silver) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p { color: var(--kc-silver-dim); }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--kc-silver-mute) !important;
        }

        /* Form controls (input, slider, toggle, etc.) */
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] > div,
        .stDateInput input, .stNumberInput input, .stTextInput input {
            background: rgba(224, 224, 224, 0.04) !important;
            border: 1px solid var(--kc-border) !important;
            color: var(--kc-silver) !important;
            border-radius: 8px !important;
        }
        .stDateInput input:focus, .stNumberInput input:focus { border-color: var(--kc-bronze) !important; }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--kc-bronze-bright) !important;
            border: 2px solid var(--kc-bg) !important;
        }
        div[data-baseweb="slider"] > div > div > div { background: var(--kc-bronze) !important; }

        /* TABS --------------------------------------------------------- */
        div[data-baseweb="tab-list"] {
            gap: 0.25rem !important;
            border-bottom: 1px solid var(--kc-border) !important;
            background: transparent !important;
        }
        button[data-baseweb="tab"] {
            font-weight: 500 !important;
            color: var(--kc-silver-dim) !important;
            padding: 0.85rem 1.15rem !important;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--kc-bronze-bright) !important; }
        div[data-baseweb="tab-highlight"] { background: var(--kc-bronze) !important; height: 2px !important; }
        div[data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

        /* EXPANDERS --------------------------------------------------- */
        div[data-testid="stExpander"] {
            background: var(--kc-card) !important;
            border: 1px solid var(--kc-border) !important;
            border-radius: 12px !important;
            margin-bottom: 0.65rem !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] details > summary {
            font-weight: 600 !important;
            color: var(--kc-silver) !important;
            padding: 0.85rem 1.15rem !important;
            font-size: 0.92rem;
        }
        div[data-testid="stExpander"] svg { fill: var(--kc-bronze) !important; }

        /* TRUST NOTE --------------------------------------------------- */
        .kc-trust {
            margin-top: 0.85rem;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            background: rgba(57, 5, 23, 0.22);
            border-left: 2px solid var(--kc-burgundy-glow);
            font-size: 0.78rem;
            color: var(--kc-silver-dim);
            line-height: 1.55;
        }
        .kc-trust b { color: var(--kc-bronze-bright); }

        /* SIDEBAR SUMMARY --------------------------------------------- */
        .kc-side-summary {
            background: rgba(224, 224, 224, 0.025);
            border: 1px solid var(--kc-border);
            border-radius: 10px;
            padding: 0.9rem 1.05rem;
            font-size: 0.85rem;
            color: var(--kc-silver);
        }
        .kc-side-summary-row {
            display: flex; justify-content: space-between;
            padding: 0.3rem 0;
            border-top: 1px solid var(--kc-border);
        }
        .kc-side-summary-row:first-child { border-top: 0; }
        .kc-side-summary-row .label {
            color: var(--kc-silver-dim);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.68rem;
            font-weight: 500;
        }
        .kc-side-summary-row .value { font-weight: 600; color: var(--kc-bronze-bright); }

        /* PIPELINE ---------------------------------------------------- */
        .kc-pipeline {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.55rem;
            margin: 0.85rem 0 0;
        }
        .kc-pipeline-step {
            background: rgba(224, 224, 224, 0.03);
            border: 1px solid var(--kc-border);
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
            font-size: 0.82rem;
            color: var(--kc-silver);
            text-align: left;
            position: relative;
        }
        .kc-pipeline-step b {
            display: block;
            color: var(--kc-bronze);
            margin-bottom: 0.3rem;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-weight: 500;
        }

        /* NOTES GRID -------------------------------------------------- */
        .kc-notes-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 0.9rem; margin-top: 0.5rem;
        }
        .kc-notes-card {
            background: rgba(224, 224, 224, 0.025);
            border: 1px solid var(--kc-border);
            border-radius: 12px;
            padding: 1rem 1.15rem;
        }
        .kc-notes-card h5 {
            margin: 0 0 0.55rem;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            color: var(--kc-bronze);
            font-weight: 500;
        }
        .kc-notes-card ul { margin: 0; padding-left: 1.15rem; color: var(--kc-silver-dim); font-size: 0.88rem; }
        .kc-notes-card ul li { margin: 0.3rem 0; line-height: 1.55; }
        .kc-notes-card ul li b { color: var(--kc-silver); font-weight: 600; }

        /* STAT STRIP -------------------------------------------------- */
        .kc-stat-strip {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem;
            margin-top: 0.4rem;
        }
        .kc-stat-chip {
            background: rgba(224, 224, 224, 0.025);
            border: 1px solid var(--kc-border);
            border-radius: 12px;
            padding: 0.9rem 1.05rem;
        }
        .kc-stat-chip .label {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.64rem;
            color: var(--kc-bronze);
            font-weight: 500;
        }
        .kc-stat-chip .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--kc-silver);
            margin-top: 0.22rem;
            letter-spacing: -0.025em;
            font-variant-numeric: tabular-nums;
        }
        .kc-stat-chip .sub { font-size: 0.7rem; color: var(--kc-silver-mute); margin-top: 0.18rem; }

        /* Streamlit alert boxes */
        div[data-testid="stAlert"] {
            background: rgba(163, 133, 96, 0.08) !important;
            border: 1px solid var(--kc-border) !important;
            border-radius: 10px !important;
            color: var(--kc-silver) !important;
        }

        /* Hide Streamlit chrome */
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------------
@st.cache_resource
def _load_artifacts():
    reg, clf, history, metadata = load_models()
    history = history.copy()
    history["event_date"] = pd.to_datetime(history["event_date"])
    return reg, clf, history, metadata


# ----------------------------------------------------------------------------
# Plotly layout helper (dark, transparent so card bg shows)
# ----------------------------------------------------------------------------
def _layout(**overrides):
    base = dict(
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=SILVER, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=14, b=30),
        xaxis=dict(
            showgrid=False, color=SILVER_DIM,
            linecolor="rgba(224,224,224,0.12)", ticks="",
            tickfont=dict(color=SILVER_DIM, size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(224,224,224,0.06)", color=SILVER_DIM,
            linecolor="rgba(224,224,224,0.12)", ticks="",
            tickfont=dict(color=SILVER_DIM, size=11),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(color=SILVER_DIM, size=11),
        ),
        hoverlabel=dict(
            font=dict(family="Inter", size=12, color=SILVER),
            bgcolor=NEAR_BLACK, bordercolor=BRONZE,
        ),
    )
    base.update(overrides)
    return base


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
HERO_HTML = """
<div class="kc-hero">
  <div class="kc-eyebrow">Kava Social Chess · Bradenton, Florida</div>
  <h1 class="kc-title">Kava<span class="kc-title-accent">Cast</span></h1>
  <div class="kc-subtitle">Attendance Intelligence for Kava Social Chess Bracket Nights</div>
  <p class="kc-desc">
    A distributed data pipeline and machine-learning system that forecasts bracket-night
    turnout from historical Kava Social chess logs, calendar patterns, Bradenton weather,
    and recent community momentum. Plan boards, clocks, and staffing with confidence.
  </p>
  <div class="kc-badges">
    <span class="kc-badge kc-badge--bronze">Bradenton · FL</span>
    <span class="kc-badge kc-badge--burgundy">Attendance Forecast</span>
    <span class="kc-badge kc-badge--silver">Kava Social Chess</span>
    <span class="kc-badge kc-badge--silver">MLflow Pipeline</span>
  </div>
</div>
"""


def _section_header(eyebrow: str, title: str) -> None:
    st.markdown(
        f'<div class="kc-section-h"><span class="eyebrow">{eyebrow}</span><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
def _sidebar(history: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            '<div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.26em; '
            'color:#A38560; font-weight:500; margin-bottom:0.25rem;">Control Panel</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h2 style="margin:0 0 1.2rem; font-size:1.25rem; color:#E0E0E0; '
            'font-weight:600;">Predict a bracket night</h2>',
            unsafe_allow_html=True,
        )

        last_event = history["event_date"].max().date()
        default_date = last_event + timedelta(days=14)
        if default_date < date.today():
            default_date = date.today() + timedelta(days=7)

        target = st.date_input(
            "Bracket-night date",
            value=default_date,
            min_value=last_event + timedelta(days=1),
            max_value=date.today() + timedelta(days=180),
            help="Pick the upcoming date you want to forecast.",
        )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.22em; '
            'color:#A38560; font-weight:500; margin:0 0 0.4rem;">Weather override</div>',
            unsafe_allow_html=True,
        )
        st.caption("Leave off to use live Bradenton forecast from Open-Meteo.")
        custom_weather = st.toggle("Use custom weather", value=False)

        weather_override = None
        if custom_weather:
            temp_high_f = st.number_input(
                "Expected high (°F)", min_value=30, max_value=110, value=82, step=1,
            )
            rain_chance = st.slider("Rain chance", 0, 100, 20, 5, format="%d%%")
            humidity = st.slider("Humidity", 0, 100, 70, 5, format="%d%%")
            temp_c = (temp_high_f - 32) * 5.0 / 9.0
            weather_override = {
                "temperature_high": temp_c,
                "average_temperature": temp_c - 3.0,
                "feels_like_temperature": temp_c + (humidity - 50) / 25.0,
                "precipitation_amount": rain_chance / 20.0,
                "rain_indicator": 1 if rain_chance >= 40 else 0,
                "thunderstorm_indicator": 1 if rain_chance >= 70 else 0,
                "severe_weather_indicator": 1 if rain_chance >= 85 else 0,
            }

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.22em; '
            'color:#A38560; font-weight:500; margin:0 0 0.5rem;">Recent activity</div>',
            unsafe_allow_html=True,
        )

        latest_att = int(history["attendance_count"].iloc[-1])
        rolling3 = history["attendance_count"].tail(3).mean()
        avg = history["attendance_count"].mean()
        med = history["attendance_count"].median()
        st.markdown(
            f"""
            <div class="kc-side-summary">
              <div class="kc-side-summary-row"><span class="label">Last event</span><span class="value">{last_event:%b %d, %Y}</span></div>
              <div class="kc-side-summary-row"><span class="label">Latest attendance</span><span class="value">{latest_att}</span></div>
              <div class="kc-side-summary-row"><span class="label">Rolling 3-event avg</span><span class="value">{rolling3:.1f}</span></div>
              <div class="kc-side-summary-row"><span class="label">Average attendance</span><span class="value">{avg:.1f}</span></div>
              <div class="kc-side-summary-row"><span class="label">Median attendance</span><span class="value">{med:.1f}</span></div>
              <div class="kc-side-summary-row"><span class="label">Total events tracked</span><span class="value">{len(history)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="kc-trust"><b>Trust note.</b> KavaCast forecasts attendance only. '
            "It does not predict chess outcomes or player performance.</div>",
            unsafe_allow_html=True,
        )

    return target, weather_override


# ----------------------------------------------------------------------------
# FORECAST COMMAND CENTER
# ----------------------------------------------------------------------------
def _category_colors(cat: str) -> tuple[str, str]:
    """Return (CSS value-color class, hex color) for a turnout category."""
    if cat == "High":
        return "kc-card-value--bronze", BRONZE_BRIGHT
    if cat == "Low":
        return "kc-card-value--burgundy", "#d4a3b1"
    # Normal
    return "", SILVER


def _forecast_command_center(pred) -> None:
    _section_header("Forecast", "Forecast command center")

    boards = math.ceil(pred.predicted_attendance_rounded / 2)
    extra_board = pred.turnout_category == "High"
    boards_text = f"{boards} + 1 spare" if extra_board else str(boards)
    clocks = max(boards, 3)
    staffing = {
        "High": "Extra staffing recommended",
        "Normal": "Standard staffing",
        "Low": "Light staffing fine",
    }[pred.turnout_category]

    prob_pct = pred.high_turnout_probability * 100
    cat_value_class, _ = _category_colors(pred.turnout_category)

    # ROW 1 — three KPI cards (Fusedash-style top strip)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kc-card kc-card--feature">
              <div class="kc-card-label">Predicted Attendance</div>
              <div class="kc-card-value kc-card-value--bronze">{pred.predicted_attendance_rounded}<span class="kc-card-unit">players</span></div>
              <div class="kc-card-sub">Regressor estimate <b>{pred.predicted_attendance:.1f}</b> · forecast for <b>{pred.event_date}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kc-card">
              <div class="kc-card-label">Turnout Category</div>
              <div class="kc-card-value kc-card-value-mid {cat_value_class}">{pred.turnout_category} turnout</div>
              <div class="kc-card-sub">High-turnout threshold &mdash; attendance &ge; <b>{pred.median_attendance_threshold:.0f}</b> (historical median).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kc-card">
              <div class="kc-card-label">High-Turnout Probability</div>
              <div class="kc-card-value kc-card-value--bronze">{prob_pct:.0f}<span class="kc-card-unit">%</span></div>
              <div class="kc-progress-track"><div class="kc-progress-fill" style="width:{min(100, max(0, prob_pct)):.0f}%"></div></div>
              <div class="kc-card-sub">Classifier probability the night clears the historical median.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ROW 2 — planning recommendation, full editorial width
    st.write("")  # 4px spacer
    st.markdown(
        f"""
        <div class="kc-card kc-card--feature">
          <div class="kc-card-label">Planning Recommendation</div>
          <div style="display:flex; gap:2.5rem; align-items:flex-start; flex-wrap:wrap;">
            <div style="flex: 2 1 360px;">
              <div class="kc-card-body" style="font-size:1.02rem;">{pred.planning_note}</div>
              <div class="kc-card-sub">Inputs synthesized from {pred.model_metadata.get('n_training_events', '?')} historical bracket nights · CV MAE {pred.model_metadata.get('regression_cv_mae', float('nan')):.2f} players.</div>
            </div>
            <div style="flex: 1 1 220px; min-width:220px;">
              <ul class="kc-card-list">
                <li><span class="label">Boards to prep</span><span class="value">{boards_text}</span></li>
                <li><span class="label">Clocks recommended</span><span class="value">{clocks}</span></li>
                <li><span class="label">Staffing</span><span class="value">{staffing}</span></li>
              </ul>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------------
def _chart_panel_open(eyebrow: str) -> None:
    st.markdown(f'<div class="kc-chart-panel"><h4>{eyebrow}</h4>', unsafe_allow_html=True)


def _chart_panel_close(caption: str | None = None) -> None:
    if caption:
        st.markdown(f'<div class="kc-chart-caption">{caption}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _trend_chart(history: pd.DataFrame, pred) -> go.Figure:
    h = history.sort_values("event_date").copy()
    h["rolling_5"] = h["attendance_count"].rolling(5, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["attendance_count"],
            mode="lines+markers", name="Actual attendance",
            line=dict(color=SILVER, width=1.7),
            marker=dict(size=5, color=SILVER, line=dict(color=DEEP_GREEN, width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["rolling_5"],
            mode="lines", name="5-event rolling average",
            line=dict(color=BRONZE, width=2.6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pd.to_datetime(pred.event_date)],
            y=[pred.predicted_attendance_rounded],
            mode="markers", name="Forecast",
            marker=dict(
                color=BRONZE_BRIGHT, size=17, symbol="diamond",
                line=dict(color=BURGUNDY, width=2),
            ),
        )
    )
    fig.update_layout(**_layout(height=380, margin=dict(l=20, r=20, t=20, b=30)))
    return fig


def _smoothed_chart(history: pd.DataFrame) -> go.Figure:
    h = history.sort_values("event_date").copy()
    h["smoothed"] = h["attendance_count"].rolling(window=7, center=True, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["attendance_count"],
            mode="markers", name="Attendance",
            marker=dict(color=SILVER_DIM, size=7, opacity=0.65, line=dict(color=DEEP_GREEN, width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["smoothed"],
            mode="lines", name="Smoothed trend (7-event centered)",
            line=dict(color=BRONZE_BRIGHT, width=3),
        )
    )
    fig.update_layout(**_layout(height=340, margin=dict(l=20, r=20, t=20, b=30)))
    return fig


def _monthly_heatmap(history: pd.DataFrame) -> go.Figure:
    h = history.copy()
    h["year"] = h["event_date"].dt.year
    h["month"] = h["event_date"].dt.month
    pivot = (
        h.groupby(["year", "month"])["attendance_count"]
        .mean()
        .reset_index()
        .pivot(index="year", columns="month", values="attendance_count")
        .reindex(columns=list(range(1, 13)))
    )
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        color_continuous_scale=[
            (0.0, NEAR_BLACK),
            (0.4, DEEP_GREEN_2),
            (0.75, BRONZE_DIM),
            (1.0, BRONZE_BRIGHT),
        ],
        aspect="auto",
        labels=dict(color="Avg attendance"),
    )
    fig.update_traces(textfont=dict(color=SILVER, family="Inter", size=11))
    fig.update_layout(
        **_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_colorbar=dict(
                title="Avg", tickfont=dict(color=SILVER_DIM),
                thickness=12, outlinewidth=0,
            ),
        )
    )
    return fig


def _events_per_month_chart(history: pd.DataFrame) -> go.Figure:
    h = history.copy()
    h["month"] = h["event_date"].dt.month
    counts = h.groupby("month").size().reindex(range(1, 13), fill_value=0)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(
        go.Bar(
            x=months, y=counts.values,
            marker=dict(color=BRONZE, line=dict(color=BURGUNDY, width=1)),
        )
    )
    fig.update_layout(**_layout(height=300, margin=dict(l=20, r=20, t=20, b=30)))
    return fig


def _feature_importance_chart(reg, feature_columns: list[str]) -> go.Figure | None:
    if not hasattr(reg, "feature_importances_"):
        return None
    importances = pd.Series(reg.feature_importances_, index=feature_columns).sort_values(ascending=True)
    top = importances.tail(12)
    pretty = (
        top.index.str.replace("_", " ")
        .str.replace(" event ", " ")
        .str.title()
    )
    fig = go.Figure(
        go.Bar(
            x=top.values, y=pretty, orientation="h",
            marker=dict(color=BRONZE, line=dict(color=BURGUNDY, width=0.6)),
        )
    )
    fig.update_layout(
        **_layout(
            height=420, margin=dict(l=20, r=20, t=20, b=30),
            xaxis=dict(
                showgrid=True, gridcolor="rgba(224,224,224,0.05)",
                color=SILVER_DIM, linecolor="rgba(224,224,224,0.12)",
            ),
            yaxis=dict(
                color=SILVER, linecolor="rgba(224,224,224,0.12)", automargin=True,
                tickfont=dict(color=SILVER, size=11),
            ),
        )
    )
    return fig


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
def _render_forecast_tab(history: pd.DataFrame, pred) -> None:
    _chart_panel_open("Historical attendance with forecast")
    st.plotly_chart(
        _trend_chart(history, pred),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    _chart_panel_close(
        "The dotted bronze line is the 5-event rolling average. "
        "The bronze diamond is KavaCast's forecast for the selected date."
    )


def _render_trends_tab(history: pd.DataFrame, reg, metadata: dict) -> None:
    _chart_panel_open("Smoothed attendance trend")
    st.plotly_chart(
        _smoothed_chart(history),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    _chart_panel_close()

    fig = _feature_importance_chart(reg, metadata["feature_columns"])
    if fig is not None:
        _chart_panel_open("Top features driving the forecast")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        _chart_panel_close(
            "Random-Forest feature importance. Calendar timing and prior-event "
            "attendance dominate &mdash; community momentum carries more signal "
            "than the calendar alone."
        )


def _render_calendar_tab(history: pd.DataFrame) -> None:
    col_a, col_b = st.columns([1.45, 1])
    with col_a:
        _chart_panel_open("Average attendance by month")
        st.plotly_chart(
            _monthly_heatmap(history), use_container_width=True,
            config={"displayModeBar": False},
        )
        _chart_panel_close()
    with col_b:
        _chart_panel_open("Event frequency by month")
        st.plotly_chart(
            _events_per_month_chart(history), use_container_width=True,
            config={"displayModeBar": False},
        )
        _chart_panel_close()

    # Seasonality strip
    h = history.copy()
    h["month"] = h["event_date"].dt.month
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall", 11: "Fall"}
    h["season"] = h["month"].map(season_map)
    seasonal = h.groupby("season")["attendance_count"].agg(["mean", "count"]).round(1)
    seasonal = seasonal.reindex(["Winter", "Spring", "Summer", "Fall"])
    chips = "".join(
        f"""<div class="kc-stat-chip">
              <div class="label">{idx} · avg</div>
              <div class="value">{row['mean']:.1f}</div>
              <div class="sub">{int(row['count'])} events</div>
            </div>"""
        for idx, row in seasonal.iterrows()
    )
    st.markdown(f'<div class="kc-stat-strip">{chips}</div>', unsafe_allow_html=True)


def _render_model_notes(metadata: dict) -> None:
    mae = metadata.get("regression_cv_mae", float("nan"))
    rmse = metadata.get("regression_cv_rmse", float("nan"))
    acc = metadata.get("classification_cv_accuracy", float("nan"))
    f1 = metadata.get("classification_cv_f1", float("nan"))
    threshold = metadata.get("median_attendance_threshold", float("nan"))
    n = metadata.get("n_training_events", 0)

    st.markdown(
        f"""
        <div class="kc-stat-strip">
          <div class="kc-stat-chip"><div class="label">CV MAE</div><div class="value">{mae:.2f}</div><div class="sub">players · regression</div></div>
          <div class="kc-stat-chip"><div class="label">CV RMSE</div><div class="value">{rmse:.2f}</div><div class="sub">players · regression</div></div>
          <div class="kc-stat-chip"><div class="label">CV Accuracy</div><div class="value">{acc * 100:.1f}%</div><div class="sub">high vs low turnout</div></div>
          <div class="kc-stat-chip"><div class="label">CV F1</div><div class="value">{f1:.3f}</div><div class="sub">classifier</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kc-notes-grid">
          <div class="kc-notes-card">
            <h5>What the model uses</h5>
            <ul>
              <li>Calendar facts: month, day-of-week, holiday-week, school-break flags</li>
              <li>Scheduling: days since last event; weekly / biweekly / long-break indicators</li>
              <li>Prior-event lag: previous attendance, rolling 3 / 5 / 10-event averages, trend deltas</li>
              <li>Prior-event momentum: previous unique players, new players, draw rate, games-per-player</li>
              <li>Bradenton, FL weather: high / low / mean temperature, feels-like, precipitation, rain &amp; storm flags</li>
            </ul>
          </div>
          <div class="kc-notes-card">
            <h5>What the model does NOT use</h5>
            <ul>
              <li>Same-night game counts, draw rates, or unique-player counts (anti-leakage)</li>
              <li>Specific players or player ratings</li>
              <li>Game outcomes, openings, or move quality</li>
              <li>External social-media or promotional signals (not in the dataset yet)</li>
            </ul>
          </div>
          <div class="kc-notes-card">
            <h5>Training &amp; evaluation</h5>
            <ul>
              <li>Random Forest regressor for attendance count + Random Forest classifier for high vs low turnout</li>
              <li>Trained on <b>{n}</b> historical bracket nights</li>
              <li>Evaluated with 4-fold <b>time-series cross-validation</b> &mdash; test folds always come after train folds</li>
              <li>Tracked in MLflow (<code>./mlruns/</code> on the local pipeline)</li>
              <li>High-turnout threshold: attendance &ge; <b>{threshold:.0f}</b> (historical median)</li>
            </ul>
          </div>
          <div class="kc-notes-card">
            <h5>Honest limitations</h5>
            <ul>
              <li>~{n} events is a small training set &mdash; treat predictions as a planning prior, not gospel</li>
              <li>Bracket rescheduling around holidays adds noise the model cannot see</li>
              <li>Player retention shifts after long breaks are only partly absorbed by rolling features</li>
              <li>Weather forecast quality degrades beyond ~10 days out</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# EXPANDERS
# ----------------------------------------------------------------------------
def _render_pipeline_overview() -> None:
    st.markdown(
        """
        <p style="color:#9BA09B; line-height:1.6; margin:0 0 0.4rem; font-size:0.92rem;">
        Raw bracket logs flow through a medallion architecture on AWS S3, are transformed by
        PySpark into typed Parquet tables, become features for a Random Forest model tracked
        in MLflow, and serve this Streamlit dashboard.
        </p>
        <div class="kc-pipeline">
          <div class="kc-pipeline-step"><b>Source</b>Kava Social game logs (Excel / TSV)</div>
          <div class="kc-pipeline-step"><b>Bronze</b>Raw TSV in S3 object storage</div>
          <div class="kc-pipeline-step"><b>Silver</b>Cleaned game-level Parquet (PySpark + pandas)</div>
          <div class="kc-pipeline-step"><b>Gold</b>Event features + Bradenton weather</div>
          <div class="kc-pipeline-step"><b>Serve</b>scikit-learn · MLflow · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_how_it_works() -> None:
    st.markdown(
        """
        <div style="color:#9BA09B; line-height:1.65; font-size:0.92rem;">
        <b style="color:#C4A77D;">1. Aggregation.</b>
        Every chess game ever recorded at a Kava Social bracket night is one row in the source
        data. KavaCast collapses those rows into one event-level record per night &mdash;
        attendance is the count of unique non-null players in either color column.<br><br>

        <b style="color:#C4A77D;">2. Feature engineering.</b>
        For each event, KavaCast builds a feature row from things knowable
        <i>before</i> people show up: calendar facts about the date, Bradenton weather for that
        date, and lag / rolling features summarising the recent past (last event's attendance,
        3-event and 5-event rolling averages, days since last event, ...).<br><br>

        <b style="color:#C4A77D;">3. Modeling.</b>
        A Random Forest regressor predicts attendance count and a separate Random Forest
        classifier predicts whether the night will clear the historical median
        ("high turnout"). Both are tracked with MLflow during training.<br><br>

        <b style="color:#C4A77D;">4. Forecasting.</b>
        When you pick a date in the sidebar, KavaCast builds the same feature row, pulls live
        weather from Open-Meteo (or your custom override), and runs both models. The bronze
        card is the regressor; the probability gauge is the classifier; the planning note
        translates the forecast into boards, clocks, and staffing.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    _inject_css()

    try:
        reg, clf, history, metadata = _load_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Model artifacts not found. Run the local pipeline first:\n\n"
            "`python src/clean_kava_chess_data.py`\n"
            "`python src/weather.py`\n"
            "`python src/feature_engineering.py`\n"
            "`python src/train_model.py`\n\n"
            f"Underlying error: {e}"
        )
        return

    st.markdown(HERO_HTML, unsafe_allow_html=True)

    target_date, weather_override = _sidebar(history)

    # ?date=YYYY-MM-DD support for the grading test script
    qp = st.query_params
    if "date" in qp:
        try:
            target_date = pd.to_datetime(qp["date"]).date()
            st.info(f"Using **?date={target_date}** from the URL.")
        except Exception:
            pass

    try:
        pred = predict_for_date(target_date, weather_override=weather_override)
    except TypeError:
        pred = predict_for_date(target_date)

    _forecast_command_center(pred)

    _section_header("Analytics", "Historical attendance & model context")
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Trends", "Calendar", "Model notes"])
    with tab1:
        _render_forecast_tab(history, pred)
    with tab2:
        _render_trends_tab(history, reg, metadata)
    with tab3:
        _render_calendar_tab(history)
    with tab4:
        _render_model_notes(metadata)

    _section_header("Reference", "Method & pipeline")
    with st.expander("Pipeline overview"):
        _render_pipeline_overview()
    with st.expander("How this forecast works"):
        _render_how_it_works()
    with st.expander("Model and data notes"):
        _render_model_notes(metadata)


if __name__ == "__main__":
    main()
