"""
KavaCast - Kava Social Chess Attendance Predictor
Streamlit UI: a warm-tournament-lounge analytics dashboard.

Run locally:
    streamlit run app/streamlit_app.py

Public URL: served via Streamlit Community Cloud.
"""
from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
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

IVORY = "#F8F4EA"
SAND = "#EFE6D2"
CREAM = "#FFF9EF"
WHITE = "#FFFFFF"
BORDER = "#E6D8BC"
ESPRESSO = "#241C17"
CHARCOAL = "#1F2937"
TAUPE = "#7A6A58"
GOLD = "#C9A227"
DEEP_GOLD = "#A97821"
TEAL = "#0F766E"
TEAL_LIGHT = "#5eead4"
GREEN = "#4F7C45"
NAVY = "#172033"
AMBER_WARN = "#B45309"


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --kc-ivory: #F8F4EA;
            --kc-sand: #EFE6D2;
            --kc-cream: #FFF9EF;
            --kc-white: #FFFFFF;
            --kc-border: #E6D8BC;
            --kc-espresso: #241C17;
            --kc-charcoal: #1F2937;
            --kc-taupe: #7A6A58;
            --kc-gold: #C9A227;
            --kc-deep-gold: #A97821;
            --kc-teal: #0F766E;
            --kc-teal-light: #5eead4;
            --kc-green: #4F7C45;
            --kc-navy: #172033;
        }

        html, body, .stApp, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
        }

        .stApp { background: var(--kc-ivory) !important; }

        .block-container {
            max-width: 1400px !important;
            padding-top: 1.1rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Hero */
        .kc-hero {
            position: relative;
            border-radius: 22px;
            padding: 2rem 2.25rem 2.1rem;
            margin: 0 0 1.5rem;
            background:
                radial-gradient(circle at 88% -10%, rgba(201,162,39,0.32) 0%, rgba(23,32,51,0) 55%),
                radial-gradient(circle at 0% 110%, rgba(15,118,110,0.20) 0%, rgba(23,32,51,0) 55%),
                linear-gradient(135deg, var(--kc-navy) 0%, var(--kc-espresso) 65%, #2a2017 100%);
            color: var(--kc-cream);
            box-shadow: 0 16px 38px rgba(20,16,12,0.22);
            overflow: hidden;
        }
        .kc-hero::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--kc-gold) 0%, var(--kc-deep-gold) 45%, var(--kc-teal) 100%);
        }
        .kc-eyebrow {
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.20em;
            color: rgba(248, 244, 234, 0.7);
            font-weight: 600;
            margin-bottom: 0.55rem;
        }
        .kc-title {
            font-size: 2.7rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin: 0;
            color: var(--kc-white);
            line-height: 1.05;
        }
        .kc-title .kc-title-accent { color: var(--kc-gold); }
        .kc-subtitle {
            font-size: 1.05rem;
            font-weight: 500;
            color: var(--kc-gold);
            margin: 0.35rem 0 0.85rem;
        }
        .kc-desc {
            font-size: 0.95rem;
            line-height: 1.6;
            max-width: 760px;
            color: rgba(248, 244, 234, 0.85);
            margin: 0 0 1.15rem 0;
        }
        .kc-badges { display: flex; gap: 0.55rem; flex-wrap: wrap; }
        .kc-badge {
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .kc-badge--gold { background: rgba(201,162,39,0.18); color: var(--kc-gold); border: 1px solid rgba(201,162,39,0.4); }
        .kc-badge--teal { background: rgba(15,118,110,0.22); color: var(--kc-teal-light); border: 1px solid rgba(15,118,110,0.48); }
        .kc-badge--cream { background: rgba(255,249,239,0.08); color: var(--kc-cream); border: 1px solid rgba(255,249,239,0.22); }

        /* Section header */
        .kc-section-h { display: flex; align-items: baseline; gap: 0.75rem; margin: 1.5rem 0 0.9rem; }
        .kc-section-h .eyebrow {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.18em;
            color: var(--kc-taupe); font-weight: 700;
        }
        .kc-section-h h3 {
            margin: 0; font-size: 1.32rem; font-weight: 700; color: var(--kc-espresso);
            letter-spacing: -0.01em;
        }

        /* Metric cards */
        .kc-card {
            position: relative;
            background: var(--kc-white);
            border: 1px solid var(--kc-border);
            border-radius: 18px;
            padding: 1.2rem 1.35rem 1.3rem;
            box-shadow: 0 4px 14px rgba(36,28,23,0.05);
            height: 100%;
            min-height: 168px;
        }
        .kc-card::before {
            content: "";
            position: absolute;
            top: 0; left: 18px; right: 18px;
            height: 3px;
            border-radius: 0 0 3px 3px;
            background: var(--kc-gold);
        }
        .kc-card--teal::before { background: var(--kc-teal); }
        .kc-card--green::before { background: var(--kc-green); }
        .kc-card--gold::before { background: var(--kc-gold); }
        .kc-card--navy::before { background: var(--kc-navy); }

        .kc-card-label {
            text-transform: uppercase;
            font-size: 0.7rem; letter-spacing: 0.16em;
            font-weight: 700;
            color: var(--kc-taupe);
            margin-bottom: 0.55rem;
        }
        .kc-card-value {
            font-size: 2.7rem;
            font-weight: 700;
            line-height: 1;
            color: var(--kc-espresso);
            letter-spacing: -0.025em;
        }
        .kc-card-value--gold { color: var(--kc-deep-gold); }
        .kc-card-value--teal { color: var(--kc-teal); }
        .kc-card-value--green { color: var(--kc-green); }
        .kc-card-unit {
            font-size: 0.95rem; font-weight: 500;
            color: var(--kc-taupe); margin-left: 0.4rem;
        }
        .kc-card-sub {
            margin-top: 0.65rem; font-size: 0.82rem;
            color: var(--kc-taupe); line-height: 1.45;
        }
        .kc-card-body {
            font-size: 0.9rem; line-height: 1.55;
            color: var(--kc-espresso); margin-bottom: 0.4rem;
        }
        .kc-card-list { margin: 0.55rem 0 0; padding: 0; list-style: none; font-size: 0.82rem; }
        .kc-card-list li {
            display: flex; justify-content: space-between;
            padding: 0.24rem 0;
            border-top: 1px dashed var(--kc-border);
        }
        .kc-card-list li:first-child { border-top: 0; }
        .kc-card-list .label { color: var(--kc-taupe); }
        .kc-card-list .value { color: var(--kc-espresso); font-weight: 600; }

        .kc-progress-track {
            height: 8px; background: #efe6d2; border-radius: 999px;
            margin-top: 0.7rem; overflow: hidden;
        }
        .kc-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--kc-teal) 0%, var(--kc-green) 100%);
            border-radius: 999px;
        }

        /* Chart panels */
        .kc-chart-panel {
            background: var(--kc-white);
            border: 1px solid var(--kc-border);
            border-radius: 18px;
            padding: 0.9rem 1rem 0.4rem;
            box-shadow: 0 4px 14px rgba(36,28,23,0.05);
            margin-bottom: 1rem;
        }
        .kc-chart-panel h4 {
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.16em;
            color: var(--kc-taupe); margin: 0 0 0.2rem 0; font-weight: 700;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: var(--kc-cream) !important;
            border-right: 1px solid var(--kc-border) !important;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--kc-espresso) !important; font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p { color: var(--kc-charcoal); }

        /* Tabs */
        div[data-baseweb="tab-list"] {
            gap: 0.4rem !important;
            border-bottom: 1px solid var(--kc-border) !important;
            background: transparent !important;
        }
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            color: var(--kc-taupe) !important;
            padding: 0.7rem 0.95rem !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--kc-espresso) !important;
        }
        div[data-baseweb="tab-highlight"] { background: var(--kc-gold) !important; }

        /* Expanders */
        details > summary,
        div[data-testid="stExpander"] summary {
            font-weight: 700 !important;
            color: var(--kc-espresso) !important;
        }

        /* Trust note */
        .kc-trust {
            margin-top: 0.6rem;
            padding: 0.65rem 0.9rem;
            border-radius: 12px;
            background: rgba(15, 118, 110, 0.06);
            border-left: 3px solid var(--kc-teal);
            font-size: 0.84rem;
            color: var(--kc-charcoal);
            line-height: 1.4;
        }

        /* Sidebar summary panel */
        .kc-side-summary {
            background: white; border: 1px solid var(--kc-border);
            border-radius: 12px; padding: 0.85rem 1rem; font-size: 0.86rem; color: var(--kc-espresso);
        }
        .kc-side-summary-row {
            display: flex; justify-content: space-between; padding: 0.2rem 0;
            border-top: 1px dashed var(--kc-border);
        }
        .kc-side-summary-row:first-child { border-top: 0; }
        .kc-side-summary-row .label { color: var(--kc-taupe); }
        .kc-side-summary-row .value { font-weight: 600; }

        /* Pipeline diagram */
        .kc-pipeline {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.55rem; margin: 0.8rem 0 0;
        }
        .kc-pipeline-step {
            background: var(--kc-cream); border: 1px solid var(--kc-border);
            border-radius: 12px; padding: 0.7rem 0.75rem; font-size: 0.84rem;
            color: var(--kc-espresso); text-align: left;
        }
        .kc-pipeline-step b {
            display: block; color: var(--kc-deep-gold);
            margin-bottom: 0.2rem; font-size: 0.68rem;
            text-transform: uppercase; letter-spacing: 0.12em;
        }

        /* Bullet lists in expanders */
        .kc-notes-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 1rem; margin-top: 0.5rem;
        }
        .kc-notes-card {
            background: var(--kc-cream); border: 1px solid var(--kc-border);
            border-radius: 14px; padding: 0.9rem 1.1rem;
        }
        .kc-notes-card h5 {
            margin: 0 0 0.4rem 0; font-size: 0.82rem;
            text-transform: uppercase; letter-spacing: 0.14em;
            color: var(--kc-deep-gold);
        }
        .kc-notes-card ul { margin: 0; padding-left: 1.1rem; color: var(--kc-charcoal); font-size: 0.9rem; }
        .kc-notes-card ul li { margin: 0.18rem 0; line-height: 1.45; }

        /* Stat strip */
        .kc-stat-strip {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem;
            margin-top: 0.4rem;
        }
        .kc-stat-chip {
            background: white; border: 1px solid var(--kc-border);
            border-radius: 14px; padding: 0.75rem 0.95rem;
        }
        .kc-stat-chip .label {
            text-transform: uppercase; letter-spacing: 0.12em;
            font-size: 0.66rem; color: var(--kc-taupe); font-weight: 700;
        }
        .kc-stat-chip .value {
            font-size: 1.35rem; font-weight: 700; color: var(--kc-espresso);
            margin-top: 0.15rem;
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
# Plotly layout helper
# ----------------------------------------------------------------------------
def _layout(**overrides):
    base = dict(
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=ESPRESSO, size=12),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin=dict(l=20, r=20, t=44, b=30),
        title=dict(font=dict(size=14, color=ESPRESSO), x=0, xanchor="left"),
        xaxis=dict(showgrid=False, color=TAUPE, linecolor=BORDER, ticks=""),
        yaxis=dict(gridcolor="#eef2f7", color=TAUPE, linecolor=BORDER, ticks=""),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(color=TAUPE, size=11),
        ),
        hoverlabel=dict(font=dict(family="Inter", size=12)),
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
  <div class="kc-subtitle">Chess Night Attendance Forecasting for Kava Social</div>
  <p class="kc-desc">
    A distributed data pipeline and machine-learning model that predicts bracket-night turnout
    using historical Kava Social chess logs, calendar patterns, Bradenton weather, and recent
    community momentum.
  </p>
  <div class="kc-badges">
    <span class="kc-badge kc-badge--gold">Bradenton, FL</span>
    <span class="kc-badge kc-badge--teal">Attendance Forecast</span>
    <span class="kc-badge kc-badge--cream">Kava Social Chess</span>
    <span class="kc-badge kc-badge--cream">MLflow Pipeline</span>
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
            '<div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.18em; '
            'color:#A97821; font-weight:700; margin-bottom:0.15rem;">Control Panel</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Predict a bracket night")

        last_event = history["event_date"].max().date()
        default_date = last_event + timedelta(days=14)
        if default_date < date.today():
            default_date = date.today() + timedelta(days=7)

        target = st.date_input(
            "Bracket-night date",
            value=default_date,
            min_value=last_event + timedelta(days=1),
            max_value=date.today() + timedelta(days=180),
            help="Pick the upcoming Sunday (or any date) you want to plan for.",
        )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.16em; '
            'color:#7A6A58; font-weight:700; margin:0 0 0.45rem;">Weather override</div>',
            unsafe_allow_html=True,
        )
        st.caption("Leave off to use the live Open-Meteo forecast for Bradenton, FL.")
        custom_weather = st.toggle("Use custom weather", value=False)

        weather_override = None
        if custom_weather:
            temp_high_f = st.number_input(
                "Expected high (°F)",
                min_value=30, max_value=110, value=82, step=1,
            )
            rain_chance = st.slider("Rain chance", 0, 100, 20, 5, format="%d%%")
            humidity = st.slider("Humidity", 0, 100, 70, 5, format="%d%%")
            # The model was trained on Open-Meteo °C values; convert.
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
            '<div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.16em; '
            'color:#7A6A58; font-weight:700; margin:0 0 0.5rem;">Recent activity</div>',
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
            '<div class="kc-trust"><b>Trust note.</b> This model forecasts attendance only. '
            "It does not predict chess outcomes or player performance.</div>",
            unsafe_allow_html=True,
        )

    return target, weather_override


# ----------------------------------------------------------------------------
# FORECAST COMMAND CENTER
# ----------------------------------------------------------------------------
def _forecast_command_center(pred, history: pd.DataFrame) -> None:
    _section_header("Forecast", "Forecast command center")

    boards = math.ceil(pred.predicted_attendance_rounded / 2)
    extra_board = pred.turnout_category == "High"
    boards_text = f"{boards}+1 spare" if extra_board else str(boards)
    clocks = max(boards, 3)
    if pred.turnout_category == "High":
        staffing = "Extra staffing recommended"
    elif pred.turnout_category == "Low":
        staffing = "Light staffing fine"
    else:
        staffing = "Standard staffing"

    prob_pct = pred.high_turnout_probability * 100

    cat_card_class = {"High": "kc-card--green", "Normal": "kc-card--teal", "Low": "kc-card--gold"}[pred.turnout_category]
    cat_value_class = {"High": "kc-card-value--green", "Normal": "kc-card-value--teal", "Low": "kc-card-value--gold"}[pred.turnout_category]

    c1, c2, c3, c4 = st.columns([1.05, 1, 1, 1.45])

    with c1:
        st.markdown(
            f"""
            <div class="kc-card kc-card--gold">
              <div class="kc-card-label">Predicted Attendance</div>
              <div class="kc-card-value kc-card-value--gold">{pred.predicted_attendance_rounded}<span class="kc-card-unit">players</span></div>
              <div class="kc-card-sub">Regressor estimate <b style="color:#241C17;">{pred.predicted_attendance:.1f}</b> · forecast for <b style="color:#241C17;">{pred.event_date}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kc-card {cat_card_class}">
              <div class="kc-card-label">Turnout Category</div>
              <div class="kc-card-value {cat_value_class}" style="font-size:2.05rem;">{pred.turnout_category} turnout</div>
              <div class="kc-card-sub">High-turnout threshold is attendance &ge; <b style="color:#241C17;">{pred.median_attendance_threshold:.0f}</b> (historical median).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kc-card kc-card--teal">
              <div class="kc-card-label">High-Turnout Probability</div>
              <div class="kc-card-value kc-card-value--teal">{prob_pct:.0f}<span class="kc-card-unit">%</span></div>
              <div class="kc-progress-track"><div class="kc-progress-fill" style="width:{min(100, max(0, prob_pct)):.0f}%"></div></div>
              <div class="kc-card-sub">Classifier probability the night clears the historical median.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kc-card kc-card--navy">
              <div class="kc-card-label">Planning Recommendation</div>
              <div class="kc-card-body">{pred.planning_note}</div>
              <ul class="kc-card-list">
                <li><span class="label">Boards to prep</span><span class="value">{boards_text}</span></li>
                <li><span class="label">Clocks recommended</span><span class="value">{clocks}</span></li>
                <li><span class="label">Staffing</span><span class="value">{staffing}</span></li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------------
def _chart_panel_open(eyebrow: str) -> None:
    st.markdown(f'<div class="kc-chart-panel"><h4>{eyebrow}</h4>', unsafe_allow_html=True)


def _chart_panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _trend_chart(history: pd.DataFrame, pred) -> go.Figure:
    h = history.sort_values("event_date").copy()
    h["rolling_5"] = h["attendance_count"].rolling(5, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["attendance_count"],
            mode="lines+markers",
            name="Actual attendance",
            line=dict(color=ESPRESSO, width=2),
            marker=dict(size=6, color=ESPRESSO),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["rolling_5"],
            mode="lines", name="5-event rolling average",
            line=dict(color=TEAL, width=3, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pd.to_datetime(pred.event_date)],
            y=[pred.predicted_attendance_rounded],
            mode="markers", name="Forecast",
            marker=dict(color=GOLD, size=15, symbol="diamond",
                        line=dict(color="white", width=2)),
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
            marker=dict(color=ESPRESSO, size=8, opacity=0.65),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h["event_date"], y=h["smoothed"],
            mode="lines", name="Smoothed trend (7-event centered)",
            line=dict(color=DEEP_GOLD, width=3),
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
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        color_continuous_scale=[
            (0.0, SAND),
            (0.5, TEAL_LIGHT),
            (1.0, TEAL),
        ],
        aspect="auto",
        labels=dict(color="Avg attendance"),
    )
    fig.update_traces(textfont=dict(color=ESPRESSO, family="Inter", size=11))
    fig.update_layout(
        **_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_colorbar=dict(title="Avg", tickfont=dict(color=TAUPE), thickness=12),
        )
    )
    return fig


def _events_per_month_chart(history: pd.DataFrame) -> go.Figure:
    h = history.copy()
    h["month"] = h["event_date"].dt.month
    counts = h.groupby("month").size().reindex(range(1, 13), fill_value=0)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(
        go.Bar(x=months, y=counts.values, marker=dict(color=GOLD, line=dict(color=DEEP_GOLD, width=1)))
    )
    fig.update_layout(**_layout(height=300, margin=dict(l=20, r=20, t=20, b=30)))
    return fig


def _feature_importance_chart(reg, feature_columns: list[str]) -> go.Figure | None:
    if not hasattr(reg, "feature_importances_"):
        return None
    importances = pd.Series(reg.feature_importances_, index=feature_columns).sort_values(ascending=True)
    top = importances.tail(12)
    pretty = top.index.str.replace("_", " ").str.replace(" event ", " ").str.title()
    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=pretty,
            orientation="h",
            marker=dict(color=TEAL, line=dict(color=NAVY, width=0.5)),
        )
    )
    fig.update_layout(
        **_layout(
            height=420, margin=dict(l=20, r=20, t=20, b=30),
            xaxis=dict(showgrid=True, gridcolor="#eef2f7", color=TAUPE, linecolor=BORDER),
            yaxis=dict(color=ESPRESSO, linecolor=BORDER, automargin=True),
        )
    )
    return fig


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
def _render_forecast_tab(history: pd.DataFrame, pred) -> None:
    _chart_panel_open("Historical attendance with forecast")
    st.plotly_chart(_trend_chart(history, pred), use_container_width=True, config={"displayModeBar": False})
    _chart_panel_close()


def _render_trends_tab(history: pd.DataFrame, reg, metadata: dict) -> None:
    _chart_panel_open("Smoothed attendance trend")
    st.plotly_chart(_smoothed_chart(history), use_container_width=True, config={"displayModeBar": False})
    _chart_panel_close()

    fig = _feature_importance_chart(reg, metadata["feature_columns"])
    if fig is not None:
        _chart_panel_open("Top features driving the attendance forecast")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Feature importance from the Random Forest regressor. Calendar timing, prior-event "
            "attendance, and weather repeatedly dominate — community momentum carries more signal "
            "than the calendar alone."
        )
        _chart_panel_close()


def _render_calendar_tab(history: pd.DataFrame) -> None:
    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        _chart_panel_open("Average attendance by month")
        st.plotly_chart(_monthly_heatmap(history), use_container_width=True, config={"displayModeBar": False})
        _chart_panel_close()
    with col_b:
        _chart_panel_open("Event frequency by month")
        st.plotly_chart(_events_per_month_chart(history), use_container_width=True, config={"displayModeBar": False})
        _chart_panel_close()

    # Seasonality summary
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
        f"""<div class="kc-stat-chip"><div class="label">{idx} avg</div>
            <div class="value">{row['mean']:.1f}</div>
            <div style="font-size:0.7rem; color:#7A6A58; margin-top:0.1rem;">{int(row['count'])} events</div></div>"""
        for idx, row in seasonal.iterrows()
    )
    st.markdown(f'<div class="kc-stat-strip">{chips}</div>', unsafe_allow_html=True)


def _render_model_notes_tab(metadata: dict) -> None:
    mae = metadata.get("regression_cv_mae", float("nan"))
    rmse = metadata.get("regression_cv_rmse", float("nan"))
    acc = metadata.get("classification_cv_accuracy", float("nan"))
    f1 = metadata.get("classification_cv_f1", float("nan"))
    threshold = metadata.get("median_attendance_threshold", float("nan"))
    n = metadata.get("n_training_events", 0)

    st.markdown(
        f"""
        <div class="kc-stat-strip">
          <div class="kc-stat-chip"><div class="label">CV MAE</div><div class="value">{mae:.2f}</div><div style="font-size:0.7rem;color:#7A6A58;">players, regression</div></div>
          <div class="kc-stat-chip"><div class="label">CV RMSE</div><div class="value">{rmse:.2f}</div><div style="font-size:0.7rem;color:#7A6A58;">players, regression</div></div>
          <div class="kc-stat-chip"><div class="label">CV Accuracy</div><div class="value">{acc * 100:.1f}%</div><div style="font-size:0.7rem;color:#7A6A58;">high vs low turnout</div></div>
          <div class="kc-stat-chip"><div class="label">CV F1</div><div class="value">{f1:.3f}</div><div style="font-size:0.7rem;color:#7A6A58;">classifier</div></div>
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
              <li>Scheduling: days since last event, weekly / biweekly / long-break indicators</li>
              <li>Prior-event lag: previous attendance, rolling 3/5/10-event averages, trend deltas</li>
              <li>Prior-event momentum: previous unique players, new players, draw rate, games-per-player</li>
              <li>Bradenton, FL weather: high/low/mean temperature, feels-like, precipitation, rain &amp; storm flags</li>
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
              <li>Evaluated with 4-fold <b>time-series cross-validation</b> — test folds always come after train folds</li>
              <li>Tracked in MLflow (<code>./mlruns/</code> on the local pipeline)</li>
              <li>High-turnout threshold: attendance &ge; <b>{threshold:.0f}</b> (historical median)</li>
            </ul>
          </div>
          <div class="kc-notes-card">
            <h5>Honest limitations</h5>
            <ul>
              <li>~{n} events is a small training set — treat predictions as a planning prior, not gospel</li>
              <li>Bracket rescheduling around holidays adds noise the model cannot see</li>
              <li>Player retention shifts after long breaks are partly absorbed by rolling features</li>
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
        <p style="color:#3A2F28; line-height:1.55; margin:0 0 0.4rem;">
        Raw bracket logs flow through a medallion architecture on AWS S3, get transformed by
        PySpark, become features for a Random Forest model tracked in MLflow, and feed this
        Streamlit dashboard.
        </p>
        <div class="kc-pipeline">
          <div class="kc-pipeline-step"><b>Source</b>Raw Kava Social game logs (Excel/TSV)</div>
          <div class="kc-pipeline-step"><b>Bronze</b>Raw TSV in S3 (cloud object store)</div>
          <div class="kc-pipeline-step"><b>Silver</b>Cleaned game-level Parquet (PySpark / pandas)</div>
          <div class="kc-pipeline-step"><b>Gold</b>Event-level features + Bradenton weather</div>
          <div class="kc-pipeline-step"><b>Serve</b>scikit-learn + MLflow → Streamlit forecast</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_how_it_works() -> None:
    st.markdown(
        """
        <div style="color:#3A2F28; line-height:1.6;">
        <b>1. Aggregation.</b> Every chess game ever recorded at a Kava Social bracket night is one
        row in the source data. KavaCast collapses those rows into one event-level record per night —
        attendance is the count of unique non-null players in either color column.<br><br>
        <b>2. Feature engineering.</b> For each event, we build a feature row from things knowable
        <i>before</i> people show up: calendar facts about the date, Bradenton weather for that date,
        and lag/rolling features summarising the recent past (last event's attendance, 3-event and
        5-event rolling averages, days since last event, ...).<br><br>
        <b>3. Modeling.</b> A Random Forest regressor predicts the attendance count and a separate
        Random Forest classifier predicts whether the night will clear the historical median
        ("high turnout"). Both are tracked with MLflow during training.<br><br>
        <b>4. Forecasting.</b> When you pick a date in the sidebar, KavaCast builds the same feature
        row, pulls live weather from Open-Meteo, and runs both models. The card on the left is the
        regressor; the probability gauge is the classifier; the planning note translates the
        forecast into boards, clocks, and staffing.
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
        # Backwards-compat: older predict.py without the kwarg
        pred = predict_for_date(target_date)

    _forecast_command_center(pred, history)

    _section_header("Analytics", "Historical attendance & model context")
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Trends", "Calendar", "Model notes"])
    with tab1:
        _render_forecast_tab(history, pred)
    with tab2:
        _render_trends_tab(history, reg, metadata)
    with tab3:
        _render_calendar_tab(history)
    with tab4:
        _render_model_notes_tab(metadata)

    _section_header("Reference", "Pipeline & method")
    with st.expander("Pipeline overview"):
        _render_pipeline_overview()
    with st.expander("How this forecast works"):
        _render_how_it_works()
    with st.expander("Model and data notes"):
        _render_model_notes_tab(metadata)


if __name__ == "__main__":
    main()
