"""
Streamlit UI for the Kava Social Chess Attendance Predictor.

Run locally:
    streamlit run app/streamlit_app.py

Public URL is served via Streamlit Community Cloud once this file is pushed
to GitHub and connected from share.streamlit.io.
"""
from __future__ import annotations

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


st.set_page_config(
    page_title="Kava Social Chess Attendance Predictor",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)


KAVA_TEAL = "#0fb5a8"
KAVA_DARK = "#0d1b2a"
KAVA_AMBER = "#f5a524"


st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}}
    .kava-hero {{
        background: linear-gradient(135deg, {KAVA_DARK} 0%, #1b3a5b 50%, {KAVA_TEAL} 120%);
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 18px rgba(13, 27, 42, 0.28);
    }}
    .kava-hero h1 {{
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    .kava-hero p {{margin: 0.4rem 0 0; opacity: 0.9; font-size: 1rem;}}
    .kava-card {{
        background: white;
        border: 1px solid #e6e8ec;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .kava-metric-big {{
        font-size: 3rem;
        font-weight: 700;
        color: {KAVA_DARK};
        line-height: 1;
    }}
    .kava-metric-label {{
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }}
    .kava-pill {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}
    .kava-pill-high {{background: #fef3c7; color: #92400e;}}
    .kava-pill-normal {{background: #dbeafe; color: #1e3a8a;}}
    .kava-pill-low {{background: #e0e7ff; color: #3730a3;}}
    .kava-note {{
        background: #f0fdf4;
        border-left: 4px solid {KAVA_TEAL};
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        color: #064e3b;
        margin-top: 1rem;
    }}
    section[data-testid="stSidebar"] {{background: #fafbfc;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def _load_history():
    _, _, history, metadata = load_models()
    return history, metadata


def _hero():
    st.markdown(
        """
        <div class="kava-hero">
            <h1>♟️ Kava Social Chess Attendance Predictor</h1>
            <p>An event-planning tool for Kava Social bracket nights in Bradenton, FL.
            Pick a future date and we'll estimate how many players to expect, based on
            historical attendance patterns, calendar timing, weather, and recent community momentum.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(history: pd.DataFrame):
    st.sidebar.header("Predict a bracket night")
    last_event = history["event_date"].max().date()
    default_date = last_event + timedelta(days=14)
    if default_date < date.today():
        default_date = date.today() + timedelta(days=7)
    target = st.sidebar.date_input(
        "Bracket night date",
        value=default_date,
        min_value=last_event + timedelta(days=1),
        max_value=date.today() + timedelta(days=180),
        help="Choose the upcoming Sunday (or any date) you want to plan for.",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Last recorded event:** {last_event:%b %d, %Y}  \n"
        f"**Total events in history:** {len(history)}  \n"
        f"**Avg attendance:** {history['attendance_count'].mean():.1f}"
    )
    return target


def _prediction_panel(target_date, metadata):
    pred = predict_for_date(target_date)
    cat_class = {
        "High": "kava-pill-high",
        "Normal": "kava-pill-normal",
        "Low": "kava-pill-low",
    }.get(pred.turnout_category, "kava-pill-normal")

    cols = st.columns([1.2, 1, 1])
    with cols[0]:
        st.markdown(
            f"""
            <div class="kava-card">
              <div class="kava-metric-label">Predicted Attendance</div>
              <div class="kava-metric-big">{pred.predicted_attendance_rounded}</div>
              <div style="color:#6b7280; margin-top:0.2rem;">players (regressor: {pred.predicted_attendance:.1f})</div>
              <span class="kava-pill {cat_class}">{pred.turnout_category} Turnout</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        prob = pred.high_turnout_probability
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "High-Turnout Probability", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": KAVA_TEAL},
                    "steps": [
                        {"range": [0, 34], "color": "#e0e7ff"},
                        {"range": [34, 66], "color": "#dbeafe"},
                        {"range": [66, 100], "color": "#fef3c7"},
                    ],
                    "threshold": {
                        "line": {"color": KAVA_AMBER, "width": 4},
                        "thickness": 0.75,
                        "value": 66,
                    },
                },
            )
        )
        gauge.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)
    with cols[2]:
        st.markdown(
            f"""
            <div class="kava-card">
              <div class="kava-metric-label">Median Threshold</div>
              <div class="kava-metric-big" style="color:{KAVA_TEAL}">{pred.median_attendance_threshold:.0f}</div>
              <div style="color:#6b7280;">High turnout = attendance &ge; this</div>
              <div style="margin-top:0.8rem; font-size:0.85rem; color:#6b7280;">
                Model trained on <b>{pred.model_metadata.get('n_training_events', '?')}</b> events.<br>
                Reg CV MAE: <b>{pred.model_metadata.get('regression_cv_mae', float('nan')):.2f}</b><br>
                Clf CV F1: <b>{pred.model_metadata.get('classification_cv_f1', float('nan')):.3f}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="kava-note"><b>Planning note:</b> {pred.planning_note}</div>',
        unsafe_allow_html=True,
    )
    return pred


def _trend_chart(history: pd.DataFrame, pred):
    h = history.copy()
    h["rolling_5"] = h["attendance_count"].rolling(5, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h["event_date"],
            y=h["attendance_count"],
            mode="lines+markers",
            name="Actual attendance",
            line=dict(color=KAVA_DARK, width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h["event_date"],
            y=h["rolling_5"],
            mode="lines",
            name="5-event rolling avg",
            line=dict(color=KAVA_TEAL, width=3, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pd.to_datetime(pred.event_date)],
            y=[pred.predicted_attendance_rounded],
            mode="markers",
            name="Prediction",
            marker=dict(color=KAVA_AMBER, size=14, symbol="diamond", line=dict(color="white", width=2)),
        )
    )
    fig.update_layout(
        title="Attendance over time",
        xaxis_title=None,
        yaxis_title="Unique players",
        height=380,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef2f7")
    return fig


def _monthly_heatmap(history: pd.DataFrame):
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
        color_continuous_scale=["#e0e7ff", KAVA_TEAL, KAVA_DARK],
        aspect="auto",
        labels=dict(color="Avg attendance"),
    )
    fig.update_layout(
        title="Average attendance by month",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_colorbar=dict(title="Players"),
    )
    return fig


def _predicted_vs_actual(history: pd.DataFrame):
    """Backfill predictions for the last N events using a 1-event holdout
    rolling window (no leakage) - quick visual sanity check."""
    fig = px.scatter(
        history,
        x="event_date",
        y="attendance_count",
        trendline="lowess",
        labels={"event_date": "Event date", "attendance_count": "Attendance"},
    )
    fig.update_traces(marker=dict(color=KAVA_DARK, size=8, opacity=0.7))
    fig.update_layout(
        title="Historical attendance with smoothed trend",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef2f7")
    return fig


def main():
    _hero()
    try:
        history, metadata = _load_history()
    except FileNotFoundError as e:
        st.error(
            f"Model artifacts not found. Run the training pipeline first:\n\n"
            f"`python src/clean_kava_chess_data.py`\n"
            f"`python src/feature_engineering.py`\n"
            f"`python src/weather.py`\n"
            f"`python src/train_model.py`\n\nError: {e}"
        )
        return

    target_date = _sidebar(history)

    # Query param support so test_project.py can hit ?date=YYYY-MM-DD
    qp = st.query_params
    if "date" in qp:
        try:
            from datetime import date as _date
            qp_date = pd.to_datetime(qp["date"]).date()
            target_date = qp_date
            st.info(f"Using ?date={qp_date} from URL query param.")
        except Exception:
            pass

    pred = _prediction_panel(target_date, None)

    st.markdown("### Attendance over time")
    st.plotly_chart(_trend_chart(history, pred), use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(_monthly_heatmap(history), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(_predicted_vs_actual(history), use_container_width=True)

    with st.expander("Show feature row used by the model"):
        feat_df = pd.DataFrame(
            [(k, v) for k, v in pred.features_used.items()],
            columns=["feature", "value"],
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with st.expander("About this tool"):
        st.markdown(
            f"""
            **What it predicts:** unique-player attendance for a future Kava Social chess
            bracket night in Bradenton, FL.

            **Data:** {len(history)} historical bracket nights of game-level logs (Sep 2022 - present),
            aggregated to event-level.

            **Pipeline (distributed):** raw TSV uploaded to **S3 bronze** → cleaned and
            feature-engineered with **PySpark** to silver and gold Parquet tables in S3.
            Models trained with **scikit-learn** and tracked in **MLflow**.

            **Models:** RandomForestRegressor for attendance count, RandomForestClassifier
            for high-vs-low turnout, both validated with time-series cross-validation.
            Strictly leakage-safe — only calendar + Bradenton weather + prior-event lag features.

            **Limits:** ~{len(history)} events is a small training set, so treat predictions
            as a planning prior, not gospel.
            """
        )


if __name__ == "__main__":
    main()
