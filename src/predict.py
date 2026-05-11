"""
Inference helper: given a future bracket-night date, build a leakage-safe
feature row using the latest event history + Bradenton weather, and run
both the attendance regressor and the high-turnout classifier.

Used by app/streamlit_app.py and tests/test_project.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class Prediction:
    event_date: str
    predicted_attendance: float
    predicted_attendance_rounded: int
    high_turnout_probability: float
    turnout_category: str
    median_attendance_threshold: float
    planning_note: str
    features_used: dict
    model_metadata: dict


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _maybe_fetch_weather(target: date) -> dict:
    """Try to pull weather for the target date from Open-Meteo. Returns empty
    dict if we can't reach the network - the model will fall back to medians."""
    try:
        from src.weather import fetch_weather_for_dates  # type: ignore
    except Exception:
        try:
            from weather import fetch_weather_for_dates  # type: ignore
        except Exception:
            return {}
    try:
        df = fetch_weather_for_dates([target])
    except Exception:
        return {}
    if df.empty:
        return {}
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    row = df.loc[df["event_date"] == pd.Timestamp(target)]
    if row.empty:
        return {}
    row = row.iloc[0].to_dict()
    return row


def load_models():
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    # Prefer the new neutral-name artifact (saved by the multi-model bake-off);
    # fall back to the legacy RF filename so older deploys keep working.
    reg_path = MODELS_DIR / "attendance_regressor.joblib"
    if not reg_path.exists():
        reg_path = MODELS_DIR / "rf_attendance_regressor.joblib"
    reg = joblib.load(reg_path)
    clf = joblib.load(MODELS_DIR / "rf_high_turnout_classifier.joblib")
    history = pd.read_parquet(MODELS_DIR / "event_history.parquet")
    history["event_date"] = pd.to_datetime(history["event_date"]).dt.normalize()
    history = history.sort_values("event_date").reset_index(drop=True)
    return reg, clf, history, metadata


def build_feature_row(target_date: date, history: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Construct the model input row for a future event date, using only
    information knowable before that date."""
    target_ts = pd.Timestamp(target_date)
    prior = history[history["event_date"] < target_ts].copy()
    if prior.empty:
        # Cold start - we'll let medians take over
        previous_event_attendance = np.nan
        previous_event_date = pd.NaT
        attendance_two_events_ago = np.nan
        rolling = {3: np.nan, 5: np.nan, 10: np.nan}
        attendance_trend_last_3 = np.nan
        attendance_trend_last_5 = np.nan
        previous_event_high_turnout = np.nan
        previous_event_num_games = np.nan
        previous_event_unique_players = np.nan
        previous_event_new_players_count = np.nan
        previous_event_returning_players_count = np.nan
        previous_event_draw_rate = np.nan
        previous_event_games_per_player = np.nan
        rolling_3_avg_num_games = np.nan
        rolling_3_avg_new_players = np.nan
        rolling_3_avg_returning_players = np.nan
        event_number_overall = 0
    else:
        previous_event_attendance = prior["attendance_count"].iloc[-1]
        previous_event_date = prior["event_date"].iloc[-1]
        attendance_two_events_ago = (
            prior["attendance_count"].iloc[-2] if len(prior) >= 2 else np.nan
        )
        rolling = {}
        for w in (3, 5, 10):
            rolling[w] = prior["attendance_count"].tail(w).mean()
        attendance_trend_last_3 = (
            prior["attendance_count"].iloc[-1] - prior["attendance_count"].iloc[-3]
            if len(prior) >= 3
            else np.nan
        )
        attendance_trend_last_5 = (
            prior["attendance_count"].iloc[-1] - prior["attendance_count"].iloc[-5]
            if len(prior) >= 5
            else np.nan
        )
        running_median_prior = prior["attendance_count"].expanding().median().iloc[-1]
        previous_event_high_turnout = int(prior["attendance_count"].iloc[-1] >= running_median_prior)
        previous_event_num_games = prior["num_games"].iloc[-1]
        previous_event_unique_players = prior["unique_players"].iloc[-1]
        previous_event_new_players_count = prior["new_players_count"].iloc[-1]
        previous_event_returning_players_count = prior["returning_players_count"].iloc[-1]
        previous_event_draw_rate = prior["draw_rate"].iloc[-1]
        previous_event_games_per_player = prior["games_per_player"].iloc[-1]
        rolling_3_avg_num_games = prior["num_games"].tail(3).mean()
        rolling_3_avg_new_players = prior["new_players_count"].tail(3).mean()
        rolling_3_avg_returning_players = prior["returning_players_count"].tail(3).mean()
        event_number_overall = len(prior)

    days_since_last_event = (
        (target_ts - previous_event_date).days if not pd.isna(previous_event_date) else np.nan
    )

    weather = _maybe_fetch_weather(target_date)

    row = {
        "month": target_ts.month,
        "day_of_month": target_ts.day,
        "day_of_week": target_ts.dayofweek,
        "is_sunday": int(target_ts.dayofweek == 6),
        "week_of_year": int(target_ts.isocalendar().week),
        "is_beginning_of_month": int(target_ts.day <= 7),
        "is_end_of_month": int(target_ts.day >= 24),
        "is_holiday_week": _holiday_week(target_ts),
        "is_school_break": _school_break(target_ts),
        "days_since_last_event": days_since_last_event,
        "biweekly_event_indicator": int(12 <= (days_since_last_event or 0) <= 16) if pd.notna(days_since_last_event) else 0,
        "weekly_event_indicator": int(5 <= (days_since_last_event or 0) <= 9) if pd.notna(days_since_last_event) else 0,
        "back_to_back_event_indicator": int((days_since_last_event or 999) <= 3) if pd.notna(days_since_last_event) else 0,
        "first_event_after_long_break": int((days_since_last_event or 0) >= 30) if pd.notna(days_since_last_event) else 0,
        "events_this_month_so_far": _events_this_month_prior(prior, target_ts),
        "event_number_overall": event_number_overall,
        "event_number_in_year": _events_in_year_prior(prior, target_ts),
        "previous_event_attendance": previous_event_attendance,
        "attendance_two_events_ago": attendance_two_events_ago,
        "rolling_3_event_attendance": rolling.get(3, np.nan),
        "rolling_5_event_attendance": rolling.get(5, np.nan),
        "rolling_10_event_attendance": rolling.get(10, np.nan),
        "attendance_trend_last_3": attendance_trend_last_3,
        "attendance_trend_last_5": attendance_trend_last_5,
        "previous_event_high_turnout": previous_event_high_turnout,
        "previous_event_num_games": previous_event_num_games,
        "previous_event_unique_players": previous_event_unique_players,
        "previous_event_new_players_count": previous_event_new_players_count,
        "previous_event_returning_players_count": previous_event_returning_players_count,
        "previous_event_draw_rate": previous_event_draw_rate,
        "previous_event_games_per_player": previous_event_games_per_player,
        "rolling_3_avg_num_games": rolling_3_avg_num_games,
        "rolling_3_avg_new_players": rolling_3_avg_new_players,
        "rolling_3_avg_returning_players": rolling_3_avg_returning_players,
        "temperature_high": weather.get("temperature_high", np.nan),
        "temperature_low": weather.get("temperature_low", np.nan),
        "average_temperature": weather.get("average_temperature", np.nan),
        "feels_like_temperature": weather.get("feels_like_temperature", np.nan),
        "precipitation_amount": weather.get("precipitation_amount", np.nan),
        "rain_indicator": weather.get("rain_indicator", np.nan),
        "thunderstorm_indicator": weather.get("thunderstorm_indicator", np.nan),
        "wind_speed": weather.get("wind_speed", np.nan),
        "severe_weather_indicator": weather.get("severe_weather_indicator", np.nan),
    }

    feature_columns = metadata["feature_columns"]
    medians = metadata["feature_medians"]
    X = pd.DataFrame([row])[feature_columns]
    X = X.fillna(pd.Series(medians))
    return X


def _holiday_week(ts: pd.Timestamp) -> int:
    try:
        import holidays as _holidays

        h = _holidays.UnitedStates(years=[ts.year])
        return int(any(((ts + pd.Timedelta(days=k)).date() in h) for k in range(-3, 4)))
    except Exception:
        return 0


def _school_break(ts: pd.Timestamp) -> int:
    m, d = ts.month, ts.day
    if m in (6, 7):
        return 1
    if m == 12 and d >= 18:
        return 1
    if m == 1 and d <= 5:
        return 1
    if m == 3 and 10 <= d <= 20:
        return 1
    return 0


def _events_this_month_prior(prior: pd.DataFrame, target: pd.Timestamp) -> int:
    if prior.empty:
        return 0
    return int(
        ((prior["event_date"].dt.year == target.year) & (prior["event_date"].dt.month == target.month)).sum()
    )


def _events_in_year_prior(prior: pd.DataFrame, target: pd.Timestamp) -> int:
    if prior.empty:
        return 0
    return int((prior["event_date"].dt.year == target.year).sum())


def predict_for_date(target_date, weather_override: dict | None = None) -> Prediction:
    """Predict attendance for `target_date`. Optional `weather_override` is a
    dict that can supply any subset of the weather feature names (e.g.
    temperature_high, rain_indicator, average_temperature, ...) — those keys
    take precedence over the auto-fetched Open-Meteo values, the rest fall
    back to forecast/history/medians as usual."""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    reg, clf, history, metadata = load_models()
    X = build_feature_row(target_date, history, metadata)
    if weather_override:
        for k, v in weather_override.items():
            if k in X.columns and v is not None:
                X.at[X.index[0], k] = v

    yhat_reg = float(reg.predict(X)[0])
    proba = float(clf.predict_proba(X)[0, 1])
    threshold = float(metadata.get("median_attendance_threshold", 12))
    rounded = int(round(yhat_reg))
    if proba >= 0.66 or yhat_reg >= threshold + 3:
        category = "High"
        note = (
            f"Expect above-average turnout (~{rounded} players). "
            "Consider extra boards, clocks, and staffing for a busy night."
        )
    elif proba <= 0.34 and yhat_reg <= threshold - 3:
        category = "Low"
        note = (
            f"Expect below-average turnout (~{rounded} players). "
            "Lighter setup should be fine; great chance to run instruction or casual play."
        )
    else:
        category = "Normal"
        note = (
            f"Expect a typical night (~{rounded} players). "
            "Standard board count and staffing recommended."
        )

    return Prediction(
        event_date=str(target_date),
        predicted_attendance=round(yhat_reg, 2),
        predicted_attendance_rounded=rounded,
        high_turnout_probability=round(proba, 3),
        turnout_category=category,
        median_attendance_threshold=threshold,
        planning_note=note,
        features_used=X.iloc[0].to_dict(),
        model_metadata={
            "trained_at": metadata.get("trained_at"),
            "n_training_events": metadata.get("n_training_events"),
            "regression_cv_mae": metadata.get("regression_cv_mae"),
            "regression_cv_rmse": metadata.get("regression_cv_rmse"),
            "classification_cv_accuracy": metadata.get("classification_cv_accuracy"),
            "classification_cv_f1": metadata.get("classification_cv_f1"),
        },
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    p = predict_for_date(args.date)
    print(json.dumps(asdict(p), indent=2, default=str))


if __name__ == "__main__":
    main()
