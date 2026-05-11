"""
Train two models for Kava Social chess attendance:
  1. RandomForestRegressor -> attendance_count
  2. RandomForestClassifier -> high_turnout

Both are tracked with MLflow (local file store under ./mlruns) and the best
model artifacts are persisted to ./models/ for the Streamlit app to load.

Run:
    python src/train_model.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PARQUET = PROJECT_ROOT / "data" / "gold" / "event_features.parquet"
WEATHER_PARQUET = PROJECT_ROOT / "data" / "gold" / "weather_bradenton.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

FEATURE_COLUMNS = [
    # calendar
    "month",
    "day_of_month",
    "day_of_week",
    "is_sunday",
    "week_of_year",
    "is_beginning_of_month",
    "is_end_of_month",
    "is_holiday_week",
    "is_school_break",
    # scheduling
    "days_since_last_event",
    "biweekly_event_indicator",
    "weekly_event_indicator",
    "back_to_back_event_indicator",
    "first_event_after_long_break",
    "events_this_month_so_far",
    "event_number_overall",
    "event_number_in_year",
    # lag / rolling attendance
    "previous_event_attendance",
    "attendance_two_events_ago",
    "rolling_3_event_attendance",
    "rolling_5_event_attendance",
    "rolling_10_event_attendance",
    "attendance_trend_last_3",
    "attendance_trend_last_5",
    "previous_event_high_turnout",
    # prior-event momentum
    "previous_event_num_games",
    "previous_event_unique_players",
    "previous_event_new_players_count",
    "previous_event_returning_players_count",
    "previous_event_draw_rate",
    "previous_event_games_per_player",
    "rolling_3_avg_num_games",
    "rolling_3_avg_new_players",
    "rolling_3_avg_returning_players",
    # weather (joined from Open-Meteo)
    "temperature_high",
    "temperature_low",
    "average_temperature",
    "feels_like_temperature",
    "precipitation_amount",
    "rain_indicator",
    "thunderstorm_indicator",
    "wind_speed",
    "severe_weather_indicator",
]


def load_dataset() -> pd.DataFrame:
    events = pd.read_parquet(GOLD_PARQUET)
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    if WEATHER_PARQUET.exists():
        weather = pd.read_parquet(WEATHER_PARQUET)
        weather["event_date"] = pd.to_datetime(weather["event_date"]).dt.normalize()
        events = events.merge(weather, on="event_date", how="left")
    else:
        print("[train] WARN: no weather parquet found - weather features will be NaN")
        for c in [
            "temperature_high",
            "temperature_low",
            "average_temperature",
            "feels_like_temperature",
            "precipitation_amount",
            "rain_indicator",
            "thunderstorm_indicator",
            "wind_speed",
            "severe_weather_indicator",
        ]:
            events[c] = np.nan
    return events


def prepare_xy(events: pd.DataFrame):
    df = events.copy()
    # Build feature matrix - fill NaNs with column medians (calendar leakage-safe;
    # the first few rows have no lag history)
    X = df[FEATURE_COLUMNS].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y_reg = df["attendance_count"].astype(float)
    y_clf = df["high_turnout"].astype(int)
    return X, y_reg, y_clf, df["event_date"], medians


def time_series_evaluate(X, y, model_factory, n_splits=4, classification=False):
    tss = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for fold, (tr, te) in enumerate(tss.split(X)):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        if classification:
            metrics_list.append(
                {
                    "fold": fold,
                    "accuracy": float(accuracy_score(y.iloc[te], pred)),
                    "f1": float(f1_score(y.iloc[te], pred, zero_division=0)),
                }
            )
        else:
            mae = mean_absolute_error(y.iloc[te], pred)
            rmse = float(np.sqrt(mean_squared_error(y.iloc[te], pred)))
            metrics_list.append({"fold": fold, "mae": float(mae), "rmse": rmse})
    return metrics_list


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
    mlflow.set_experiment("kava_chess_attendance")

    events = load_dataset()
    X, y_reg, y_clf, dates, medians = prepare_xy(events)

    # ----- Regression -----
    with mlflow.start_run(run_name=f"rf_regressor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "n_estimators": 400,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "n_features": len(FEATURE_COLUMNS),
                "n_events": len(events),
            }
        )

        def _reg_factory():
            return RandomForestRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

        cv_metrics = time_series_evaluate(X, y_reg, _reg_factory, classification=False)
        mae_mean = float(np.mean([m["mae"] for m in cv_metrics]))
        rmse_mean = float(np.mean([m["rmse"] for m in cv_metrics]))
        mlflow.log_metric("cv_mae", mae_mean)
        mlflow.log_metric("cv_rmse", rmse_mean)
        for m in cv_metrics:
            mlflow.log_metric(f"fold{m['fold']}_mae", m["mae"])
            mlflow.log_metric(f"fold{m['fold']}_rmse", m["rmse"])

        reg = _reg_factory()
        reg.fit(X, y_reg)
        mlflow.sklearn.log_model(reg, "model")
        joblib.dump(reg, MODELS_DIR / "rf_attendance_regressor.joblib")
        print(f"[train] regression CV  MAE={mae_mean:.2f}  RMSE={rmse_mean:.2f}")

    # ----- Classification -----
    with mlflow.start_run(run_name=f"rf_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": 400,
                "max_depth": 6,
                "min_samples_leaf": 2,
                "target": "high_turnout",
                "median_attendance_threshold": float(events["_median_attendance_used"].iloc[0]),
            }
        )

        def _clf_factory():
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        cv_metrics = time_series_evaluate(X, y_clf, _clf_factory, classification=True)
        acc_mean = float(np.mean([m["accuracy"] for m in cv_metrics]))
        f1_mean = float(np.mean([m["f1"] for m in cv_metrics]))
        mlflow.log_metric("cv_accuracy", acc_mean)
        mlflow.log_metric("cv_f1", f1_mean)
        for m in cv_metrics:
            mlflow.log_metric(f"fold{m['fold']}_accuracy", m["accuracy"])
            mlflow.log_metric(f"fold{m['fold']}_f1", m["f1"])

        clf = _clf_factory()
        clf.fit(X, y_clf)
        try:
            auc = float(roc_auc_score(y_clf, clf.predict_proba(X)[:, 1]))
            mlflow.log_metric("train_auc", auc)
        except Exception:
            auc = float("nan")
        mlflow.sklearn.log_model(clf, "model")
        joblib.dump(clf, MODELS_DIR / "rf_high_turnout_classifier.joblib")
        print(f"[train] classifier CV ACC={acc_mean:.3f}  F1={f1_mean:.3f}  train_AUC={auc:.3f}")

    # Persist supporting artifacts the Streamlit app needs
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": medians.to_dict(),
        "median_attendance_threshold": float(events["_median_attendance_used"].iloc[0]),
        "n_training_events": int(len(events)),
        "regression_cv_mae": mae_mean,
        "regression_cv_rmse": rmse_mean,
        "classification_cv_accuracy": acc_mean,
        "classification_cv_f1": f1_mean,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Persist the event-level table that the app uses to compute lag features on
    # the fly for any future date
    events_for_app = events[
        [
            "event_date",
            "attendance_count",
            "num_games",
            "num_draws",
            "unique_players",
            "new_players_count",
            "returning_players_count",
            "games_per_player",
            "draw_rate",
            "high_turnout",
        ]
    ].copy()
    events_for_app.to_parquet(MODELS_DIR / "event_history.parquet", index=False)
    print(f"[train] saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
