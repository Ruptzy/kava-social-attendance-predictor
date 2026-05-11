"""
Train and compare attendance models for the Kava Social Chess Club.

Compared candidates:
  1. Naive baseline   -> last event's attendance (fallback: mean)
  2. Ridge regression -> simple linear interpretable benchmark
  3. Random Forest    -> non-linear tabular model
  4. Gradient Boosting-> strong structured-data candidate

Evaluation:
  - 4-fold TimeSeriesSplit cross-validation on the training portion
    (test folds always come AFTER train folds - no leakage).
  - Chronological holdout: the last `HOLDOUT_N` events are kept out and
    used as the deciding test set. The model with the lowest holdout MAE
    that also beats the naive baseline is selected and saved.

All runs are tracked in MLflow under `./mlruns/`.

Run:
    python src/train_model.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PARQUET = PROJECT_ROOT / "data" / "gold" / "event_features.parquet"
WEATHER_PARQUET = PROJECT_ROOT / "data" / "gold" / "weather_bradenton.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

HOLDOUT_N = 14  # number of most-recent events held out as the chronological test set

FEATURE_COLUMNS = [
    # calendar
    "month", "day_of_month", "day_of_week", "is_sunday",
    "week_of_year", "is_beginning_of_month", "is_end_of_month",
    "is_holiday_week", "is_school_break",
    # scheduling
    "days_since_last_event",
    "biweekly_event_indicator", "weekly_event_indicator",
    "back_to_back_event_indicator", "first_event_after_long_break",
    "events_this_month_so_far", "event_number_overall", "event_number_in_year",
    # lag / rolling attendance
    "previous_event_attendance", "attendance_two_events_ago",
    "rolling_3_event_attendance", "rolling_5_event_attendance",
    "rolling_10_event_attendance",
    "attendance_trend_last_3", "attendance_trend_last_5",
    "previous_event_high_turnout",
    # prior-event momentum
    "previous_event_num_games", "previous_event_unique_players",
    "previous_event_new_players_count", "previous_event_returning_players_count",
    "previous_event_draw_rate", "previous_event_games_per_player",
    "rolling_3_avg_num_games", "rolling_3_avg_new_players",
    "rolling_3_avg_returning_players",
    # weather (from Open-Meteo merge)
    "temperature_high", "temperature_low", "average_temperature",
    "feels_like_temperature", "precipitation_amount", "rain_indicator",
    "thunderstorm_indicator", "wind_speed", "severe_weather_indicator",
]

# Feature family map (used in the app for narrative grouping)
FEATURE_FAMILY = {
    # Recent attendance momentum
    **{c: "Recent attendance momentum" for c in [
        "previous_event_attendance", "attendance_two_events_ago",
        "rolling_3_event_attendance", "rolling_5_event_attendance",
        "rolling_10_event_attendance",
        "attendance_trend_last_3", "attendance_trend_last_5",
        "previous_event_high_turnout",
    ]},
    # Calendar timing
    **{c: "Calendar timing" for c in [
        "month", "day_of_month", "day_of_week", "is_sunday",
        "week_of_year", "is_beginning_of_month", "is_end_of_month",
        "days_since_last_event", "biweekly_event_indicator",
        "weekly_event_indicator", "back_to_back_event_indicator",
        "first_event_after_long_break", "events_this_month_so_far",
        "event_number_overall", "event_number_in_year",
    ]},
    # Schedule & holiday context
    **{c: "Schedule & holiday context" for c in [
        "is_holiday_week", "is_school_break",
    ]},
    # Bradenton weather
    **{c: "Bradenton weather" for c in [
        "temperature_high", "temperature_low", "average_temperature",
        "feels_like_temperature", "precipitation_amount", "rain_indicator",
        "thunderstorm_indicator", "wind_speed", "severe_weather_indicator",
    ]},
    # Community momentum
    **{c: "Community momentum" for c in [
        "previous_event_num_games", "previous_event_unique_players",
        "previous_event_new_players_count", "previous_event_returning_players_count",
        "previous_event_draw_rate", "previous_event_games_per_player",
        "rolling_3_avg_num_games", "rolling_3_avg_new_players",
        "rolling_3_avg_returning_players",
    ]},
}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class NaiveLastPredictor:
    """Predict attendance = previous event's attendance; fallback to mean."""

    def __init__(self) -> None:
        self.fallback_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLastPredictor":
        self.fallback_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = X["previous_event_attendance"]
        return col.fillna(self.fallback_).to_numpy()

    @property
    def feature_importances_(self):  # pragma: no cover - compatibility shim
        return None


@dataclass
class Candidate:
    name: str
    label: str
    short_blurb: str
    model: Any


def _build_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="naive_last",
            label="Naive (last event)",
            short_blurb="Always predicts the previous night's attendance.",
            model=NaiveLastPredictor(),
        ),
        Candidate(
            name="ridge",
            label="Ridge Regression",
            short_blurb="Simple interpretable linear model with L2 regularization.",
            model=RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 50.0]),
        ),
        Candidate(
            name="random_forest",
            label="Random Forest",
            short_blurb="Non-linear tree ensemble; captures interactions.",
            model=RandomForestRegressor(
                n_estimators=400, max_depth=8,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            ),
        ),
        Candidate(
            name="gradient_boosting",
            label="Gradient Boosting",
            short_blurb="Sequential trees that correct each others' errors.",
            model=GradientBoostingRegressor(
                n_estimators=400, max_depth=3,
                learning_rate=0.05, random_state=42,
            ),
        ),
    ]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
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
            "temperature_high", "temperature_low", "average_temperature",
            "feels_like_temperature", "precipitation_amount", "rain_indicator",
            "thunderstorm_indicator", "wind_speed", "severe_weather_indicator",
        ]:
            events[c] = np.nan
    return events.sort_values("event_date").reset_index(drop=True)


def prepare_xy(events: pd.DataFrame):
    df = events.copy()
    X = df[FEATURE_COLUMNS].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y_reg = df["attendance_count"].astype(float)
    y_clf = df["high_turnout"].astype(int)
    return X, y_reg, y_clf, df["event_date"], medians


def cv_regression(X, y, model_factory, n_splits=4):
    """Return list of {fold, mae, rmse} dicts from TimeSeriesSplit CV."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for fold, (tr, te) in enumerate(tss.split(X)):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        metrics_list.append({
            "fold": fold,
            "mae": float(mean_absolute_error(y.iloc[te], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y.iloc[te], pred))),
        })
    return metrics_list


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
    mlflow.set_experiment("kava_chess_attendance")

    events = load_dataset()
    X, y_reg, y_clf, dates, medians = prepare_xy(events)

    # ---- chronological holdout ----
    split = max(1, len(X) - HOLDOUT_N)
    X_train, y_train = X.iloc[:split], y_reg.iloc[:split]
    X_test, y_test = X.iloc[split:], y_reg.iloc[split:]
    print(
        f"[train] events={len(X)} | train={len(X_train)} (until "
        f"{dates.iloc[split-1]:%Y-%m-%d}) | holdout={len(X_test)} (from "
        f"{dates.iloc[split]:%Y-%m-%d})"
    )

    # ---- compare candidates ----
    comparison: list[dict[str, Any]] = []
    candidates = _build_candidates()
    for cand in candidates:
        with mlflow.start_run(run_name=f"{cand.name}_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_params({
                "model_class": type(cand.model).__name__,
                "model_label": cand.label,
                "n_features": len(FEATURE_COLUMNS),
                "n_events": len(events),
                "holdout_n": HOLDOUT_N,
            })

            # CV on the training portion only (still chronological)
            cv_metrics = cv_regression(
                X_train, y_train,
                lambda c=cand: type(c.model)(**{}) if isinstance(c.model, NaiveLastPredictor)
                else type(c.model)(**c.model.get_params()),
            )
            cv_mae = float(np.mean([m["mae"] for m in cv_metrics]))
            cv_rmse = float(np.mean([m["rmse"] for m in cv_metrics]))
            mlflow.log_metric("cv_mae", cv_mae)
            mlflow.log_metric("cv_rmse", cv_rmse)

            # Fit on full train, evaluate on holdout
            model = cand.model
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, test_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            mlflow.log_metric("holdout_mae", test_mae)
            mlflow.log_metric("holdout_rmse", test_rmse)

            print(
                f"[train] {cand.label:22s}  CV MAE {cv_mae:5.2f}  "
                f"holdout MAE {test_mae:5.2f}  RMSE {test_rmse:5.2f}"
            )
            comparison.append({
                "name": cand.name, "label": cand.label, "blurb": cand.short_blurb,
                "cv_mae": cv_mae, "cv_rmse": cv_rmse,
                "holdout_mae": test_mae, "holdout_rmse": test_rmse,
            })

    # ---- pick winner ----
    naive_mae = next(c["holdout_mae"] for c in comparison if c["name"] == "naive_last")
    beats_naive = [c for c in comparison if c["name"] != "naive_last" and c["holdout_mae"] < naive_mae]
    if beats_naive:
        winner = min(beats_naive, key=lambda c: c["holdout_mae"])
    else:
        winner = min(comparison, key=lambda c: c["holdout_mae"])
        print(f"[train] WARN: no learned model beat the naive baseline; selecting {winner['label']}")

    print(f"[train] WINNER: {winner['label']}  holdout MAE {winner['holdout_mae']:.2f}")

    # ---- refit winner on full data and persist ----
    winner_cand = next(c for c in candidates if c.name == winner["name"])
    winner_model = winner_cand.model
    # Recreate fresh to avoid any prior partial state
    if not isinstance(winner_model, NaiveLastPredictor):
        winner_model = type(winner_model)(**winner_model.get_params())
    winner_model.fit(X, y_reg)

    joblib.dump(winner_model, MODELS_DIR / "attendance_regressor.joblib")
    # legacy filename for backward compat with the deployed predict.py
    joblib.dump(winner_model, MODELS_DIR / "rf_attendance_regressor.joblib")

    # ---- classifier (kept as RandomForest; not the focus of comparison) ----
    print("[train] training high-turnout classifier (RandomForestClassifier)...")
    with mlflow.start_run(run_name=f"rf_classifier_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_params({
            "model_class": "RandomForestClassifier",
            "n_estimators": 400, "max_depth": 6, "min_samples_leaf": 2,
            "median_attendance_threshold": float(events["_median_attendance_used"].iloc[0]),
        })
        tss = TimeSeriesSplit(n_splits=4)
        cv_metrics = []
        for fold, (tr, te) in enumerate(tss.split(X)):
            m = RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            m.fit(X.iloc[tr], y_clf.iloc[tr])
            pred = m.predict(X.iloc[te])
            cv_metrics.append({
                "fold": fold,
                "accuracy": float(accuracy_score(y_clf.iloc[te], pred)),
                "f1": float(f1_score(y_clf.iloc[te], pred, zero_division=0)),
            })
        acc_mean = float(np.mean([m["accuracy"] for m in cv_metrics]))
        f1_mean = float(np.mean([m["f1"] for m in cv_metrics]))
        mlflow.log_metric("cv_accuracy", acc_mean)
        mlflow.log_metric("cv_f1", f1_mean)
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(X, y_clf)
        try:
            auc = float(roc_auc_score(y_clf, clf.predict_proba(X)[:, 1]))
            mlflow.log_metric("train_auc", auc)
        except Exception:
            auc = float("nan")
        joblib.dump(clf, MODELS_DIR / "rf_high_turnout_classifier.joblib")
        print(f"[train] classifier CV ACC {acc_mean:.3f}  F1 {f1_mean:.3f}  train AUC {auc:.3f}")

    # ---- metadata & persisted history ----
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_family": FEATURE_FAMILY,
        "feature_medians": {k: float(v) if pd.notna(v) else None for k, v in medians.to_dict().items()},
        "median_attendance_threshold": float(events["_median_attendance_used"].iloc[0]),
        "n_training_events": int(len(events)),
        "holdout_n": HOLDOUT_N,
        "selected_model_name": winner["name"],
        "selected_model_label": winner["label"],
        "selected_model_blurb": winner_cand.short_blurb,
        "model_comparison": comparison,
        "naive_baseline_mae": naive_mae,
        "regression_cv_mae": winner["cv_mae"],
        "regression_cv_rmse": winner["cv_rmse"],
        "regression_holdout_mae": winner["holdout_mae"],
        "regression_holdout_rmse": winner["holdout_rmse"],
        "classification_cv_accuracy": acc_mean,
        "classification_cv_f1": f1_mean,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Rich event history: include weather + momentum columns so the app can
    # build the "Weather", "Community Momentum" and similar tabs without a
    # second join at request time.
    history_cols = [
        "event_date", "attendance_count", "num_games", "num_draws",
        "unique_players", "new_players_count", "returning_players_count",
        "games_per_player", "draw_rate", "high_turnout",
        "days_since_last_event", "is_holiday_week", "is_school_break",
        "previous_event_attendance", "previous_event_unique_players",
        "previous_event_new_players_count", "previous_event_returning_players_count",
        "rolling_3_event_attendance", "rolling_5_event_attendance",
        "rolling_3_avg_num_games", "rolling_3_avg_new_players",
        "rolling_3_avg_returning_players",
        "temperature_high", "temperature_low", "average_temperature",
        "feels_like_temperature", "precipitation_amount", "rain_indicator",
        "thunderstorm_indicator", "wind_speed", "severe_weather_indicator",
    ]
    history_cols = [c for c in history_cols if c in events.columns]
    events[history_cols].to_parquet(MODELS_DIR / "event_history.parquet", index=False)

    print(f"[train] saved model artifacts to {MODELS_DIR}")
    print(f"[train] metadata: selected_model={winner['label']}  "
          f"holdout MAE={winner['holdout_mae']:.2f} players "
          f"(naive baseline was {naive_mae:.2f})")


if __name__ == "__main__":
    main()
