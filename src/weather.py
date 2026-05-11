"""
Fetch Bradenton, FL weather from Open-Meteo and build the per-date features
the model + dashboard use.

Why this is more careful than a daily aggregate:
  Bracket nights start at 8:00 PM. A 5-minute afternoon thunderstorm at
  noon should NOT be considered "rainy" for a bracket night that
  happened seven hours later under a clear sky. So we pull hourly
  precipitation, temperature, humidity and weather codes, then
  summarise the EVENT WINDOW (7 PM - 11 PM local time, America/New_York)
  into per-date features. We also de-emphasize the binary
  rain/no-rain flag - subtropical Florida makes a yes/no rain feature
  noisy. Instead the headline weather feature is a five-component
  comfort/discomfort score.

Output columns (per date):
  Daily aggregates:
    - temperature_high / temperature_low / average_temperature
    - feels_like_temperature
    - precipitation_amount (daily total mm)
    - daily_rain_amount    (daily rain mm; kept for honesty)
    - wind_speed           (daily max km/h)
    - weather_code         (daily, for reference)
    - daily_humidity_max   (max hourly humidity %, derived)

  Event-window aggregates (7 PM - 11 PM):
    - event_window_temp_c
    - event_window_precip_mm
    - event_window_wind_kmh
    - event_window_weather_code
    - event_window_humidity   (avg humidity 7-11 PM)
    - temperature_at_8pm      (the 8 PM hourly temp reading)

  Derived flags:
    - rain_indicator           (event_window_precip_mm >= 0.5 mm)
    - thunderstorm_indicator   (event-window code in {95, 96, 99})
    - severe_weather_indicator (event-window code in {65, 75, 82, 95, 96, 99})
    - weather_discomfort_score (0-5, see comfort_score below)

The comfort score is the headline weather feature in the UI:
  +1 if daily_humidity_max          >= 80 %
  +1 if temperature_high (Celsius)  >= 31     (about 88 °F)
  +1 if precipitation_amount (mm)   >= 2.5    (about 0.10 in)
  +1 if wind_speed (km/h max)       >= 24     (about 15 mph)
  +1 if thunderstorm_indicator      == 1

Run:
    python src/weather.py
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_PARQUET = PROJECT_ROOT / "data" / "gold" / "weather_bradenton.parquet"

BRADENTON_LAT = 27.4989
BRADENTON_LON = -82.5748

# Event window: 7 PM - 11 PM local. Inclusive of 19, 20, 21, 22 (4 hours).
EVENT_WINDOW_START_HOUR = 19
EVENT_WINDOW_END_HOUR_EXCLUSIVE = 23
EVENT_PEAK_HOUR = 20  # 8 PM, the exact tip-off

# Threshold for the rain-near-8PM flag. ~0.5 mm = 0.02 inches.
RAIN_THRESHOLD_MM = 0.5

# Comfort score thresholds (units match Open-Meteo defaults: °C, mm, km/h)
DISCOMFORT_HUMIDITY_PCT = 80
DISCOMFORT_TEMP_HIGH_C = 31      # ~ 88 °F
DISCOMFORT_PRECIP_MM = 2.5       # ~ 0.10 in
DISCOMFORT_WIND_KMH = 24         # ~ 15 mph

DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "weathercode",
]

HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weathercode",
    "windspeed_10m",
]

DAILY_RENAMES = {
    "time": "event_date",
    "temperature_2m_max": "temperature_high",
    "temperature_2m_min": "temperature_low",
    "temperature_2m_mean": "average_temperature",
    "apparent_temperature_max": "feels_like_temperature",
    "precipitation_sum": "precipitation_amount",
    "rain_sum": "daily_rain_amount",
    "windspeed_10m_max": "wind_speed",
    "weathercode": "weather_code",
}

THUNDERSTORM_CODES = {95, 96, 99}
SEVERE_CODES = {65, 75, 82, 95, 96, 99}


def _hourly_to_event_window(hourly: dict) -> pd.DataFrame:
    """Build a per-date DataFrame of event-window aggregates."""
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["event_date"])
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    h["hour"] = h["time"].dt.hour
    h["date"] = h["time"].dt.normalize()

    # The 8 PM (peak) snapshot per date
    peak = h.loc[h["hour"] == EVENT_PEAK_HOUR, ["date", "temperature_2m", "relative_humidity_2m"]].copy()
    peak = peak.rename(columns={
        "date": "event_date",
        "temperature_2m": "temperature_at_8pm",
        "relative_humidity_2m": "humidity_at_8pm",
    })

    # Daily max humidity computed from hourly readings (Open-Meteo doesn't
    # expose this as a daily field directly).
    daily_humidity_max = (
        h.groupby("date")["relative_humidity_2m"].max()
        .reset_index().rename(columns={"date": "event_date", "relative_humidity_2m": "daily_humidity_max"})
    )

    in_window = (h["hour"] >= EVENT_WINDOW_START_HOUR) & (h["hour"] < EVENT_WINDOW_END_HOUR_EXCLUSIVE)
    win = h.loc[in_window].copy()
    if win.empty:
        agg = pd.DataFrame(columns=["event_date"])
    else:
        agg = win.groupby("date").agg(
            event_window_temp_c=("temperature_2m", "mean"),
            event_window_humidity=("relative_humidity_2m", "mean"),
            event_window_precip_mm=("precipitation", "sum"),
            event_window_wind_kmh=("windspeed_10m", "max"),
            event_window_weather_code=("weathercode", "max"),
        ).reset_index().rename(columns={"date": "event_date"})

    out = daily_humidity_max
    if not agg.empty:
        out = out.merge(agg, on="event_date", how="outer")
    out = out.merge(peak, on="event_date", how="outer")
    return out


def _comfort_score(row: pd.Series) -> int:
    """Five-point weather discomfort score for one date.

    0 = comfortable evening
    5 = miserable: muggy + hot + wet + windy + thunderstorm
    """
    score = 0
    h = row.get("daily_humidity_max")
    if pd.notna(h) and h >= DISCOMFORT_HUMIDITY_PCT:
        score += 1
    th = row.get("temperature_high")
    if pd.notna(th) and th >= DISCOMFORT_TEMP_HIGH_C:
        score += 1
    p = row.get("precipitation_amount")
    if pd.notna(p) and p >= DISCOMFORT_PRECIP_MM:
        score += 1
    w = row.get("wind_speed")
    if pd.notna(w) and w >= DISCOMFORT_WIND_KMH:
        score += 1
    t = row.get("thunderstorm_indicator")
    if pd.notna(t) and t >= 1:
        score += 1
    return int(score)


def _apply_flags_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the boolean flags + the comfort score."""
    df = df.copy()
    df["rain_indicator"] = (df["event_window_precip_mm"].fillna(0) >= RAIN_THRESHOLD_MM).astype(int)
    if "event_window_weather_code" in df.columns:
        df["thunderstorm_indicator"] = df["event_window_weather_code"].isin(THUNDERSTORM_CODES).astype(int)
        df["severe_weather_indicator"] = df["event_window_weather_code"].isin(SEVERE_CODES).astype(int)
    else:
        df["thunderstorm_indicator"] = 0
        df["severe_weather_indicator"] = 0
    df["weather_discomfort_score"] = df.apply(_comfort_score, axis=1)
    return df


def _fetch(url: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": BRADENTON_LAT,
        "longitude": BRADENTON_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_FIELDS),
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "America/New_York",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily", {})
    daily_df = pd.DataFrame(daily)
    if daily_df.empty:
        return daily_df
    daily_df["time"] = pd.to_datetime(daily_df["time"])
    daily_df = daily_df.rename(columns=DAILY_RENAMES)

    hourly_agg = _hourly_to_event_window(js.get("hourly", {}))

    merged = daily_df.merge(hourly_agg, on="event_date", how="left")
    merged = _apply_flags_and_score(merged)
    return merged


def fetch_historical(start: str, end: str) -> pd.DataFrame:
    return _fetch("https://archive-api.open-meteo.com/v1/archive", start, end)


def fetch_forecast(start: str, end: str) -> pd.DataFrame:
    return _fetch("https://api.open-meteo.com/v1/forecast", start, end)


def fetch_weather_for_dates(dates: list[date]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    today = date.today()
    start = min(dates)
    end = max(dates)
    frames: list[pd.DataFrame] = []
    if start <= today - timedelta(days=1):
        hist_end = min(end, today - timedelta(days=1))
        frames.append(fetch_historical(start.isoformat(), hist_end.isoformat()))
        time.sleep(0.5)
    if end >= today:
        fc_start = max(start, today)
        fc_end = min(end, today + timedelta(days=15))
        if fc_end >= fc_start:
            frames.append(fetch_forecast(fc_start.isoformat(), fc_end.isoformat()))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["event_date"]).reset_index(drop=True)
    return out


def main() -> None:
    GOLD = PROJECT_ROOT / "data" / "gold"
    GOLD.mkdir(parents=True, exist_ok=True)
    silver = pd.read_parquet(PROJECT_ROOT / "data" / "silver" / "games.parquet")
    event_dates = pd.to_datetime(silver["event_date"]).dt.date.unique().tolist()
    event_dates = sorted(event_dates)
    print(f"[weather] fetching hourly+daily for {len(event_dates)} event dates "
          f"({event_dates[0]} .. {event_dates[-1]})")
    df = fetch_weather_for_dates(event_dates + [date.today() + timedelta(days=14)])

    if not df.empty:
        n = len(df)
        print(
            f"[weather] event-window rain (>= {RAIN_THRESHOLD_MM} mm 7-11 PM): "
            f"{int(df['rain_indicator'].sum())} / {n} days "
            f"({100 * df['rain_indicator'].mean():.1f}%)"
        )
        score_counts = df["weather_discomfort_score"].value_counts().sort_index()
        print("[weather] comfort-score distribution (0 comfortable .. 5 rough):")
        for s, c in score_counts.items():
            print(f"    score={s}  ->  {int(c)} days")
        if "daily_humidity_max" in df.columns:
            print(f"[weather] daily humidity max  mean={df['daily_humidity_max'].mean():.1f}%")
        if "event_window_humidity" in df.columns:
            print(f"[weather] 7-11 PM humidity    mean={df['event_window_humidity'].mean():.1f}%")

    df.to_parquet(WEATHER_PARQUET, index=False)
    print(f"[weather] wrote {len(df)} daily rows -> {WEATHER_PARQUET}")


if __name__ == "__main__":
    main()
