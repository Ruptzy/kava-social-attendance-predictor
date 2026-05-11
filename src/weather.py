"""
Fetch Bradenton, FL weather from Open-Meteo (free, no API key) and aggregate
it around the Kava Social chess bracket event window.

Why this is more careful than a daily aggregate:
  Bracket nights start at 8:00 PM. A 5-minute afternoon thunderstorm at
  noon should NOT be considered "rainy" for a bracket night that
  happened seven hours later under a clear sky. So we pull hourly
  precipitation, temperature, and weather codes, then summarise the
  EVENT WINDOW (default 7 PM - 11 PM local time, America/New_York)
  into per-date features.

Output columns (per date):
  - temperature_high / low / mean / feels_like_temperature  (daily aggregates)
  - precipitation_amount                                     (daily total mm)
  - event_window_temp_c                                      (avg temp 7-11 PM)
  - event_window_precip_mm                                   (sum precip 7-11 PM)
  - rain_indicator           = event_window_precip_mm >= 0.5 mm (~0.02 in)
  - thunderstorm_indicator   = event-window weather code in {95, 96, 99}
  - severe_weather_indicator = event-window weather code in {65, 75, 82, 95, 96, 99}
  - wind_speed                                               (daily max)
  - weather_code                                             (daily, for reference)

When hourly data is missing for some rows (e.g. the API returned only
daily for a far-future date), the event-window features fall back to NaN
and the model fills them with the training-set medians.

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

# Event window is 7:00 PM - 11:00 PM local time. We aggregate hours
# 19, 20, 21, 22 (inclusive).
EVENT_WINDOW_START_HOUR = 19
EVENT_WINDOW_END_HOUR_EXCLUSIVE = 23

# Anything at or above this many millimetres in the event window counts
# as a real rainy night. ~0.5 mm is roughly 0.02 inches - enough to feel
# but not a single passing drop.
RAIN_THRESHOLD_MM = 0.5

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hourly_to_event_window(hourly: dict) -> pd.DataFrame:
    """Given Open-Meteo's hourly response dict, build a per-date DataFrame
    with event-window aggregates (precipitation, temperature, weather code)."""
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["event_date"])
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    h["hour"] = h["time"].dt.hour
    h["date"] = h["time"].dt.normalize()

    in_window = (h["hour"] >= EVENT_WINDOW_START_HOUR) & (h["hour"] < EVENT_WINDOW_END_HOUR_EXCLUSIVE)
    win = h.loc[in_window].copy()
    if win.empty:
        return pd.DataFrame(columns=["event_date"])

    # Sum precipitation over the four event-window hours (mm).
    precip_col = "precipitation" if "precipitation" in win.columns else "rain"
    if precip_col not in win.columns:
        win[precip_col] = np.nan

    agg = win.groupby("date").agg(
        event_window_precip_mm=(precip_col, "sum"),
        event_window_temp_c=("temperature_2m", "mean") if "temperature_2m" in win.columns else (precip_col, "size"),
        event_window_wind_kmh=("windspeed_10m", "max") if "windspeed_10m" in win.columns else (precip_col, "size"),
    ).reset_index()
    agg = agg.rename(columns={"date": "event_date"})

    # Max weather code seen in the event window (single number per date).
    if "weathercode" in win.columns:
        wc = (
            win.groupby("date")["weathercode"]
            .max()
            .reset_index()
            .rename(columns={"date": "event_date", "weathercode": "event_window_weather_code"})
        )
        agg = agg.merge(wc, on="event_date", how="left")
    else:
        agg["event_window_weather_code"] = np.nan

    return agg


def _apply_event_window_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the boolean event-window indicators from the aggregates."""
    df = df.copy()
    df["rain_indicator"] = (df["event_window_precip_mm"].fillna(0) >= RAIN_THRESHOLD_MM).astype(int)
    if "event_window_weather_code" in df.columns:
        df["thunderstorm_indicator"] = df["event_window_weather_code"].isin(THUNDERSTORM_CODES).astype(int)
        df["severe_weather_indicator"] = df["event_window_weather_code"].isin(SEVERE_CODES).astype(int)
    else:
        df["thunderstorm_indicator"] = 0
        df["severe_weather_indicator"] = 0
    return df


def _fetch(url: str, start: str, end: str) -> pd.DataFrame:
    """Fetch BOTH daily and hourly weather in a single call; assemble a
    per-date frame with event-window features merged in."""
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
    merged = _apply_event_window_flags(merged)
    return merged


def fetch_historical(start: str, end: str) -> pd.DataFrame:
    return _fetch("https://archive-api.open-meteo.com/v1/archive", start, end)


def fetch_forecast(start: str, end: str) -> pd.DataFrame:
    return _fetch("https://api.open-meteo.com/v1/forecast", start, end)


def fetch_weather_for_dates(dates: list[date]) -> pd.DataFrame:
    """Cover the min..max range of the given dates, splitting between
    historical archive (for past dates) and forecast (for future / today)."""
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
        # Forecast API allows up to ~16 days ahead
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
        # Quick sanity print for the rainy-night threshold (event-window rain).
        n_rain = int(df["rain_indicator"].sum())
        n_total = len(df)
        print(
            f"[weather] event-window rain (>= {RAIN_THRESHOLD_MM} mm 7-11 PM): "
            f"{n_rain} / {n_total} days ({100 * n_rain / n_total:.1f}%)"
        )
        # Also report daily-trace rain for context.
        if "daily_rain_amount" in df.columns:
            n_trace = int((df["daily_rain_amount"].fillna(0) > 0.1).sum())
            print(
                f"[weather] daily ANY-trace rain (> 0.1 mm anytime): "
                f"{n_trace} / {n_total} days ({100 * n_trace / n_total:.1f}%) "
                "<-- the old, overcounting metric, for comparison"
            )

    df.to_parquet(WEATHER_PARQUET, index=False)
    print(f"[weather] wrote {len(df)} daily rows -> {WEATHER_PARQUET}")


if __name__ == "__main__":
    main()
