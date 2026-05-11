"""
Fetch historical and forecast daily weather for Bradenton, FL from Open-Meteo
(free, no API key). Used to enrich event-level features.

Run:
    python src/weather.py
"""
from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import time

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_PARQUET = PROJECT_ROOT / "data" / "gold" / "weather_bradenton.parquet"

BRADENTON_LAT = 27.4989
BRADENTON_LON = -82.5748


def fetch_historical(start: str, end: str) -> pd.DataFrame:
    """Pull daily history from Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": BRADENTON_LAT,
        "longitude": BRADENTON_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "apparent_temperature_max",
                "precipitation_sum",
                "rain_sum",
                "windspeed_10m_max",
                "weathercode",
            ]
        ),
        "timezone": "America/New_York",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    daily = js.get("daily", {})
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "time": "event_date",
            "temperature_2m_max": "temperature_high",
            "temperature_2m_min": "temperature_low",
            "temperature_2m_mean": "average_temperature",
            "apparent_temperature_max": "feels_like_temperature",
            "precipitation_sum": "precipitation_amount",
            "rain_sum": "rain_amount",
            "windspeed_10m_max": "wind_speed",
            "weathercode": "weather_code",
        }
    )
    df["rain_indicator"] = (df["rain_amount"].fillna(0) > 0.1).astype(int)
    df["thunderstorm_indicator"] = df["weather_code"].isin([95, 96, 99]).astype(int)
    df["severe_weather_indicator"] = df["weather_code"].isin([95, 96, 99, 65, 75, 82]).astype(int)
    return df


def fetch_forecast(start: str, end: str) -> pd.DataFrame:
    """Pull daily forecast (up to ~16 days out) from Open-Meteo forecast API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": BRADENTON_LAT,
        "longitude": BRADENTON_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "apparent_temperature_max",
                "precipitation_sum",
                "rain_sum",
                "windspeed_10m_max",
                "weathercode",
            ]
        ),
        "timezone": "America/New_York",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    daily = js.get("daily", {})
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "time": "event_date",
            "temperature_2m_max": "temperature_high",
            "temperature_2m_min": "temperature_low",
            "temperature_2m_mean": "average_temperature",
            "apparent_temperature_max": "feels_like_temperature",
            "precipitation_sum": "precipitation_amount",
            "rain_sum": "rain_amount",
            "windspeed_10m_max": "wind_speed",
            "weathercode": "weather_code",
        }
    )
    df["rain_indicator"] = (df["rain_amount"].fillna(0) > 0.1).astype(int)
    df["thunderstorm_indicator"] = df["weather_code"].isin([95, 96, 99]).astype(int)
    df["severe_weather_indicator"] = df["weather_code"].isin([95, 96, 99, 65, 75, 82]).astype(int)
    return df


def fetch_weather_for_dates(dates: list[date]) -> pd.DataFrame:
    """Fetch weather covering the min..max range of the given dates, splitting
    into historical vs forecast as appropriate."""
    if not dates:
        return pd.DataFrame()
    today = date.today()
    start = min(dates)
    end = max(dates)
    frames = []
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
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["event_date"]).reset_index(drop=True)


def main() -> None:
    GOLD = PROJECT_ROOT / "data" / "gold"
    GOLD.mkdir(parents=True, exist_ok=True)
    silver = pd.read_parquet(PROJECT_ROOT / "data" / "silver" / "games.parquet")
    event_dates = (
        pd.to_datetime(silver["event_date"]).dt.date.unique().tolist()
    )
    event_dates = sorted(event_dates)
    print(f"[weather] Fetching weather for {len(event_dates)} event dates "
          f"({event_dates[0]} .. {event_dates[-1]})")
    df = fetch_weather_for_dates(event_dates + [date.today() + timedelta(days=14)])
    df.to_parquet(WEATHER_PARQUET, index=False)
    print(f"[weather] Wrote {len(df)} daily rows -> {WEATHER_PARQUET}")


if __name__ == "__main__":
    main()
