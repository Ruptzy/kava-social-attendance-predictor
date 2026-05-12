"""
Gold layer: event-level attendance + leakage-safe feature table.

Reads cleaned game-level rows from data/silver/games.parquet and writes the
event-level model-ready table to data/gold/event_features.parquet.

Attendance rule:
  attendance_count(event_date) = # of unique non-null players who appear in
  either White or Black on that date.

Leakage rule:
  Predictive features for predicting attendance on event_date E are derived
  ONLY from events strictly before E (or from the calendar itself). Same-night
  game counts, draw rates, etc. are stored for analytics/visualization but
  are NOT used as model inputs.

Run:
    python src/feature_engineering.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_PARQUET = PROJECT_ROOT / "data" / "silver" / "games.parquet"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
GOLD_PARQUET = GOLD_DIR / "event_features.parquet"
GOLD_CSV = GOLD_DIR / "event_features.csv"


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


STANDINGS_SUPPLEMENT_PATH = PROJECT_ROOT / "data" / "raw" / "standings_supplement.csv"

# Reuse the same name-merge rules the silver cleaner uses so player names
# from standings paste-ins line up with player names from the game logs.
_SUPPLEMENT_NAME_MERGES = {
    "harold": "Gonzalez, Harold",
    "gonzalez, harold": "Gonzalez, Harold",
    "cruz, omar": "Cruz, Omar",
}


def _normalize_supplement_name(raw: str) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "null":
        return None
    key = s.lower()
    if key in _SUPPLEMENT_NAME_MERGES:
        return _SUPPLEMENT_NAME_MERGES[key]
    return s


def _load_standings_supplement() -> pd.DataFrame:
    """Read the manually-pasted Swiss-standings supplement.

    The supplement records dates where we only have a player list + total
    attendance (no game-by-game logs). It is merged into the event-level
    table after the silver-game aggregation - it is NEVER injected into
    the silver game-level pipeline. The columns it produces in the gold
    table are: attendance_count + the player set (for new/returning
    accounting); fields like num_games / draw_rate / games_per_player
    are intentionally NaN because we genuinely don't know them.

    Each row carries `source_type = "standings_summary"` so downstream
    consumers can tell where each row came from.
    """
    if not STANDINGS_SUPPLEMENT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(STANDINGS_SUPPLEMENT_PATH)
    if df.empty:
        return df
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df["players_list"] = df["players"].fillna("").apply(
        lambda s: {n for n in (_normalize_supplement_name(p) for p in str(s).split(",")) if n}
    )
    df["source_type"] = "standings_summary"
    df["source_note"] = df.get("source_note", "")
    df["attendance_count"] = df["attendance_count"].astype(int)
    return df[["event_date", "attendance_count", "players_list", "source_type", "source_note"]]


def build_event_table(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-level rows to one row per event_date, then merge in
    the standings supplement for dates that the game logs don't cover."""
    g = games.copy()
    g["event_date"] = pd.to_datetime(g["event_date"]).dt.normalize()

    # Unique attendees per event (non-null only)
    def _attendees(sub: pd.DataFrame) -> set[str]:
        names = pd.concat([sub["white_player"], sub["black_player"]]).dropna()
        return set(names.tolist())

    rows = []
    for ev_date, sub in g.groupby("event_date", sort=True):
        attendees = _attendees(sub)
        num_games = len(sub)
        num_draws = (sub["result_type"] == "draw").sum()
        unique_players = len(attendees)
        games_per_player = num_games / unique_players if unique_players else 0.0
        draw_rate = num_draws / num_games if num_games else 0.0
        rows.append(
            {
                "event_date": ev_date,
                "attendance_count": unique_players,
                "num_games": num_games,
                "num_draws": int(num_draws),
                "unique_players": unique_players,
                "games_per_player": games_per_player,
                "draw_rate": draw_rate,
                "_attendees": attendees,
                "source_type": "game_logs",
                "source_note": "",
            }
        )

    ev = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)

    # Merge in standings-summary dates that are NOT already covered by the
    # game logs. Existing game-logs dates always win - we never overwrite a
    # real game-level count with a standings count.
    sup = _load_standings_supplement()
    if not sup.empty:
        existing = set(ev["event_date"].dt.normalize())
        added = 0
        for _, srow in sup.iterrows():
            if srow["event_date"] in existing:
                continue
            ev = pd.concat([ev, pd.DataFrame([{
                "event_date": srow["event_date"],
                "attendance_count": int(srow["attendance_count"]),
                # We don't know game counts from a standings paste-in.
                # Leave NaN so downstream median-fill handles them honestly.
                "num_games": np.nan,
                "num_draws": np.nan,
                "unique_players": int(srow["attendance_count"]),
                "games_per_player": np.nan,
                "draw_rate": np.nan,
                "_attendees": set(srow["players_list"]),
                "source_type": "standings_summary",
                "source_note": str(srow["source_note"]),
            }])], ignore_index=True)
            added += 1
        if added:
            print(f"[gold] merged {added} standings-summary event(s) "
                  "(date(s) absent from the game-level logs)")
        ev = ev.sort_values("event_date").reset_index(drop=True)

    return ev


def add_new_returning(events: pd.DataFrame) -> pd.DataFrame:
    """Add new_players_count / returning_players_count using the running set of
    attendees seen in PRIOR events only (no leakage)."""
    seen_before: set[str] = set()
    new_counts: list[int] = []
    returning_counts: list[int] = []
    for _, row in events.iterrows():
        attendees: set[str] = row["_attendees"]
        new_n = len(attendees - seen_before)
        ret_n = len(attendees & seen_before)
        new_counts.append(new_n)
        returning_counts.append(ret_n)
        seen_before |= attendees
    events = events.copy()
    events["new_players_count"] = new_counts
    events["returning_players_count"] = returning_counts
    return events


def add_calendar_features(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["year"] = e["event_date"].dt.year
    e["month"] = e["event_date"].dt.month
    e["day_of_month"] = e["event_date"].dt.day
    e["day_of_week"] = e["event_date"].dt.dayofweek  # 0=Mon
    e["is_sunday"] = (e["day_of_week"] == 6).astype(int)
    e["week_of_year"] = e["event_date"].dt.isocalendar().week.astype(int)
    e["season"] = e["month"].apply(_season)
    e["is_beginning_of_month"] = (e["day_of_month"] <= 7).astype(int)
    e["is_end_of_month"] = (e["day_of_month"] >= 24).astype(int)
    # US holidays (Bradenton, FL)
    try:
        import holidays as _holidays

        us_h = _holidays.UnitedStates(years=sorted(e["year"].unique().tolist()))
        e["is_holiday_week"] = e["event_date"].apply(
            lambda d: int(any((d + pd.Timedelta(days=k)).date() in us_h for k in range(-3, 4)))
        )
    except Exception:
        e["is_holiday_week"] = 0
    e["is_school_break"] = (
        ((e["month"] == 6) | (e["month"] == 7))  # summer
        | ((e["month"] == 12) & (e["day_of_month"] >= 18))  # winter break
        | ((e["month"] == 1) & (e["day_of_month"] <= 5))
        | ((e["month"] == 3) & (e["day_of_month"].between(10, 20)))  # spring break window
    ).astype(int)
    return e


def add_lag_rolling(events: pd.DataFrame) -> pd.DataFrame:
    """Add prior-event lag and rolling features (strictly shifted to avoid leakage)."""
    e = events.copy().sort_values("event_date").reset_index(drop=True)

    e["previous_event_attendance"] = e["attendance_count"].shift(1)
    e["attendance_two_events_ago"] = e["attendance_count"].shift(2)

    for w in (3, 5, 10):
        e[f"rolling_{w}_event_attendance"] = (
            e["attendance_count"].shift(1).rolling(window=w, min_periods=1).mean()
        )

    e["attendance_trend_last_3"] = (
        e["attendance_count"].shift(1) - e["attendance_count"].shift(3)
    )
    e["attendance_trend_last_5"] = (
        e["attendance_count"].shift(1) - e["attendance_count"].shift(5)
    )

    # Prior-event community momentum (shifted - no leakage)
    e["previous_event_num_games"] = e["num_games"].shift(1)
    e["previous_event_unique_players"] = e["unique_players"].shift(1)
    e["previous_event_new_players_count"] = e["new_players_count"].shift(1)
    e["previous_event_returning_players_count"] = e["returning_players_count"].shift(1)
    e["previous_event_draw_rate"] = e["draw_rate"].shift(1)
    e["previous_event_games_per_player"] = e["games_per_player"].shift(1)

    e["rolling_3_avg_num_games"] = (
        e["num_games"].shift(1).rolling(window=3, min_periods=1).mean()
    )
    e["rolling_3_avg_new_players"] = (
        e["new_players_count"].shift(1).rolling(window=3, min_periods=1).mean()
    )
    e["rolling_3_avg_returning_players"] = (
        e["returning_players_count"].shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Gap / scheduling features (based purely on past dates, not future)
    e["days_since_last_event"] = (
        e["event_date"] - e["event_date"].shift(1)
    ).dt.days
    e["biweekly_event_indicator"] = (
        e["days_since_last_event"].between(12, 16)
    ).astype("Int64")
    e["weekly_event_indicator"] = (
        e["days_since_last_event"].between(5, 9)
    ).astype("Int64")
    e["back_to_back_event_indicator"] = (
        e["days_since_last_event"] <= 3
    ).astype("Int64")
    e["first_event_after_long_break"] = (
        e["days_since_last_event"] >= 30
    ).astype("Int64")

    # previous_event_high_turnout (relative to running median up to the prior event)
    running_median = e["attendance_count"].expanding().median().shift(1)
    e["previous_event_high_turnout"] = (
        (e["attendance_count"].shift(1) >= running_median).astype("Int64")
    )

    # events_this_month_so_far: prior-event count in same month/year
    counts = []
    seen = {}  # (year, month) -> count
    for _, row in e.iterrows():
        key = (row["event_date"].year, row["event_date"].month)
        counts.append(seen.get(key, 0))
        seen[key] = seen.get(key, 0) + 1
    e["events_this_month_so_far"] = counts

    e["event_number_overall"] = np.arange(len(e))  # 0-indexed, prior count
    e["event_number_in_year"] = (
        e.groupby("year").cumcount()
    )

    return e


def add_target(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    median_attendance = e["attendance_count"].median()
    e["high_turnout"] = (e["attendance_count"] >= median_attendance).astype(int)
    e["_median_attendance_used"] = median_attendance
    return e


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(SILVER_PARQUET)
    events = build_event_table(games)
    events = add_new_returning(events)
    events = add_calendar_features(events)
    events = add_lag_rolling(events)
    events = add_target(events)
    events = events.drop(columns=["_attendees"])
    events.to_parquet(GOLD_PARQUET, index=False)
    events.to_csv(GOLD_CSV, index=False)
    print(f"[gold] Wrote {len(events):,} event rows -> {GOLD_PARQUET}")
    print(f"[gold] Attendance summary:")
    print(events["attendance_count"].describe().to_string())
    print(f"[gold] Median attendance (used for high_turnout target): {events['_median_attendance_used'].iloc[0]}")


if __name__ == "__main__":
    main()
