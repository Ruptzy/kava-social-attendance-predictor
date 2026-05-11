"""
Silver layer: clean game-level Kava Social chess bracket data.

Reads raw tab-separated bracket logs from data/raw/kava_chess_games.tsv and
produces a tidy, deduplicated game-level dataset at data/silver/games.parquet.

Rules baked in (per project spec):
- Repeated dates are NOT duplicates (one bracket night has many games).
- A row is a duplicate only if Date + Time + White + Black + Winner all match.
- "Null" in White or Black means a bye / missing opponent. We keep the
  non-null player as an attendee. Both-Null rows are dropped.
- "Draw" in Winner is preserved as a draw result.
- Conservative name normalization: case + whitespace fixes only, plus the
  one merge the user confirmed: "Gonzalez, Harold" / "Harold" -> Harold.
  Ambiguous variants (e.g. Omar vs Cruz, Omar vs Omar Azab) are kept distinct.

Run:
    python src/clean_kava_chess_data.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "kava_chess_games.tsv"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
SILVER_PARQUET = SILVER_DIR / "games.parquet"
SILVER_CSV = SILVER_DIR / "games.csv"

NAME_MERGES = {
    "harold": "Gonzalez, Harold",
    "gonzalez, harold": "Gonzalez, Harold",
    "cruz, omar": "Cruz, Omar",
}


def _normalize_name(raw: str | float) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower() == "null":
        return None
    s = re.sub(r"\s+", " ", s)
    key = s.lower()
    if key in NAME_MERGES:
        return NAME_MERGES[key]
    return s


def _parse_winner(raw: str | float) -> tuple[str | None, str]:
    """Return (winner_name_or_None, result_type) where result_type in
    {white, black, draw, unknown}. Winner column may be 'Draw', a name, or blank."""
    if raw is None:
        return None, "unknown"
    if isinstance(raw, float) and pd.isna(raw):
        return None, "unknown"
    s = str(raw).strip()
    if not s:
        return None, "unknown"
    if s.lower() == "draw":
        return None, "draw"
    return _normalize_name(s), "decisive"


def load_raw(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            f"Paste your tab-separated bracket logs into that file first."
        )
    # The pasted data may contain two header rows (one per copy-paste chunk).
    # We detect repeated header rows and drop them after read.
    df = pd.read_csv(
        raw_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]
    expected = {"Date", "Time", "White", "Black", "Winner"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {missing}. Found: {df.columns.tolist()}")
    # Drop any rows that are just repeated headers
    mask_header = (df["Date"].str.strip() == "Date") & (df["White"].str.strip() == "White")
    df = df.loc[~mask_header].copy()
    # Drop completely empty rows
    df = df.loc[df["Date"].str.strip() != ""].copy()
    return df


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Parse event_date robustly across "Jan 19, 2025" style
    df["event_date"] = pd.to_datetime(df["Date"].str.strip(), format="mixed", errors="coerce")
    bad_dates = df["event_date"].isna().sum()
    if bad_dates:
        print(f"[clean] WARN: {bad_dates} rows with unparseable dates dropped")
    df = df.loc[df["event_date"].notna()].copy()

    # Parse game_time as a clock time string (e.g. "5:00 PM") -> time-of-day
    df["game_time_raw"] = df["Time"].str.strip()
    df["game_time"] = pd.to_datetime(
        df["game_time_raw"], format="%I:%M %p", errors="coerce"
    ).dt.time

    # Normalize player names
    df["white_player"] = df["White"].apply(_normalize_name)
    df["black_player"] = df["Black"].apply(_normalize_name)

    # Parse winner + result type
    parsed = df["Winner"].apply(_parse_winner)
    df["winner_player"] = parsed.apply(lambda t: t[0])
    df["result_kind"] = parsed.apply(lambda t: t[1])  # decisive | draw | unknown

    # result_type: which color won, or 'draw'
    def _result_type(row):
        if row["result_kind"] == "draw":
            return "draw"
        w = row["winner_player"]
        if w is None:
            return "unknown"
        if w == row["white_player"]:
            return "white_won"
        if w == row["black_player"]:
            return "black_won"
        # winner doesn't match either side (e.g. Charle vs Charle (albert friend))
        return "unknown"

    df["result_type"] = df.apply(_result_type, axis=1)

    # Drop rows with no actual players (both null)
    df = df.loc[df[["white_player", "black_player"]].notna().any(axis=1)].copy()

    df["comments"] = df.get("Comments", "").fillna("").astype(str).str.strip()

    out = df[
        [
            "event_date",
            "game_time",
            "game_time_raw",
            "white_player",
            "black_player",
            "winner_player",
            "result_type",
            "result_kind",
            "comments",
        ]
    ].copy()

    # Dedup: exact same Date+Time+White+Black+Winner = duplicate
    before = len(out)
    out = out.drop_duplicates(
        subset=["event_date", "game_time_raw", "white_player", "black_player", "winner_player"],
        keep="first",
    ).reset_index(drop=True)
    after = len(out)
    if before != after:
        print(f"[clean] Dropped {before - after} exact-duplicate rows")

    out = out.sort_values(["event_date", "game_time_raw"]).reset_index(drop=True)
    return out


def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    silver = clean(raw)
    silver.to_parquet(SILVER_PARQUET, index=False)
    silver.to_csv(SILVER_CSV, index=False)
    print(f"[clean] Wrote {len(silver):,} game rows -> {SILVER_PARQUET}")
    print(f"[clean] Distinct events: {silver['event_date'].nunique()}")
    print(f"[clean] Distinct players: {pd.unique(pd.concat([silver['white_player'], silver['black_player']]).dropna()).size}")


if __name__ == "__main__":
    main()
