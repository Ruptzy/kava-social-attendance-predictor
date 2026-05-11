"""
One-off ingestion: read the latest Kava Social chess ranking xlsx and write a
clean TSV bronze file at data/raw/kava_chess_games.tsv with the canonical
columns (Date, Time, White, Black, Winner, Comments).

The xlsx workbooks each have one sheet per bracket night (or per season).
We scan every sheet, pull rows that match the bracket-game schema, and emit
one flat TSV.

Run:
    python src/xlsx_to_raw.py
    python src/xlsx_to_raw.py --xlsx "C:/path/to/your.xlsx"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, date

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "kava_chess_games.tsv"

CANONICAL_COLUMNS = ["Date", "Time", "White", "Black", "Winner", "Comments"]


def _looks_like_bracket_sheet(df: pd.DataFrame) -> bool:
    """A bracket sheet has columns including White, Black, Winner (or close)."""
    cols = {str(c).strip().lower() for c in df.columns}
    return ("white" in cols and "black" in cols and "winner" in cols)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        cs = str(c).strip()
        lc = cs.lower()
        if lc == "date":
            rename[c] = "Date"
        elif lc == "time":
            rename[c] = "Time"
        elif lc == "white":
            rename[c] = "White"
        elif lc == "black":
            rename[c] = "Black"
        elif lc == "winner":
            rename[c] = "Winner"
        elif lc.startswith("comment"):
            rename[c] = "Comments"
    df = df.rename(columns=rename)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[CANONICAL_COLUMNS]


def _coerce_date_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v).strftime("%b %-d, %Y") if sys.platform != "win32" else pd.Timestamp(v).strftime("%b %#d, %Y")
    s = str(v).strip()
    # Try to parse mixed string formats and re-emit canonical
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%b %#d, %Y") if sys.platform == "win32" else ts.strftime("%b %-d, %Y")
    except Exception:
        pass
    return s


def _coerce_time_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(v)
        return ts.strftime("%-I:%M %p") if sys.platform != "win32" else ts.strftime("%#I:%M %p")
    if hasattr(v, "hour") and hasattr(v, "minute"):
        try:
            h, m = int(v.hour), int(v.minute)
            ts = pd.Timestamp(year=2000, month=1, day=1, hour=h, minute=m)
            return ts.strftime("%#I:%M %p") if sys.platform == "win32" else ts.strftime("%-I:%M %p")
        except Exception:
            pass
    return str(v).strip()


def read_workbook(xlsx_path: Path) -> pd.DataFrame:
    print(f"[xlsx] reading {xlsx_path}")
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    frames = []
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet, dtype=object)
        except Exception as e:
            print(f"[xlsx]   skip sheet {sheet!r}: {e}")
            continue
        if df.empty:
            continue
        # Find the header row: scan first 5 rows for one that contains White/Black/Winner
        header_row = None
        for i in range(min(len(df), 6)):
            row_vals = {str(v).strip().lower() for v in df.iloc[i].tolist() if pd.notna(v)}
            if "white" in row_vals and "black" in row_vals and "winner" in row_vals:
                header_row = i
                break
        if header_row is not None and header_row > 0:
            df = xl.parse(sheet, dtype=object, header=header_row)
        if not _looks_like_bracket_sheet(df):
            continue
        df = _normalize_columns(df)
        # Strip rows with no players at all
        df = df[(df["White"].astype(str).str.strip() != "") | (df["Black"].astype(str).str.strip() != "")]
        if df.empty:
            continue
        # If Date column is empty on every row but the sheet name encodes a date, fill it
        if (df["Date"].astype(str).str.strip() == "").all():
            try:
                ts = pd.to_datetime(sheet, errors="raise")
                df["Date"] = ts.strftime("%#b %#d, %Y") if sys.platform == "win32" else ts.strftime("%b %-d, %Y")
            except Exception:
                pass
        df["Date"] = df["Date"].apply(_coerce_date_cell)
        df["Time"] = df["Time"].apply(_coerce_time_cell)
        for col in ("White", "Black", "Winner", "Comments"):
            df[col] = df[col].fillna("").astype(str).str.strip()
        frames.append(df)
        print(f"[xlsx]   sheet {sheet!r}: kept {len(df)} rows")
    if not frames:
        raise RuntimeError(f"No bracket sheets found in {xlsx_path}")
    out = pd.concat(frames, ignore_index=True)
    # Drop fully empty Date rows
    out = out[out["Date"].astype(str).str.strip() != ""]
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xlsx",
        default=str(
            Path.home()
            / "Desktop"
            / "Chess league Kava Social"
            / "Chess rankings NEW 2025"
            / "2026-04-05 2026-03-22 2025-11-30 11_2_2025 chess ranking Backup (1) Backup - Copy Backup.xlsx"
        ),
    )
    ap.add_argument("--out", default=str(RAW_PATH))
    args = ap.parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        print(f"ERROR: xlsx not found: {xlsx_path}", file=sys.stderr)
        return 1
    df = read_workbook(xlsx_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[xlsx] wrote {len(df):,} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
