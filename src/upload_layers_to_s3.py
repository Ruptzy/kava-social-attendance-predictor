"""
Upload the silver + gold Parquet tables to S3, alongside the bronze TSV.
This materializes the medallion architecture in cloud storage:

  s3://<bucket>/bronze/kava_chess_games.tsv
  s3://<bucket>/silver/games.parquet
  s3://<bucket>/gold/event_features.parquet
  s3://<bucket>/gold/weather_bradenton.parquet

Run:
    python src/upload_layers_to_s3.py --bucket kava-chess-pipeline-352435704328
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOADS = [
    ("data/raw/kava_chess_games.tsv", "bronze/kava_chess_games.tsv"),
    ("data/silver/games.parquet", "silver/games.parquet"),
    ("data/silver/games.csv", "silver/games.csv"),
    ("data/gold/event_features.parquet", "gold/event_features.parquet"),
    ("data/gold/event_features.csv", "gold/event_features.csv"),
    ("data/gold/weather_bradenton.parquet", "gold/weather_bradenton.parquet"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    for rel, key in UPLOADS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"  skip (missing): {rel}")
            continue
        s3.upload_file(str(path), args.bucket, key)
        print(f"  uploaded: {rel} -> s3://{args.bucket}/{key}")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
