"""
Bronze ingestion: upload the raw bracket-log TSV to S3.

This is the project's cloud-object-store landing zone (bronze).

Run:
    python src/upload_to_s3.py --bucket kava-chess-pipeline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "kava_chess_games.tsv"


def ensure_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code not in ("404", "NoSuchBucket"):
            raise
    print(f"[s3] creating bucket {bucket} in {region}")
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="bronze")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args(argv)

    if not RAW_PATH.exists():
        print(f"ERROR: raw file missing: {RAW_PATH}", file=sys.stderr)
        return 1

    s3 = boto3.client("s3", region_name=args.region)
    ensure_bucket(s3, args.bucket, args.region)

    key = f"{args.prefix}/kava_chess_games.tsv"
    s3.upload_file(str(RAW_PATH), args.bucket, key)
    print(f"[s3] uploaded {RAW_PATH} -> s3://{args.bucket}/{key}")
    print(f"[s3] bronze URI: s3://{args.bucket}/{args.prefix}/kava_chess_games.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
