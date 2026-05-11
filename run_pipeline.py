"""
One-shot runner for the full pipeline:
  bronze (S3, optional)  ->  silver  ->  weather  ->  gold  ->  train  ->  test

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-weather   # offline
    python run_pipeline.py --upload-s3 kava-chess-pipeline  # also push bronze to S3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print(f"!! step failed with code {res.returncode}")
        sys.exit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-weather", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--upload-s3", default=None, help="if set, upload bronze TSV to this S3 bucket")
    args = ap.parse_args()

    py = sys.executable

    if args.upload_s3:
        run([py, "src/upload_to_s3.py", "--bucket", args.upload_s3])

    run([py, "src/clean_kava_chess_data.py"])
    if not args.skip_weather:
        run([py, "src/weather.py"])
    run([py, "src/feature_engineering.py"])
    if not args.skip_train:
        run([py, "src/train_model.py"])
    if not args.skip_test:
        run([py, "tests/test_project.py"])


if __name__ == "__main__":
    main()
