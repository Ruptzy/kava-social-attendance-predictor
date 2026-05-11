"""
End-to-end test script for the Kava Social Chess Attendance Predictor.

What it does:
  1. Imports the local prediction module and runs a sample prediction for a
     future Sunday, printing the result.
  2. (Optional) If KAVA_APP_URL is set in the environment, also makes an HTTP
     GET to the live Streamlit app with ?date=... and confirms it returns 200.

Exit codes:
  0 on success, non-zero on any failure.

Requirements:
    pip install -r requirements.txt

Usage:
    python tests/test_project.py
    KAVA_APP_URL=https://your-app.streamlit.app python tests/test_project.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _next_sunday(d: date) -> date:
    days_ahead = (6 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


def _test_local_prediction() -> dict:
    try:
        from src.predict import predict_for_date
    except Exception as e:
        print(f"FAIL: could not import src.predict ({e})")
        raise

    sample_date = _next_sunday(date.today() + timedelta(days=14))
    print(f"[test] running local prediction for {sample_date}")
    pred = predict_for_date(sample_date)
    out = asdict(pred)

    assert isinstance(out["predicted_attendance"], (int, float)), "predicted_attendance must be numeric"
    assert out["predicted_attendance_rounded"] > 0, "predicted attendance should be > 0"
    assert 0.0 <= out["high_turnout_probability"] <= 1.0, "probability must be in [0,1]"
    assert out["turnout_category"] in ("Low", "Normal", "High"), "category must be Low/Normal/High"

    print("[test] LOCAL PREDICTION:")
    print(json.dumps(
        {k: v for k, v in out.items() if k not in ("features_used",)},
        indent=2,
        default=str,
    ))
    return out


def _test_live_url() -> bool:
    url = os.environ.get("KAVA_APP_URL")
    if not url:
        print("[test] KAVA_APP_URL not set - skipping live URL check.")
        return True
    sample_date = _next_sunday(date.today() + timedelta(days=14))
    full_url = f"{url.rstrip('/')}/?date={sample_date.isoformat()}"
    try:
        import requests
    except ImportError:
        print("[test] requests not installed - skipping live URL check.")
        return True
    print(f"[test] GET {full_url}")
    r = requests.get(full_url, timeout=30)
    print(f"[test] response status: {r.status_code}")
    if r.status_code != 200:
        print(f"FAIL: live URL returned {r.status_code}")
        return False
    body = r.text.lower()
    if "kava" not in body and "attendance" not in body:
        print("FAIL: response body did not look like the Kava app")
        return False
    print("[test] LIVE URL OK")
    return True


def main() -> int:
    print("=" * 60)
    print("Kava Social Chess Attendance Predictor - End-to-end test")
    print("=" * 60)
    try:
        _test_local_prediction()
    except AssertionError as e:
        print(f"FAIL: assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"FAIL: unexpected error in local prediction: {e}")
        return 2

    if not _test_live_url():
        return 3

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
