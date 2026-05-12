# Kava Chess Clock

### Attendance Forecasting for Kava Social Chess Club

**Project 2 — Distributed Systems for Data Science · New College of Florida · Spring 2026**

A complete distributed data pipeline that predicts **turnout** for Kava Social
chess bracket nights in Bradenton, FL. The end product is a public web app
where a user picks a future bracket date and gets back a predicted number of
attendees, a high/normal/low turnout label, a confidence indicator, and a
planning note for the organizer.

This is **not** a chess engine, opening predictor, or rating model. It is a
small, honest tool for the people who run the bracket to plan boards,
clocks, and staffing.

- **Author:** Harold Gonzalez
- **Live app:** https://kava-social-attendance-predictor-cwlaxs6ygbxud7z48884zq.streamlit.app/
- **Predicts:** unique-player attendance for an upcoming Kava Social chess bracket night
- **Does NOT predict:** chess games, winners, player strength, openings, or individual performance
- **Data source:** historical Kava Social chess bracket / game logs, aggregated into event-level attendance
- **Venue value calculator (built in):** the app also ships with a conservative player-only drink-revenue estimator that turns predicted, historical-average, or manual attendance into per-night / monthly / yearly / cumulative dollar figures — useful for venue-partnership conversations

---

## What the model predicts

- **Primary target:** `attendance_count` for a future bracket night (regression, MAE / RMSE).
- **Secondary target:** `high_turnout` — `1` if the predicted attendance is at or above
  the historical median, otherwise `0` (classification, accuracy / F1).

Both heads use the same feature row, built from calendar facts about the target
date, Bradenton weather (Open-Meteo), and **prior-event** attendance / activity
features. No same-night features are used as predictors (anti-leakage).

## Pipeline architecture (medallion, distributed)

```
raw bracket logs (TSV)
        │
        ▼
   ┌────────────┐    upload_to_s3.py     ┌──────────────────┐
   │   local    │ ─────────────────────▶ │ S3 bronze bucket │
   └────────────┘                        └────────┬─────────┘
                                                  │
                                            spark_transform.py (PySpark)
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │  S3 silver/gold  │
                                         │ Parquet tables   │
                                         └────────┬─────────┘
                                                  │
                                       train_model.py + MLflow
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │  models/*.joblib │
                                         └────────┬─────────┘
                                                  │
                                          app/streamlit_app.py
                                                  │
                                                  ▼
                                         Streamlit Cloud (public URL)
```

**Two cloud / distributed stages** (rubric requirement):
1. **Bronze ingestion → S3** — raw TSV lands in `s3://<bucket>/bronze/`.
2. **Silver + Gold transformation in PySpark** — distributed compute reads from
   S3, cleans the game-level table, builds the event-level feature table, and
   writes Parquet back to S3 (`silver/`, `gold/`).

A pandas mirror of the same logic runs locally for fast iteration. Both paths
produce the same schema so the Streamlit app does not care which one fed it.

## Repository layout

```
kava-social-attendance-predictor/
├── README.md
├── requirements.txt
├── run_pipeline.py                # one-shot end-to-end runner
├── data/
│   ├── raw/kava_chess_games.tsv   # bronze input (raw bracket logs)
│   ├── silver/games.parquet       # cleaned game-level table
│   └── gold/event_features.parquet  # event-level model features
├── src/
│   ├── upload_to_s3.py            # bronze ingestion to S3
│   ├── clean_kava_chess_data.py   # silver cleaner (pandas)
│   ├── feature_engineering.py     # gold feature table (pandas, leakage-safe)
│   ├── weather.py                 # Open-Meteo Bradenton weather enrichment
│   ├── spark_transform.py         # PySpark mirror that reads/writes S3
│   ├── train_model.py             # RF regressor + RF classifier, MLflow tracked
│   └── predict.py                 # inference helper for any future date
├── app/streamlit_app.py           # public web UI
├── tests/test_project.py          # end-to-end test script
├── models/                        # trained .joblib + metadata.json + event_history.parquet
├── mlruns/                        # MLflow tracking store
└── reports/one_page_writeup.pdf   # one-page project summary
```

## Running it locally

### 0. Save the raw data

Paste your bracket-log TSV into `data/raw/kava_chess_games.tsv`. The cleaner
tolerates trailing tabs, duplicate header rows, and Null opponents.

### 1. Create the Python 3.12 environment

PySpark / MLflow / scikit-learn do not yet support Python 3.14, so a 3.12
virtualenv is required. From the project root:

```bash
# from the project root
C:\python-uv\cpython-3.12.13-windows-x86_64-none\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# requirements.txt is the slim app-runtime set (used by Streamlit Cloud).
# requirements-pipeline.txt adds pyspark, boto3, mlflow, reportlab, openpyxl
# for running the local distributed pipeline.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-pipeline.txt
```

### 2. Run the whole pipeline

```bash
.\.venv\Scripts\python.exe run_pipeline.py
```

This runs: silver clean → Bradenton weather fetch → gold feature engineering →
MLflow-tracked model training → end-to-end test.

To include the S3 bronze upload step:

```bash
.\.venv\Scripts\python.exe run_pipeline.py --upload-s3 kava-chess-pipeline
```

To run the **distributed PySpark transform** against S3:

```bash
.\.venv\Scripts\python.exe src/spark_transform.py \
    --bronze s3://kava-chess-pipeline/bronze/kava_chess_games.tsv \
    --silver s3://kava-chess-pipeline/silver/games \
    --gold   s3://kava-chess-pipeline/gold/event_features
```

### 3. Launch the Streamlit app

```bash
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

### 4. Run the test script

```bash
.\.venv\Scripts\python.exe tests/test_project.py
# Or, against the live deployed URL:
$env:KAVA_APP_URL = "https://your-app.streamlit.app"
.\.venv\Scripts\python.exe tests/test_project.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Create new app:
   - **Repository:** `your-username/kava-social-attendance-predictor`
   - **Branch:** `main`
   - **Main file:** `app/streamlit_app.py`
   - **Python version:** 3.12
4. Streamlit Cloud will install `requirements.txt` automatically. The model
   artifacts in `models/` are committed so the app can serve predictions out
   of the box (the `.gitignore` keeps `*.joblib` out by default — you'll need
   to either commit them or have the app retrain on startup; see the
   "Deploy with committed model" note below).
5. Once deployed, paste the public URL into this README (top of file) and into
   the one-page write-up.

**Deploy with committed model.** For simplicity at deploy time, comment out
the `models/*.joblib` and `models/event_history.parquet` lines in
`.gitignore` before committing, so the trained artifacts ship with the repo.
The app loads them directly — no S3 / MLflow round-trip at request time.

## Anti-leakage notes (important for the model rubric)

Same-night activity (`num_games`, `draw_rate`, `unique_players` of the night
we're predicting) is **never** used as a model input. Only:
- calendar facts knowable in advance (month, day-of-week, holiday window, ...),
- prior-event lag and rolling features (`previous_event_attendance`,
  `rolling_3_event_attendance`, ...),
- prior-event community momentum (`previous_event_new_players_count`, ...),
- and Bradenton, FL weather (historical or forecast) for the target date.

The classifier's "high vs low" threshold is the historical median attendance
computed once over the gold table — see `models/metadata.json`.

## Model bake-off & evaluation

`train_model.py` compares four candidates head-to-head:

| Model | Holdout MAE (last 14 events) | Notes |
| --- | --- | --- |
| Naive (last event) | 3.43 players | Always predicts the previous night's count. The baseline to beat. |
| Ridge Regression | 4.71 players | Simple interpretable linear model — too rigid for this signal. |
| **Random Forest** | **2.30 players** ✅ | Non-linear tree ensemble. **Selected.** |
| Gradient Boosting | 2.56 players | Strong second place. |

The selected Random Forest model improved over the naive baseline by about
1.1 players on the chronological holdout set. All runs are tracked in MLflow
under `./mlruns/`. The classifier (for the "high turnout" label) is kept as
a Random Forest. Exact values live in `models/metadata.json`.

**Anti-leakage.** The model only uses information that would be known *before*
the event starts — calendar facts, weather forecast for that date, and what
happened on previous bracket nights. No same-night counts are used as
predictors. See `src/feature_engineering.py` for the `.shift(1)` operations
that enforce this.

## Credentials

No AWS keys, Databricks tokens, or API keys are committed. The S3 client uses
the default AWS credentials chain (`~/.aws/credentials`). Open-Meteo requires
no key.

## Acknowledgements

Course: Distributed Systems for Data Science (NCF, Spring 2026), instructor
Gil + Paige. Data: my own Kava Social chess bracket logs.
