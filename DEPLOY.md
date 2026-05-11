# Deployment — final mile checklist

The repo is committed locally as `main`. Three things left:

## 1. Push to GitHub

Create a new **public** repo on the GitHub account you want to grade against
(your `ruptzy` account or whichever — your call):

```powershell
# in PowerShell, from the project root
git remote add origin https://github.com/<your-username>/kava-social-attendance-predictor.git
git push -u origin main
```

If `gh` CLI is installed and logged in to the right account, you can do it in
one step:

```powershell
gh repo create kava-social-attendance-predictor --public --source=. --remote=origin --push
```

If the project repo lives on a different GitHub account than your `git config`
identity, push will prompt for credentials — pick the account you want.

## 2. Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app** → select your repo → branch `main` → main file:
   `app/streamlit_app.py`.
3. Click **Advanced settings**, set **Python version = 3.12**.
4. Deploy. Streamlit Cloud will install `requirements.txt` and start the app.
5. Copy the public URL.

## 3. Paste the URL into the grading artifacts

1. Open `README.md` → replace the `<paste your Streamlit Community Cloud URL here once deployed>` line with the real URL.
2. Open `reports/one_page_writeup.md` → top of the file, add the URL on the byline.
3. Regenerate the PDF:
   ```powershell
   .\.venv\Scripts\python.exe src\make_writeup_pdf.py
   ```
4. Commit + push:
   ```powershell
   git add README.md reports/
   git commit -m "Add live app URL to README and write-up"
   git push
   ```

## Smoke-test the live app

Once deployed, run the test script against the live URL:

```powershell
$env:KAVA_APP_URL = "https://<your-app>.streamlit.app"
.\.venv\Scripts\python.exe tests\test_project.py
```

Expected output: `PASS`, exit code 0.

## What the grader gets

| Artifact | Where |
| --- | --- |
| Live URL | `README.md` top + `reports/one_page_writeup.pdf` |
| Test script | `tests/test_project.py` (run it locally, no env var needed) |
| One-page PDF | `reports/one_page_writeup.pdf` |
| GitHub repo | the URL you pushed to |

## S3 evidence (already done)

The medallion layers are live in S3 at:
`s3://kava-chess-pipeline-352435704328/{bronze,silver,gold,silver_spark,gold_spark}/`

A grader with read access can confirm via:
```
aws s3 ls s3://kava-chess-pipeline-352435704328/ --recursive
```

The PySpark transformation has been run end-to-end producing
`silver_spark/games/part-*.snappy.parquet` (2,726 rows) and
`gold_spark/event_features/part-*.snappy.parquet` (72 rows) from the bronze
TSV at `s3://kava-chess-pipeline-352435704328/bronze/kava_chess_games.tsv`.

## If something breaks at the last minute

- **App won't deploy**: most common cause is a Python version mismatch on
  Streamlit Cloud. Pin `3.12` in advanced settings.
- **App deploys but predictions fail**: check the deploy logs in
  share.streamlit.io. The `models/` directory must be committed and pushed
  (the `.gitignore` is already configured to include it).
- **Streamlit Cloud is rate-limiting**: the test script's local-prediction path
  still passes without `KAVA_APP_URL`, exit code 0.
