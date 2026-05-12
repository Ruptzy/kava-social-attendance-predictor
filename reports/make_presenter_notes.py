"""
Generate `reports/presenter_notes.docx` - cheat-sheet style notes for the
Kava Chess Clock final-project demo. One block per slide, each block fits
in a single glance.

Run:
    python reports/make_presenter_notes.py
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = Path(__file__).resolve().parent / "presenter_notes.docx"

# Same palette as the app, for the doc accents
ESPRESSO = RGBColor(0x24, 0x1C, 0x17)
BRONZE = RGBColor(0xA9, 0x78, 0x21)
BRONZE_DARK = RGBColor(0x6F, 0x5A, 0x41)
NAVY = RGBColor(0x17, 0x20, 0x33)
GREY = RGBColor(0x55, 0x55, 0x55)


def _set_run(run, *, size=11, bold=False, italic=False, color=None, name="Calibri"):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def _eyebrow(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text.upper())
    _set_run(r, size=9, bold=True, color=BRONZE)
    r.font.all_caps = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(0)


def _title(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    _set_run(r, size=16, bold=True, color=ESPRESSO)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)


def _say(doc, text):
    """The one-liner you actually say out loud when the slide comes up."""
    p = doc.add_paragraph()
    r = p.add_run("Say: ")
    _set_run(r, size=10, bold=True, color=NAVY)
    r2 = p.add_run(text)
    _set_run(r2, size=10, color=ESPRESSO)
    p.paragraph_format.space_after = Pt(2)


def _bullet(doc, text, sub=None):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    _set_run(r, size=10, color=ESPRESSO)
    p.paragraph_format.space_after = Pt(1)
    if sub:
        p2 = doc.add_paragraph()
        r2 = p2.add_run("    " + sub)
        _set_run(r2, size=9, italic=True, color=GREY)
        p2.paragraph_format.space_after = Pt(1)


def _gotcha(doc, text):
    """A "if they ask" answer - the things the grader might probe on."""
    p = doc.add_paragraph()
    r = p.add_run("If asked: ")
    _set_run(r, size=9, bold=True, color=BRONZE_DARK)
    r2 = p.add_run(text)
    _set_run(r2, size=9, italic=True, color=GREY)
    p.paragraph_format.space_after = Pt(2)


def _rubric_tag(doc, text):
    p = doc.add_paragraph()
    r = p.add_run("Rubric line: ")
    _set_run(r, size=9, bold=True, color=NAVY)
    r2 = p.add_run(text)
    _set_run(r2, size=9, color=GREY)
    p.paragraph_format.space_after = Pt(4)


def _hr(doc):
    p = doc.add_paragraph("─" * 90)
    _set_run(p.runs[0], size=8, color=BRONZE_DARK)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)


def main() -> None:
    doc = Document()
    # Tighten the margins so the cheat sheet fits more per page
    for section in doc.sections:
        section.top_margin = Inches(0.55)
        section.bottom_margin = Inches(0.55)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    # ---------------- Header ----------------
    h = doc.add_paragraph()
    r = h.add_run("Kava Chess Clock — Presenter Notes")
    _set_run(r, size=18, bold=True, color=ESPRESSO)
    h.paragraph_format.space_after = Pt(0)

    sub = doc.add_paragraph()
    rsub = sub.add_run("Attendance Forecasting for Kava Social Chess Club  ·  "
                       "NCF Distributed Systems for Data Science  ·  Spring 2026")
    _set_run(rsub, size=10, italic=True, color=GREY)
    sub.paragraph_format.space_after = Pt(2)

    intro = doc.add_paragraph()
    ri = intro.add_run("One slide per page. Glance at the bold lines. The italic "
                       "'If asked' lines are answers to questions the grader might lob at you.")
    _set_run(ri, size=9, italic=True, color=GREY)
    _hr(doc)

    # ============== SLIDE 1: TITLE ==============
    _eyebrow(doc, "Slide 1 · Title")
    _title(doc, "Kava Chess Clock")
    _say(doc, "I built an attendance-forecasting dashboard for the chess club "
              "I help run at Kava Social in Bradenton.")
    _bullet(doc, "Predicts how many players will show up on a future bracket night.")
    _bullet(doc, "Bracket nights start at 8:00 PM — the model uses anything knowable BEFORE then.")
    _bullet(doc, "Live URL is on the slide. Github repo is public.")
    _gotcha(doc, "Solo project. Domain is mine — I help run these nights.")
    _hr(doc)

    # ============== SLIDE 2: WHAT IT PREDICTS ==============
    _eyebrow(doc, "Slide 2 · What it predicts")
    _title(doc, "Attendance only. Not chess outcomes.")
    _say(doc, "This is an event-planning tool, not a chess engine. It predicts "
              "the number of players, not who wins.")
    _bullet(doc, "Target: unique-player attendance for a future bracket night.")
    _bullet(doc, "Two heads: regression for the count + classifier for high / normal / low.")
    _bullet(doc, "Useful for boards-to-prep, clocks-to-charge, staffing.")
    _gotcha(doc, "Why not predict game outcomes? Out of scope and a small dataset (~89 events). "
                 "The honest, useful prediction is attendance.")
    _rubric_tag(doc, "Clarity of prediction question (Write-up 10%).")
    _hr(doc)

    # ============== SLIDE 3: ARCHITECTURE ==============
    _eyebrow(doc, "Slide 3 · Architecture")
    _title(doc, "Bronze → Silver → Gold, with PySpark and S3.")
    _say(doc, "Standard medallion pipeline. Raw logs land in S3 as bronze. "
              "PySpark cleans them into silver. Feature engineering builds the gold table. "
              "scikit-learn + MLflow train the model. Streamlit serves it.")
    _bullet(doc, "Two cloud / distributed stages:",
            sub="(1) AWS S3 for bronze + silver + gold storage   (2) PySpark for the distributed transformation")
    _bullet(doc, "89 events total = 72 from game-level logs + 17 recovered from Swiss-standings paste-ins.",
            sub="The 17 carry a source_type='standings_summary' lineage column. Game-logs counts always win on conflict.")
    _bullet(doc, "All four candidate models are tracked in MLflow under ./mlruns/.")
    _gotcha(doc, "What's the distributed compute? PySpark with window functions (lag, rolling). "
                 "I ran it locally against S3 bronze; the same script runs on a cluster unchanged.")
    _gotcha(doc, "Why two layers in S3 instead of just one? Bronze = raw (audit trail). "
                 "Silver = cleaned + deduped. Gold = the model-ready feature table.")
    _rubric_tag(doc, "Pipeline Architecture 25% — distributed pipeline + medallion + appropriate infra.")
    _hr(doc)

    # ============== SLIDE 4: DASHBOARD ==============
    _eyebrow(doc, "Slide 4 · The forecast dashboard")
    _title(doc, "One number, then the why.")
    _say(doc, "The forecast lands on the page as one clear number plus a planning "
              "recommendation. Below that, six tabs explain every signal the model is reading.")
    _bullet(doc, "Forecast Summary: predicted attendance + turnout category + chance of a busy night.")
    _bullet(doc, "Six narrative tabs:",
            sub="Summary chart · Recent attendance · Calendar timing · Bradenton weather · Community momentum · Venue value · Model & method")
    _bullet(doc, "Every chart has a 'Takeaway' card explaining what to conclude — no chart left as a bare visual.")
    _bullet(doc, "Polished dark theme with subtle chess-knight watermark. Plotly charts inside translucent cards.")
    _gotcha(doc, "Is the UI responsive? Streamlit handles the breakpoints. The app uses wide layout, max-width 1500px.")
    _gotcha(doc, "What about the 'Venue Value' tab? That's the business layer — covered on the next slide.")
    _rubric_tag(doc, "Web Application / UI 20% — polished, responsive, meaningful visualization.")
    _hr(doc)

    # ============== SLIDE 5: MODEL ==============
    _eyebrow(doc, "Slide 5 · Model bake-off")
    _title(doc, "Random Forest beat three other candidates.")
    _say(doc, "I trained four candidates and held out the most recent 14 events as the "
              "test set. Random Forest had the lowest error AND beat the simple baseline by about one player.")
    _bullet(doc, "Naive baseline (predict last night): MAE 3.64 players — the bar to beat.")
    _bullet(doc, "Ridge Regression: 4.44 — too rigid for non-linear signals.")
    _bullet(doc, "Random Forest: 2.68 ← selected.",
            sub="Improves over the baseline by ~1 player; tracked in MLflow.")
    _bullet(doc, "Gradient Boosting: 3.33 — strong second place.")
    _bullet(doc, "Chronological holdout, never a random shuffle. 4-fold TimeSeriesSplit for CV on the training portion.")
    _bullet(doc, "Anti-leakage: every prior-event feature is shift(1)'d. No same-night number is ever an input.")
    _gotcha(doc, "Why Random Forest? Best holdout MAE and beats the naive baseline by ~1.0 player. "
                 "Gradient Boosting was close — would be fine too.")
    _gotcha(doc, "Why is the classifier accuracy ~52%? The high-turnout threshold sits right at the median (15 players). "
                 "Half the events are right on the line by definition, so accuracy near 50% is honest noise.")
    _gotcha(doc, "What metric? MAE — easy to explain: 'usually off by about 2.7 players.' RMSE and CV are also in metadata.")
    _rubric_tag(doc, "Predictive Model 20% — trained, tracked in MLflow, served via the live endpoint, metric reported.")
    _hr(doc)

    # ============== SLIDE 6: VENUE VALUE ==============
    _eyebrow(doc, "Slide 6 · Venue Value (the business layer)")
    _title(doc, "Attendance → drink revenue, conservatively.")
    _say(doc, "I added a venue-value calculator on top of the forecast. It turns "
              "predicted attendance into a per-night, monthly, and yearly drink-revenue estimate. "
              "The drink price defaults to $13.75 — the average of the public Kava Social DoorDash menu.")
    _bullet(doc, "Inputs in the tab: drinks per player, average drink price slider, attendance source, bracket nights per month.")
    _bullet(doc, "Outputs: per-night / monthly / yearly / total tracked revenue cards.")
    _bullet(doc, "Four charts: scenario bars, historical per-night line, monthly bars, cumulative area.")
    _bullet(doc, "Conservative by design: player-only. Excludes friends, spectators, food, merch.",
            sub="Built as a planning floor for venue-partnership conversations.")
    _gotcha(doc, "Where did $13.75 come from? Mean of OG / House Quad / Tropic Wave / Citrus Tsunami doubles "
                 "from the public DoorDash menu. The slider lets the user adjust because DoorDash itself flags "
                 "that delivery and in-person pricing can differ.")
    _rubric_tag(doc, "Originality + Craft 10% — creative data source + business value layer.")
    _hr(doc)

    # ============== SLIDE 7: LEARNED ==============
    _eyebrow(doc, "Slide 7 · What I learned")
    _title(doc, "Three things that bit harder than expected.")
    _say(doc, "These three lessons shaped a lot of the design decisions.")
    _bullet(doc, "Leakage is a habit, not a feature.",
            sub="Every interesting same-night number (game count, draw rate, unique players) is only knowable AFTER people show up. "
                "Every prior-event feature uses shift(1). Every CV split is chronological.")
    _bullet(doc, "Florida rain is too noisy for a yes/no flag.",
            sub="My first version flagged 52% of days as 'rainy' (any trace of daily precipitation). "
                "Fixed by switching to a 7-11 PM event-window threshold of 0.5 mm — now ~15% of nights, realistic for Bradenton. "
                "Headline weather feature became a 0-5 comfort score, not a binary.")
    _bullet(doc, "Plain language beats jargon.",
            sub="The first UI revision had 'CV MAE 3.67' and 'ŷ regressor estimate' labels. The redesign replaced them with "
                "'usually off by about 2.7 players' and 'expected players'. The dashboard suddenly felt like a tool, "
                "not a class project.")
    _gotcha(doc, "What would you change if you rebuilt it? "
                 "(1) Weekly retraining via GitHub Actions. "
                 "(2) Real FastAPI endpoint behind the Streamlit UI. "
                 "(3) Track event-reschedule flags as a feature — holiday-week reschedules add noise the model can't see.")
    _rubric_tag(doc, "Write-up 10% (learnings) + Originality + Craft 10% (honest about limitations).")
    _hr(doc)

    # ============== CLOSING ==============
    _eyebrow(doc, "Quick numbers to remember")
    _bullet(doc, "89 events (72 game logs + 17 standings recoveries)")
    _bullet(doc, "Random Forest holdout MAE 2.68 · Naive baseline 3.64 · improvement ~1 player")
    _bullet(doc, "48 features in 5 plain-language families: attendance momentum, calendar timing, "
                "schedule context, Bradenton weather, community momentum")
    _bullet(doc, "Default drink price $13.75 (DoorDash menu mean)")
    _bullet(doc, "Two cloud / distributed stages: S3 + PySpark")

    _eyebrow(doc, "If everything goes wrong")
    _bullet(doc, "Live app refuses to load → the GitHub README has the URL, the repo has the test script + PDF.")
    _bullet(doc, "Test script: `python test_project.py` from repo root. Exits 0 on success.")
    _bullet(doc, "One-pager PDF is at the repo root as `one_page_writeup.pdf`.")

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
