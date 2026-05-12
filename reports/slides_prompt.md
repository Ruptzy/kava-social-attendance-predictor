# Prompt to paste into Claude (for Artifacts)

Copy everything below the next line into a new Claude chat and ask Claude to
build a slide deck as an HTML/React Artifact. The deck should match the Kava
Chess Clock app's "luxury chess lounge" aesthetic.

---

Build me a 7-slide presentation deck for my final project, as a single full-screen
HTML/CSS/JS artifact (or React component if you prefer). The slides should match
the dark luxury aesthetic of my live Streamlit app exactly. Use arrow keys / space
to advance between slides.

**Project**: Kava Chess Clock — Attendance Forecasting for Kava Social Chess Club
**Live app**: https://kava-social-attendance-predictor-cwlaxs6ygbxud7z48884zq.streamlit.app/
**Repo**: github.com/Ruptzy/kava-social-attendance-predictor
**Author**: Harold Gonzalez (NCF, Distributed Systems for Data Science, Spring 2026)

## Design system (match the app exactly)

Palette (CSS variables, use these for everything):
- `--bg: #03110D`           page background (near-black green)
- `--surface: #0a1a14`      sidebar / sub-panels
- `--card: #16302B`         card surface (deep green)
- `--burgundy: #390517`     state warmth
- `--burgundy-glow: #5a1124`
- `--bronze: #A38560`       primary metallic accent
- `--bronze-bright: #C4A77D`
- `--bronze-dim: #6F5A41`
- `--silver: #E0E0E0`       main text
- `--silver-dim: #9BA09B`
- `--silver-mute: #6B7570`
- `--border: rgba(163, 133, 96, 0.18)`

Typography: Inter (Google Fonts), weights 300/400/500/600/700. Headlines bold,
body 400, eyebrows uppercase with `letter-spacing: 0.22em`.

Slide canvas: 16:9, full viewport, dark gradient background:
`radial-gradient(ellipse 1100px 600px at 0% 0%, rgba(57,5,23,0.18), transparent 55%),
 radial-gradient(ellipse 900px 500px at 100% 100%, rgba(163,133,96,0.08), transparent 55%),
 #03110D`

Each slide should have:
- A faint chess-knight glyph (Unicode `♞`) at ~5% opacity in the bottom-right
  corner, rotated -8°, around 18rem font-size — purely decorative, behind text
- A 1px bronze hairline at the top
- Small uppercase bronze eyebrow text above the slide title
- Slide number in the bottom-left as `01 / 07` in silver-dim

Cards (for stats/metrics) use:
- `background: linear-gradient(180deg, rgba(22,48,43,0.86), rgba(18,41,34,0.88))`
- `border: 1px solid var(--border); border-radius: 16px`
- `box-shadow: 0 8px 24px rgba(0,0,0,0.32)`
- Bronze hairline accent across the top: `linear-gradient(90deg, transparent, var(--bronze) 50%, transparent)`

Numbers: tabular-nums, 3.2rem bold for hero metrics. Labels: uppercase silver-dim,
0.66rem, letter-spacing 0.22em.

No emojis. No clip-art chess pieces beyond the single ♞ watermark. No light
backgrounds. Quiet, editorial, never busy.

## Slide-by-slide content

### Slide 1 — Title
- Eyebrow: KAVA SOCIAL CHESS · BRADENTON, FLORIDA
- Title: **Kava Chess <span style="color:bronze">Clock</span>**
  (use light-weight 300 on "Clock", bold 700 on "Kava Chess")
- Subtitle in bronze-bright: Attendance Forecasting for Kava Social Chess Club
- Below: "Harold Gonzalez · Distributed Systems for Data Science · NCF Spring 2026"
- 4 thin badges across the bottom: Bradenton FL · Bracket starts 8:00 PM ·
  MLflow Pipeline · Streamlit Cloud

### Slide 2 — What it predicts
- Eyebrow: WHAT THIS PREDICTS
- Title: Plain attendance for an upcoming bracket night.
- Two side-by-side cards:
  - Card A "Predicts" (bronze accent): unique-player attendance for a future
    Kava Social bracket night
  - Card B "Does NOT predict" (burgundy accent, faded): chess games, winners,
    player strength, openings, individual performance
- Footer line in silver-dim: "Built for the people who run the bracket — to plan
  boards, clocks, and staffing."

### Slide 3 — Architecture
- Eyebrow: PIPELINE
- Title: Raw bracket logs → S3 → PySpark → MLflow → Streamlit
- Horizontal flow diagram with 5 boxes, bronze hairlines between them:
  `SOURCE → BRONZE → SILVER → GOLD → SERVE`
  Each box: bronze eyebrow label + one-line description in silver
  - SOURCE: Kava Social game logs (Excel + standings paste-ins)
  - BRONZE: Raw TSV in AWS S3
  - SILVER: Cleaned game-level Parquet (PySpark + pandas)
  - GOLD: Event-level features + Bradenton weather
  - SERVE: scikit-learn · MLflow · Streamlit
- Below the flow, a small line: "Two cloud / distributed stages: S3 storage +
  PySpark transformation. 89 events tracked (72 game logs + 17 recovered from
  Swiss standings)."

### Slide 4 — The dashboard in action
- Eyebrow: THE FORECAST
- Title: One number, then the why.
- Show 4 stacked metric cards (the actual cards from my app):
  - Card 1 (bronze top): "PREDICTED ATTENDANCE" → big "15" → "expected players"
  - Card 2: "EXPECTED TURNOUT" → "Normal"
  - Card 3: "CHANCE OF A BUSY NIGHT" → "62%" with a tiny progress bar
  - Card 4 (navy top, full width): "PLANNING RECOMMENDATION" with bullet list:
    "8 boards to prep · 4 clocks ready · Standard staffing"
- Small note at the bottom: "Six narrative tabs explain every signal: recent
  attendance, calendar timing, Bradenton weather, community momentum,
  venue value, model & method."

### Slide 5 — Model bake-off
- Eyebrow: MODEL
- Title: Four candidates, chronological holdout, lowest error wins.
- A clean comparison table (4 rows × 3 columns):
  | Model | Holdout MAE | Note |
  | Naive (last event) | 3.64 | Baseline to beat |
  | Ridge Regression | 4.44 | Too rigid |
  | **Random Forest** | **2.68** ✓ | **Selected** |
  | Gradient Boosting | 3.33 | Second place |
- Highlight the Random Forest row with a bronze-tinted background + a bronze
  pill that says "SELECTED"
- Below the table, three short lines in silver-dim:
  - "Improves over baseline by ~1 player"
  - "4-fold time-series cross-validation, never random shuffle"
  - "Anti-leakage: only signals knowable before the bracket starts"
- All runs tracked in MLflow

### Slide 6 — Venue Value (the bonus tab)
- Eyebrow: BUSINESS LAYER
- Title: Turning attendance into venue revenue.
- 3 polished revenue cards across the top:
  - "Per bracket night": $206
  - "Monthly estimate": $412
  - "Yearly estimate": $4,944
- One paragraph in silver-dim under the cards:
  "Conservative player-only drink revenue, built on top of the attendance
  forecast. Uses public Kava Social DoorDash menu pricing as the default,
  fully adjustable. Built to support venue-partnership conversations."
- Small trust note in burgundy-tinted card: "Excludes friends, spectators,
  food, and merch — read it as a floor, not the full event value."

### Slide 7 — What I learned
- Eyebrow: TAKEAWAYS
- Title: Three things that bit harder than expected.
- Three cards stacked vertically:
  1. **Leakage is a habit, not a feature.** Every interesting same-night number
     is information you only have after people show up. `.shift(1)` everywhere,
     `TimeSeriesSplit` always — no random shuffling.
  2. **Florida rain is too noisy for a yes/no flag.** Switched from a daily
     trace threshold (52% of days "rainy") to a 7–11 PM event window with
     ~0.5 mm threshold (15% rainy). Headline weather feature became a
     comfort score, not a binary.
  3. **Plain language beats jargon.** Replaced "CV MAE" / "regression output"
     with "usually off by about 2.7 players" / "expected turnout". The
     dashboard suddenly felt like a tool, not a class project.
- Footer line: "Live URL · GitHub repo · test_project.py · one_page_writeup.pdf"

## Navigation requirements
- Arrow keys / space to advance, arrow keys to go back
- Slide number in bottom-left ("03 / 07")
- A subtle bronze progress bar across the bottom of the viewport
- Smooth fade or slide transition (250ms, ease-out), nothing flashy

Make it feel like the live app: quiet, composed, never decorative for its own
sake. The chess knight should whisper, not shout.
