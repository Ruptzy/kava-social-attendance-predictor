"""
Render reports/one_page_writeup.md to reports/one_page_writeup.pdf using
ReportLab. Single page, 11pt body, 0.75 inch margins.

Run:
    python src/make_writeup_pdf.py
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD = PROJECT_ROOT / "reports" / "one_page_writeup.md"
PDF = PROJECT_ROOT / "reports" / "one_page_writeup.pdf"


def md_inline_to_html(text: str) -> str:
    # very small subset: **bold**, `code`, *italic*
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', text)
    text = re.sub(r"\*\*([^\*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^\*]+)\*", r"<i>\1</i>", text)
    text = text.replace("→", "&rarr;").replace("≈", "~").replace("≥", "&ge;").replace("•", "&bull;")
    return text


def main() -> None:
    text = MD.read_text(encoding="utf-8")

    body = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=16, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                        textColor="#0d1b2a", spaceBefore=6, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=body, fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                         textColor="#555555", spaceAfter=6)

    flow = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flow.append(Spacer(1, 2))
            continue
        if s.startswith("# "):
            flow.append(Paragraph(md_inline_to_html(s[2:].strip()), h1))
        elif s.startswith("## "):
            flow.append(Paragraph(md_inline_to_html(s[3:].strip()), h2))
        elif s.startswith("```"):
            # skip code fences - keep content as monospace
            continue
        else:
            # treat **bold** chunks at start as subtitle-ish
            if s.startswith("**") and s.endswith("**"):
                flow.append(Paragraph(md_inline_to_html(s), sub))
            else:
                flow.append(Paragraph(md_inline_to_html(s), body))

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Kava Social Chess Attendance Predictor - One Page Write-Up",
        author="Harold Gonzalez",
    )
    doc.build(flow)
    print(f"[pdf] wrote {PDF}")


if __name__ == "__main__":
    main()
