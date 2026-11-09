"""Build the submission PDF: progress report, executed notebook, then source code.

Usage:
    python scripts/build_pdf.py      (run the notebook first so it has outputs)
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import nbformat
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
REPORT_FILE = ROOT / "submission" / "progress_report.md"
NOTEBOOK_FILE = ROOT / "notebooks" / "progress_demo.ipynb"
OUTPUT_FILE = ROOT / "submission" / "Progress Report - Sabesan Shanmugabhavananthan.pdf"
SOURCE_FILES = [
    "scout/data.py",
    "scout/tools.py",
    "scout/prompts.py",
    "scout/guardrails.py",
    "scout/memory.py",
    "scout/graph.py",
    "scripts/fetch_data.py",
]
CODE_LINE_WIDTH = 118


def register_mono_font() -> str:
    """Use Menlo (covers accented names like Jokić); fall back to Courier."""
    try:
        pdfmetrics.registerFont(TTFont("Mono", "/System/Library/Fonts/Menlo.ttc", subfontIndex=0))
        return "Mono"
    except Exception:
        return "Courier"


MONO = register_mono_font()
STYLES = getSampleStyleSheet()
TITLE = STYLES["Title"]
HEADING = STYLES["Heading2"]
SUBHEADING = STYLES["Heading4"]
BODY = ParagraphStyle("Body", parent=STYLES["BodyText"], fontSize=10.5, leading=14.5, alignment=4)
CODE = ParagraphStyle("Code", fontName=MONO, fontSize=7, leading=8.6, spaceAfter=6,
                      backColor=colors.HexColor("#f3f3f3"), borderPadding=4)
OUTPUT = ParagraphStyle("Output", fontName=MONO, fontSize=7, leading=8.6, spaceAfter=8,
                        leftIndent=8, textColor=colors.HexColor("#222222"))


def wrap_code(text: str) -> str:
    """Hard-wrap long lines, keeping each continuation indented under its line."""
    wrapped = []
    for line in text.rstrip().splitlines() or [""]:
        indent = " " * (len(line) - len(line.lstrip()) + 4)
        wrapped += textwrap.wrap(line, CODE_LINE_WIDTH, subsequent_indent=indent,
                                 drop_whitespace=False, replace_whitespace=False) or [""]
    return "\n".join(wrapped)


def markdown_to_markup(text: str) -> str:
    """Convert the bits of Markdown we use (**bold**, `code`) into ReportLab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", rf'<font face="{MONO}" size="9">\1</font>', text)


def report_section() -> list[Flowable]:
    flowables: list[Flowable] = []
    for block in REPORT_FILE.read_text(encoding="utf-8").strip().split("\n\n"):
        if block.startswith("# "):
            flowables.append(Paragraph(markdown_to_markup(block[2:]), TITLE))
        elif block.startswith("_"):
            flowables.append(Paragraph(f"<i>{markdown_to_markup(block.strip('_'))}</i>", STYLES["Normal"]))
            flowables.append(Spacer(1, 8))
        else:
            flowables.append(Paragraph(markdown_to_markup(" ".join(block.split())), BODY))
    return flowables


def cell_output_text(output: dict) -> str:
    if output.get("output_type") == "error":
        return f"{output['ename']}: {output['evalue']}"
    return output.get("text") or output.get("data", {}).get("text/plain", "")


def notebook_section() -> list[Flowable]:
    notebook = nbformat.read(NOTEBOOK_FILE, as_version=4)
    flowables: list[Flowable] = [
        PageBreak(),
        Paragraph("Appendix A - Executed demo notebook (notebooks/progress_demo.ipynb)", HEADING),
    ]
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            for line in filter(str.strip, cell.source.splitlines()):
                style = SUBHEADING if line.startswith("#") else BODY
                flowables.append(Paragraph(markdown_to_markup(line.lstrip("# ")), style))
            continue

        flowables.append(Preformatted(wrap_code(cell.source), CODE))
        for output in cell.get("outputs", []):
            text = cell_output_text(output)
            if text:
                flowables.append(Preformatted(wrap_code(text), OUTPUT))
    return flowables


def source_code_section() -> list[Flowable]:
    flowables: list[Flowable] = [PageBreak(), Paragraph("Appendix B - Source code (work in progress)", HEADING)]
    for relative_path in SOURCE_FILES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        flowables += [Paragraph(relative_path, SUBHEADING), Preformatted(wrap_code(source), CODE)]
    return flowables


def main() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=LETTER,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Progress Report - Basketball Scouting & Game-Plan Coordinator",
        author="Sabesan Shanmugabhavananthan",
    )
    document.build(report_section() + notebook_section() + source_code_section())
    print(f"wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
