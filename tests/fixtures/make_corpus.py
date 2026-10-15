"""Build a tiny synthetic corpus: two FICTIONAL banks, a few pages each.

Used by tests and by `bra demo` to exercise the full pipeline without real
reports or API keys. Every figure here is invented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

FILLER = (
    "The Group continued to execute its strategic plan, with a focus on operational efficiency, "
    "client experience and the digital transformation of its retail networks. "
)

BANKS = {
    "alpha": {
        "name": "Banque Alpha",
        "pages": [
            ("Message from the Chairman", FILLER * 4),
            ("Key figures 2025",
             "Banque Alpha reported net income attributable to the Group of EUR 9.4 billion in 2025, "
             "up 6% compared with 2024. Revenues reached EUR 48.1 billion."),
            ("Capital and solvency",
             "At 31 December 2025, the fully loaded Common Equity Tier 1 (CET1) ratio of Banque Alpha "
             "stood at 12.9%, well above the regulatory requirement of 10.2%. The leverage ratio was 4.4%."),
            ("Risk management", FILLER * 3 + "The cost of risk is detailed in the table below."),
            ("Cost of risk by business line", None),  # table page
            ("Sustainability", "Banque Alpha financed EUR 31 billion of renewable energy projects in 2025. " + FILLER),
        ],
        "table": [["Business line", "2025 (EUR m)", "2024 (EUR m)"],
                  ["Retail banking", "1,210", "1,105"],
                  ["Corporate banking", "640", "702"],
                  ["Total", "1,850", "1,807"]],
    },
    "beta": {
        "name": "Beta Bank",
        "pages": [
            ("Letter to shareholders", FILLER * 4),
            ("Financial highlights",
             "Beta Bank posted net income, Group share, of EUR 4.1 billion for the 2025 financial year, "
             "a decrease of 3% year on year, on revenues of EUR 26.7 billion."),
            ("Prudential ratios",
             "Beta Bank's CET1 ratio reached 13.4% at end-December 2025, against 13.1% a year earlier. "
             "The total capital ratio was 18.0%."),
            ("Employees", "At the end of 2025 Beta Bank employed 117,000 people in 62 countries. " + FILLER),
        ],
        "table": None,
    },
}


def draw_table(page: pymupdf.Page, rows: list[list[str]], top: float = 120) -> None:
    x0, col_w, row_h = 60, 160, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + c * col_w, top + r * row_h, x0 + (c + 1) * col_w, top + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (4, 5, -4, -2), cell, fontsize=10)


def build(out_dir: Path, year: int = 2025) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for code, spec in BANKS.items():
        doc = pymupdf.open()
        for title, body in spec["pages"]:
            page = doc.new_page()
            page.insert_text((60, 70), f"{spec['name']} - {title}", fontsize=15)
            if body is None:
                draw_table(page, spec["table"])
            else:
                page.insert_textbox(pymupdf.Rect(60, 100, 540, 780), body, fontsize=11)
        path = out_dir / f"{code}_{year}_urd.pdf"
        doc.save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build(Path(sys.argv[1] if len(sys.argv) > 1 else "data/demo")):
        print(p)
