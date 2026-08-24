"""Stand-ins for the tests: a small made-up annual report."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from analyst.reports import Report

SAMPLE_PAGES = [
    "Baltic Mills S.A.\nAnnual report 2025\n\n"
    "Baltic Mills is a bakery group in northern Poland.\n"
    "As at 31 December 2025 the Group operated 412 bakeries, compared with 390 a year earlier.\n"
    "In 2025 the Group opened 27 new bakeries and closed 5.",

    "Financial highlights (all figures in PLN thousands)\n\n"
    "2025 2024\n"
    "Sales revenue 1 250 400 1 100 200\n"
    "Operating profit 118 900 101 300\n"
    "Net profit 85 300 72 100\n\n"
    "The Group employed 5,120 people at the end of 2025.",

    "Dividend\n\n"
    "The Management Board recommends a dividend of PLN 2.50 per share for 2025.\n"
    "In 2024 the dividend was PLN 2.10 per share.",
]

SAMPLE_REPORT = Report(
    report_id="baltic-2025", company="Baltic Mills", fiscal_year=2025, title="Baltic Mills annual report 2025",
    url="https://example.org/baltic-2025.pdf", sha256="0" * 64, pages=len(SAMPLE_PAGES),
)


def write_sample_pdf(path: Path, pages: list[str] = SAMPLE_PAGES) -> Path:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((60, 80), text, fontname="cour", fontsize=10)
    document.save(path)
    document.close()
    return path
