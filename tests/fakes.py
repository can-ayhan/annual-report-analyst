"""Stand-ins for the tests."""

from __future__ import annotations

from analyst.reports import Report

SAMPLE_REPORT = Report(
    report_id="baltic-2025", company="Baltic Mills", fiscal_year=2025, title="Baltic Mills annual report 2025",
    url="https://example.org/baltic-2025.pdf", sha256="0" * 64, pages=3,
)
