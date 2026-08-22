import hashlib
import re
from dataclasses import replace

import pytest

from analyst.reports import ChecksumMismatch, fetch, load_reports
from fakes import SAMPLE_REPORT


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def read(self) -> bytes:
        return self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_manifest_lists_reports_with_checksums():
    reports = load_reports()
    assert len({r.report_id for r in reports}) == len(reports) >= 4
    for report in reports:
        assert report.url.startswith("https://") and report.url.endswith(".pdf")
        assert re.fullmatch(r"[0-9a-f]{64}", report.sha256)
        assert report.pages > 0 and 2000 < report.fiscal_year < 2100


def test_one_company_has_two_years():
    years = {}
    for report in load_reports():
        years.setdefault(report.company, set()).add(report.fiscal_year)
    assert any(len(found) > 1 for found in years.values())


def test_a_download_is_kept_when_the_checksum_matches(tmp_path):
    content = b"%PDF-1.7 pretend"
    report = replace(SAMPLE_REPORT, sha256=hashlib.sha256(content).hexdigest())
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return FakeResponse(content)

    path = fetch(report, tmp_path, opener=opener)
    assert path.read_bytes() == content
    assert fetch(report, tmp_path, opener=opener) == path
    assert calls == [report.url]                      # the second call found the file and did not download


def test_a_changed_file_is_refused(tmp_path):
    report = replace(SAMPLE_REPORT, sha256="a" * 64)
    with pytest.raises(ChecksumMismatch, match="baltic-2025"):
        fetch(report, tmp_path, opener=lambda request, timeout: FakeResponse(b"something else"))
    assert not report.path(tmp_path).exists()
