"""The reports the system knows about (data/reports.yml) and how to fetch them.

The PDFs are not in the repository. They are downloaded from the publishers and
checked against the checksum recorded when the list was made.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from analyst.config import REPORTS_DIR, REPORTS_FILE

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; annual-report-analyst)"


@dataclass(frozen=True)
class Report:
    report_id: str
    company: str
    fiscal_year: int
    title: str
    url: str
    sha256: str
    pages: int

    @property
    def label(self) -> str:
        return f"{self.company} {self.fiscal_year}"

    def path(self, directory: Path = REPORTS_DIR) -> Path:
        return directory / f"{self.report_id}.pdf"


class ChecksumMismatch(RuntimeError):
    pass


def load_reports(path: Path = REPORTS_FILE) -> list[Report]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Report(report_id=item.pop("id"), **item) for item in data["reports"]]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(report: Report, directory: Path = REPORTS_DIR, opener=urllib.request.urlopen) -> Path:
    """Download the PDF unless a copy with the right checksum is already there."""
    target = report.path(directory)
    if target.exists() and sha256_of(target) == report.sha256:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(report.url, headers={"User-Agent": USER_AGENT})
    log.info("downloading %s", report.url)
    with opener(request, timeout=120) as response:
        content = response.read()
    found = hashlib.sha256(content).hexdigest()
    if found != report.sha256:
        # the publisher replaced the file: page numbers in old answers and in the evaluation may be off
        raise ChecksumMismatch(
            f"{report.report_id}: expected sha256 {report.sha256[:12]}…, got {found[:12]}…")
    target.write_bytes(content)
    return target
