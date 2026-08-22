"""Where things are."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_FILE = ROOT / "data" / "reports.yml"
REPORTS_DIR = ROOT / "data" / "reports"
