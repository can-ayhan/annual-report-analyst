"""Text of a PDF, page by page."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf


def clean(text: str) -> str:
    """Tidy extracted text without changing what it says: odd spaces, trailing blanks, runs of empty lines."""
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def page_texts(path: Path) -> list[str]:
    """One string per page, in the order of the file. Page 1 is index 0."""
    with pymupdf.open(path) as document:
        return [clean(page.get_text("text")) for page in document]
