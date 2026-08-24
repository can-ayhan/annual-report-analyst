"""Shared fixtures."""

from __future__ import annotations

import pytest

from fakes import write_sample_pdf


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    return write_sample_pdf(tmp_path_factory.mktemp("reports") / "baltic-2025.pdf")

