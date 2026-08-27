"""Shared fixtures. Tests that need PostgreSQL use their own database and are skipped without one."""

from __future__ import annotations

import os

import psycopg
import pytest

from analyst import db
from analyst.config import database_conninfo
from analyst.embeddings import HashingEmbedder
from analyst.index import index_report
from analyst.search import ReportSearch
from fakes import SAMPLE_REPORT, write_sample_pdf

TEST_DB = "reports_test"


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    return write_sample_pdf(tmp_path_factory.mktemp("reports") / "baltic-2025.pdf")


@pytest.fixture(scope="session")
def test_database() -> str:
    try:
        admin = psycopg.connect(database_conninfo(), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no PostgreSQL to test against: {exc}")
    with admin:
        admin.execute(f"drop database if exists {TEST_DB} with (force)")
        admin.execute(f"create database {TEST_DB}")
    os.environ["PGDATABASE"] = TEST_DB
    return database_conninfo()


@pytest.fixture(scope="session")
def search(test_database, sample_pdf) -> ReportSearch:
    """The sample report indexed with the hashing embedder: no model download, same code path."""
    conn = db.connect(test_database)
    db.ensure_schema(conn)
    embedder = HashingEmbedder()
    index_report(conn, SAMPLE_REPORT, embedder, pdf_path=sample_pdf)
    return ReportSearch(conn, embedder)
