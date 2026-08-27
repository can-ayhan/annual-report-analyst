"""Put a report into the database: pages as they are, passages with their embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import psycopg

from analyst.chunking import split_page
from analyst.config import EMBEDDING_DIMENSION
from analyst.embeddings import Embedder
from analyst.pdf import page_texts
from analyst.reports import Report

log = logging.getLogger(__name__)

BATCH = 16          # passages per embedding call: larger batches are no faster on a CPU and use more memory


def embedding_input(report: Report, passage: str) -> str:
    """What is embedded: the passage with the company and year in front, so that a
    question naming a company lands on that company's pages."""
    return f"{report.company}, annual report {report.fiscal_year}\n{passage}"


def index_report(conn: psycopg.Connection, report: Report, embedder: Embedder,
                 pdf_path: Path | None = None) -> dict[str, int]:
    """Replace whatever is stored for the report with a fresh extraction."""
    if embedder.dimension != EMBEDDING_DIMENSION:
        raise ValueError(f"the embedding model has {embedder.dimension} dimensions, the passage table "
                         f"is built for {EMBEDDING_DIMENSION}")
    pages = page_texts(pdf_path or report.path())
    passages = [
        (page_no, position, text)
        for page_no, page in enumerate(pages, start=1)
        for position, text in enumerate(split_page(page))
    ]
    with conn.transaction():
        conn.execute("delete from report where report_id = %s", (report.report_id,))
        conn.execute(
            """insert into report (report_id, company, fiscal_year, title, source_url, sha256, pages)
               values (%s, %s, %s, %s, %s, %s, %s)""",
            (report.report_id, report.company, report.fiscal_year, report.title, report.url,
             report.sha256, len(pages)),
        )
        with conn.cursor() as cur:
            cur.executemany(
                "insert into page (report_id, page_no, text) values (%s, %s, %s)",
                [(report.report_id, page_no, page) for page_no, page in enumerate(pages, start=1)],
            )
            for start in range(0, len(passages), BATCH):
                batch = passages[start:start + BATCH]
                vectors = embedder.passages([embedding_input(report, text) for _, _, text in batch])
                cur.executemany(
                    """insert into passage (report_id, page_no, position, text, embedding)
                       values (%s, %s, %s, %s, %s)""",
                    [(report.report_id, page_no, position, text, np.asarray(vector, dtype=np.float32))
                     for (page_no, position, text), vector in zip(batch, vectors, strict=True)],
                )
    log.info("%s: %s pages, %s passages", report.report_id, len(pages), len(passages))
    return {"pages": len(pages), "passages": len(passages)}
