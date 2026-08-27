"""The three things that can be done with the indexed reports: list them, search them, read a page.

The agents call these directly and the MCP server exposes the same three.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import psycopg

from analyst.embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    passage_id: int
    report_id: str
    company: str
    fiscal_year: int
    page_no: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.company} {self.fiscal_year}, p. {self.page_no}"

    def as_dict(self) -> dict:
        return {**asdict(self), "citation": self.citation}


NEAREST = """
select p.passage_id, p.report_id, r.company, r.fiscal_year, p.page_no, p.text,
       1 - (p.embedding <=> %(vector)s) as score
from passage p join report r using (report_id)
where (%(company)s::text is null or lower(r.company) = lower(%(company)s))
  and (%(fiscal_year)s::int is null or r.fiscal_year = %(fiscal_year)s)
order by p.embedding <=> %(vector)s, p.passage_id
limit %(limit)s
"""


class ReportSearch:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder):
        self.conn = conn
        self.embedder = embedder

    def reports(self) -> list[dict]:
        rows = self.conn.execute(
            "select report_id, company, fiscal_year, title, pages from report order by company, fiscal_year"
        ).fetchall()
        self.conn.commit()
        return [dict(zip(("report_id", "company", "fiscal_year", "title", "pages"), row, strict=True))
                for row in rows]

    def search(self, query: str, company: str | None = None, fiscal_year: int | None = None,
               limit: int = 6) -> list[Hit]:
        vector = np.asarray(self.embedder.query(query), dtype=np.float32)
        rows = self.conn.execute(NEAREST, {
            "vector": vector, "company": company or None, "fiscal_year": fiscal_year, "limit": limit,
        }).fetchall()
        self.conn.commit()
        return [Hit(*row[:6], score=float(row[6])) for row in rows]

    def page(self, report_id: str, page_no: int) -> str | None:
        row = self.conn.execute("select text from page where report_id = %s and page_no = %s",
                                (report_id, page_no)).fetchone()
        self.conn.commit()
        return row[0] if row else None
