"""The three things that can be done with the indexed reports: list them, search them, read a page.

The agents call these directly and the MCP server exposes the same three.

Search is hybrid. Meaning is matched by the embedding and exact words (names,
line items, figures) by PostgreSQL full-text search; the two rankings are merged
by reciprocal rank fusion, so a passage that does well in either one comes up.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import psycopg

from analyst.embeddings import Embedder

CANDIDATES = 30         # how many passages each ranking contributes
RRF_K = 60              # the usual constant of reciprocal rank fusion


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


HYBRID = """
with scope as (
    select p.passage_id, p.embedding, p.tsv
    from passage p join report r using (report_id)
    where (%(company)s::text is null or lower(r.company) = lower(%(company)s))
      and (%(fiscal_year)s::int is null or r.fiscal_year = %(fiscal_year)s)
),
by_meaning as (
    select passage_id, row_number() over (order by embedding <=> %(vector)s) as position
    from scope order by embedding <=> %(vector)s limit %(candidates)s
),
by_words as (
    select passage_id, row_number() over (order by ts_rank_cd(tsv, query) desc) as position
    from scope, to_tsquery('english', %(words)s) as query
    where %(words)s <> '' and tsv @@ query
    order by ts_rank_cd(tsv, query) desc limit %(candidates)s
),
merged as (
    select passage_id,
           coalesce(1.0 / (%(k)s + m.position), 0) + coalesce(1.0 / (%(k)s + w.position), 0) as score
    from by_meaning m full join by_words w using (passage_id)
)
select p.passage_id, p.report_id, r.company, r.fiscal_year, p.page_no, p.text, merged.score
from merged join passage p using (passage_id) join report r using (report_id)
order by merged.score desc, p.passage_id
limit %(limit)s
"""


def word_query(text: str) -> str:
    """The words of a question as an OR query: any of them may match, more of them rank higher."""
    words = dict.fromkeys(re.findall(r"[A-Za-z0-9]{2,}", text.lower()))
    return " | ".join(words)


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
        rows = self.conn.execute(HYBRID, {
            "vector": vector, "words": word_query(query), "company": company or None,
            "fiscal_year": fiscal_year, "candidates": CANDIDATES, "k": RRF_K, "limit": limit,
        }).fetchall()
        self.conn.commit()
        return [Hit(*row[:6], score=float(row[6])) for row in rows]

    def page(self, report_id: str, page_no: int) -> str | None:
        row = self.conn.execute("select text from page where report_id = %s and page_no = %s",
                                (report_id, page_no)).fetchone()
        self.conn.commit()
        return row[0] if row else None
