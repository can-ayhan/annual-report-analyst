"""The report tools as an MCP server.

Three tools, the same three the agents use: list the reports, search them, read a
page. Any MCP client can start this over stdio (`analyst mcp`) and work with the
indexed reports without the rest of the system.
"""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.mcpserver import MCPServer

from analyst.search import ReportSearch

INSTRUCTIONS = """\
Company annual reports, indexed page by page. Use list_reports to see what is available,
search_reports to find passages (each comes with its report and page), and get_page to read
a whole page when a passage is cut short. Cite the company, year and page for anything you state."""


def default_search() -> ReportSearch:
    from analyst import db
    from analyst.embeddings import FastEmbedEmbedder

    return ReportSearch(db.connect(), FastEmbedEmbedder())


def build_server(make_search: Callable[[], ReportSearch] = default_search) -> MCPServer:
    """make_search is called on first use, so the server answers the handshake before the model loads."""
    server = MCPServer("annual-reports", instructions=INSTRUCTIONS)
    holder: dict[str, ReportSearch] = {}

    def search() -> ReportSearch:
        if "search" not in holder:
            holder["search"] = make_search()
        return holder["search"]

    @server.tool()
    def list_reports() -> list[dict]:
        """The annual reports in the library: id, company, fiscal year, title and number of pages."""
        return search().reports()

    @server.tool()
    def search_reports(query: str, company: str | None = None, fiscal_year: int | None = None,
                       limit: int = 5) -> list[dict]:
        """Find passages about a topic. Returns the text with its report, page and a ready citation.

        query: words likely to appear on the page (line items, names, units)
        company: restrict to one company, written as in list_reports
        fiscal_year: restrict to one fiscal year
        limit: number of passages, at most 10
        """
        hits = search().search(query, company=company, fiscal_year=fiscal_year, limit=max(1, min(limit, 10)))
        return [hit.as_dict() for hit in hits]

    @server.tool()
    def get_page(report_id: str, page: int) -> dict:
        """The full text of one page of a report. page is the position in the PDF, starting at 1."""
        text = search().page(report_id, page)
        if text is None:
            return {"report_id": report_id, "page": page, "found": False, "text": ""}
        return {"report_id": report_id, "page": page, "found": True, "text": text}

    return server


def main() -> None:
    build_server().run("stdio")
