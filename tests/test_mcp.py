"""The MCP server, exercised through an MCP client: in memory, and once over a real stdio pipe."""

import asyncio
import json
import os
import sys
import textwrap

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from analyst.mcp_server import build_server


def structured(result):
    """What a tool returned, whichever way the SDK hands it over."""
    if result.structured_content is not None:
        content = result.structured_content
        return content["result"] if set(content) == {"result"} else content
    return json.loads(result.content[0].text)


def call(search, tool: str, arguments: dict | None = None):
    async def run():
        async with Client(build_server(lambda: search)) as client:
            return await client.call_tool(tool, arguments or {})

    return structured(asyncio.run(run()))


def test_three_tools_are_advertised_with_their_parameters(search):
    async def run():
        async with Client(build_server(lambda: search)) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in asyncio.run(run())}
    assert set(tools) == {"list_reports", "search_reports", "get_page"}
    assert set(tools["search_reports"].input_schema["properties"]) == {"query", "company", "fiscal_year", "limit"}
    assert tools["search_reports"].input_schema["required"] == ["query"]
    assert "page" in tools["get_page"].description.lower()


def test_list_reports(search):
    assert call(search, "list_reports") == [{"report_id": "baltic-2025", "company": "Baltic Mills", "fiscal_year": 2025,
                                             "title": "Baltic Mills annual report 2025", "pages": 3}]


def test_search_returns_passages_with_a_citation(search):
    hits = call(search, "search_reports", {"query": "dividend per share", "company": "Baltic Mills", "limit": 2})
    assert len(hits) == 2
    assert hits[0]["citation"] == "Baltic Mills 2025, p. 3" and hits[0]["page_no"] == 3
    assert "PLN 2.50 per share" in hits[0]["text"]


def test_get_page_and_a_page_that_does_not_exist(search):
    page = call(search, "get_page", {"report_id": "baltic-2025", "page": 2})
    assert page["found"] and "Net profit 85 300 72 100" in page["text"]
    assert call(search, "get_page", {"report_id": "baltic-2025", "page": 40}) == {
        "report_id": "baltic-2025", "page": 40, "found": False, "text": ""}


def test_the_store_is_only_opened_when_a_tool_is_used():
    opened = []

    def make_search():
        opened.append(True)
        raise AssertionError("not needed for the handshake")

    async def run():
        async with Client(build_server(make_search)) as client:
            return (await client.list_tools()).tools

    assert len(asyncio.run(run())) == 3 and opened == []


def test_over_stdio_as_a_separate_process(search, test_database):
    # a real subprocess speaking MCP on stdin/stdout, with the test database and the hashing embedder
    program = textwrap.dedent("""
        from analyst import db
        from analyst.embeddings import HashingEmbedder
        from analyst.mcp_server import build_server
        from analyst.search import ReportSearch
        build_server(lambda: ReportSearch(db.connect(), HashingEmbedder())).run("stdio")
    """)
    # the child gets this process's environment, so it connects to the test database too
    server = StdioServerParameters(command=sys.executable, args=["-c", program], env=dict(os.environ))

    async def run():
        async with Client(stdio_client(server)) as client:
            names = [tool.name for tool in (await client.list_tools()).tools]
            hits = await client.call_tool("search_reports", {"query": "number of bakeries", "limit": 1})
            return names, structured(hits)

    names, hits = asyncio.run(run())
    assert sorted(names) == ["get_page", "list_reports", "search_reports"]
    assert hits[0]["citation"] == "Baltic Mills 2025, p. 1"
