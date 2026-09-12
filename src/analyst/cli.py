"""Command line.

    analyst fetch                      download the reports listed in data/reports.yml
    analyst index                      extract, split, embed and store them
    analyst search "store count"       try the retrieval on its own
    analyst ask "How many ...?"        run the agents on a question
    analyst mcp                        serve the report tools over MCP (stdio)
"""

from __future__ import annotations

import argparse
import logging
import sys

from analyst import config


def _search():
    from analyst import db
    from analyst.embeddings import FastEmbedEmbedder
    from analyst.search import ReportSearch

    conn = db.connect()
    db.ensure_schema(conn)
    return ReportSearch(conn, FastEmbedEmbedder())


def _chat(model: str):
    from analyst.llm import GroqChat

    return GroqChat(model)


def cmd_fetch(args) -> int:
    from analyst.reports import fetch, load_reports

    for report in load_reports():
        path = fetch(report)
        print(f"{report.report_id:<16} {path.stat().st_size / 1e6:5.1f} MB  {report.title}")
    return 0


def cmd_index(args) -> int:
    from analyst import db
    from analyst.embeddings import FastEmbedEmbedder
    from analyst.index import index_report
    from analyst.reports import load_reports

    reports = [r for r in load_reports() if not args.report or r.report_id in args.report]
    missing = [r.report_id for r in reports if not r.path().exists()]
    if missing:
        raise SystemExit(f"not downloaded yet: {', '.join(missing)}. Run `analyst fetch` first.")
    conn = db.connect()
    db.ensure_schema(conn)
    embedder = FastEmbedEmbedder()
    for report in reports:
        counts = index_report(conn, report, embedder)
        print(f"{report.report_id:<16} {counts['pages']:>4} pages  {counts['passages']:>5} passages")
    return 0


def cmd_search(args) -> int:
    for hit in _search().search(args.query, company=args.company, fiscal_year=args.year, limit=args.limit):
        print(f"\n[{hit.citation}]  score {hit.score:.4f}\n{hit.text}")
    return 0


def cmd_ask(args) -> int:
    from analyst.graph import ask

    llm = _chat(config.chat_model())
    state = ask(args.question, llm, _search(), args.variant)
    if args.trace:
        print("searches:")
        for search in state.get("searches_run", []):
            print(f"  {search.query!r}  company={search.company}  year={search.fiscal_year}")
        print("passages:", ", ".join(f"{p.label} {p.citation}" for p in state.get("passages", [])))
        for claim in state.get("checked", []):
            mark = "accepted" if claim.supported else f"rejected ({claim.reason})"
            print(f"  {mark}: {claim.text}")
        usage = state["usage"]
        print(f"{usage['calls']} model calls, {usage['prompt_tokens'] + usage['completion_tokens']} tokens, "
              f"{state['seconds']} s\n")
    print(state["answer"].text)
    return 0


def cmd_mcp(args) -> int:
    from analyst import mcp_server

    mcp_server.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyst", description="Questions over company annual reports")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the reports").set_defaults(func=cmd_fetch)

    p = sub.add_parser("index", help="extract, split, embed and store the reports")
    p.add_argument("report", nargs="*", help="report ids; default: all")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search the indexed passages")
    p.add_argument("query")
    p.add_argument("--company")
    p.add_argument("--year", type=int)
    p.add_argument("-k", "--limit", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("--variant", default="baseline", help="analyst prompt to use, see agents/prompts.py")
    p.add_argument("--trace", action="store_true", help="show searches, passages and verdicts")
    p.set_defaults(func=cmd_ask)

    sub.add_parser("mcp", help="serve the report tools over MCP (stdio)").set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command != "mcp":           # the MCP server owns stdout; everything else may print
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
