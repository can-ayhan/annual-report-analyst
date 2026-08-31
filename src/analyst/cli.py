"""Command line.

    analyst fetch                      download the reports listed in data/reports.yml
    analyst index                      extract, split, embed and store them
    analyst search "store count"       try the retrieval on its own
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

    return parser


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
