"""Command line.

    analyst fetch                      download the reports listed in data/reports.yml
    analyst index                      extract, split, embed and store them
    analyst search "store count"       try the retrieval on its own
    analyst ask "How many ...?"        run the agents on a question
    analyst eval --model NAME          run the question set
    analyst report                     table of all saved evaluation runs
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


def _chat(model: str, cache: bool, longest_wait: float | None = None):
    from analyst.evaluation import ModelSpec, load_models
    from analyst.llm import LONGEST_WAIT, GroqChat

    spec = load_models().get(model) or ModelSpec(name=model)
    cache_dir = config.ROOT / ".cache" / "llm" if cache else None
    return GroqChat(model, cache_dir=cache_dir, options=spec.options,
                    longest_wait=longest_wait or LONGEST_WAIT), spec


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

    llm, _ = _chat(config.chat_model(), cache=False)
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


def cmd_eval(args) -> int:
    from analyst import evaluation

    model = args.model or config.chat_model()
    llm, spec = _chat(model, cache=not args.no_cache, longest_wait=args.wait)
    questions = [q for q in evaluation.load_questions() if not args.only or q.id in args.only]
    directory = evaluation.RESULTS_DIR / args.out if args.out else evaluation.RESULTS_DIR

    def show(row: dict) -> None:
        mark = "ok  " if row["correct"] else "MISS"
        cited = {True: "page ok", False: "page MISS", None: ""}[row["cited"]]
        tokens = row["prompt_tokens"] + row["completion_tokens"]
        print(f"{mark} {row['id']:<18} {cited:<9} {tokens:>6} tokens {row['model_seconds']:>5.1f} s  "
              f"{row['answer'][:90]!r}", flush=True)

    result = evaluation.evaluate(llm, _search(), args.variant, spec, questions, on_row=show)
    summary = result["summary"]
    print(f"\n{summary['model']} / {summary['variant']}: "
          f"{summary['correct']}/{summary['questions']} correct, "
          f"{summary['cited']}/{summary['answerable']} with the right page, "
          f"{summary['tokens_per_question']} tokens and {summary['median_seconds']} s per question, "
          f"${summary['cost_per_100_questions_usd']} per 100 questions at list price")
    if "stopped" in result:
        print(f"stopped early: {result['stopped']}")
        return 1
    if args.out or not args.only:
        print("saved to", evaluation.save(result, directory))
    return 0


def cmd_report(args) -> int:
    from analyst import evaluation

    directory = evaluation.RESULTS_DIR / args.out if args.out else evaluation.RESULTS_DIR
    print(evaluation.comparison_table(directory))
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

    p = sub.add_parser("eval", help="run the question set")
    p.add_argument("--model", help="default: ANALYST_MODEL")
    p.add_argument("--variant", default="baseline")
    p.add_argument("--only", nargs="*", help="question ids; a partial run is only saved with --out")
    p.add_argument("--out", help="save under evals/results/OUT, to keep a run on part of the set apart")
    p.add_argument("--wait", type=float, help="longest rate-limit wait to sit out, in seconds (default 90)")
    p.add_argument("--no-cache", action="store_true", help="do not replay identical model calls from .cache")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="table of saved evaluation runs")
    p.add_argument("--out", help="the runs saved under evals/results/OUT")
    p.set_defaults(func=cmd_report)
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
