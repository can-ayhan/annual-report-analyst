# Annual Report Analyst

A multi-agent system that answers questions over public company annual reports and cites the
page behind every claim. What it cannot back with a passage from a report, it does not say.

```
$ analyst ask "How many stores did Dino Polska have at the end of 2025, and how many did it open during the year?"
As at 31 December 2025, the Dino Polska store network consisted of 3,033 stores. [Dino Polska 2025, p. 10; Dino Polska 2025, p. 7; Dino Polska 2025, p. 8]
In 2025, Dino Polska opened 345 new stores. [Dino Polska 2025, p. 10]

$ analyst ask "What was Allegro's revenue in 2025?"
The indexed reports do not contain an answer to this.
```

```
                 ┌──────────────── LangGraph ─────────────────┐
  question ──>  planner ──> retriever ──> analyst ──> verifier ──> answer with pages
                               │             ^            │
                               │             └── revise ──┘   (once, if a claim was rejected)
                    list / search / read page
                               │
                  PostgreSQL + pgvector  <── the same three tools ──>  MCP server
```

| | |
|---|---|
| Agents | LangGraph: planner, retriever, analyst, verifier |
| Chat model | Groq API, any model with tool use; the free tier is enough |
| Retrieval | pgvector on PostgreSQL 16, hybrid with full-text search; passages never cross a page |
| Embeddings | a small ONNX model on the CPU (fastembed), no service and no key |
| Tools for other clients | MCP server over stdio |
| Evaluation | 18 questions with known answers and pages; accuracy, cost and latency per model and prompt |

Nothing here costs money to run: the embeddings are local, the database is a container and the
chat model is on a free tier.

## The library

Four reports from three companies listed in Warsaw, one of them for two years (`data/reports.yml`):

| Report | Pages | Passages |
|---|---:|---:|
| CD PROJEKT, Management Board report 2024 | 146 | 481 |
| CD PROJEKT, Management Board report 2025 | 154 | 481 |
| Dino Polska, Management Board activity report 2025 | 103 | 466 |
| InPost, Integrated Annual Report 2025 | 289 | 941 |
| | 692 | 2,369 |

The PDFs are not in the repository. `analyst fetch` downloads them from the publishers and checks
each against a recorded checksum, because a replaced file would shift the pages that get cited.

## Run it

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d                 # PostgreSQL with pgvector on localhost:5436
cp .env.example .env                 # add GROQ_API_KEY and ANALYST_MODEL to ask questions

analyst fetch                        # four PDFs, 27 MB
analyst index                        # about ten minutes on a laptop CPU, once
analyst search "number of stores" --company "Dino Polska"
analyst ask "How many employees did InPost have at the end of 2025?" --trace
```

`fetch`, `index`, `search` and the MCP server need no key. `--trace` shows the searches, the
passages and the verifier's verdict on every claim.

## How a question is answered

| Agent | Model call | What it does |
|---|---|---|
| Planner | one | Turns the question into up to four searches, each aimed at a company and year where it can. A comparison becomes one search per side. |
| Retriever | one | Runs the searches, looks at the start of each passage found, and may ask for up to two more searches. Hands on the six best passages. |
| Analyst | one, two after a revision | Writes the answer as claims. Each claim is one checkable sentence with the passages that state it. |
| Verifier | one | Accepts or rejects each claim against the passages it cites. A rejected claim goes back to the analyst once, with the reason. |

Every agent asks the model for a Pydantic object, not for text (`src/analyst/llm.py`): the request
forces a tool call with the object's schema, the reply is validated, and a reply that does not fit
is sent back once with the error.

A typical question is four model calls, five to six thousand tokens and about two seconds of
model time.

### Retrieval

- A page is split into passages of at most 180 words, along line breaks, because in these reports a
  line is often a table row. A passage never leaves its page, which is what makes the citation exact.
- Search is hybrid. The embedding matches meaning; PostgreSQL full-text search matches the exact
  words (names, line items, figures) that an embedding blurs. The two rankings are merged by
  reciprocal rank fusion (`src/analyst/search.py`).
- The question itself is always one of the searches, next to the planner's.

### Verification

A claim reaches the answer only if it passes three checks (`src/analyst/agents/verifier.py`):

1. it cites passages, and only passages that were retrieved;
2. every figure in it occurs in those passages as written, so a rounded, converted or invented
   number fails without a model being asked;
3. a model reads the claim against those passages, and to accept it has to quote the words that
   back it. The quote is then looked up in the passage: words that are not there, or not in that
   order, do not count.

Check 2 is plain string work: a claim of "PLN 867 million" against a table that says `866 989`
(in thousands) is rejected because 867 is not in the passage as written.

A rejection from the saved runs:

```
The CD PROJEKT Group's net profit margin was 60.1% in 2025.
  -> accepted on the strength of '60.1%' alone, which is too short to show what the figure refers to
```

That is a correct claim that was lost: the verifier model agreed with it and showed nothing but
the number. It is the price of the rule, and it shows up in the results below.

## MCP server

`analyst mcp` serves the three functions the agents use to any MCP client, over stdio:

| Tool | Returns |
|---|---|
| `list_reports()` | the reports in the library |
| `search_reports(query, company?, fiscal_year?, limit?)` | passages with report, page and a ready citation |
| `get_page(report_id, page)` | the full text of one page |

A client is pointed at it with the usual entry:

```json
{"mcpServers": {"annual-reports": {"command": "/path/to/.venv/bin/analyst", "args": ["mcp"]}}}
```

The database and the embedding model are opened on first use, so the handshake does not wait for
them. `tests/test_mcp.py` talks to the server through an MCP client, in memory and as a subprocess.

## Evaluation

`evals/questions.yml` has 18 questions written by reading the reports. Sixteen have the expected
figures and every page that states them; one of those needs two reports. Two are about companies
the library does not have, where the right answer is to say so.

- **Correct**: the expected figures are in the answer, and no known wrong reading of them is.
  For the two questions without an answer: nothing was claimed.
- **Right page cited**: the answer cites a page on which the figure is stated.
- **Cost**: tokens of every model call at the provider's list price (`evals/models.yml`).
- **Model time**: duration of the model calls; waiting for the free tier's rate limit is left out.

```
analyst eval --model NAME [--variant quote_first] [--only ID ...]
analyst report
```

### Three models, same prompts

All 18 questions, baseline prompt:

| Model | Prompt | Correct | Right page cited | Claims rejected | Tokens / question | Median model time | Cost per 100 questions |
|---|---|---:|---:|---:|---:|---:|---:|
| openai/gpt-oss-120b | baseline | 18/18 | 16/16 | 0 | 5,231 | 2.2 s | $0.101 |
| openai/gpt-oss-20b | baseline | 16/18 | 15/16 | 2 | 5,135 | 1.7 s | $0.048 |
| qwen/qwen3.8-27b | baseline | 17/18 | 15/16 | 0 | 6,209 | 2.0 s | $0.623 |

### Three ways of instructing the analyst

The smallest model on eight of the questions, which is what one day of the free tier leaves room for after the
runs above (planner and retriever calls are replayed from the baseline run):

| Model | Prompt | Correct | Right page cited | Claims rejected | Tokens / question | Median model time | Cost per 100 questions |
|---|---|---:|---:|---:|---:|---:|---:|
| openai/gpt-oss-20b | baseline | 6/8 | 6/7 | 2 | 5,138 | 1.7 s | $0.049 |
| openai/gpt-oss-20b | minimal | 6/8 | 6/7 | 2 | 5,020 | 1.7 s | $0.048 |
| openai/gpt-oss-20b | quote_first | 8/8 | 7/7 | 0 | 5,311 | 2.1 s | $0.051 |

What the runs show:

- **No invented answers and no wrong pages.** All three models said the library has no answer for
  the two companies it does not hold. Every answer that was given cites a page that states its
  figure; the two misses under "Right page cited" are questions left unanswered.
- **The misses are of three kinds**, and only the first is a wrong statement reaching the answer:
  - the small model read a table in PLN thousands as millions (`PLN 2,546,313 million`), and the
    verifier let it pass;
  - the small model's verifier agreed with a correct profit margin but quoted only `60.1%`, twice,
    so the claim was rejected and the question went unanswered;
  - Qwen had the right page as its first passage and declined to read the "Total" row of the revenue
    breakdown as the Group's revenue.
- **Cost differs far more than accuracy.** At list price Qwen costs six times the large GPT-OSS
  model and thirteen times the small one for the same four calls. Model time is about two seconds
  per question for all three.
- **Copying the passage first is the one prompt change that mattered.** With `quote_first` the
  small model got all eight right: the copied table row carries its unit, and it gives the
  verifier something to quote. It costs 3 % more tokens and 0.4 s. The two-sentence `minimal`
  prompt did exactly as well as the careful baseline, which says the structure (claims, citations,
  a verifier) does more of the work than the wording.

Eight and eighteen questions show large effects and failures worth fixing. They do not rank models
that are one answer apart.

## What went wrong along the way

- **A tool schema one model could not use.** Pydantic writes an optional field as `anyOf` with a
  default. One of the three models answered every such request with an unparsable tool call. With
  the field written as a type list and every property required, it works on all three.
- **A tool call that came out as prose.** A reasoning model sometimes writes its thoughts where the
  tool call should be, and at temperature 0 does it again on a retry. After two such failures the
  same schema is requested as a JSON response.
- **Correct claims rejected over a table row.** The row reads `Total headcount 49,887 55,856`; the
  verifier quoted `Total headcount 55,856`, which is honest and is not a substring. The quoted words
  now have to appear in order and close together, and each part of a quote with an ellipsis is
  looked up on its own.
- **The right page not asked for.** A planner query of two words missed the page that the question,
  searched as it was asked, finds first. And a follow-up search for "InPost Group" found nothing
  because the library calls the company "InPost". Both are now handled without a model.
- **A right figure with the wrong scale.** `PLN 2,546,313 million` for a table in thousands passed
  every check: the number is in the passage. The scoring now counts such readings as wrong; the
  verifier still lets them through (see Limits).

## Layout

```
data/reports.yml           the reports: publisher URL, checksum, pages
sql/schema.sql             report, page and passage tables with the vector and full-text indexes
src/analyst/
    reports.py pdf.py chunking.py embeddings.py index.py     from PDF to indexed passages
    search.py                                                 list, search, read a page
    llm.py                                                    structured model calls, cache, rate limits
    agents/                                                   planner, retriever, analyst, verifier, prompts
    graph.py                                                  the LangGraph graph
    mcp_server.py                                             the tools over MCP
    evaluation.py cli.py                                      question set runner and commands
evals/                     questions, compared models, saved runs
tests/                     98 tests; none needs a key or the real PDFs
```

## Limits

- **Scale and units.** The verifier checks that a figure is in the passage, not that "thousand" or
  "million" next to it is right. The unit is often in a table header, away from the number.
- **Eighteen questions** are enough to catch breakage and to see large differences, not to rank
  models that are one question apart.
- **Three companies, one language.** Other reports will have layouts the page splitter has not
  seen. Scanned PDFs are not handled.
- **No calculations.** The analyst is told to report figures as written. "By how much did revenue
  grow" is answered with the two figures, or with the growth rate if the report states it.
- **Citations are PDF pages**, the position in the file, which is what a viewer shows. The number
  printed on the page can differ.
