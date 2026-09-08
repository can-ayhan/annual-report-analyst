# Annual Report Analyst

Questions about public companies answered from their annual reports, with the page behind every claim.

The idea: index the reports page by page, let a small team of agents find the passages, write the
answer and check it, and refuse to state anything a retrieved passage does not back.

## Plan

1. A list of reports to index, fetched from the publishers
2. Pages into passages, passages into pgvector
3. Agents on top: plan the searches, retrieve, answer, verify
4. The search and page tools as an MCP server
5. An evaluation: questions with known answers, several models

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d              # PostgreSQL with pgvector on localhost:5436
analyst fetch                     # download the reports in data/reports.yml and check their checksums
analyst index                     # extract, split, embed and store them (about ten minutes on a laptop CPU)
analyst search "number of stores" --company "Dino Polska"
```

The embeddings come from a small model that runs locally, so nothing here needs a key.

Asking a question needs a chat model. Copy `.env.example` to `.env`, add a Groq key (the free tier
is enough) and the model to use, then:

```
analyst ask "How many stores did Dino Polska have at the end of 2025?" --trace
```

Four agents work on it in a LangGraph graph: a planner turns the question into searches, a retriever
runs them and decides whether the results are enough, an analyst writes the answer as claims with
the passages they come from, and a verifier rejects every claim that its passages do not back.

## Status

Work in progress: steps 1 to 3 are done.
