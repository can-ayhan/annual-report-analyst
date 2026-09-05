"""Settings come from the environment; a .env file in the project root is read first."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_FILE = ROOT / "data" / "reports.yml"
REPORTS_DIR = ROOT / "data" / "reports"
SCHEMA_FILE = ROOT / "sql" / "schema.sql"

# Small, runs on a laptop CPU, 384 dimensions (the width of passage.embedding in sql/schema.sql).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=value lines; variables that are already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def database_conninfo() -> str:
    env = os.environ.get
    return (
        f"host={env('PGHOST', 'localhost')} port={env('PGPORT', '5436')} "
        f"dbname={env('PGDATABASE', 'reports')} user={env('PGUSER', 'analyst')} "
        f"password={env('PGPASSWORD', 'analyst')}"
    )


def embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def chat_model() -> str:
    """The Groq model the agents use. There is no default: models come and go, so the choice is explicit."""
    model = os.environ.get("ANALYST_MODEL")
    if not model:
        raise SystemExit("ANALYST_MODEL is not set. Put the id of a Groq chat model with tool use in .env.")
    return model
