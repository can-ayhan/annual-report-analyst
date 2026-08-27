"""PostgreSQL connection with the pgvector types registered."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from analyst.config import SCHEMA_FILE, database_conninfo


def connect(conninfo: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(conninfo or database_conninfo())
    conn.execute("create extension if not exists vector")
    conn.commit()
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.commit()
