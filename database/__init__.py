"""PostgreSQL + PostGIS warehouse: schema, loader, analytical views (Phase 2)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import psycopg
from sqlalchemy import Engine, create_engine

from config import load_env

LOG = logging.getLogger(__name__)

DEFAULT_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5434/postgres"


def get_db_url() -> str:
    """SQLAlchemy URL from DB_URL in the environment/.env (falls back to the docker-compose default)."""
    load_env()
    return os.environ.get("DB_URL", DEFAULT_DB_URL)


def psycopg_dsn() -> str:
    """The same URL without the SQLAlchemy driver suffix, for ``psycopg.connect``."""
    return get_db_url().replace("postgresql+psycopg://", "postgresql://", 1)


def connect(**kwargs: Any) -> psycopg.Connection:
    """Open a psycopg connection to the warehouse."""
    return psycopg.connect(psycopg_dsn(), **kwargs)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Shared SQLAlchemy engine (use for ``pandas.read_sql``)."""
    return create_engine(get_db_url(), pool_pre_ping=True)


def db_available(timeout_s: int = 3) -> bool:
    """True if the warehouse accepts connections (used to skip DB tests and fall back to parquet)."""
    try:
        with psycopg.connect(psycopg_dsn(), connect_timeout=timeout_s) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error:
        return False
    return True
