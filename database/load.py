"""Load data/processed/*.parquet into PostgreSQL/PostGIS.

Usage:
    python -m database.load            # TRUNCATE + COPY all tables in one transaction, assert row counts
    python -m database.load --schema   # apply database/schema.sql only
    python -m database.load --views    # (re)create database/views.sql only
    python -m database.load --all      # schema -> load -> views

The load is idempotent: tables are truncated and refilled inside a single transaction, so a failed
load leaves the previous contents untouched.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time

import pandas as pd
import psycopg

from config import ROOT, get_path
from database import connect

LOG = logging.getLogger(__name__)

SQL_DIR = ROOT / "database"
CHUNK_ROWS = 200_000

# Child tables first when truncating; parents first when loading.
LOAD_ORDER = ["districts", "weather_daily", "satellite_16d", "crops_annual"]
COPY_COLUMNS: dict[str, list[str]] = {
    "weather_daily": [
        "district_id",
        "date",
        "rain_mm",
        "tmax_c",
        "tmin_c",
        "rh_pct",
        "wind_ms",
        "solar_mj",
        "is_synthetic",
    ],
    "satellite_16d": ["district_id", "date", "ndvi", "soil_moisture", "is_synthetic"],
    "crops_annual": [
        "district_id",
        "year",
        "season",
        "crop",
        "area_ha",
        "production_t",
        "yield_t_per_ha",
        "yield_flag",
        "is_synthetic",
    ],
}


def run_sql_file(conn: psycopg.Connection, name: str) -> None:
    """Execute a whole .sql file (multiple statements) on ``conn``."""
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    LOG.info("applied %s", name)


def _copy_frame(conn: psycopg.Connection, table: str, frame: pd.DataFrame) -> None:
    columns = COPY_COLUMNS[table]
    statement = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '')"
    with conn.cursor() as cur, cur.copy(statement) as copy:
        for start in range(0, len(frame), CHUNK_ROWS):
            chunk = frame.iloc[start : start + CHUNK_ROWS][columns]
            buffer = io.StringIO()
            chunk.to_csv(buffer, index=False, header=False, na_rep="", date_format="%Y-%m-%d")
            copy.write(buffer.getvalue())


def _load_districts(conn: psycopg.Connection, frame: pd.DataFrame) -> None:
    rows = [
        (
            r.district_id,
            r.state,
            r.district,
            r.centroid_lon,
            r.centroid_lat,
            r.geometry_wkt,
            r.area_km2,
            getattr(r, "member_districts", None),
            getattr(r, "icrisat_name", None),
        )
        for r in frame.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO districts (district_id, state, district, centroid, geom, area_km2, member_districts, icrisat_name)
            VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    ST_Multi(ST_MakeValid(ST_GeomFromText(%s, 4326))), %s, %s, %s)
            """,
            rows,
        )


def read_parquet_tables() -> dict[str, pd.DataFrame]:
    """Read the four processed tables; raises FileNotFoundError naming the missing file."""
    processed = get_path("data_processed")
    tables = {}
    for name in LOAD_ORDER:
        path = processed / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m etl.run --all` first")
        tables[name] = pd.read_parquet(path)
    return tables


def load_all(conn: psycopg.Connection) -> dict[str, int]:
    """Truncate and reload every table; returns the row counts read back from the database.

    Raises:
        AssertionError: if any table's row count differs from its parquet file.
    """
    tables = read_parquet_tables()
    started = time.time()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE weather_daily, satellite_16d, crops_annual, districts")
    _load_districts(conn, tables["districts"])
    for name in LOAD_ORDER[1:]:
        _copy_frame(conn, name, tables[name])
        LOG.info("loaded %s (%d rows)", name, len(tables[name]))
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for name in LOAD_ORDER:
            cur.execute(f"SELECT count(*) FROM {name}")
            counts[name] = int(cur.fetchone()[0])
    for name, expected in ((n, len(t)) for n, t in tables.items()):
        assert counts[name] == expected, (
            f"{name}: database has {counts[name]} rows, parquet has {expected}"
        )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT matviewname FROM pg_matviews WHERE schemaname = 'public' ORDER BY matviewname"
        )
        views = [row[0] for row in cur.fetchall()]
    for view in views:
        with conn.cursor() as cur:
            cur.execute(f"REFRESH MATERIALIZED VIEW {view}")
        LOG.info("refreshed materialized view %s", view)
    LOG.info("loaded all tables in %.1fs", time.time() - started)
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--schema", action="store_true", help="apply schema.sql only")
    group.add_argument("--views", action="store_true", help="create views.sql only")
    group.add_argument("--all", action="store_true", help="schema, load, views")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        with connect() as conn:
            if args.schema or args.all:
                run_sql_file(conn, "schema.sql")
                conn.commit()
            if args.views:
                run_sql_file(conn, "views.sql")
                conn.commit()
            if not (args.schema or args.views):
                counts = load_all(conn)
                conn.commit()
                print("row counts:", counts)  # noqa: T201 - CLI summary
                if args.all:
                    run_sql_file(conn, "views.sql")
                    conn.commit()
    except psycopg.OperationalError as exc:
        LOG.error(
            "cannot reach the database (%s). Start it with `make db-up` and check DB_URL in .env.",
            exc,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
