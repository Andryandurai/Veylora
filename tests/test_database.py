"""Phase 2 tests. Skipped automatically when the warehouse is not reachable or not loaded."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from config import get_path
from database import connect, db_available

pytestmark = pytest.mark.skipif(
    not db_available(), reason="PostgreSQL warehouse not reachable (make db-up)"
)


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        with connection.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.v_training_table'), to_regclass('public.weather_daily')"
            )
            training, weather = cur.fetchone()
        if training is None or weather is None:
            pytest.skip("warehouse not loaded: run `make db-all`")
        yield connection


def scalar(conn, sql: str, params: tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


@pytest.mark.parametrize("table", ["districts", "weather_daily", "satellite_16d", "crops_annual"])
def test_row_counts_equal_parquet(conn, table: str) -> None:
    parquet = get_path("data_processed") / f"{table}.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet.name} not built")
    assert scalar(conn, f"SELECT count(*) FROM {table}") == len(pd.read_parquet(parquet))


def test_postgis_enabled_and_geometries_valid(conn) -> None:
    assert scalar(conn, "SELECT PostGIS_Version()")
    assert scalar(conn, "SELECT count(*) FROM districts WHERE NOT ST_IsValid(geom)") == 0
    assert scalar(conn, "SELECT count(*) FROM districts WHERE ST_SRID(geom) <> 4326") == 0


def test_training_table_one_row_per_key_and_no_null_labels(conn) -> None:
    rows = scalar(conn, "SELECT count(*) FROM v_training_table")
    keys = scalar(
        conn,
        "SELECT count(*) FROM (SELECT DISTINCT district_id, year, season, crop FROM v_training_table) k",
    )
    assert rows > 0 and rows == keys
    assert scalar(conn, "SELECT count(*) FROM v_training_table WHERE yield_label IS NULL") == 0
    assert scalar(conn, "SELECT count(*) FROM v_training_table WHERE yield_label <= 0") == 0


def test_lag_features_use_only_earlier_years(conn) -> None:
    mismatches = scalar(
        conn,
        """
        SELECT count(*) FROM v_training_table t
        JOIN crops_annual c ON c.district_id = t.district_id AND c.season = t.season AND c.crop = t.crop
                           AND c.year = t.year - 1
        WHERE t.prev_year_yield IS DISTINCT FROM c.yield_t_per_ha
        """,
    )
    assert mismatches == 0
    # Rows without a real previous year must have a NULL lag (no silent fill from an older year).
    assert (
        scalar(
            conn,
            """
            SELECT count(*) FROM v_training_table t
            WHERE t.prev_year_yield IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM crops_annual c WHERE c.district_id = t.district_id
                              AND c.season = t.season AND c.crop = t.crop AND c.year = t.year - 1)
            """,
        )
        == 0
    )


def test_lead_windows_are_nested(conn) -> None:
    """Rain over the first 30 days cannot exceed rain over 60, 90 days or the season."""
    bad = scalar(
        conn,
        "SELECT count(*) FROM v_season_weather WHERE rain_d30 > rain_d60 + 1e-3 OR rain_d60 > rain_d90 + 1e-3 OR rain_d90 > total_rain + 1e-3",
    )
    assert bad == 0
    assert (
        scalar(
            conn,
            "SELECT count(*) FROM v_season_weather WHERE dry_spell_d30 > 30 OR dry_spell_d90 > 90",
        )
        == 0
    )


def test_satellite_features_only_use_finished_composites(conn) -> None:
    """The NDVI at sowing must come from a composite whose last day (date + 15) is <= season start."""
    violations = scalar(
        conn,
        """
        SELECT count(*) FROM v_season_weather w
        WHERE w.ndvi_at_sowing IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM satellite_16d s WHERE s.district_id = w.district_id
              AND s.ndvi = w.ndvi_at_sowing AND s.date + 15 <= w.season_start)
        """,
    )
    assert violations == 0


def test_spatial_neighbours_known_pair_and_speed(conn) -> None:
    sql = """
        SELECT b.district_id FROM districts a JOIN districts b ON b.district_id <> a.district_id
         AND ST_DWithin(a.geom::geography, b.geom::geography, 100000)
        WHERE a.district_id = %s
    """
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, ("tamil_nadu__thanjavur",))
        neighbours = {row[0] for row in cur.fetchall()}
    assert time.perf_counter() - started < 1.0
    assert "tamil_nadu__trichy" in neighbours  # Thanjavur and Tiruchirappalli share a border
    assert "karnataka__belgaum" not in neighbours  # ~900 km away
    assert "tamil_nadu__thanjavur" not in neighbours
