"""Shared helpers for IMD gridded data (rainfall 0.25 deg, Tmax/Tmin 1 deg) via the imdlib package.

imdlib downloads one binary GrADS file per variable and year from the IMD Pune server. This module
adds what the plan needs on top of it: resumable downloads (skip complete files, retry flaky ones),
a size-based completeness check, and a loader that returns an ``xarray.DataArray`` with NaN for
missing cells, ready for area-weighted aggregation to district polygons.

Run ``python -m etl.imd_common --years 1991 2024`` to pre-download everything.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import xarray as xr

from config import get_path

LOG = logging.getLogger(__name__)

# (n_lat, n_lon) of each IMD product; used to verify that a download is complete.
GRID_SHAPE: dict[str, tuple[int, int]] = {"rain": (135, 129), "tmax": (31, 31), "tmin": (31, 31)}
_FLOAT32_BYTES = 4
VARIABLES = ("rain", "tmax", "tmin")


def default_raw_dir() -> Path:
    """Directory where IMD yearly files are cached (data/raw/imd)."""
    return get_path("data_raw") / "imd"


def _candidate_paths(var: str, year: int, raw_dir: Path) -> list[Path]:
    folder = raw_dir / var
    return [folder / f"{year}.grd", folder / f"{year}.GRD"]


def find_file(var: str, year: int, raw_dir: Path | None = None) -> Path | None:
    """Return the cached file for (var, year) if it exists, else None."""
    raw_dir = raw_dir or default_raw_dir()
    for path in _candidate_paths(var, year, raw_dir):
        if path.is_file():
            return path
    return None


def is_complete(var: str, year: int, raw_dir: Path | None = None) -> bool:
    """True when the cached file has exactly days x lat x lon float32 values."""
    path = find_file(var, year, raw_dir)
    if path is None:
        return False
    days = 366 if calendar.isleap(year) else 365
    n_a, n_b = GRID_SHAPE[var]
    return path.stat().st_size == days * n_a * n_b * _FLOAT32_BYTES


def ensure_year(
    var: str, year: int, raw_dir: Path | None = None, retries: int = 4, pause_s: float = 20.0
) -> Path:
    """Download (var, year) from IMD unless a complete file is already cached.

    Raises:
        RuntimeError: if the download still fails after ``retries`` attempts.
    """
    import imdlib as imd

    raw_dir = raw_dir or default_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    if is_complete(var, year, raw_dir):
        return find_file(var, year, raw_dir)  # type: ignore[return-value]
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            for stale in _candidate_paths(var, year, raw_dir):
                stale.unlink(missing_ok=True)  # never keep a truncated file
            imd.get_data(var, year, year, fn_format="yearwise", file_dir=str(raw_dir))
            if is_complete(var, year, raw_dir):
                LOG.info("IMD %s %d downloaded", var, year)
                return find_file(var, year, raw_dir)  # type: ignore[return-value]
            raise RuntimeError("downloaded file has an unexpected size")
        except Exception as exc:
            last_error = exc
            LOG.warning("IMD %s %d attempt %d/%d failed: %s", var, year, attempt, retries, exc)
            time.sleep(pause_s * attempt)
    raise RuntimeError(f"IMD download failed for {var} {year}: {last_error}")


def ensure_years(
    var: str, years: Iterable[int], raw_dir: Path | None = None
) -> tuple[list[int], list[int]]:
    """Download several years; return (available_years, failed_years) without aborting on failures."""
    available: list[int] = []
    failed: list[int] = []
    for year in years:
        try:
            ensure_year(var, year, raw_dir)
            available.append(year)
        except RuntimeError as exc:
            LOG.error("%s", exc)
            failed.append(year)
    return available, failed


def open_year(var: str, year: int, raw_dir: Path | None = None) -> xr.DataArray:
    """Open one cached year as a (time, lat, lon) DataArray; missing cells are NaN."""
    import imdlib as imd

    raw_dir = raw_dir or default_raw_dir()
    if not is_complete(var, year, raw_dir):
        raise FileNotFoundError(f"IMD {var} {year} not cached in {raw_dir}; run etl.imd_common")
    dataset = imd.open_data(var, year, year, "yearwise", str(raw_dir)).get_xarray()
    array = dataset[var].astype("float32")
    # imdlib already maps the -999 fill value to NaN; guard against odd versions anyway.
    return array.where(array > -900.0)


def main() -> None:
    """CLI: pre-download IMD files for a year range."""
    parser = argparse.ArgumentParser(description="Download IMD gridded files (resumable).")
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"), required=True)
    parser.add_argument("--vars", nargs="+", default=list(VARIABLES), choices=VARIABLES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    years = range(args.years[0], args.years[1] + 1)
    for var in args.vars:
        ok, bad = ensure_years(var, years)
        LOG.info("IMD %s: %d years available, failed=%s", var, len(ok), bad)


if __name__ == "__main__":
    main()
