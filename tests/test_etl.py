"""Phase 1 tests. No network, no credentials: everything runs on tiny fixtures or toy geometry."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from etl import boundaries, crop_yield, gee_auth_check, gee_satellite, grid, nasa_power, validate
from etl.names import canonical_key, make_district_id, name_key, split_aliases, state_key
from etl.synthetic.crops import generate_crops
from etl.synthetic.satellite import generate_satellite
from etl.synthetic.weather import generate_weather
from etl.weather import assemble_weather

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = {
    "states": ["Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana"],
    "crops": ["rice", "wheat", "cotton", "sugarcane"],
    "crop_season": {"rice": "kharif", "wheat": "rabi", "cotton": "kharif", "sugarcane": "kharif"},
    "years": {"history_start": 2008, "start": 2008, "end": 2012},
    "seasons": {
        "kharif": {"start_month": 6, "end_month": 10},
        "rabi": {"start_month": 11, "end_month": 3},
    },
}


# ---------------------------------------------------------------- names
def test_name_normalisation_handles_diacritics_and_punctuation() -> None:
    assert name_key("Kadapa(YSR)") == "kadapa ysr"
    assert state_key("Tamil Nādu") == "tamil nadu"
    assert state_key("Orissa") == "odisha"
    assert make_district_id("Tamil Nādu", "Tiruchirappalli") == "tamil_nadu__tiruchirappalli"
    assert split_aliases("North Arcot / Vellore") == ["North Arcot", "Vellore"]


def test_alias_map_resolves_spelling_variants() -> None:
    assert canonical_key("Tamil Nadu", "Trichy") == canonical_key("Tamil Nadu", "Tiruchirapalli")
    assert canonical_key("Tamil Nadu", "Tiruchirappalli") == name_key("Tiruchirappalli")
    assert canonical_key("Karnataka", "Shimoge") == "shimoga"
    assert canonical_key("Telangana", "Hyderabad") == "hydrabad"
    assert canonical_key("Karnataka", "Unknown Place") == "unknown place"


# ---------------------------------------------------------------- grid aggregation
def _toy_grid() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.5, 1.5]), np.array([0.5, 1.5])  # 2 x 2 cells of 1 degree


def test_area_weighted_mean_on_2x2_grid() -> None:
    lat, lon = _toy_grid()
    polygons = gpd.GeoSeries(
        [box(0, 0, 2, 1), box(0.5, 0.5, 1.5, 1.5), box(0, 0, 1, 2), box(0, 0, 0.5, 1)],
        crs="EPSG:4326",
    )
    weights = grid.build_weights(polygons, lat, lon)
    values = np.array(
        [[1.0, 2.0, 3.0, 4.0]]
    )  # cells in row-major order: (0,0)=1 (0,1)=2 (1,0)=3 (1,1)=4
    out = grid.aggregate(values, weights)[0]
    assert out[0] == pytest.approx(1.5, rel=1e-3)  # bottom row
    assert out[1] == pytest.approx(2.5, rel=1e-2)  # quarter of every cell
    assert out[2] == pytest.approx(2.0, rel=1e-2)  # left column (cells 1 and 3)
    assert out[3] == pytest.approx(1.0, rel=1e-6)  # half of cell (0,0) only


def test_aggregate_renormalises_over_missing_cells() -> None:
    lat, lon = _toy_grid()
    weights = grid.build_weights(gpd.GeoSeries([box(0, 0, 2, 1)], crs="EPSG:4326"), lat, lon)
    values = np.array([[1.0, np.nan, 3.0, 4.0], [np.nan, np.nan, 3.0, 4.0]])
    out = grid.aggregate(values, weights)
    assert out[0, 0] == pytest.approx(1.0)  # NaN cell ignored, weights renormalised
    assert np.isnan(out[1, 0])  # no valid cell on that day


def test_polygon_without_valid_overlap_uses_nearest_valid_cell() -> None:
    lat, lon = _toy_grid()
    valid = np.array([False, False, False, True])
    weights = grid.build_weights(
        gpd.GeoSeries([box(0.1, 0.1, 0.4, 0.4)], crs="EPSG:4326"), lat, lon, valid
    )
    assert weights[0, 3] == 1.0
    assert weights.nnz == 1


# ---------------------------------------------------------------- boundaries / unit assignment
def _toy_modern() -> gpd.GeoDataFrame:
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    geoms = [box(i, 0, i + 1, 1) for i in range(5)]
    return gpd.GeoDataFrame(
        {"state": "Tamil Nadu", "state_key": "tamil nadu", "district": names},
        geometry=geoms,
        crs="EPSG:4326",
    )


def test_units_anchor_by_name_then_nearest_and_overrides() -> None:
    units = pd.DataFrame(
        {
            "state": ["Tamil Nadu"] * 2,
            "name": ["Alpha", "Delta / Epsilon"],
            "lat": [0.5, 0.5],
            "lon": [0.5, 3.5],
        }
    )
    assigned = boundaries.assign_units(_toy_modern(), units, overrides={})
    unit = dict(zip(assigned["district"], assigned["unit"], strict=True))
    assert unit["Alpha"] == "Alpha"
    assert unit["Delta"] == unit["Epsilon"] == "Delta / Epsilon"
    assert unit["Beta"] == "Alpha"  # adjacent to the Alpha anchor
    assert unit["Gamma"] == "Delta / Epsilon"
    forced = boundaries.assign_units(
        _toy_modern(), units, overrides={("tamil nadu", "gamma"): "Alpha"}
    )
    assert dict(zip(forced["district"], forced["unit"], strict=True))["Gamma"] == "Alpha"


def test_build_districts_dissolves_units_and_follows_contract() -> None:
    units = pd.DataFrame(
        {
            "state": ["Tamil Nadu"] * 2,
            "name": ["Alpha", "Delta / Epsilon"],
            "lat": [0.5, 0.5],
            "lon": [0.5, 3.5],
        }
    )
    districts = boundaries.build_districts(
        boundaries.assign_units(_toy_modern(), units, overrides={})
    )
    assert len(districts) == 2
    assert set(districts["district_id"]) == {"tamil_nadu__alpha", "tamil_nadu__epsilon"}
    for column in (
        "district_id",
        "state",
        "district",
        "centroid_lat",
        "centroid_lon",
        "geometry_wkt",
        "area_km2",
    ):
        assert column in districts
    assert districts["geometry_wkt"].str.startswith("MULTIPOLYGON").all()
    assert (districts["area_km2"] > 1e4).all()


# ---------------------------------------------------------------- crops
def _districts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district_id": ["tamil_nadu__trichy", "karnataka__raichur", "telangana__adilabad"],
            "state": ["Tamil Nadu", "Karnataka", "Telangana"],
            "icrisat_name": ["Tiruchirapalli / Trichy", "Raichur", "Adilabad"],
            "member_districts": ["Karur|Tiruchirappalli", "Raichur|Yadgir", "Adilabad"],
        }
    )


def _icrisat_wide() -> pd.DataFrame:
    base = {"State Name": "Tamil Nadu", "Dist Name": "Tiruchirapalli / Trichy"}
    rows = [
        # 2 districts-years: consistent yield, mismatching reported yield, zero-area wheat, other-state row
        {**base, "Year": 2010, "RICE AREA": 100.0, "RICE PRODUCTION": 300.0, "RICE YIELD": 3000.0},
        {**base, "Year": 2011, "RICE AREA": 100.0, "RICE PRODUCTION": 300.0, "RICE YIELD": 2000.0},
        {
            "State Name": "Karnataka",
            "Dist Name": "Raichur",
            "Year": 2010,
            "RICE AREA": 50.0,
            "RICE PRODUCTION": 100.0,
            "RICE YIELD": 2000.0,
        },
        {
            "State Name": "Punjab",
            "Dist Name": "Ludhiana",
            "Year": 2010,
            "RICE AREA": 10.0,
            "RICE PRODUCTION": 40.0,
            "RICE YIELD": 4000.0,
        },
    ]
    frame = pd.DataFrame(rows)
    for crop in ("WHEAT", "COTTON", "SUGARCANE"):
        frame[f"{crop} AREA"] = 0.0
        frame[f"{crop} PRODUCTION"] = 0.0
        frame[f"{crop} YIELD"] = 0.0
    frame.loc[0, ["WHEAT AREA", "WHEAT PRODUCTION", "WHEAT YIELD"]] = [2.0, 3.0, 1500.0]
    return frame


def test_icrisat_units_conversion_zero_area_and_yield_flag() -> None:
    tidy = crop_yield.icrisat_to_tidy(_icrisat_wide(), _districts(), SETTINGS)
    assert set(tidy.columns) == {*crop_yield.CROP_COLUMNS, "mismatch_rel"}
    rice = tidy[tidy["crop"] == "rice"].set_index(["district_id", "year"])
    assert rice.loc[("tamil_nadu__trichy", 2010), "area_ha"] == pytest.approx(100_000.0)
    assert rice.loc[("tamil_nadu__trichy", 2010), "production_t"] == pytest.approx(300_000.0)
    assert rice.loc[("tamil_nadu__trichy", 2010), "yield_t_per_ha"] == pytest.approx(3.0)
    assert not rice.loc[("tamil_nadu__trichy", 2010), "yield_flag"]
    assert rice.loc[("tamil_nadu__trichy", 2011), "yield_flag"]  # reported 2.0 vs recomputed 3.0
    assert "punjab" not in " ".join(tidy["district_id"])  # unit lookup drops non-study districts
    assert set(tidy["crop"]) == {"rice", "wheat"}  # zero-area crops are dropped
    assert tidy.loc[tidy["crop"] == "wheat", "season"].iloc[0] == "rabi"
    assert not tidy["is_synthetic"].any()


def test_apy_csv_maps_modern_names_aggregates_and_reports_unmatched() -> None:
    tidy, unmatched, coverage = crop_yield.apy_csv_to_tidy(
        FIXTURES / "apy_sample.csv", _districts(), SETTINGS
    )
    key = ["district_id", "year", "season", "crop"]
    rice = tidy[(tidy["crop"] == "rice")].set_index(key)
    # Tiruchirappalli (1000 ha) + Karur (500 ha) are both inside the Trichy unit and sum up.
    assert rice.loc[("tamil_nadu__trichy", 2010, "kharif", "rice"), "area_ha"] == pytest.approx(
        1500.0
    )
    assert rice.loc[
        ("tamil_nadu__trichy", 2010, "kharif", "rice"), "production_t"
    ] == pytest.approx(4250.0)
    assert ("tamil_nadu__trichy", 2010, "rabi", "rice") in rice.index
    cane = tidy[tidy["crop"] == "sugarcane"].iloc[0]
    assert cane["season"] == "kharif" and cane["year"] == 2011  # whole-year mapped via crop_season
    cotton = tidy[tidy["crop"] == "cotton"].iloc[0]
    assert cotton["production_t"] == pytest.approx(1000 * 0.17)  # bales -> tonnes
    assert "banana" not in set(tidy["crop"])
    # Telangana district listed under Andhra Pradesh is found through the state-agnostic fallback.
    assert "telangana__adilabad" in set(tidy["district_id"])
    assert set(unmatched["District_Name"]) == {"NOWHERE", "NOWHEREVILLE"}
    assert 0.5 < coverage < 1.0


def test_crop_data_unavailable_error_explains_where_to_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(force: bool = False) -> pd.DataFrame:
        raise crop_yield.icrisat.IcrisatUnavailableError("offline")

    monkeypatch.setattr(crop_yield.icrisat, "fetch_apy", boom)
    monkeypatch.setattr(crop_yield.icrisat, "raw_dir", lambda: tmp_path)
    with pytest.raises(crop_yield.CropDataUnavailableError, match="data.gov.in"):
        crop_yield.build_crops(_districts())


# ---------------------------------------------------------------- NASA POWER and GEE parsing
def test_power_parser_maps_names_and_missing_values() -> None:
    frame = nasa_power.parse_response(json.loads((FIXTURES / "power_sample.json").read_text()))
    assert list(frame.columns) == ["date", *nasa_power.OUTPUT_COLUMNS]
    assert len(frame) == 3
    assert np.isnan(frame.loc[1, "rh_pct"]) and np.isnan(frame.loc[2, "p_tmax_c"])  # -999 -> NaN
    assert frame.loc[0, "rh_pct"] == pytest.approx(60.5)
    assert frame["solar_mj"].dtype == np.float32


def test_gee_feature_parser_keeps_masked_values_as_nan() -> None:
    features = json.loads((FIXTURES / "gee_features.json").read_text())
    frame = gee_satellite.parse_features(features, "ndvi_raw")
    assert len(frame) == 3
    assert frame.loc[0, "ndvi_raw"] == pytest.approx(5500.5)
    assert frame["ndvi_raw"].isna().sum() == 2
    assert frame["date"].dtype.kind == "M"


def test_gee_auth_check_fails_cleanly_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEE_PROJECT", "")
    ok, message = gee_auth_check.check_auth()
    assert not ok and "GEE_PROJECT" in message
    assert (
        "earthengine authenticate" in gee_auth_check.INSTRUCTIONS
        or "ee authenticate" in gee_auth_check.INSTRUCTIONS
    )


# ---------------------------------------------------------------- weather assembly
def _tiny_districts() -> pd.DataFrame:
    return pd.DataFrame(
        {"district_id": ["d1", "d2"], "centroid_lat": [11.0, 17.0], "centroid_lon": [78.0, 80.0]}
    )


def _frame(
    columns: dict[str, list[float]], district: str = "d1", start: str = "2020-01-01"
) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    data = {"district_id": district, "date": pd.date_range(start, periods=n, freq="D")}
    data.update({k: np.array(v, dtype="float32") for k, v in columns.items()})
    return pd.DataFrame(data)


def test_weather_prefers_imd_then_power_without_synthetic() -> None:
    days = 3  # 2020-01-01 .. 2020-01-03 inside a one-year grid is not needed: use a 3-day window
    districts = _tiny_districts().iloc[:1]
    imd_rain = _frame({"rain_mm": [1.0, np.nan, 3.0]})
    imd_temp = _frame({"tmax_c": [30.0, 31.0, np.nan], "tmin_c": [20.0, 21.0, 22.0]})
    power = _frame(
        {
            "rh_pct": [60.0] * days,
            "wind_ms": [2.0] * days,
            "solar_mj": [18.0] * days,
            "p_tmax_c": [99.0, 99.0, 33.0],
            "p_tmin_c": [99.0] * days,
            "p_rain_mm": [9.0, 2.0, 9.0],
        }
    )
    full = pd.date_range("2020-01-01", "2020-12-31")
    power = power.set_index("date").reindex(full).reset_index().rename(columns={"index": "date"})
    power["district_id"] = "d1"
    power[["rh_pct", "wind_ms", "solar_mj", "p_tmax_c", "p_tmin_c", "p_rain_mm"]] = power[
        ["rh_pct", "wind_ms", "solar_mj", "p_tmax_c", "p_tmin_c", "p_rain_mm"]
    ].fillna(5.0)
    weather, provenance = assemble_weather(
        districts, 2020, 2020, imd_rain, imd_temp, power, allow_synthetic=False
    )
    first_three = weather.set_index("date").loc["2020-01-01":"2020-01-03"]
    assert list(first_three["rain_mm"]) == [1.0, 2.0, 3.0]  # IMD, POWER fill, IMD
    assert list(first_three["tmax_c"]) == [30.0, 31.0, 33.0]  # IMD, IMD, POWER fill
    assert not weather["is_synthetic"].any()
    assert provenance["rain_mm"]["imd"] > 0 and provenance["rain_mm"]["nasa_power"] > 0
    assert provenance["rain_mm"]["synthetic"] == 0.0


def test_weather_falls_back_to_flagged_synthetic_and_keeps_tmin_below_tmax() -> None:
    districts = _tiny_districts()
    weather, provenance = assemble_weather(
        districts, 2019, 2019, None, None, None, allow_synthetic=True, seed=42
    )
    assert len(weather) == 2 * 365
    assert weather["is_synthetic"].all()
    assert provenance["rain_mm"]["synthetic"] == pytest.approx(1.0)
    assert (weather["tmin_c"] <= weather["tmax_c"]).all()
    again, _ = assemble_weather(
        districts, 2019, 2019, None, None, None, allow_synthetic=True, seed=42
    )
    pd.testing.assert_frame_equal(weather, again)  # seeded -> reproducible


# ---------------------------------------------------------------- synthetic generators
def test_synthetic_weather_is_monsoon_shaped_and_physical() -> None:
    weather = generate_weather(_tiny_districts(), 2000, 2004, seed=42)
    assert weather["is_synthetic"].all()
    assert weather["rain_mm"].between(0, 1000).all()
    assert weather["tmax_c"].between(0, 55).all()
    monthly = weather.assign(month=weather["date"].dt.month).groupby("month")["rain_mm"].sum()
    assert monthly.loc[[6, 7, 8, 9]].sum() > 0.5 * monthly.sum()  # SW monsoon dominates
    assert weather.groupby(weather["date"].dt.month)["tmax_c"].mean().idxmax() in (4, 5, 6)


def test_synthetic_satellite_follows_rainfall_with_lag() -> None:
    weather = generate_weather(_tiny_districts().iloc[:1], 2000, 2005, seed=1)
    sat = generate_satellite(weather, 2001, 2004, seed=1)
    assert sat["is_synthetic"].all()
    assert sat["ndvi"].between(-0.2, 1.0).all() and sat["soil_moisture"].between(0, 1).all()
    wet = sat[sat["date"].dt.month.isin([8, 9])]["ndvi"].mean()
    dry = sat[sat["date"].dt.month.isin([3, 4])]["ndvi"].mean()
    assert wet > dry


def test_synthetic_crops_are_flagged_and_reproducible() -> None:
    districts = _tiny_districts()
    weather = generate_weather(districts, 2005, 2012, seed=3)
    crops_a = generate_crops(districts, weather, None, 2006, 2011, seed=3)
    crops_b = generate_crops(districts, weather, None, 2006, 2011, seed=3)
    pd.testing.assert_frame_equal(crops_a, crops_b)
    assert crops_a["is_synthetic"].all()
    assert (crops_a["yield_t_per_ha"] > 0).all()
    assert set(crops_a.columns) >= set(crop_yield.CROP_COLUMNS)


# ---------------------------------------------------------------- validation
def _valid_tables() -> dict[str, pd.DataFrame]:
    districts = pd.DataFrame(
        {
            "district_id": ["d1", "d2"],
            "centroid_lat": [11.0, 17.0],
            "centroid_lon": [78.0, 80.0],
            "area_km2": [5000.0, 7000.0],
        }
    )
    weather = pd.concat(
        [
            _frame(
                {
                    "rain_mm": [0.0, 5.0],
                    "tmax_c": [30.0, 31.0],
                    "tmin_c": [20.0, 21.0],
                    "rh_pct": [60, 61],
                    "wind_ms": [2, 2],
                    "solar_mj": [18, 17],
                },
                d,
            )
            for d in ("d1", "d2")
        ],
        ignore_index=True,
    )
    weather["is_synthetic"] = False
    satellite = pd.DataFrame(
        {
            "district_id": ["d1"],
            "date": [pd.Timestamp("2020-01-01")],
            "ndvi": [0.5],
            "soil_moisture": [0.3],
            "is_synthetic": [False],
        }
    )
    crops = pd.DataFrame(
        {
            "district_id": ["d1", "d2"],
            "year": [2010, 2010],
            "season": ["kharif", "kharif"],
            "crop": ["rice", "rice"],
            "area_ha": [100.0, 200.0],
            "production_t": [300.0, 500.0],
            "yield_t_per_ha": [3.0, 2.5],
            "yield_flag": [False, False],
            "is_synthetic": [False, False],
        }
    )
    return {
        "districts": districts,
        "weather_daily": weather,
        "satellite_16d": satellite,
        "crops_annual": crops,
    }


def test_validation_passes_on_clean_tables() -> None:
    checks = validate.run_checks(_valid_tables(), {"join_coverage": 1.0})
    failed = [c for c in checks if not c.passed]
    assert not failed, failed


def test_validation_flags_duplicates_out_of_range_orphans_and_low_coverage() -> None:
    tables = _valid_tables()
    weather = tables["weather_daily"].copy()
    weather.loc[0, "rain_mm"] = 1500.0  # impossible daily rain
    weather.loc[1, "tmax_c"] = 60.0  # impossible Tmax
    weather = pd.concat([weather, weather.iloc[[2]]], ignore_index=True)  # duplicate key
    weather.loc[3, "district_id"] = "ghost"  # orphan
    tables["weather_daily"] = weather
    crops = tables["crops_annual"].copy()
    crops["district_id"] = ["x1", "x2"]  # nothing joins
    tables["crops_annual"] = crops
    checks = validate.run_checks(tables, {"join_coverage": 0.2})
    failed = {(c.table, c.name.split(" ")[0]) for c in checks if not c.passed}
    assert ("weather_daily", "unique") in failed
    assert ("weather_daily", "rain_mm") in failed
    assert ("weather_daily", "tmax_c") in failed
    assert ("weather_daily", "district_id") in failed
    assert ("crops_annual", "join") in failed


def test_data_quality_report_mentions_synthetic_share() -> None:
    tables = _valid_tables()
    tables["satellite_16d"]["is_synthetic"] = True
    checks = validate.run_checks(tables, {"join_coverage": 1.0})
    text = validate.render_report(
        tables,
        checks,
        {"sources": ["fixture"], "coverage_by_source": {"fixture": 1.0}, "unmatched": []},
        {},
        "note",
    )
    assert "WARNING: parts of the data are synthetic" in text
    assert "satellite_16d" in text


def test_filter_unreliable_drops_contradictory_and_implausible_rows() -> None:
    settings = {
        "etl": {
            "max_yield_mismatch": 0.5,
            "max_plausible_yield_t_ha": {"wheat": 4.5},
            "min_plausible_yield_t_ha": {"wheat": 0.1},
        }
    }
    rows = pd.DataFrame(
        {
            "district_id": ["a", "b", "c", "d", "e"],
            "year": [2010] * 5,
            "season": ["rabi"] * 5,
            "crop": ["wheat", "wheat", "wheat", "rice", "wheat"],
            "area_ha": [100.0, 100.0, 100.0, 100.0, 100.0],
            "production_t": [120.0, 0.05, 480.0, 900.0, 120.0],
            "yield_t_per_ha": [1.2, 0.0005, 4.8, 9.0, 1.2],
            "yield_flag": [False, True, False, False, True],
            "is_synthetic": [False] * 5,
            "mismatch_rel": [0.01, 0.0, 0.0, np.nan, 0.8],
        }
    )
    kept, dropped = crop_yield.filter_unreliable(rows, settings)
    assert list(kept["district_id"]) == ["a", "d"]  # rice has no bound configured -> kept
    assert dict(zip(dropped["district_id"], dropped["reason"], strict=True)) == {
        "b": "implausibly low yield",
        "c": "implausibly high yield",
        "e": "reported yield disagrees with production/area",
    }
    assert "mismatch_rel" not in kept
