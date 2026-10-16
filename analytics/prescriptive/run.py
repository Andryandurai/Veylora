"""Command line: irrigation plan and crop ranking for one district-season.

    python -m analytics.prescriptive.run --district tamil_nadu__thanjavur --crop rice --season kharif --budget 350

Prints the weekly LP schedule, a strategy comparison scored by the daily water-balance simulator, the yield
estimate (ML potential x water-stress ratio) and the crop-switch ranking. All outputs are decision-support
estimates; see the provenance notes in config/crop_params.yaml and config/economics.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from analytics.prescriptive import crop_switch
from analytics.prescriptive import scenario as S
from analytics.prescriptive.params import load_crop

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m analytics.prescriptive.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--district", required=True, help="district_id, e.g. tamil_nadu__thanjavur")
    parser.add_argument("--crop", required=True, choices=["rice", "wheat", "cotton", "sugarcane"])
    parser.add_argument("--season", required=True, choices=["kharif", "rabi"])
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="seasonal irrigation budget in mm (default: per-crop value in crop_params.yaml)",
    )
    parser.add_argument(
        "--weekly-cap", type=float, default=None, help="maximum irrigation per week in mm"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="reference weather year (default: the typical training year)",
    )
    parser.add_argument("--rain-multiplier", type=float, default=1.0)
    parser.add_argument("--temperature-delta", type=float, default=0.0)
    parser.add_argument(
        "--soil", default=None, help="soil class (sandy_loam, loam, clay_loam, clay)"
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="skip the yield forecast and crop ranking (water balance only)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of text"
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Compute the plan; returns a JSON-serialisable dict."""
    budget = args.budget if args.budget is not None else load_crop(args.crop).default_budget_mm
    inputs = S.load_season_inputs(
        args.district,
        args.crop,
        args.season,
        args.year,
        args.rain_multiplier,
        args.temperature_delta,
        args.soil,
    )
    table, details = S.evaluate_strategies(inputs, budget, args.weekly_cap)
    out: dict[str, Any] = {
        "district": args.district,
        "crop": args.crop,
        "season": args.season,
        "weather_year": inputs.year,
        "rain_multiplier": args.rain_multiplier,
        "temperature_delta": args.temperature_delta,
        "budget_mm": budget,
        "taw_mm": inputs.taw,
        "lp_solve_seconds": details["lp"].solve_seconds,
        "strategies": table.drop(columns=["weekly_irrigation_mm"]).round(3).to_dict("records"),
        "weekly_irrigation_mm": {
            r["strategy"]: r["weekly_irrigation_mm"] for r in table.to_dict("records")
        },
    }
    if not args.no_ml:
        try:
            from analytics.predictive.predict import predict_yield

            ml = predict_yield(
                args.district,
                args.crop,
                args.season,
                {"temperature_delta": args.temperature_delta},
                year=inputs.year,
            )
            yields = S.strategy_yields(inputs, budget, ml, args.weekly_cap)
            out["ml_potential_t_ha"] = {k: ml[k] for k in ("yield_t_ha", "lo", "hi")}
            out["strategies"] = (
                yields.drop(columns=["weekly_irrigation_mm"]).round(3).to_dict("records")
            )
            ranking = crop_switch.recommend(
                args.district,
                args.season,
                {
                    "rain_multiplier": args.rain_multiplier,
                    "temperature_delta": args.temperature_delta,
                },
                year=inputs.year,
            )
            out["crop_ranking"] = ranking.round(1).to_dict("records")
            out["crop_ranking_note"] = crop_switch.DISCLAIMER
        except Exception as exc:  # models/data missing: still deliver the water-balance part
            LOG.warning("yield forecast and crop ranking unavailable: %s", exc)
            out["ml_error"] = str(exc)
    return out


def print_text(result: dict[str, Any]) -> None:
    """Human-readable report on stdout."""
    pd.set_option("display.width", 200)
    print(
        f"{result['district']} | {result['crop']} {result['season']} | weather year {result['weather_year']} | rain x{result['rain_multiplier']} | dT {result['temperature_delta']:+.1f} C"
    )  # noqa: T201
    print(
        f"budget {result['budget_mm']:.0f} mm, root-zone TAW {result['taw_mm']:.0f} mm, LP solved in {result['lp_solve_seconds']:.2f} s\n"
    )  # noqa: T201
    frame = pd.DataFrame(result["strategies"])
    columns = [
        c
        for c in (
            "strategy",
            "water_used_mm",
            "eta_over_etc",
            "rel_yield",
            "water_saved_vs_baseline_pct",
            "yield_t_ha",
            "yield_lo",
            "yield_hi",
        )
        if c in frame
    ]
    print(frame[columns].to_string(index=False))  # noqa: T201
    print(
        "\noptimized weekly irrigation (mm):",
        np.round(result["weekly_irrigation_mm"]["optimized"], 0).astype(int).tolist(),
    )  # noqa: T201
    if "crop_ranking" in result:
        print("\ncrop ranking by expected net return (INR/ha):")  # noqa: T201
        print(
            pd.DataFrame(result["crop_ranking"])[
                [
                    "crop",
                    "yield_mid_t_ha",
                    "net_mid_inr_ha",
                    "net_lower_inr_ha",
                    "rank_mid",
                    "rank_lower",
                ]
            ].to_string(index=False)
        )  # noqa: T201
        print("(" + result["crop_ranking_note"] + ")")  # noqa: T201
    if "ml_error" in result:
        print("\nyield forecast unavailable:", result["ml_error"])  # noqa: T201


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        result = run(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if args.json:
        print(json.dumps(result, indent=1, default=float))  # noqa: T201
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
